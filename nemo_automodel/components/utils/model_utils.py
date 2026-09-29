# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import inspect
import logging
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from nemo_automodel.shared.import_utils import safe_import
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

HAVE_TORCHAO, torch_ao = safe_import("torchao")


import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _get_cached_forward_signature(forward_callable: Callable[..., Any]) -> inspect.Signature | None:
    """Best-effort retrieval cached by the underlying ``forward`` callable."""
    try:
        return inspect.signature(forward_callable)
    except (ValueError, TypeError):
        return None


def _get_forward_signature(model: nn.Module) -> inspect.Signature | None:
    """Retrieve ``model.forward`` once per callable, preserving live patches."""
    forward = getattr(model, "forward", None)
    if not callable(forward):
        return None
    # Cache only ordinary bound methods. Caching callable instances or partials
    # can retain their owning module (and GPU parameters), while partialmethod
    # may create a fresh callable on every attribute access.
    class_forward = inspect.getattr_static(type(model), "forward", None)
    if inspect.ismethod(forward) and getattr(forward, "__func__", None) is class_forward:
        signature = _get_cached_forward_signature(forward.__func__)
        if signature is None:
            return None
        parameters = tuple(signature.parameters.values())
        return signature.replace(parameters=parameters[1:]) if parameters else signature

    try:
        return inspect.signature(forward)
    except (ValueError, TypeError):
        return None


def _supports_seq_lens(model: nn.Module) -> bool:
    """
    Check if the model's forward() accepts seq_lens.

    Returns True if:
    - forward() has an explicit `seq_lens` parameter, OR
    - forward() has **kwargs (so it won't crash if seq_lens is passed)

    Returns False otherwise (passing seq_lens would cause "unexpected kwarg" error).
    """
    sig = _get_forward_signature(model)
    if sig is None:
        return False
    params = sig.parameters
    # Check for explicit seq_lens parameter
    if "seq_lens" in params:
        return True
    # Check for **kwargs (VAR_KEYWORD)
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


# Umbrella of multimodal kwarg names used by VLM forwards across families.
# Mirrors the discovery pattern of ``fake_image._VISION_TOKEN_ID_ATTRS``: every
# consumer iterates this list and uses whichever keys the live batch contains,
# so callers don't need a per-model branch. New VLM family -> append its keys.
VLM_INPUT_KEYS: tuple[str, ...] = (
    "input_ids",
    # Image
    "pixel_values",
    "image_flags",
    "imgs_sizes",
    "image_position_ids",
    "mm_token_type_ids",
    "image_grid_hws",
    "image_grid_thw",
    "image_sizes",
    "patch_pixel_values",
    "num_patches",
    "patch_newline_mask",
    "image_embeds",
    # Video
    "pixel_values_videos",
    "video_grid_thw",
    # Audio / sound
    "sound_features",
    "sound_attention_mask",
    "audio_input_values",
    "audio_attention_mask",
)


def filter_forward_kwargs(model: nn.Module, kwargs: dict) -> dict:
    """Drop kwargs that ``model.forward`` does not accept.

    If the model exposes ``**kwargs`` or its signature cannot be inspected, the
    input kwargs are returned unchanged. The original dict is never mutated.
    """
    sig = _get_forward_signature(model)
    if sig is None:
        return dict(kwargs)

    params = sig.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)

    allowed = set(params.keys())
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        logger.debug("Dropping unsupported forward kwargs for %s: %s", type(model).__name__, dropped)
    return filtered


@torch.no_grad()
def _get_model_param_stats(model: nn.Module) -> tuple[int, int, float]:
    """
    Get the number of trainable parameters and the L2 norm of the model.

    Args:
        model: Model to analyze

    Returns:
        total_params: int
        trainable_params: int
        local_sq_norm: float
    """
    total_params = 0
    trainable_params = 0
    local_sq_norm = 0.0

    for p in model.parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
        try:
            local_sq_norm += p.detach().norm(2) ** 2
        except Exception:
            pass
    if isinstance(local_sq_norm, torch.Tensor):
        local_sq_norm = local_sq_norm.item()
    return total_params, trainable_params, local_sq_norm


@contextmanager
def skip_random_init():
    """
    Context manager to skip random weight initialization when loading pretrained models.
    """
    try:
        mod = __import__("transformers.initialization", fromlist=["_init_weights"])
    except ImportError:
        mod = __import__("transformers.modeling_utils", fromlist=["_init_weights"])
    prev = getattr(mod, "_init_weights", True)
    mod._init_weights = False
    try:
        yield
    finally:
        mod._init_weights = prev


def resolve_trust_remote_code(pretrained_model_name_or_path):
    """
    Whitelist NVIDIA models to allow remote code execution.

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model.

    Returns:
        bool: True if the model should be loaded with trust_remote_code, False otherwise.
    """
    if not pretrained_model_name_or_path:
        return False
    # pretrained_model_name_or_path can be something like nvidia/NVIDIA-Nemotron-Nano-9B-v2
    return not os.path.isdir(pretrained_model_name_or_path) and pretrained_model_name_or_path.startswith("nvidia/")


@torch.no_grad()
def print_trainable_parameters(model: nn.Module, name: str = "Model") -> tuple[int, int]:
    """Print the number of trainable parameters in the model.

    Args:
        model: Model to analyze
        name: Label for the summary header (e.g. ``"Draft"`` to distinguish the
            draft model from the target in speculative-decoding training).

    Returns:
        trainable_params: int
        total_params: int
    """
    total_params, trainable_params, local_sq_norm = _get_model_param_stats(model)

    try:
        # TODO(@akoumparouli): make this sharding aware.
        local_sq_norm = float(local_sq_norm**0.5)
        trainable_pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

        logging.info(f"{name} summary:")
        logging.info("--------------------------------")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters percentage: {trainable_pct:.2f}%")
        logging.info(f"Param L2 norm: {local_sq_norm:.4f}")
        logging.info("--------------------------------")
    except Exception:
        logging.info(f"{name} summary: <unavailable>")

    return trainable_params, total_params


def _freeze_module_by_attribute_and_patterns(model, attribute_name, name_patterns):
    """Freeze a legacy model attribute and modules matching name substrings."""
    if attribute_name is not None and hasattr(model, attribute_name):
        getattr(model, attribute_name).requires_grad_(False)

    for name, module in model.named_modules():
        if any(pattern in name.lower() for pattern in name_patterns):
            module.requires_grad_(False)


@dataclass(frozen=True)
class ModuleSelector:
    """Select modules by exact canonical path or case-sensitive shell-style glob.

    Exactly one of ``path`` or ``glob`` must be set. Both match against the full
    canonical module path (activation-checkpoint and ``torch.compile`` wrapper
    components stripped): ``path`` is an exact match, while ``glob`` uses
    :func:`fnmatch.fnmatchcase` semantics where ``*`` also crosses ``.``
    separators. Matching is recursive: a selected module's entire subtree is
    frozen or unfrozen.
    """

    path: str | None = None
    glob: str | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one matching mode is configured."""
        if (self.path is None) == (self.glob is None):
            raise ValueError(
                f"ModuleSelector requires exactly one of `path` or `glob`; got path={self.path!r}, glob={self.glob!r}."
            )
        value = self.path if self.path is not None else self.glob
        if not isinstance(value, str) or not value:
            raise ValueError(f"ModuleSelector values must be non-empty strings; got {value!r}.")

    def matches(self, module_path: str) -> bool:
        """Return whether this selector matches a canonical module path.

        Args:
            module_path: Canonical fully qualified module path.

        Returns:
            Whether the selector matches ``module_path``.
        """
        if self.path is not None:
            return module_path == self.path
        assert self.glob is not None
        return fnmatch.fnmatchcase(module_path, self.glob)

    def describe(self) -> str:
        """Return the selector in its ``key: value`` configuration form."""
        key, value = ("path", self.path) if self.path is not None else ("glob", self.glob)
        return f"{key}: {value}"


@dataclass(frozen=True)
class FreezeConfig:
    """Typed schema for the ``freeze_config`` recipe section.

    Trainability semantics, in application order:

    1. Full fine-tuning preserves the model's existing trainability state;
       PEFT establishes a LoRA-trainable, base-frozen baseline.
    2. ``freeze_modules`` recursively freezes the selected modules.
    3. ``unfreeze_modules`` recursively unfreezes the selected modules and wins
       on overlap.
    4. Framework-required freezes (dead K/V projections, indexer parameters)
       are applied by the infrastructure afterwards and remain protected.
    5. The final trainable parameter set is validated before the optimizer is
       constructed.

    The legacy modality booleans remain supported. ``freeze_vision_tower``
    defaults to ``True`` only for legacy-only configurations; a configuration
    that declares ``freeze_modules`` or ``unfreeze_modules`` uses
    explicit-selector semantics and does not implicitly freeze vision modules.
    """

    freeze_modules: list[ModuleSelector] | None = None
    unfreeze_modules: list[ModuleSelector] | None = None
    freeze_vision_tower: bool | None = None
    freeze_audio_tower: bool = False
    freeze_language_model: bool = False
    freeze_video_embedder: bool = False

    def __post_init__(self) -> None:
        """Validate selectors and legacy compatibility options."""
        for field_name in ("freeze_modules", "unfreeze_modules"):
            selectors = getattr(self, field_name)
            if selectors is None:
                continue
            if not isinstance(selectors, list):
                raise TypeError(f"FreezeConfig.{field_name} must be a list; got {type(selectors).__name__}.")
            invalid = [selector for selector in selectors if not isinstance(selector, ModuleSelector)]
            if invalid:
                raise TypeError(f"FreezeConfig.{field_name} entries must be ModuleSelector instances; got {invalid!r}.")

        if self.freeze_vision_tower is not None and not isinstance(self.freeze_vision_tower, bool):
            raise TypeError(
                f"FreezeConfig.freeze_vision_tower must be a boolean or None; got {self.freeze_vision_tower!r}."
            )
        for field_name in ("freeze_audio_tower", "freeze_language_model", "freeze_video_embedder"):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(f"FreezeConfig.{field_name} must be a boolean; got {value!r}.")

    def has_generic_selectors(self) -> bool:
        """Return whether either generic selector field was explicitly declared."""
        return self.freeze_modules is not None or self.unfreeze_modules is not None


_FREEZE_CONFIG_OPTIONS = {
    "freeze_modules",
    "unfreeze_modules",
    "freeze_vision_tower",
    "freeze_audio_tower",
    "freeze_language_model",
    "freeze_video_embedder",
}


def _parse_module_selector(entry: Any, *, field_name: str) -> ModuleSelector:
    """Parse one ``freeze_modules``/``unfreeze_modules`` entry into a ModuleSelector.

    Args:
        entry: Mapping with exactly one of ``path`` or ``glob``, or an existing
            ModuleSelector.
        field_name: Owning list field name, used in error messages.

    Returns:
        The validated ModuleSelector.

    Raises:
        ValueError: If the entry is not a ``{path: ...}`` / ``{glob: ...}``
            mapping or contains unknown keys.
    """
    if isinstance(entry, ModuleSelector):
        return entry
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"freeze_config.{field_name} entries must be mappings with exactly one of `path` or `glob`, "
            f"e.g. `- path: vision_tower` or `- glob: '*.audio_encoder'`; got {entry!r}."
        )
    unknown = set(entry) - {"path", "glob"}
    if unknown:
        raise ValueError(
            f"freeze_config.{field_name} entry {dict(entry)!r} has unsupported key(s) {sorted(unknown)}; "
            "use exactly one of `path` or `glob`."
        )
    return ModuleSelector(path=entry.get("path"), glob=entry.get("glob"))


def parse_freeze_config(config: FreezeConfig | Mapping[str, Any] | None) -> FreezeConfig | None:
    """Validate raw ``freeze_config`` YAML data and return a typed FreezeConfig.

    Args:
        config: A FreezeConfig (returned unchanged), a raw mapping from the
            recipe config, or None.

    Returns:
        The validated FreezeConfig, or None when no freeze configuration was
        provided.

    Raises:
        TypeError: If ``config`` is neither a mapping nor a FreezeConfig.
        ValueError: If ``config`` contains unknown options, malformed
            selectors, or non-boolean legacy options.
    """
    if config is None or isinstance(config, FreezeConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError(f"freeze_config must be a mapping of freeze options; got {type(config).__name__}: {config!r}.")

    unknown = set(config) - _FREEZE_CONFIG_OPTIONS
    if unknown:
        raise ValueError(
            f"freeze_config has unsupported option(s) {sorted(unknown)}; "
            f"valid options are {sorted(_FREEZE_CONFIG_OPTIONS)}."
        )

    def _selectors(field_name: str) -> list[ModuleSelector] | None:
        if field_name not in config:
            return None
        entries = config[field_name]
        if not isinstance(entries, list):
            raise ValueError(
                f"freeze_config.{field_name} must be a list of `path`/`glob` selector mappings; got {entries!r}."
            )
        return [_parse_module_selector(entry, field_name=field_name) for entry in entries]

    legacy_booleans = {}
    for option in ("freeze_vision_tower", "freeze_audio_tower", "freeze_language_model", "freeze_video_embedder"):
        value = config.get(option, None)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"freeze_config.{option} must be a boolean; got {value!r}.")
        legacy_booleans[option] = value

    return FreezeConfig(
        freeze_modules=_selectors("freeze_modules"),
        unfreeze_modules=_selectors("unfreeze_modules"),
        **legacy_booleans,
    )


def enable_radio_vit_fused_attn(model):
    """Route RADIO ViT attention through ``F.scaled_dot_product_attention``.

    RADIO's timm Attention blocks default to ``fused_attn=False``, which
    materializes the full ``(B, H, seq, seq)`` attention tensor (~5 GiB per
    block at RADIO-v2-H + dynamic-resolution patch counts). Flipping
    ``fused_attn=True`` matches the Megatron-Bridge path which sets
    ``vision_config.use_flash_attn=True`` via
    ``attn_implementation="flash_attention_2"``.

    No-op when the model has no RADIO vision tower.

    Args:
        model: The model to patch in place.
    """
    vision_model = getattr(model, "vision_model", None) or getattr(getattr(model, "model", None), "vision_model", None)
    vit_blocks = getattr(
        getattr(getattr(vision_model, "radio_model", None), "model", None),
        "blocks",
        None,
    )
    if vit_blocks is None:
        return

    flipped = 0
    for block in vit_blocks:
        attn = getattr(block, "attn", None)
        if attn is not None:
            attn.fused_attn = True
            flipped += 1
    if flipped:
        logger.info("Enabled fused_attn on %d RADIO ViT blocks", flipped)


def _apply_module_selectors(
    model: nn.Module,
    selectors: list[ModuleSelector],
    *,
    requires_grad: bool,
    strict: bool,
    field_name: str,
) -> None:
    """Set ``requires_grad`` on every module matched by ``selectors``, recursively.

    Module paths are canonicalized (activation-checkpoint and ``_orig_mod``
    wrapper components stripped) so selectors keep matching after model surgery
    that wraps or renames parameter-holding modules.

    Args:
        model: The model (or pipeline-parallel model part) to modify.
        selectors: Typed selectors to resolve against the module hierarchy.
        requires_grad: Trainability to apply to matched modules.
        strict: When True, raise if a selector matches no parameters. Rebinding
            after parallelization uses False because sharding may relocate or
            regroup the selected modules (e.g. pipeline stages hold only a part
            of the model).
        field_name: Owning FreezeConfig field name, used in error messages.

    Raises:
        ValueError: If ``strict`` and a selector matches no parameters.
    """
    named_modules = [
        (canonical_parameter_fqn(name).replace("_orig_mod.", ""), module)
        for name, module in model.named_modules(remove_duplicate=False)
        if name
    ]
    for selector in selectors:
        matched_params = 0
        for name, module in named_modules:
            if selector.matches(name):
                matched_params += sum(1 for _ in module.parameters())
                module.requires_grad_(requires_grad)
        if strict and matched_params == 0:
            raise ValueError(
                f"freeze_config.{field_name} selector `{selector.describe()}` matched no parameters in the model. "
                "Selectors must match the full canonical module path."
            )


def apply_parameter_freezing(
    model: nn.Module,
    freeze_config: FreezeConfig | Mapping[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Apply parameter freezing based on a typed FreezeConfig.

    Application order: legacy modality booleans and ``freeze_modules`` freeze,
    then ``unfreeze_modules`` unfreezes and wins on overlap. Framework-required
    freezes (dead K/V projections, indexer parameters) are applied separately
    by the infrastructure after this function, before and after sharding.

    Args:
        model: The model to apply freezing to.
        freeze_config: Typed freeze configuration or a raw mapping retained for
            compatibility with direct callers. Raw mappings are validated and
            converted to FreezeConfig before use.
        strict: When True, raise if a ``freeze_modules``/``unfreeze_modules``
            selector matches no parameters. Set False when rebinding the policy
            onto a post-parallelization model part.

    Legacy modality booleans:
        - freeze_vision_tower: bool (default True in legacy-only configurations)
        - freeze_audio_tower: bool (default False)
        - freeze_language_model: bool (default False)
        - freeze_video_embedder: bool (default False)
    """
    parsed_freeze_config = parse_freeze_config(freeze_config)
    if parsed_freeze_config is None:
        return
    freeze_config = parsed_freeze_config

    # Preserve the legacy attribute and substring matching independently of the
    # fnmatch semantics used by the generic selectors.
    freeze_vision_tower = freeze_config.freeze_vision_tower
    if freeze_vision_tower is None:
        # Explicit-selector configurations establish trainability solely through
        # freeze_modules/unfreeze_modules and skip the legacy implicit vision freeze.
        freeze_vision_tower = not freeze_config.has_generic_selectors()
    # freeze_video_embedder: NemotronOmni RADIO's patch_generator.video_embedder is only
    # exercised on video inputs; on image-only training it sits in the optimizer without
    # state (no grad -> no lazy init), so dcp.load on resume raises
    # "Missing key in checkpoint state_dict: optim.state.<...>.video_embedder.weight.step".
    # Independent of freeze_vision_tower so the image encoder can stay trainable
    # while the video branch is frozen out.
    legacy_freeze_aliases = (
        (freeze_vision_tower, "vision_tower", ("vision", "visual", "image_encoder")),
        (freeze_config.freeze_audio_tower, "audio_tower", ("audio", "audio_encoder", "speech", "sound")),
        (freeze_config.freeze_language_model, "language_model", ("language", "text", "llm")),
        (freeze_config.freeze_video_embedder, None, ("patch_generator.video_embedder",)),
    )
    for enabled, attribute_name, name_patterns in legacy_freeze_aliases:
        if enabled:
            _freeze_module_by_attribute_and_patterns(model, attribute_name, name_patterns)

    _apply_module_selectors(
        model, freeze_config.freeze_modules or [], requires_grad=False, strict=strict, field_name="freeze_modules"
    )
    _apply_module_selectors(
        model, freeze_config.unfreeze_modules or [], requires_grad=True, strict=strict, field_name="unfreeze_modules"
    )

    # Phi4MM: cast internal fp32 LoRA adapters to bf16 for FSDP2 compatibility,
    # and disable KV cache (remote code uses legacy DynamicCache.key_cache
    # attribute removed in transformers v5.x).
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type == "phi4mm":
        cast_mixed_dtype_params_to_bf16(model)
        if hasattr(model, "config"):
            model.config.use_cache = False


def freeze_deepseek_v4_indexer_params(model):
    """Freeze DeepSeek V4 indexer params that only feed discrete top-k masks."""
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "deepseek_v4":
        return

    frozen_count = 0
    for name, param in model.named_parameters():
        if ".self_attn.compressor.indexer." in name:
            param.requires_grad_(False)
            frozen_count += 1

    if frozen_count > 0:
        logger.info("Froze %d DeepSeek V4 indexer parameters.", frozen_count)


def cast_mixed_dtype_params_to_bf16(model):
    """Cast fp32 parameters and buffers to bf16 for FSDP2 compatibility."""
    for p in model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    for b in model.buffers():
        if b.dtype == torch.float32:
            b.data = b.data.to(torch.bfloat16)


def squeeze_input_for_thd(input_ids, position_ids, padding_mask, attn_kwargs, seqlens_padding_value=-1000):
    """
    Squeeze batch dimension and prepare inputs for THD (total, hidden, depth) format.

    This function removes the batch dimension from input tensors and processes attention
    kwargs for use with Transformer Engine's THD format. It's typically used when the
    batch has already been converted to THD format (with batch_size=1 as a placeholder
    dimension) and that dimension needs to be removed.

    The function performs three key operations:
    1. Removes the batch dimension (dim 0) from input tensors
    2. Filters out padding values from cumulative sequence length tensors
    3. Converts max_seqlen from tensor to scalar if needed

    Args:
        input_ids (torch.Tensor or None): Input token IDs with shape [1, total_tokens]
            or [1, total_tokens, hidden_dim]. The first dimension will be squeezed.
            ``None`` is permitted when the caller is feeding the model via
            ``inputs_embeds`` instead — embeddings are squeezed inside the model
            forward (the ``squeezed_for_thd`` branch in ``NemotronHModel.forward``
            and analogous code paths), so this helper has nothing to squeeze and
            simply returns ``None`` for the ``input_ids`` slot.
        position_ids (torch.Tensor): Position IDs with shape [1, total_tokens].
            The first dimension will be squeezed.
        padding_mask (torch.Tensor): Padding mask with shape [1, total_tokens].
            The first dimension will be squeezed.
        attn_kwargs (dict): Dictionary of attention-related tensors. May contain:
            - cu_seqlens: Cumulative sequence lengths [1, num_seqs+1]
            - cu_seqlens_padded: Cumulative padded sequence lengths [1, num_seqs+1]
            - max_seqlen: Maximum sequence length (tensor or int)
            - Other attention parameters (will be squeezed if tensors)
        seqlens_padding_value (int): Sentinel value used to indicate padding in
            cu_seqlens and cu_seqlens_padded tensors. These values will be filtered
            out. Default: -1000.

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Input IDs with batch dimension removed [total_tokens]
                or [total_tokens, hidden_dim]
            - position_ids (torch.Tensor): Position IDs with batch dimension removed [total_tokens]
            - padding_mask (torch.Tensor): Padding mask with batch dimension removed [total_tokens]
            - attn_kwargs (dict): Updated attention kwargs with:
                - Batch dimensions removed from all tensor values
                - Padding values filtered from cu_seqlens and cu_seqlens_padded
                - max_seqlen converted to scalar if it was a tensor

    Example:
        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [1, 5]
        >>> position_ids = torch.tensor([[0, 1, 2, 3, 4]])  # [1, 5]
        >>> padding_mask = torch.tensor([[False, False, False, False, False]])  # [1, 5]
        >>> attn_kwargs = {
        ...     'cu_seqlens': torch.tensor([[0, 3, 5, -1000]]),  # [1, 4] with padding
        ...     'cu_seqlens_padded': torch.tensor([[0, 3, 5, -1000]]),
        ...     'max_seqlen': torch.tensor([3])
        ... }
        >>> ids, pos, mask, kwargs = squeeze_input_for_thd(
        ...     input_ids, position_ids, padding_mask, attn_kwargs
        ... )
        >>> ids.shape
        torch.Size([5])
        >>> kwargs['cu_seqlens']  # Padding value filtered out
        tensor([0, 3, 5])
        >>> kwargs['max_seqlen']  # Converted to scalar
        3

    Note:
        This function modifies attn_kwargs in-place. If you need to preserve the original
        dictionary, pass a copy.
    """
    if input_ids is not None:
        input_ids = input_ids.squeeze(0)
    position_ids = position_ids.squeeze(0)
    if isinstance(padding_mask, torch.Tensor):
        padding_mask = padding_mask.squeeze(0)
    for key, value in attn_kwargs.items():
        if isinstance(value, torch.Tensor):
            attn_kwargs[key] = value.squeeze(0)
        if key in ["cu_seqlens", "cu_seqlens_padded"]:
            attn_kwargs[key] = value[value != seqlens_padding_value].contiguous()
        if key == "max_seqlen" and isinstance(value, torch.Tensor):
            attn_kwargs[key] = value.item()

    return input_ids, position_ids, padding_mask, attn_kwargs


# taken and edited from https://github.com/huggingface/transformers/blob/32a58e31463e238c967207bf73772490c353551a/src/transformers/integrations/accelerate.py#L53-L158
@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.

    Example:

    ```python
    import torch.nn as nn
    from nemo_automodel.components.utils.model_utils import init_empty_weights

    with init_empty_weights():
        tst = nn.Linear(100, 100)  # on `cuda` device
    ```
    """
    device = torch.device("meta")
    fp8_parameter_mapping = {
        "_linear_mm_config": "linear_mm_config",
        "_dtype": "dtype",
        "_precomputed_scale": "precomputed_scale",
    }
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            if HAVE_TORCHAO and isinstance(
                module._parameters[name], torch_ao.float8.fsdp_utils.WeightWithDynamicFloat8CastTensor
            ):
                kwargs = {}
                for k in module._parameters[name].__dict__:
                    if k in fp8_parameter_mapping:
                        kwargs[fp8_parameter_mapping[k]] = getattr(module._parameters[name], k)
                is_hf_initialized = kwargs.pop("_is_hf_initialized", None)
            else:
                # Standard nn.Parameter only accepts requires_grad, not arbitrary __dict__ attributes
                # (e.g., TransformerEngine sets tensor_model_parallel on weights)
                if param_cls is nn.Parameter:
                    kwargs = {"requires_grad": param.requires_grad}
                    is_hf_initialized = None
                else:
                    kwargs = module._parameters[name].__dict__.copy()
                    kwargs["requires_grad"] = param.requires_grad
                    is_hf_initialized = kwargs.pop("_is_hf_initialized", None)
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            if is_hf_initialized is not None:
                setattr(module._parameters[name], "_is_hf_initialized", is_hf_initialized)

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
