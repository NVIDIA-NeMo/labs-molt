# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Model resolution and initialization helpers.

Functions for resolving which model class to use (custom vs HF), downloading
weights, applying config overrides, and instantiating the model.
"""

import gc
import glob
import inspect
import json
import logging
import os
import shutil
import threading
from contextlib import contextmanager
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import PretrainedConfig

try:
    from huggingface_hub.errors import StrictDataclassClassValidationError
except ImportError:
    StrictDataclassClassValidationError = ValueError
from transformers.modeling_utils import PreTrainedModel

# For models that still accesses config.pad_token_id after v5 removes it in PretrainedConfig
if not hasattr(PretrainedConfig, "pad_token_id"):
    PretrainedConfig.pad_token_id = None

# Shim: some trust_remote_code dLLM model code (e.g. Nemotron-Labs-Diffusion)
# imports check_model_inputs from transformers.utils.generic, which is not
# present in released transformers versions. Install a no-op fallback.
import transformers.utils.generic as _generic_utils

if not hasattr(_generic_utils, "check_model_inputs"):

    def _check_model_inputs(func):
        return func

    _generic_utils.check_model_inputs = _check_model_inputs

from nemo_automodel._transformers.auto_config import NeMoAutoConfig as AutoConfig
from nemo_automodel._transformers.utils import apply_qwen3_omni_config_patch

apply_qwen3_omni_config_patch()

import nemo_automodel.components.checkpoint.utils as checkpoint_utils
import nemo_automodel.components.distributed.utils as dist_utils
from nemo_automodel._transformers.registry import ModelRegistry, resolve_custom_config_cls
from nemo_automodel.components.distributed.init_utils import get_local_world_size_preinit, get_world_size_safe
from nemo_automodel.components.models.common.gated_delta_net_fp32 import (
    has_gated_delta_net_fp32_checkpoint_contract,
    is_gated_delta_net_fp32_param_key,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import (
    HUB_LOADING_KWARGS,
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
    restore_pretrained_generation_config,
)
from nemo_automodel.components.utils.model_utils import resolve_trust_remote_code, skip_random_init
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)

# Thread-local: when True, HF's get_init_context must not add torch.device("meta")
# so that model init runs on real device (used when retrying after "Cannot copy out of meta tensor").
_hf_meta_device_disabled = threading.local()


def _get_hf_meta_device_disabled():
    return getattr(_hf_meta_device_disabled, "value", False)


@contextmanager
def no_hf_meta_device():
    """Disable HuggingFace's meta device in get_init_context so model is built on real device."""
    prev = _get_hf_meta_device_disabled()
    _hf_meta_device_disabled.value = True
    try:
        yield
    finally:
        _hf_meta_device_disabled.value = prev


def _filter_meta_device_from_init_context(contexts):
    """Remove torch.device('meta') from HF init context list when we want real-device init."""
    return [c for c in contexts if not (isinstance(c, torch.device) and getattr(c, "type", None) == "meta")]


def _patched_get_init_context(cls, *args, **kwargs):
    """Wrapper around PreTrainedModel.get_init_context that strips meta device when requested."""
    original = _patched_get_init_context.__wrapped__
    contexts = original(cls, *args, **kwargs)
    if _get_hf_meta_device_disabled():
        return _filter_meta_device_from_init_context(contexts)
    return contexts


# Bind original and install patch (classmethod-safe)
_original_get_init_context = PreTrainedModel.get_init_context.__func__
_patched_get_init_context.__wrapped__ = _original_get_init_context
PreTrainedModel.get_init_context = classmethod(_patched_get_init_context)


def _get_mixin_wrapped_class(model_class: type) -> type:
    """
    Get a class that combines HFCheckpointingMixin with the original model class.

    If the class already has the mixin, returns it unchanged.

    Args:
        model_class: The original model class (e.g., LlamaForCausalLM)

    Returns:
        A class that inherits from both HFCheckpointingMixin and model_class
    """
    # Custom models already inherit HFCheckpointingMixin
    if issubclass(model_class, HFCheckpointingMixin):
        return model_class

    # Create wrapper class that looks identical to original
    return type(
        model_class.__name__,
        (HFCheckpointingMixin, model_class),
        {
            "__module__": model_class.__module__,
            "__qualname__": model_class.__qualname__,
        },
    )


@contextmanager
def local_torch_dtype(
    dtype: torch.dtype, model_class_name: str | None = None, default_dtype: torch.dtype = torch.bfloat16
):
    """
    Locally change the torch default dtype to `dtype`, and restore the old one upon exiting the context.
    If `model_class_name` is provided, it's used to provide a more helpful error message if `dtype` is not valid.
    """
    # Just a more helping error before we set `torch.set_default_dtype` later on which would crash in this case
    if isinstance(dtype, str):
        dtype = default_dtype
    if not dtype.is_floating_point:
        if model_class_name is not None:
            error_message = (
                f"{model_class_name} cannot be instantiated under `dtype={dtype}` as it's not a floating-point dtype"
            )
        else:
            error_message = f"Cannot set `{dtype}` as torch's default as it's not a floating-point dtype"
        raise ValueError(error_message)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def _propagate_torch_dtype_to_subconfigs(hf_config, torch_dtype: torch.dtype) -> None:
    """Recursively set ``torch_dtype`` on ``hf_config`` and all nested sub-configs.

    Multimodal configs (e.g. ``Gemma4ForConditionalGeneration``) hold nested
    sub-configs such as ``text_config``, ``vision_config``, and ``audio_config``.
    During model construction, HF builds sub-modules via
    ``AutoModel.from_config(sub_config)``, which reads ``torch_dtype`` from the
    sub-config rather than the parent. Without propagation, those sub-modules
    keep the checkpoint dtype while directly instantiated ``nn.Linear`` modules
    take the requested default dtype, producing a mixed-dtype model that FSDP2's
    uniform-original-dtype check rejects.

    Args:
        hf_config: The top-level HuggingFace config to update in place.
        torch_dtype: The dtype to assign to every nested ``PretrainedConfig``.
    """
    seen: set[int] = set()

    def _recurse(cfg) -> None:
        if id(cfg) in seen:
            return
        seen.add(id(cfg))
        cfg.torch_dtype = torch_dtype
        for value in vars(cfg).values():
            if isinstance(value, PretrainedConfig):
                _recurse(value)

    _recurse(hf_config)


def _is_config_compatible_with_custom_model(arch_name: str, config) -> bool:
    """
    Check if a HuggingFace config is compatible with our custom model implementation.

    Some architectures (e.g., NemotronHForCausalLM) are shared between different model versions
    (v2 vs v3) but our custom implementation only supports specific versions. This function
    validates that the config has the required attributes for the custom implementation.

    Args:
        arch_name: The architecture name (e.g., "NemotronHForCausalLM")
        config: The HuggingFace config object

    Returns:
        True if the config is compatible with our custom implementation, False otherwise
    """
    # NemotronHForCausalLM is shared by the MoE ("v3", has n_routed_experts) and the
    # dense ("v2"-style, e.g. Nano 4B/9B/12B BF16) Nemotron-H variants. The custom
    # implementation now handles both. Dense is recognized by its hybrid layer pattern:
    # key off layers_block_type, which is what the custom model actually consumes. (A raw
    # hybrid_override_pattern alone is not enough; the model never normalizes it, so a
    # config with no layers_block_type falls back to HF instead of routing here.)
    if arch_name == "NemotronHForCausalLM":
        if getattr(config, "n_routed_experts", None) is not None:
            return True
        return bool(getattr(config, "layers_block_type", None))

    # All other architectures are assumed compatible
    return True


def _resolve_custom_model_cls_for_config(config):
    """Resolve the custom model class for *config*, if the config is compatible."""
    architectures = get_architectures(config)
    if not architectures:
        return None

    arch_name = architectures[0]
    if not ModelRegistry.has_custom_model(arch_name):
        return None

    # Some architecture names are shared across multiple upstream variants.
    # Screen them here before asking the registry for the custom implementation.
    if not _is_config_compatible_with_custom_model(arch_name, config):
        return None

    return ModelRegistry.resolve_custom_model_cls(arch_name, config)


def _load_registered_custom_config(pretrained_model_name_or_path, attn_implementation, **kwargs):
    """Load a config through Automodel's registry before falling back to HF AutoConfig."""
    config_kwargs = kwargs.copy()
    config_kwargs["_from_auto"] = True
    config_kwargs["name_or_path"] = pretrained_model_name_or_path
    config_kwargs.pop("code_revision", None)

    try:
        config_dict, unused_kwargs = AutoConfig.get_config_dict(pretrained_model_name_or_path, **config_kwargs)
    except Exception:
        logger.debug("Could not pre-read config for %s", pretrained_model_name_or_path, exc_info=True)
        return None

    model_type = config_dict.get("model_type")
    if not isinstance(model_type, str):
        return None

    config_cls = resolve_custom_config_cls(model_type)
    if config_cls is None:
        return None

    unused_kwargs.setdefault("attn_implementation", attn_implementation)
    return config_cls.from_dict(config_dict, **unused_kwargs)


def get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs):
    """
    Get the HF config for the model.
    """
    kwargs = kwargs.copy()
    trust_remote_code = kwargs.pop("trust_remote_code", resolve_trust_remote_code(pretrained_model_name_or_path))
    hf_config = kwargs.get("config", None)
    # A plain-dict ``config`` (e.g. from CLI ``--model.config.num_hidden_layers 16``,
    # which the config loader materializes as a dict, not a PretrainedConfig) is a set
    # of config *overrides*, not a finished config object. Treat it as such: drop it
    # from ``config`` and fold its keys into the kwargs that AutoConfig.from_pretrained
    # applies as overrides. Returning the raw dict instead would break every downstream
    # consumer that expects a config object (e.g. get_architectures -> the custom-model
    # resolver, which would then mis-dispatch to the stock-HF path and reject custom
    # kwargs like ``backend``).
    if isinstance(hf_config, dict):
        kwargs.pop("config", None)
        kwargs.update(hf_config)
        hf_config = None
    if hf_config is None:
        # Filter out nested dict kwargs before passing to AutoConfig.from_pretrained.
        # Nested dicts (e.g. text_config={"key": val}) would replace entire sub-configs
        # with incomplete dicts, losing all other fields. These nested overrides are
        # instead handled by _consume_config_overrides which deep-merges them.
        nested_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if isinstance(kwargs[k], dict)}  # noqa: F841
        hf_config = _load_registered_custom_config(pretrained_model_name_or_path, attn_implementation, **kwargs)
    if hf_config is None:
        try:
            hf_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
        except (ValueError, StrictDataclassClassValidationError) as e:
            err = str(e)
            if "does not recognize this architecture" in err:
                raise ValueError(
                    f"{e}\n\n"
                    f"The checkpoint '{pretrained_model_name_or_path}' has a model type not "
                    f"recognized by the installed version of NeMo Automodel. "
                    f"This usually means your installed package is out of date.\n\n"
                    f"To fix this, try upgrading:\n"
                    f"  pip install --upgrade nemo_automodel\n"
                    f"or install from source:\n"
                    f"  pip install git+https://github.com/NVIDIA-NeMo/Automodel.git"
                ) from e
            # Some upstream configs (e.g. stepfun-ai/Step-3.5-Flash) ship
            # layer_types longer than num_hidden_layers, which newer transformers
            # versions reject during config instantiation. huggingface_hub wraps
            # the validator's ValueError in StrictDataclassClassValidationError
            # (not a ValueError subclass), so both exception types must be caught.
            if "num_hidden_layers" in err and ("layer_types" in err or "layer types" in err):
                hf_config = _load_config_with_layer_types_fix(
                    pretrained_model_name_or_path,
                    attn_implementation,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
            else:
                raise
    return hf_config


def _load_config_with_layer_types_fix(pretrained_model_name_or_path, attn_implementation, trust_remote_code, **kwargs):
    """Load an HF config after truncating ``layer_types`` to ``num_hidden_layers``.

    Works around buggy upstream configs whose ``layer_types`` list is longer than
    ``num_hidden_layers`` (e.g. stepfun-ai/Step-3.5-Flash).
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config_dict, _ = AutoConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
    n = config_dict.get("num_hidden_layers")
    lt = config_dict.get("layer_types")
    if isinstance(n, int) and isinstance(lt, list) and len(lt) > n:
        logger.warning(
            "Truncating layer_types (len=%d) to num_hidden_layers=%d for %s",
            len(lt),
            n,
            pretrained_model_name_or_path,
        )
        config_dict["layer_types"] = lt[:n]

    config_cls = None
    auto_map = config_dict.get("auto_map") or {}
    if trust_remote_code and "AutoConfig" in auto_map:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        code_kwargs = {
            key: kwargs[key]
            for key in ("revision", "code_revision", "cache_dir", "token", "local_files_only", "force_download")
            if key in kwargs
        }
        if config_dict.get("_commit_hash") is not None:
            code_kwargs["revision"] = config_dict["_commit_hash"]
        config_cls = get_class_from_dynamic_module(auto_map["AutoConfig"], pretrained_model_name_or_path, **code_kwargs)
    if config_cls is None:
        model_type = config_dict.get("model_type")
        config_cls = CONFIG_MAPPING.get(model_type)
    if config_cls is None:
        raise ValueError(
            f"Could not resolve config class for {pretrained_model_name_or_path} "
            f"(model_type={config_dict.get('model_type')!r})"
        )
    return config_cls.from_dict(config_dict, attn_implementation=attn_implementation)


def get_is_hf_model(config, force_hf):
    """Determine whether the model should use the HF (not custom) implementation."""
    if force_hf:
        return True
    return _resolve_custom_model_cls_for_config(config) is None


def _download_model_weights(
    hf_config: PretrainedConfig,
    pretrained_model_name_or_path: str,
    process_group: torch.distributed.ProcessGroup | None = None,
    **kwargs: Any,
) -> None:
    """Download weights at the config's resolved commit, retaining Hub options."""
    if not os.path.isdir(pretrained_model_name_or_path):
        if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            logger.info(
                "HF_HUB_OFFLINE=1: skipping weight download for %s (using cached weights)",
                pretrained_model_name_or_path,
            )
            return
        num_nodes = (get_world_size_safe() % get_local_world_size_preinit()) + 1  # 1-indexed
        if num_nodes > 1:
            logger.info(
                "Downloading model weights on %d nodes. This incurs high storage usage. "
                "It is recommended to download once with `hf download` and pass in the "
                "downloaded path to the `pretrained_model_name_or_path` argument.",
                num_nodes,
            )
        # Import via module reference (vs bound name) so unit tests can patch
        # `nemo_automodel.components.distributed.utils.FirstRankPerNode`.
        with dist_utils.FirstRankPerNode(group=process_group):
            download_kwargs = {
                key: kwargs[key]
                for key in ("cache_dir", "revision", "token", "local_files_only", "force_download")
                if key in kwargs
            }
            if hf_config._commit_hash is not None:
                download_kwargs["revision"] = hf_config._commit_hash
            snapshot_download(pretrained_model_name_or_path, **download_kwargs)


def _prepopulate_remote_code_cache(hf_config, pretrained_model_name_or_path, kwargs, process_group=None):
    """Fully populate HF's dynamic-module cache on the model-local rank 0 first.

    ``get_cached_module_file`` copies a custom-code file plus its *direct* relative
    imports, but ``get_class_in_module`` later validates the *transitive* closure. A
    checkpoint whose ``modeling_*.py`` reaches a shim only transitively (e.g. DeciLM:
    ``modeling_decilm -> configuration_decilm -> transformers_4_44_2__configuration_llama``)
    then dies with ``FileNotFoundError`` because that file was never copied into the
    model's cache dir; multi-rank reloads from a shared filesystem also race on the
    partial copy. Copy every ``.py`` from the checkpoint dir into the resolved cache
    dir under ``FirstRankPerNode`` so the closure is complete before any rank reads it.
    Best-effort: any failure falls through to HF's own (un-serialized) resolution.
    """
    if not pretrained_model_name_or_path or not os.path.isdir(str(pretrained_model_name_or_path)):
        return
    auto_map = getattr(hf_config, "auto_map", None)
    if not isinstance(auto_map, dict) or not auto_map:
        return
    try:
        from transformers.dynamic_module_utils import HF_MODULES_CACHE, get_cached_module_file
    except Exception:
        return
    src_dir = str(pretrained_model_name_or_path)
    module_files = set()
    for value in auto_map.values():
        for ref in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(ref, str) and "." in ref:
                module_files.add(ref.rsplit(".", 1)[0] + ".py")
    src_py = glob.glob(os.path.join(src_dir, "*.py"))
    with dist_utils.FirstRankPerNode(group=process_group):
        for module_file in module_files:
            try:
                cached = get_cached_module_file(src_dir, module_file)
                cache_dir = os.path.dirname(os.path.join(HF_MODULES_CACHE, cached))
                for src in src_py:
                    dst = os.path.join(cache_dir, os.path.basename(src))
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
            except Exception:
                pass


def _setup_bnb_loading_kwargs(kwargs: dict) -> None:
    """Configure kwargs for HF from_pretrained to work with BitsAndBytes quantization.

    Sets ``device_map`` so HF loads+quantizes per-shard on the current GPU, and
    disables the async weight loader introduced in transformers v5 which can
    materialize many full-precision tensors concurrently before the quantizer
    runs, causing OOM on memory-constrained systems.
    """
    kwargs.setdefault("device_map", {"": torch.cuda.current_device()})
    prev = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
    if prev is None:
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        logger.info("Set HF_DEACTIVATE_ASYNC_LOAD=1 for BnB-compatible synchronous weight loading.")
    logger.info("BnB loading: device_map=%s", kwargs["device_map"])


# Fraction of TOTAL CUDA memory that the estimated BF16 footprint may occupy
# before we refuse the load and point the user at the streaming workaround.
# We budget against total (not free) memory so the verdict is deterministic
# across ranks and unaffected by transient allocations from co-located
# processes; the reserved remainder (~30%) covers the CUDA context, NCCL /
# process-group buffers, loader scratch, and load-time activations.
_FP8_PREFLIGHT_FOOTPRINT_THRESHOLD = 0.70
_FP8_PREFLIGHT_DISABLE_ENV = "NEMO_AUTOMODEL_DISABLE_FP8_PREFLIGHT"


def _get_quant_attr(quantization_config, key):
    """Read ``key`` from a quant config that may be a dict or an object."""
    if quantization_config is None:
        return None
    if isinstance(quantization_config, dict):
        return quantization_config.get(key)
    return getattr(quantization_config, key, None)


def _quant_config_is_fp8_full_materialize(quantization_config) -> bool:
    """True if this config asks HF to dequantize the FP8 checkpoint to BF16 in full.

    Keys off the canonical ``dequantize`` flag set by transformers'
    ``FineGrainedFP8Config`` (and by ``_maybe_dequantize_fp8_for_peft`` on the
    native ``hf_config.quantization_config``). ``quant_method`` may be a plain
    string or the ``QuantizationMethod.FP8`` enum, both of which compare equal
    to ``"fp8"``.
    """
    if _get_quant_attr(quantization_config, "quant_method") != "fp8":
        return False
    return _get_quant_attr(quantization_config, "dequantize") is True


def _param_count_from_local_safetensors(pretrained_model_name_or_path) -> int | None:
    """Exact parameter count from an already-local safetensors checkpoint.

    Sums the element count of every tensor by reading only safetensors headers
    (no weight data is materialized), which is exact regardless of GQA, MoE,
    tied embeddings, or on-disk dtype. For an FP8 checkpoint the stored element
    count equals the dequantized BF16 element count, so multiplying by 2 bytes
    downstream yields the true BF16 footprint (scale tensors are tiny and only
    make the estimate marginally conservative).

    Returns ``None`` — so the caller falls back to the coarse formula — when the
    checkpoint is not already on disk (e.g. a not-yet-downloaded repo id), has no
    safetensors files, or cannot be read.
    """
    if not pretrained_model_name_or_path:
        return None
    try:
        model_dir = _resolve_model_dir(pretrained_model_name_or_path)
    except Exception:
        # Not cached locally (snapshot_download(local_files_only=True) raised) or
        # otherwise unresolvable: defer to the formula.
        return None
    if not os.path.isdir(model_dir) or not _has_safetensors(model_dir):
        return None

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                index = json.load(f)
            shard_files = list(dict.fromkeys(index["weight_map"].values()))
        except (OSError, ValueError, KeyError):
            return None
    else:
        shard_files = ["model.safetensors"]

    try:
        from safetensors import safe_open

        total = 0
        for shard in shard_files:
            with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
                for key in f.keys():
                    n = 1
                    for dim in f.get_slice(key).get_shape():
                        n *= dim
                    total += n
    except Exception:
        return None
    return total or None


def _get_hf_param_count_estimate(hf_config, pretrained_model_name_or_path=None) -> int | None:
    """Best-effort parameter count, most accurate source first.

    Order of preference:
      1. Exact element count from an already-local safetensors checkpoint
         (``_param_count_from_local_safetensors``) — dtype / GQA / MoE-agnostic.
      2. An explicit ``hf_config.num_parameters`` override, if a caller set one.
         (HF does not populate this on configs; it is only an escape hatch.)
      3. A coarse transformer formula as a last resort, used when the checkpoint
         is not yet on disk:
         ``L * (attn + n_experts * 3*H*I) + V*H`` where the attention term is
         ``2*H*(n_heads*head_dim) + 2*H*(n_kv*head_dim)`` so it tracks GQA and
         collapses to ``4*H*H`` for multi-head attention. It assumes gated MLPs
         and ignores biases / norms / routers, so it is approximate — a sanity
         check, not a memory planner.

    Sums over any text/vision/audio sub-configs for multimodal wrappers.
    Returns ``None`` when no estimate can be formed.
    """
    exact = _param_count_from_local_safetensors(pretrained_model_name_or_path)
    if exact:
        return exact

    if hf_config is None:
        return None

    explicit = getattr(hf_config, "num_parameters", None)
    if isinstance(explicit, int) and explicit > 0:
        return explicit

    total = 0
    seen: set[int] = set()

    def _walk(cfg) -> None:
        nonlocal total
        if cfg is None or id(cfg) in seen:
            return
        seen.add(id(cfg))

        # Multimodal wrappers (e.g. Gemma4, Mistral3-VLM) park real layer
        # counts on text_config / vision_config / audio_config. Recurse into
        # those and skip the wrapper itself so we don't double count.
        sub_configs = [getattr(cfg, attr, None) for attr in ("text_config", "vision_config", "audio_config")]
        sub_configs = [s for s in sub_configs if isinstance(s, PretrainedConfig)]
        if sub_configs:
            for sub in sub_configs:
                _walk(sub)
            return

        layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or 0
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None) or 0
        vocab = getattr(cfg, "vocab_size", None) or 0
        inter = getattr(cfg, "intermediate_size", None) or getattr(cfg, "ffn_hidden_size", None) or (4 * hidden)
        n_experts = (
            getattr(cfg, "num_local_experts", None)
            or getattr(cfg, "num_experts", None)
            or getattr(cfg, "n_routed_experts", None)
            or 1
        )

        if not layers or not hidden:
            return

        # Attention: Q and O are H*(n_heads*head_dim); K and V are
        # H*(n_kv*head_dim). For GQA n_kv < n_heads shrinks K/V; for plain
        # multi-head attention (n_kv == n_heads, head_dim == H/n_heads) this
        # collapses to the familiar 4*H*H.
        n_heads = (
            getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "n_head", None)
            or getattr(cfg, "num_heads", None)
            or 0
        )
        n_kv = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_kv_heads", None) or n_heads
        head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads if n_heads else 0)
        if n_heads and head_dim:
            attn = 2 * hidden * (n_heads * head_dim) + 2 * hidden * (n_kv * head_dim)
        else:
            attn = 4 * hidden * hidden  # head config unavailable: assume multi-head

        per_layer = attn + n_experts * 3 * hidden * inter
        total += layers * per_layer + vocab * hidden

    _walk(hf_config)
    return total if total > 0 else None


def _has_streaming_fp8_checkpoint_load(hf_config) -> bool:
    """Return whether the registered custom model owns this FP8 checkpoint load."""
    model_cls = _resolve_custom_model_cls_for_config(hf_config)
    if not getattr(model_cls, "_supports_streaming_fp8_checkpoint_load", False):
        return False
    native_quantization_config = getattr(hf_config, "quantization_config", None)
    return (
        _get_quant_attr(native_quantization_config, "quant_method") == "fp8"
        and _get_quant_attr(native_quantization_config, "weight_block_size") is None
    )


def _check_fp8_dequantize_will_fit(
    hf_config, quantization_config, pretrained_model_name_or_path, force_hf: bool = True
) -> None:
    """Refuse FP8-dequantize loads that won't fit in CUDA HBM.

    The forced-HF and HF-fallback routes both materialize the entire model in
    BF16 inside ``_from_pretrained_parent_class`` on every rank *before* any
    Automodel sharding kicks in. For models > available HBM (Devstral 123B,
    Mistral Medium 128B, ...) this OOMs deep inside the HF loader with no useful
    log line. See https://github.com/NVIDIA-NeMo/Automodel/issues/2114.

    This pre-flight estimates the BF16 footprint and compares it to a budget of
    ``_FP8_PREFLIGHT_FOOTPRINT_THRESHOLD`` x TOTAL device memory. Budgeting
    against total (not free) memory keeps the verdict deterministic across ranks
    and immune to transient allocations from co-located processes; the reserved
    remainder absorbs the CUDA context, NCCL buffers, loader scratch, and
    load-time activations. The footprint prefers an exact count read from the
    on-disk safetensors checkpoint, falling back to a coarse config formula, so
    it neither systematically over- nor under-counts.

    No-ops when CUDA isn't available, the config isn't an FP8 full-materialize
    config, the param count can't be estimated, or
    ``NEMO_AUTOMODEL_DISABLE_FP8_PREFLIGHT=1`` is set.

    The guard runs from two load paths: the explicit ``force_hf=True`` branch
    and the no-custom-class HF fallback. ``force_hf`` only picks the wording of
    the diagnostic, since "drop force_hf" is not a useful suggestion on the
    fallback path.

    Raises:
        RuntimeError: if the estimated BF16 footprint exceeds the safe budget.
    """
    if os.environ.get(_FP8_PREFLIGHT_DISABLE_ENV) == "1":
        return None
    if not _quant_config_is_fp8_full_materialize(quantization_config):
        return None
    if not torch.cuda.is_available():
        return None

    n_params = _get_hf_param_count_estimate(hf_config, pretrained_model_name_or_path)
    if not n_params:
        logger.warning(
            "FP8 dequantize pre-flight: could not estimate parameter count for %s; skipping memory check.",
            pretrained_model_name_or_path,
        )
        return None

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, AssertionError) as exc:
        logger.warning(
            "FP8 dequantize pre-flight: torch.cuda.mem_get_info() failed (%s); skipping memory check for %s.",
            exc,
            pretrained_model_name_or_path,
        )
        return None

    bf16_bytes = n_params * 2
    budget_bytes = int(total_bytes * _FP8_PREFLIGHT_FOOTPRINT_THRESHOLD)
    if bf16_bytes <= budget_bytes:
        return None

    gib = 1024**3
    route_explanation = (
        "force_hf=True selects Transformers' full-materialize loader"
        if force_hf
        else "the custom-model route was unavailable, so Automodel fell back to Transformers' full-materialize loader"
    )
    if _has_streaming_fp8_checkpoint_load(hf_config):
        force_hf_action = "Remove force_hf=True and " if force_hf else ""
        recovery = (
            "Automodel has a streaming FP8 checkpoint loader for this architecture. "
            f"{force_hf_action}do not pass FineGrainedFP8Config(dequantize=True); leave the checkpoint's native "
            "FP8 quantization_config unchanged so the custom path can dequantize each local shard."
        )
    else:
        recovery = (
            "Automodel has no streaming FP8 checkpoint loader registered for this architecture, so BF16 "
            "dequantized loading is unsupported at this model size."
        )
    raise RuntimeError(
        f"FP8 dequantize pre-flight failed for {pretrained_model_name_or_path!r}.\n"
        f"  Estimated BF16 footprint: {bf16_bytes / gib:.1f} GiB "
        f"(~{n_params / 1e9:.1f}B parameters x 2 bytes).\n"
        f"  Total CUDA memory: {total_bytes / gib:.1f} GiB "
        f"(safe budget at {int(_FP8_PREFLIGHT_FOOTPRINT_THRESHOLD * 100)}%: {budget_bytes / gib:.1f} GiB; "
        f"free right now: {free_bytes / gib:.1f} GiB).\n"
        f"  With quant_method='fp8' + dequantize=True, {route_explanation}. Transformers materializes the full "
        f"BF16 model on every rank inside _from_pretrained_parent_class BEFORE Automodel sharding runs, so "
        f"tensor parallelism cannot prevent the OOM for any "
        f"model larger than one device's HBM.\n"
        f"  Next step: {recovery}\n"
        f"  Tracking issue: https://github.com/NVIDIA-NeMo/Automodel/issues/2114\n"
        f"  Bypass (use only if you're sure the estimate is wrong): "
        f"export {_FP8_PREFLIGHT_DISABLE_ENV}=1."
    )


def _resolve_model_dir(
    pretrained_model_name_or_path: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    subfolder: str = "",
) -> str:
    """Resolve a HF repo id or local path to a local directory with model files."""
    model_dir = pretrained_model_name_or_path
    if not os.path.isdir(model_dir):
        model_dir = snapshot_download(
            pretrained_model_name_or_path, revision=revision, cache_dir=cache_dir, local_files_only=True
        )
    return os.path.join(model_dir, subfolder) if subfolder else model_dir


def _has_safetensors(model_dir: str) -> bool:
    """Check whether a model directory contains safetensors checkpoint files."""
    if os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
        return True
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return True
    return False


def _stream_load_bnb_weights(model, model_dir, device, torch_dtype, *, disable_mmap: bool = False):
    """Load safetensor shards one-at-a-time, quantizing BnB Params4bit on the fly.

    Peak memory ≈ (accumulated quantized weights) + (one bf16 weight tensor)
    instead of (full bf16 model) with standard HF loading.
    When disable_mmap=True, one tensor at a time is cloned to anonymous CPU
    memory before CUDA transfer to avoid UMA page-migration stalls.
    """
    import bitsandbytes as bnb

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = list(dict.fromkeys(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    # Build name → (module, attr_name, param_or_buffer) index
    param_map: dict[str, tuple] = {}
    for name, param in model.named_parameters():
        parts = name.rsplit(".", 1)
        mod = model.get_submodule(parts[0]) if len(parts) == 2 else model
        param_map[name] = (mod, parts[-1], param)
    for name, buf in model.named_buffers():
        if name not in param_map:
            parts = name.rsplit(".", 1)
            mod = model.get_submodule(parts[0]) if len(parts) == 2 else model
            param_map[name] = (mod, parts[-1], buf)

    loaded_keys: set[str] = set()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    def consume_tensor(key: str, tensor: torch.Tensor) -> None:
        """Install one checkpoint tensor into the target model.

        Args:
            key: Fully-qualified state-dict key. The key determines the tensor's rank and axis semantics.
            tensor: Tensor from the active safetensors shard. This function takes ownership and may cast,
                quantize, move, or install it as a parameter/buffer.

        Returns:
            None.
        """
        if key not in param_map:
            logger.debug("Skipping key not in model: %s", key)
            del tensor
            return

        mod, attr, old_param = param_map[key]

        if isinstance(old_param, bnb.nn.Params4bit):
            if torch_dtype is not None:
                tensor = tensor.to(dtype=torch_dtype)
            new_param = bnb.nn.Params4bit(
                data=tensor,
                requires_grad=False,
                compress_statistics=old_param.compress_statistics,
                quant_type=old_param.quant_type,
                quant_storage=old_param.quant_storage,
                module=mod if isinstance(mod, bnb.nn.Linear4bit) else None,
                bnb_quantized=False,
            )
            del tensor
            new_param._quantize(device)
            mod._parameters[attr] = new_param
        else:
            target_dtype = torch_dtype if torch_dtype is not None else tensor.dtype
            materialized = tensor.to(device=device, dtype=target_dtype)
            del tensor
            if isinstance(old_param, torch.nn.Parameter):
                mod._parameters[attr] = torch.nn.Parameter(materialized, requires_grad=old_param.requires_grad)
            else:
                mod._buffers[attr] = materialized

        loaded_keys.add(key)

    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = os.path.join(model_dir, shard_file)
        logger.info(
            "Streaming BnB shard %d/%d: %s",
            shard_idx + 1,
            len(shard_files),
            shard_file,
        )

        from safetensors import safe_open

        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if disable_mmap and key in param_map:
                    # Avoid full-shard materialization while detaching the next
                    # CUDA/quantization copy from mmap-backed checkpoint pages.
                    tensor = tensor.clone()
                try:
                    consume_tensor(key, tensor)
                finally:
                    del tensor

        gc.collect()
        torch.cuda.empty_cache()

    # Tie weights before validating: safetensors typically stores only one copy
    # of a tied pair (e.g. Llama's lm_head.weight tied to embed_tokens.weight),
    # so the untied sibling is still on meta at this point. tie_weights()
    # re-establishes the Python-level alias so both sides point at the loaded
    # tensor.
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # Any param/buffer still on meta after load+tie is a real missing key —
    # forward pass would silently produce NaN.  Fail loudly instead.
    missing: list[str] = []
    for name, (_, _, _) in param_map.items():
        if name in loaded_keys:
            continue
        current = _get_model_tensor(model, name)
        if current is None or (hasattr(current, "device") and current.device.type == "meta"):
            missing.append(name)

    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise RuntimeError(
            f"Streaming BnB load left {len(missing)} tensor(s) unmaterialized after tie_weights: {preview}{more}"
        )

    logger.info(
        "Streaming BnB load complete: %d tensors loaded (%d additional tied after load)",
        len(loaded_keys),
        len(param_map) - len(loaded_keys),
    )


def _get_bnb_modules_to_not_convert(model, quantization_config):
    """Mirror HF BnB's default skip-list semantics before streaming replacement."""
    from transformers.quantizers.base import get_keys_to_not_convert

    skip_modules = getattr(quantization_config, "llm_int8_skip_modules", None)
    keep_in_fp32_modules = getattr(model, "_keep_in_fp32_modules", None)

    modules_to_not_convert = get_keys_to_not_convert(model) if skip_modules is None else []
    if skip_modules is not None:
        modules_to_not_convert.extend(skip_modules)
    if keep_in_fp32_modules is not None:
        modules_to_not_convert.extend(keep_in_fp32_modules)

    return list(set(modules_to_not_convert))


def _streaming_bnb_supported(cls, hf_config) -> bool:
    """Whether streaming BnB can safely load HF safetensors directly into the target class.

    The streaming loader maps safetensors keys 1:1 onto ``model.named_parameters()``.
    Two cases break that 1:1 assumption and must fall back to the standard HF loader:

    1. Automodel's custom implementations fuse projections (e.g. MoE
       ``mlp.experts.gate_up_proj``) and rely on a ``state_dict_adapter`` to translate
       HF-style keys on load. Detected via the ``HFCheckpointingMixin`` marker.
    2. Vanilla HF classes whose safetensors use a legacy layout that HF's loader
       reshapes/renames at load time (e.g. Mixtral ``block_sparse_moe.experts.*.w1`` →
       fused ``mlp.experts.gate_up_proj``). Detected via HF's per-model-type
       ``get_checkpoint_conversion_mapping`` — any non-empty mapping means the streaming
       path would leave fused tensors on meta device.
    """
    try:
        model_cls = cls._model_mapping[type(hf_config)]
    except (KeyError, TypeError):
        return False
    try:
        from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

        if issubclass(model_cls, HFCheckpointingMixin):
            return False
    except ImportError:
        pass
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    except ImportError:
        return True
    model_type = getattr(hf_config, "model_type", None)
    if model_type and get_checkpoint_conversion_mapping(model_type):
        return False
    return True


def _init_model_bnb_streaming(
    cls, pretrained_model_name_or_path, hf_config, attn_implementation, torch_dtype, quantization_config, **kwargs
):
    """Create model on meta device, replace Linear→Linear4bit, stream-load+quantize.

    This avoids materializing the full bf16 model in memory, which is critical
    for unified-memory systems (e.g. DGX Spark) where CPU and GPU share the
    same physical memory pool.

    Returns ``(is_custom_model=False, model)`` so the caller treats it like an
    HF-loaded model with weights already present.
    """
    from transformers.initialization import no_init_weights
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear

    from nemo_automodel.components.utils.model_utils import init_empty_weights

    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = dtype_from_str(torch_dtype)
    if torch_dtype == "auto":
        torch_dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)
        if isinstance(torch_dtype, str):
            torch_dtype = dtype_from_str(torch_dtype)

    device = torch.cuda.current_device()

    # 1. Download weights if needed
    disable_mmap = bool(kwargs.pop("disable_mmap", False))
    _download_model_weights(hf_config, pretrained_model_name_or_path, **kwargs)

    # 2. Resolve to local directory & verify safetensors
    model_dir = _resolve_model_dir(
        pretrained_model_name_or_path,
        revision=hf_config._commit_hash,
        cache_dir=kwargs.get("cache_dir"),
        subfolder=kwargs.get("subfolder", ""),
    )
    if not _has_safetensors(model_dir):
        raise FileNotFoundError(f"Streaming BnB loading requires safetensors checkpoint, but none found in {model_dir}")

    # 3. Create model skeleton on meta device (zero memory)
    logger.info("Creating model skeleton on meta device for streaming BnB quantization")
    with no_init_weights(), init_empty_weights():
        model = cls._from_config_parent_class(
            hf_config,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

    # 4. Replace nn.Linear → bnb.nn.Linear4bit (still on meta, no memory)
    modules_to_not_convert = _get_bnb_modules_to_not_convert(model, quantization_config)
    model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )

    # 5. Stream-load weights, quantizing each tensor on the fly
    _stream_load_bnb_weights(model, model_dir, device, torch_dtype, disable_mmap=disable_mmap)

    # 6. Store quantization_config on the model (HF convention)
    model.config.quantization_config = quantization_config
    model.is_quantized = True

    # 7. Wrap with HFCheckpointingMixin
    try:
        hf_model_cls = cls._model_mapping[type(hf_config)]
    except KeyError:
        hf_model_cls = type(model)
    model.__class__ = _get_mixin_wrapped_class(hf_model_cls)

    return False, model


def _get_model_tensor(model, name: str):
    """Return a parameter or buffer by its fully-qualified state-dict key."""
    try:
        return model.get_parameter(name)
    except (AttributeError, ValueError):
        pass
    try:
        return model.get_buffer(name)
    except (AttributeError, ValueError):
        return None


def _restore_loaded_model_dtype(
    model, pretrained_model_name_or_path, hf_config, quantization_config, load_kwargs, requested_dtype=None
) -> None:
    """Unify each loaded tensor's dtype after HuggingFace's mixed-dtype load.

    Some modules allocate parameters in a wider dtype than the checkpoint.
    HuggingFace then copies the checkpoint tensor into that existing tensor,
    which upcasts the loaded value, leaving the model with a mix of dtypes that
    trips FSDP2's uniform-original-dtype check. We fix that by re-inspecting the
    checkpoint tensor dtypes per key and unifying each loaded parameter/buffer.

    The unification target depends on ``requested_dtype``:

      * ``None`` (the user passed ``torch_dtype="auto"``): restore each tensor to
        the exact dtype stored in the checkpoint -- faithfully mirror the file.
      * an explicit ``torch.dtype``: restore each floating tensor to the *wider*
        of (checkpoint dtype, requested dtype). This honors an explicit fp32
        request (so parameters can serve as fp32 master weights) while preserving
        intrinsically-fp32 checkpoint params (e.g. ``A_log``) even under a bf16
        request, and is a no-op for the common bf16/auto case.

    Args:
        requested_dtype: The resolved ``torch_dtype`` the caller requested, or
            None when ``"auto"`` was requested.
    """
    if quantization_config is not None or getattr(hf_config, "quantization_config", None) is not None:
        return

    try:
        checkpoint_dtypes = checkpoint_utils._get_checkpoint_tensor_dtypes(
            pretrained_model_name_or_path, hf_config, load_kwargs
        )
    except Exception as exc:
        logger.warning(
            "Failed to inspect checkpoint tensor dtypes for %s; leaving loaded dtypes unchanged: %s",
            pretrained_model_name_or_path,
            exc,
        )
        return

    if not checkpoint_dtypes:
        return

    preserve_gdn_fp32_params = has_gated_delta_net_fp32_checkpoint_contract(hf_config)
    restored_dtype_by_tensor_id: dict[int, torch.dtype] = {}
    restored_count = 0
    for name, checkpoint_dtype in checkpoint_dtypes.items():
        tensor = _get_model_tensor(model, name)
        if tensor is None:
            continue

        effective_checkpoint_dtype = (
            torch.float32
            if checkpoint_dtype.is_floating_point
            and preserve_gdn_fp32_params
            and is_gated_delta_net_fp32_param_key(name)
            else checkpoint_dtype
        )

        # Record the checkpoint's original dtype on the tensor as the compute-dtype
        # hint. Storage may be upcast below (fp32 master weights), which erases the
        # dtype HF intended for compute; downstream sharding (fully_shard_by_dtype)
        # reads ``_hf_compute_dtype`` to keep intrinsically-fp32 params (e.g. ``A_log``)
        # computing in fp32 while the bulk computes in mp_policy.param_dtype.
        if effective_checkpoint_dtype.is_floating_point and tensor.dtype.is_floating_point:
            tensor._hf_compute_dtype = effective_checkpoint_dtype

        # Pick the unification target. For an explicit floating request, take the
        # wider of (checkpoint, requested) so explicit fp32 is honored as master
        # weights while intrinsically-fp32 checkpoint params survive a bf16 request.
        # For "auto" (requested_dtype is None) or non-floating tensors, mirror the
        # checkpoint dtype exactly (preserves today's behavior).
        target_dtype = effective_checkpoint_dtype
        if (
            requested_dtype is not None
            and effective_checkpoint_dtype.is_floating_point
            and tensor.dtype.is_floating_point
        ):
            target_dtype = torch.promote_types(effective_checkpoint_dtype, requested_dtype)

        if tensor.dtype == target_dtype:
            continue

        seen_dtype = restored_dtype_by_tensor_id.get(id(tensor))
        if seen_dtype is not None and seen_dtype != target_dtype:
            logger.warning(
                "Skipping conflicting checkpoint dtypes for aliased tensor %s: %s vs %s",
                name,
                seen_dtype,
                target_dtype,
            )
            continue

        try:
            tensor.data = tensor.data.to(dtype=target_dtype)
        except (RuntimeError, TypeError) as exc:
            logger.warning("Failed to restore checkpoint dtype for %s to %s: %s", name, target_dtype, exc)
            continue

        restored_dtype_by_tensor_id[id(tensor)] = target_dtype
        restored_count += 1

    if restored_count > 0:
        logger.info("Restored checkpoint dtypes for %d tensors from %s", restored_count, pretrained_model_name_or_path)


def __init_model(
    cls,
    pretrained_model_name_or_path_or_config,
    attn_implementation,
    torch_dtype,
    quantization_config,
    force_hf,
    *model_args,
    **kwargs,
):
    # Private recipe-level toggle: when False, skip ``_restore_loaded_model_dtype``
    # so the caller's explicit ``torch_dtype`` is preserved as the master-weight
    # dtype (needed for mixed-precision training that wants an fp32 master copy).
    # Default ``True`` keeps existing behavior for every recipe that doesn't set
    # it. Pop here so the flag never reaches HF's ``from_pretrained``.
    restore_loaded_dtype = kwargs.pop("_restore_loaded_dtype", True)
    process_group = kwargs.pop("_process_group", None)
    torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch_dtype
    is_pretrained_init = isinstance(pretrained_model_name_or_path_or_config, str)  # The caller is .from_pretrained
    hf_config = (
        get_hf_config(pretrained_model_name_or_path_or_config, attn_implementation, **kwargs)
        if is_pretrained_init
        else pretrained_model_name_or_path_or_config
    )
    pretrained_model_name_or_path = (
        pretrained_model_name_or_path_or_config if is_pretrained_init else getattr(hf_config, "name_or_path")
    )
    if is_pretrained_init and hf_config._commit_hash is not None:
        kwargs["revision"] = hf_config._commit_hash
        kwargs["_commit_hash"] = hf_config._commit_hash
    # A plain-dict ``config`` override (e.g. from ``--model.config.num_hidden_layers``)
    # has already been folded into ``hf_config`` by ``get_hf_config`` above. Drop it from
    # kwargs so it is not also forwarded into the model constructor / HF from_pretrained
    # (custom path passes ``hf_config`` positionally as ``config`` -> would collide;
    # stock-HF path does not accept a dict ``config``).
    # Its keys are explicit overrides all the same, and nothing downstream can tell
    # a deliberate value from an unset one once it equals a default, so keep them.
    dict_config_overrides: set[str] = set()
    if isinstance(kwargs.get("config"), dict):
        dict_config_overrides = set(kwargs.pop("config"))
    architectures = get_architectures(hf_config)

    # Propagate the user-requested dtype to the top-level config and every nested
    # sub-config (text/vision/audio). Multimodal models like Gemma4 build their
    # sub-towers via AutoModel.from_config(sub_config), which reads torch_dtype
    # from the sub-config; without this, sub-towers stay at the checkpoint dtype
    # while directly instantiated modules (lm_head, embed_vision, embed_audio)
    # take the requested dtype, tripping FSDP2's uniform-dtype check.
    if torch_dtype != "auto":
        _propagate_torch_dtype_to_subconfigs(hf_config, torch_dtype)

    # Streaming BnB loading: when quantization is requested and we're loading from a
    # pretrained checkpoint, use streaming quantization to avoid materializing the full
    # bf16 model in memory. This is critical for unified-memory systems (DGX Spark)
    # and large models (70B+). Can be disabled with AUTOMODEL_BNB_STREAMING=0.
    _bnb_streaming = os.environ.get("AUTOMODEL_BNB_STREAMING", "1") != "0"
    if (
        quantization_config is not None
        and is_pretrained_init
        and not force_hf
        and _bnb_streaming
        and _streaming_bnb_supported(cls, hf_config)
    ):
        try:
            logger.info("Using streaming BnB quantization for memory-efficient loading")
            return _init_model_bnb_streaming(
                cls,
                pretrained_model_name_or_path,
                hf_config,
                attn_implementation,
                torch_dtype,
                quantization_config,
                **kwargs,
            )
        except FileNotFoundError:
            logger.warning(
                "Streaming BnB loading unavailable (no safetensors checkpoint); falling back to standard HF loading."
            )

    # 1. if force_hf is True, use HF model class wrapped with mixin
    if force_hf:
        # Refuse early if HF's loader would dequantize an FP8 checkpoint to a
        # full BF16 model that won't fit in HBM (issue #2114). Both the
        # user-supplied quant config and the one baked into the HF config
        # (e.g. set by _maybe_dequantize_fp8_for_peft) can trigger this path.
        # Only checked for from_pretrained: a from_config build never touches
        # the checkpoint, so there's nothing to dequantize.
        if is_pretrained_init:
            _check_fp8_dequantize_will_fit(hf_config, quantization_config, pretrained_model_name_or_path, force_hf=True)
            _check_fp8_dequantize_will_fit(
                hf_config,
                getattr(hf_config, "quantization_config", None),
                pretrained_model_name_or_path,
                force_hf=True,
            )
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
            _setup_bnb_loading_kwargs(kwargs)
        if is_pretrained_init:
            with skip_random_init():
                model = cls._from_pretrained_parent_class(
                    pretrained_model_name_or_path,
                    *model_args,
                    dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    **kwargs,
                )
            if restore_loaded_dtype:
                _restore_loaded_model_dtype(
                    model,
                    pretrained_model_name_or_path,
                    hf_config,
                    quantization_config,
                    kwargs,
                    requested_dtype=(torch_dtype if torch_dtype != "auto" else None),
                )
        else:
            model = cls._from_config_parent_class(
                hf_config,
                *model_args,
                dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        # Get HF model class and wrap with mixin
        hf_model_cls = type(model)
        try:
            if len(architectures) > 0 and architectures[0] != "NemotronHForCausalLM":
                hf_model_cls = cls._model_mapping[type(hf_config)]
        except KeyError:
            pass  # fallback to use the model class from the model object
        model.__class__ = _get_mixin_wrapped_class(hf_model_cls)
        return False, model

    # 2. If we have a custom model implementation available, we prioritize that over HF
    model_cls = _resolve_custom_model_cls_for_config(hf_config)
    if model_cls is not None:
        if quantization_config is not None:
            # BnB quantization is tightly integrated with HF's from_pretrained weight
            # loading pipeline.  Custom model constructors only create the architecture
            # (no weight loading, no quantization), so we must fall through to the HF
            # path which handles load + quantize atomically.
            logger.info(
                "BnB quantization requested; using HuggingFace model loader for %s "
                "(custom implementations do not support BnB quantization natively).",
                architectures[0],
            )
        else:
            # Download model weights on local rank 0; skip for from_config or local paths
            if pretrained_model_name_or_path:
                _download_model_weights(hf_config, pretrained_model_name_or_path, process_group=process_group, **kwargs)
            logger.info(f"Using custom model implementation for {architectures[0]}")
            kwargs.pop("trust_remote_code", None)
            # Keep the hub loading options before the constructor-argument filter drops
            # them: the generation-config restore below has to read from the same
            # subfolder/revision the weights came from.
            loading_kwargs = {key: kwargs[key] for key in HUB_LOADING_KWARGS if key in kwargs}
            for key in (*HUB_LOADING_KWARGS, "_commit_hash", "code_revision", "use_auth_token"):
                kwargs.pop(key, None)
            # Treat config-related kwargs as config overrides (HF behavior) and
            # avoid forwarding them into model __init__.
            init_param_names = _get_init_param_names(model_cls)
            config_overrides = dict_config_overrides | _consume_config_overrides(
                hf_config, kwargs, init_param_names=init_param_names
            )
            kwargs = _filter_kwargs_for_init(model_cls, kwargs)
            # Coerce plain-dict backend (e.g. from CLI --model.backend.attn sdpa) to BackendConfig
            if "backend" in kwargs and isinstance(kwargs["backend"], dict):
                backend_config_resolver = getattr(model_cls, "backend_config_resolver", None)
                if backend_config_resolver is not None:
                    kwargs["backend"] = backend_config_resolver(kwargs["backend"])
                else:
                    from nemo_automodel.components.models.common.utils import BackendConfig

                    kwargs["backend"] = BackendConfig(**kwargs["backend"])
            with local_torch_dtype(torch_dtype, model_cls.__name__):
                model = model_cls(hf_config, *model_args, **kwargs)
            if is_pretrained_init:
                # Custom constructors only see the config. Restore the checkpoint's
                # generation settings the way PreTrainedModel.from_pretrained does:
                # generation_config.json carries stop tokens and sampling defaults that
                # the config lacks, and the consolidated export writes
                # model.generation_config back out.
                restore_pretrained_generation_config(
                    model, pretrained_model_name_or_path, config_overrides=config_overrides, **loading_kwargs
                )
            return True, model

    # 3. fallback to HF model class wrapped with mixin
    model = None
    # Same FP8-dequantize OOM guard as the force_hf branch (issue #2114): an
    # fp8 checkpoint with dequantize=True but no custom streaming model class
    # also reaches HF's full-materialize loader below (e.g. when PEFT set
    # dequantize=True on the native config). Tensor parallelism can't save it.
    # Skipped for from_config builds, which never materialize a checkpoint.
    if is_pretrained_init:
        _check_fp8_dequantize_will_fit(hf_config, quantization_config, pretrained_model_name_or_path, force_hf=False)
        _check_fp8_dequantize_will_fit(
            hf_config,
            getattr(hf_config, "quantization_config", None),
            pretrained_model_name_or_path,
            force_hf=False,
        )
    # Serialize HF custom-code cache population across ranks to avoid a partial-copy race.
    _prepopulate_remote_code_cache(hf_config, pretrained_model_name_or_path, kwargs, process_group=process_group)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        _setup_bnb_loading_kwargs(kwargs)
    # For trust_remote_code custom configs, pre-resolve the model class so we
    # can strip yaml-level config-attr kwargs (e.g. ``dlm_paradigm``) that the
    # custom ``__init__`` may not accept. Without this, HF forwards them as
    # model __init__ kwargs and the call fails if the remote class tightened
    # its signature. ``_consume_config_overrides`` applies these to hf_config
    # and removes them from kwargs. ``getattr`` guards against test mocks
    # where ``cls`` may not expose ``__name__``.
    cls_name = getattr(cls, "__name__", None)
    if isinstance(cls_name, str):
        target_auto_class_name = cls_name[4:] if cls_name.startswith("NeMo") else cls_name
        remote_model_cls = _try_get_remote_code_model_cls(
            hf_config, pretrained_model_name_or_path, target_auto_class_name, kwargs
        )
        if remote_model_cls is not None:
            init_param_names = _get_init_param_names(remote_model_cls)
            _consume_config_overrides(hf_config, kwargs, init_param_names=init_param_names)
    if is_pretrained_init:
        with skip_random_init():
            model = cls._from_pretrained_parent_class(
                pretrained_model_name_or_path,
                *model_args,
                dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        if restore_loaded_dtype:
            _restore_loaded_model_dtype(
                model,
                pretrained_model_name_or_path,
                hf_config,
                quantization_config,
                kwargs,
                requested_dtype=(torch_dtype if torch_dtype != "auto" else None),
            )
    else:
        model = cls._from_config_parent_class(
            hf_config,
            *model_args,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    # Get HF model class and wrap with mixin
    hf_model_cls = type(model)
    try:
        if len(architectures) > 0 and architectures[0] != "NemotronHForCausalLM":
            hf_model_cls = cls._model_mapping[type(hf_config)]
    except KeyError:
        pass  # fallback to use the model class from the model object
    model.__class__ = _get_mixin_wrapped_class(hf_model_cls)
    return False, model


def _tie_weights_nemo(model):
    if not hasattr(model, "_nemo_tied_weights_keys"):
        return

    # Only re-tie when the controlling config flag actually requests tied
    # embeddings. ``_nemo_tied_weights_keys`` names the *candidate* tied keys
    # (pre-v5 list-form ``_tied_weights_keys`` semantics); it does not mean the
    # model is tied. Re-tying an untied model here would alias away the trained
    # ``lm_head.weight`` that ``from_pretrained`` just loaded (see #2941).
    config = getattr(model, "config", None)
    if config is not None and not checkpoint_utils.is_tied_word_embeddings(model):
        return

    def get_module_by_fqn(model, fqn):
        from functools import reduce

        fqn = fqn.split(".")
        if fqn[-1] == "weight":
            fqn = fqn[:-1]
        return reduce(getattr, fqn, model)

    for k, v in model._nemo_tied_weights_keys.items():
        get_module_by_fqn(model, k).weight = get_module_by_fqn(model, v).weight


def _apply_backend_module_overrides(model: torch.nn.Module, backend: BackendConfig) -> None:
    """Apply generic backend choices to standard modules left by model constructors.

    Model-owned constructors remain responsible for specialized projections and
    normalization layers. This pass only replaces exact PyTorch ``Linear`` and
    ``RMSNorm`` modules, preserving their parameter objects so checkpoint keys,
    tied weights, optimizer-visible identities, dtype, and device remain unchanged.

    Args:
        model: Newly constructed model whose standard child modules may need a
            backend-specific implementation.
        backend: Backend selection to apply.
    """
    replacements: dict[int, torch.nn.Module] = {}
    visited: set[int] = set()
    pending = [model]

    while pending:
        parent = pending.pop()
        if id(parent) in visited:
            continue
        visited.add(id(parent))

        for name, child in tuple(parent._modules.items()):
            if child is None:
                continue

            replacement = replacements.get(id(child))
            if replacement is None and backend.linear == "quack" and type(child) is torch.nn.Linear:
                replacement = initialize_linear_module(
                    "quack",
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                replacement.weight = child.weight
                replacement.bias = child.bias
            elif replacement is None and backend.rms_norm == "quack" and type(child) is torch.nn.RMSNorm:
                normalized_shape = tuple(child.normalized_shape)
                if len(normalized_shape) == 1:
                    parameter = child.weight
                    replacement = initialize_rms_norm_module(
                        "quack",
                        normalized_shape[0],
                        eps=child.eps,
                        device=parameter.device if parameter is not None else None,
                        dtype=parameter.dtype if parameter is not None else torch.get_default_dtype(),
                    )
                    replacement.weight = parameter

            if replacement is not None:
                replacement.train(child.training)
                replacements[id(child)] = replacement
                setattr(parent, name, replacement)
            else:
                pending.append(child)


def _init_model(
    cls,
    pretrained_model_name_or_path_or_config,
    attn_implementation,
    torch_dtype,
    quantization_config,
    force_hf,
    *model_args,
    **kwargs,
):
    requested_backend = kwargs.get("backend")
    if isinstance(requested_backend, dict):
        requested_backend = BackendConfig(**requested_backend)

    is_custom_model, model = __init_model(
        cls,
        pretrained_model_name_or_path_or_config,
        attn_implementation,
        torch_dtype,
        quantization_config,
        force_hf,
        *model_args,
        **kwargs,
    )
    if is_custom_model and requested_backend is not None:
        _apply_backend_module_overrides(model, requested_backend)
        if not hasattr(model, "backend"):
            model.backend = requested_backend

    # https://github.com/NVIDIA-NeMo/Automodel/blob/a3a57176f68add7917faaa32f19228f49fcbb1ba/examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml#L41
    # this happens in nemotron_flash, where we load using force_hf, and the model is pre 5.x
    #
    # for safety, we tied weights after _model_init. We could do the tying in post_init, but it could be overwritten.
    # So the sequence is roughly:
    #   1. HF constructs NemotronFlashForCausalLM(config).
    #   2. Inside that constructor, self.post_init() runs.
    #   3. Only after construction returns does from_pretrained() finish loading/applying checkpoint weights.
    #   4. That later load can assign lm_head.weight and model.embed_tokens.weight separately, which breaks any alias we create inside post_init().

    if hasattr(model, "_nemo_tied_weights_keys"):
        _tie_weights_nemo(model)
    return is_custom_model, model


def get_architectures(hf_config):
    """
    Get the architectures from the HF config.
    """
    architectures = []
    if hasattr(hf_config, "architectures"):
        architectures = hf_config.architectures or []
    return architectures


def _get_init_param_names(model_cls) -> set[str]:
    """
    Best-effort extraction of explicit __init__ parameter names (excluding `self`).

    Returns an empty set if the signature cannot be inspected.
    """
    try:
        sig = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return set()
    return {k for k in sig.parameters.keys() if k != "self"}


def _try_get_remote_code_model_cls(hf_config, pretrained_model_name_or_path, target_auto_class_name, kwargs):
    """Resolve the model class for a ``trust_remote_code`` custom config.

    Looks up ``hf_config.auto_map`` for ``target_auto_class_name`` (falling back
    to ``AutoModel``) and loads the referenced class via HF's dynamic-module
    loader. Returns ``None`` if the lookup or load fails for any reason — the
    caller should treat that as "no pre-resolution available" and fall through
    to HF's standard handling.
    """
    if not kwargs.get("trust_remote_code"):
        return None
    if hf_config is None or pretrained_model_name_or_path is None:
        return None
    auto_map = getattr(hf_config, "auto_map", None)
    if not auto_map:
        return None
    class_ref = auto_map.get(target_auto_class_name) or auto_map.get("AutoModel")
    if class_ref is None:
        return None
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        return get_class_from_dynamic_module(
            class_ref,
            pretrained_model_name_or_path,
            revision=kwargs.get("revision"),
            code_revision=kwargs.get("code_revision"),
            cache_dir=kwargs.get("cache_dir"),
            local_files_only=kwargs.get("local_files_only", False),
        )
    except Exception:
        return None


def _consume_config_overrides(config, kwargs: dict, *, init_param_names: set[str] | None = None) -> set[str]:
    """
    Mimic HF from_pretrained behavior: treat config-related kwargs as config overrides,
    not model __init__ kwargs.

    For custom model implementations we instantiate via `model_cls(config, **kwargs)`,
    so passing config flags like `output_hidden_states` would crash. This helper moves
    such keys onto the config and removes them from `kwargs`.

    Returns:
        The config field names the caller explicitly set. A value that equals the
        field's default is indistinguishable from an unset one afterwards, so
        consumers that must honor deliberate values need this set.
    """
    applied: set[str] = set()
    if init_param_names is None:
        init_param_names = set()
    # Prefer `to_dict()` to capture the canonical set of config fields.
    try:
        config_keys = set(config.to_dict().keys())
    except Exception:
        config_keys = set(getattr(config, "__dict__", {}).keys())

    for k in list(kwargs.keys()):
        # If the model explicitly declares this kwarg, keep it for __init__.
        if k in init_param_names:
            continue
        # Otherwise, if it looks like a config field, apply it to config.
        if k in config_keys:
            val = kwargs.pop(k)
            # Deep-merge dict overrides into existing sub-config objects (e.g.
            # text_config={"router_aux_loss_coef": 0}) instead of replacing the
            # entire sub-config, which would lose all other fields.
            applied.add(k)
            if isinstance(val, dict):
                existing = getattr(config, k, None)
                if existing is not None and hasattr(existing, "to_dict"):
                    for sub_k, sub_v in val.items():
                        setattr(existing, sub_k, sub_v)
                    continue
            setattr(config, k, val)
    return applied


def _filter_kwargs_for_init(model_cls, kwargs: dict) -> dict:
    """
    Filter kwargs down to what `model_cls.__init__` explicitly accepts.

    If the constructor has a `**kwargs` parameter (VAR_KEYWORD) or signature cannot be
    inspected, returns kwargs unchanged.
    """
    try:
        sig = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    # We pass `config` positionally.
    allowed.discard("config")
    return {k: v for k, v in kwargs.items() if k in allowed}


def resolve_sdpa_method(
    sdpa_method: list | None = None,
    device_mesh=None,
    activation_checkpointing: bool = False,
) -> list["SDPBackend"] | None:  # noqa: F821
    """Resolve SDPA backend list from config strings or runtime constraints.

    When *sdpa_method* is provided (e.g. from YAML), string values are
    converted to :class:`torch.nn.attention.SDPBackend` enum members.
    Already-resolved ``SDPBackend`` values are passed through unchanged.
    When ``None``, automatic defaults are applied based on context
    parallelism and activation checkpointing settings.

    Valid string values (case-insensitive): ``flash_attention``,
    ``efficient_attention``, ``math``, ``cudnn_attention``.

    Args:
        sdpa_method: List of backend name strings or SDPBackend enum values,
            or ``None`` to use automatic defaults.
        device_mesh: Device mesh for distributed training.
        activation_checkpointing: Whether activation checkpointing is enabled.

    Returns:
        Ordered list of :class:`SDPBackend` members, or ``None`` to use
        PyTorch's default selection.
    """
    from torch.nn.attention import SDPBackend

    _NAME_TO_BACKEND = dict(SDPBackend.__members__)

    if sdpa_method is not None:
        backends = []
        for entry in sdpa_method:
            if isinstance(entry, str):
                key = entry.upper()
                if key not in _NAME_TO_BACKEND:
                    raise ValueError(f"Unknown SDPA backend '{entry}'. Valid values: {sorted(_NAME_TO_BACKEND.keys())}")
                backends.append(_NAME_TO_BACKEND[key])
            else:
                backends.append(entry)
        return backends

    # Auto-select based on runtime constraints
    cp_size = 1
    if device_mesh is not None and "cp" in device_mesh.mesh_dim_names:
        cp_size = device_mesh["cp"].size()

    if cp_size > 1:
        # CP with DTensor only supports flash and efficient backends;
        # MATH is not compatible with DTensor.
        return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    elif activation_checkpointing:
        # For activation checkpointing, disable cudnn SDPA backend because
        # it may not be selected during recomputation, causing:
        # "Recomputed values have different metadata than during forward pass."
        return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    return None
