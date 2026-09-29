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

"""NeMo Auto Model classes.

Drop-in replacements for ``transformers.AutoModelFor*`` that add custom-kernel
patching, distributed infrastructure, PEFT, quantization, and checkpointing.

Heavy-lifting helpers live in sibling modules:

* ``kernel_patches`` -- SDPA / Liger kernel patching
* ``model_init`` -- model class resolution and instantiation
* ``infrastructure`` -- MeshContext, sharding, PEFT/quant application
"""

import gc
import inspect
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch.nn.attention import SDPBackend

from nemo_automodel.shared.torch_patches import apply_torch_patches

apply_torch_patches()
from huggingface_hub import constants as hf_constants  # noqa: E402
from huggingface_hub import snapshot_download
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
)
from transformers.initialization import no_init_weights  # noqa: E402
from transformers.models.auto.auto_factory import _BaseAutoModelClass  # noqa: E402
from transformers.utils import ContextManagers  # noqa: E402

from nemo_automodel.components.distributed.config import DistributedSetup  # noqa: E402
from nemo_automodel.components.distributed.init_utils import get_world_size_safe  # noqa: E402
from nemo_automodel.components.utils.model_utils import (  # noqa: E402
    init_empty_weights,
    resolve_trust_remote_code,
)
from nemo_automodel.shared.utils import dtype_from_str  # noqa: E402

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.quantization.fp8 import FP8Config
    from nemo_automodel.components.utils.compile_utils import CompileConfig

#  Re-exports from sibling modules (backward compatibility)
# Backward-compat shim for trust_remote_code models (e.g. DeciLM)
# that import NEED_SETUP_CACHE_CLASSES_MAPPING from transformers.generation.utils.
import transformers.generation.utils as _gen_utils  # noqa: E402

from nemo_automodel._transformers.auto_config import NeMoAutoConfig as AutoConfig
from nemo_automodel._transformers.infrastructure import (
    MeshContext,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from nemo_automodel._transformers.kernel_patches import (
    DEFAULT_ATTN_IMPLEMENTATION,
    _apply_preload_overrides,
    _get_next_fallback_attn,
    _patch_attention,
    _patch_liger_kernel,
    _verify_sdpa_support,
    apply_model_runtime_patches,
)
from nemo_automodel._transformers.model_init import (
    _consume_config_overrides,
    _init_model,
    get_hf_config,
    get_is_hf_model,
    no_hf_meta_device,
    resolve_sdpa_method,
)
from nemo_automodel.components.models.common.tie_word_embeddings import reject_tie_word_embeddings_flip

if not hasattr(_gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
    from transformers.cache_utils import StaticCache

    _gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache}


logger = logging.getLogger(__name__)

_MAX_BUILD_RETRIES = 5

_remote_code_compat_applied = False
_DISTRIBUTED_SETUP_ONLY_KWARGS = {
    "moe_mesh",
    "distributed_config",
    "pipeline_config",
    "moe_config",
    "activation_checkpointing",
    "tp_plan",
}


def _reject_separate_distributed_kwargs(kwargs: dict) -> None:
    provided = sorted(_DISTRIBUTED_SETUP_ONLY_KWARGS & set(kwargs))
    if provided:
        raise TypeError(
            "Distributed settings must be passed with distributed_setup; "
            f"separate distributed kwargs are not accepted: {provided}"
        )


def _resolve_distributed_setup(
    *,
    distributed_setup: DistributedSetup | None,
    device_mesh: Optional["DeviceMesh"] = None,
) -> DistributedSetup:
    """Return a setup, upcasting raw mesh inputs into topology-only setup."""
    if distributed_setup is not None:
        if device_mesh is not None:
            raise ValueError("Pass either distributed_setup or device_mesh, not both")
        return distributed_setup

    if isinstance(device_mesh, MeshContext):
        raise TypeError("device_mesh expects a DeviceMesh; pass DistributedSetup for MeshContext or MoE topology")

    return DistributedSetup(mesh_context=MeshContext.from_meshes(device_mesh))


def _patch_remote_code_compat():
    """Patch ``_finalize_model_loading`` for remote-code models written against older transformers.

    Remote-code models (``trust_remote_code=True``) may be incompatible with
    the installed transformers in several ways:

    1. Missing ``all_tied_weights_keys`` -- set in ``post_init()`` which the
       model may never call.
    2. Overridden ``tie_weights()`` with an old signature that doesn't accept
       the ``missing_keys`` kwarg added in newer transformers.

    This one-time patch wraps ``_finalize_model_loading`` to fix these issues
    on the fly.  For models that are already compatible the guards are no-ops.
    """
    global _remote_code_compat_applied
    if _remote_code_compat_applied:
        return
    _orig_finalize = PreTrainedModel._finalize_model_loading

    def _compat_finalize(model, load_config, loading_info):
        # 1. Ensure all_tied_weights_keys exists
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(all_submodels=True)

        # 2. Wrap tie_weights if it doesn't accept `missing_keys`
        model_cls = type(model)
        if model_cls.tie_weights is not PreTrainedModel.tie_weights:
            sig = inspect.signature(model_cls.tie_weights)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if "missing_keys" not in sig.parameters and not has_var_kw:
                _orig_tie = model_cls.tie_weights

                def _compat_tie(self, **kwargs):
                    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    return _orig_tie(self, **accepted)

                model_cls.tie_weights = _compat_tie

        # 3. Set missing config defaults that older remote code models expect
        _config_defaults = {"use_cache": False}
        for attr, default in _config_defaults.items():
            if not hasattr(model.config, attr):
                setattr(model.config, attr, default)

        return _orig_finalize(model, load_config, loading_info)

    PreTrainedModel._finalize_model_loading = staticmethod(_compat_finalize)
    _remote_code_compat_applied = True


_AUTO_CONFIG_HUB_KWARG_KEYS = (
    "revision",
    "_commit_hash",
    "force_download",
    "subfolder",
    "token",
    "use_auth_token",
    "cache_dir",
    "local_files_only",
    "code_revision",
)


def _alias_remote_auto_map_for_target(args, kwargs, target_key):
    """Return a config with ``auto_map[target_key]`` aliased from ``auto_map[AutoModel]``.

    When a trust-remote-code model card ships ``auto_map`` with only an
    ``AutoModel`` entry (and no ``target_key`` such as
    ``AutoModelForCausalLM``), HF's auto resolution fails. For models that
    are otherwise causal LMs this helper copies the existing ``AutoModel``
    class reference into ``target_key`` so resolution succeeds on retry.

    Returns the patched ``PretrainedConfig`` if patching applied, else
    ``None``.
    """
    if not kwargs.get("trust_remote_code"):
        return None
    config = kwargs.get("config")
    if config is None:
        if not args:
            return None
        pretrained_path = args[0]
        hub_kwargs = {k: kwargs[k] for k in _AUTO_CONFIG_HUB_KWARG_KEYS if k in kwargs}
        try:
            config = AutoConfig.from_pretrained(
                pretrained_path,
                trust_remote_code=True,
                **hub_kwargs,
            )
        except Exception:
            return None
    auto_map = getattr(config, "auto_map", None)
    if not auto_map or target_key in auto_map or "AutoModel" not in auto_map:
        return None
    config.auto_map = dict(auto_map)
    config.auto_map[target_key] = auto_map["AutoModel"]
    return config


def _maybe_dequantize_fp8_for_peft(hf_native_quant_cfg, peft_config, pretrained_path):
    """Set ``dequantize=True`` on FP8 quantization configs when PEFT is requested.

    Returns True if the config was mutated, False otherwise.
    """
    if peft_config is not None and isinstance(pretrained_path, str):
        if isinstance(hf_native_quant_cfg, dict) and hf_native_quant_cfg.get("quant_method") == "fp8":
            hf_native_quant_cfg["dequantize"] = True
            logger.info("FP8 model with PEFT: setting dequantize=True for compatibility")
            return True
    return False


def _maybe_reject_tie_word_embeddings_flip(pretrained_model_name_or_path, hf_config, kwargs):
    """Reject a from_pretrained request that flips tie_word_embeddings from the checkpoint.

    Re-reads the checkpoint's raw config (no user value-overrides) and compares its
    controlling tie flag to the requested ``hf_config`` via
    :func:`reject_tie_word_embeddings_flip`. Conservative by design: path-like sources
    are normalized with :func:`os.fspath`, non-path sources are skipped, and it silently
    returns if the raw config cannot be re-read, so it never blocks a load except on a
    genuine flip.

    Args:
        pretrained_model_name_or_path: The from_pretrained source (``str`` and
            ``os.PathLike`` are checked; anything else is skipped).
        hf_config: The resolved config with user overrides applied (the requested value).
        kwargs: The from_pretrained kwargs (hub-locating keys are reused for the raw load).
    """
    if isinstance(pretrained_model_name_or_path, os.PathLike):
        pretrained_model_name_or_path = os.fspath(pretrained_model_name_or_path)
    if not isinstance(pretrained_model_name_or_path, str):
        # Non-path source (e.g. bytes fspath or preloaded object): nothing to re-read.
        return
    hub_kwargs = {k: kwargs[k] for k in _AUTO_CONFIG_HUB_KWARG_KEYS if k in kwargs}
    try:
        raw_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", resolve_trust_remote_code(pretrained_model_name_or_path)),
            **hub_kwargs,
        )
    except Exception:
        # Cannot re-read the raw config (offline / custom loader); do not block the load.
        return
    architectures = getattr(hf_config, "architectures", None) or []
    model_class_name = architectures[0] if architectures else type(hf_config).__name__
    reject_tie_word_embeddings_flip(raw_config, hf_config, model_class_name)


class _BaseNeMoAutoModelClass(_BaseAutoModelClass):
    """
    Drop-in replacement for ``_BaseAutoModelClass`` that includes custom-kernels.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    TODO(@akoumpa): extend this beyond liger_kernel.

    Notes:
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.
    """

    @classmethod
    def _from_pretrained_parent_class(cls, *args, **kwargs):
        name = cls.__name__
        if name.startswith("NeMo"):
            cls.__name__ = name[4:]
        try:
            model = super().from_pretrained(*args, **kwargs)
        except (AttributeError, TypeError) as e:
            if "all_tied_weights_keys" in str(e) or (isinstance(e, TypeError) and "tie_weights" in str(e)):
                logger.warning(
                    "Remote code model incompatible with installed transformers (%s). "
                    "Applying compatibility patches and retrying.",
                    e,
                )
                _patch_remote_code_compat()
                model = super().from_pretrained(*args, **kwargs)
            else:
                raise
        except ValueError as e:
            # Some trust-remote-code model cards ship config.json with only
            # auto_map["AutoModel"] and omit the AutoModelForCausalLM (etc.)
            # entry, which makes HF's resolution fail. If the model is
            # otherwise a causal LM, alias the existing AutoModel mapping to
            # the requested target key and retry.
            target_key = cls.__name__
            if "Unrecognized configuration class" in str(e) and target_key in str(e):
                patched = _alias_remote_auto_map_for_target(args, kwargs, target_key)
                if patched is not None:
                    logger.warning(
                        "Model config.json missing auto_map[%s]; aliasing from auto_map[AutoModel] and retrying.",
                        target_key,
                    )
                    kwargs = dict(kwargs, config=patched)
                    model = super().from_pretrained(*args, **kwargs)
                else:
                    raise
            else:
                raise
        except OSError:
            if kwargs.get("use_safetensors") is not False:
                logger.warning(
                    "Checkpoint resolution failed; retrying with use_safetensors=False "
                    "(the model may only provide .bin checkpoints)."
                )
                kwargs["use_safetensors"] = False
                model = super().from_pretrained(*args, **kwargs)
            else:
                raise
        finally:
            cls.__name__ = name
        return model

    @classmethod
    def _from_config_parent_class(cls, *args, **kwargs):
        name = cls.__name__
        if name.startswith("NeMo"):
            cls.__name__ = name[4:]
        model = super().from_config(*args, **kwargs)
        cls.__name__ = name
        return model

    @classmethod
    def _build_model(
        cls,
        pretrained_model_name_or_path_or_config,
        *model_args,
        is_hf_model,
        use_liger_kernel,
        use_sdpa_patching,
        sdpa_method,
        torch_dtype,
        attn_implementation,
        quantization_config,
        force_hf,
        model_wrapper,
        parallelize_fn,
        mesh,
        peft_config,
        fp8_config,
        compile_config,
        load_base_model,
        _retry_depth=0,
        **kwargs,
    ):
        """Shared model building logic for ``from_pretrained`` and ``from_config``.

        Handles pre-load overrides, meta-device initialization, model init with
        attention-fallback retry, kernel patching (Liger, SDPA) with retry, and
        full infrastructure application (sharding, PEFT, quantization, checkpointing).

        All caller-specific setup (config resolution, infrastructure instantiation,
        ``is_hf_model`` determination) is done by ``from_pretrained`` / ``from_config``
        before delegating here.
        """
        # Extract values consumed by pop; preserve them for retry.
        kwargs = dict(kwargs)  # Defensive copy so retries get clean state
        has_packed_sequence = kwargs.pop("has_packed_sequence", False)
        freeze_config = kwargs.pop("freeze_config", None)
        cache_dir = kwargs.pop("cache_dir", hf_constants.HF_HUB_CACHE)
        if isinstance(pretrained_model_name_or_path_or_config, str):
            # Hub loaders need the cache location; from_config constructors do not.
            kwargs["cache_dir"] = cache_dir

        def _retry(**override):
            """Re-enter ``_build_model`` with overridden parameters."""
            if _retry_depth >= _MAX_BUILD_RETRIES:
                raise
            retry_kwargs = {
                **kwargs,
                "has_packed_sequence": has_packed_sequence,
                "freeze_config": freeze_config,
                "cache_dir": cache_dir,
            }
            return cls._build_model(
                pretrained_model_name_or_path_or_config,
                *model_args,
                is_hf_model=is_hf_model,
                use_liger_kernel=override.get("use_liger_kernel", use_liger_kernel),
                use_sdpa_patching=override.get("use_sdpa_patching", use_sdpa_patching),
                sdpa_method=sdpa_method,
                torch_dtype=torch_dtype,
                attn_implementation=override.get("attn_implementation", attn_implementation),
                quantization_config=quantization_config,
                force_hf=force_hf,
                model_wrapper=model_wrapper,
                parallelize_fn=parallelize_fn,
                mesh=mesh,
                peft_config=peft_config,
                fp8_config=fp8_config,
                compile_config=compile_config,
                load_base_model=load_base_model,
                _retry_depth=_retry_depth + 1,
                **retry_kwargs,
            )

        # ``attn_implementation="te"`` is a NeMo extension: route through SDPA
        # for model init and inject TE DotProductAttention post-init.
        inject_te_attention = attn_implementation == "te"
        if inject_te_attention:
            logger.info("attn_implementation='te' requested: using 'sdpa' for model init and will inject TE post-init.")
            attn_implementation = "sdpa"

        # FFPA backend setup (validate + register) lives in ffpa_attention; run it
        # before _apply_preload_overrides, which would otherwise rewrite ffpa → sdpa/
        # flash_attention_2 for HF models.
        if attn_implementation == "ffpa":
            from nemo_automodel.components.attention.ffpa_attention import setup_ffpa_backend

            setup_ffpa_backend(mesh.cp_size, has_packed_sequence)

        if is_hf_model:
            attn_implementation, use_liger_kernel = _apply_preload_overrides(
                mesh.tp_size,
                mesh.cp_size,
                has_packed_sequence,
                attn_implementation,
                use_liger_kernel,
            )
            # If preload overrides changed attn_implementation away from "sdpa"
            # (e.g., packed sequence forces "flash_attention_2"), TE injection
            # cannot intercept F.scaled_dot_product_attention; skip it.
            if inject_te_attention and attn_implementation != "sdpa":
                logger.warning(
                    "TE attention injection requires SDPA but attn_implementation was overridden to '%s'. "
                    "Skipping TE injection.",
                    attn_implementation,
                )
                inject_te_attention = False
        device = torch.cuda.current_device()

        # When PEFT is requested, force dequantization of FP8-quantized models.
        # FP8 linear modules (e.g. transformers FP8Linear) have scalar parameters
        # incompatible with FSDP2, and their custom forward doesn't compose with
        # LoRA patching. Setting dequantize=True tells transformers to convert
        # FP8 weights to bf16 during loading.
        _hf_config = (
            get_hf_config(pretrained_model_name_or_path_or_config, attn_implementation, **kwargs)
            if isinstance(pretrained_model_name_or_path_or_config, str)
            else pretrained_model_name_or_path_or_config
        )
        _hf_native_quant_cfg = getattr(_hf_config, "quantization_config", None)
        if _maybe_dequantize_fp8_for_peft(_hf_native_quant_cfg, peft_config, pretrained_model_name_or_path_or_config):
            # Only HF's from_pretrained needs `config` in kwargs (it would otherwise
            # re-read config from disk and lose the in-memory dequantize=True mutation).
            # Custom models receive _hf_config positionally in model_init.py and would
            # collide with kwargs["config"] (issue #2164).
            if is_hf_model:
                kwargs["config"] = _hf_config

        # Use meta device initialization when:
        # - either multi-GPU (world_size > 1) or single-GPU custom model (not HF)
        # - AND not using quantization (we let HF handle BitsAndBytes/FP8; don't init meta device)
        #   For non-HF models, native quant config is ignored.
        is_meta_device = all(
            [
                get_world_size_safe() > 1 or not is_hf_model,
                quantization_config is None and (_hf_native_quant_cfg is None or not is_hf_model),
            ]
        )
        init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()

        model = None  # Ensure 'model' is always bound for the except handler
        is_custom_model = None
        process_group = getattr(mesh, "process_group", None)
        try:
            with init_ctx:
                is_custom_model, model = _init_model(
                    cls,
                    pretrained_model_name_or_path_or_config,
                    attn_implementation,
                    torch_dtype,
                    quantization_config,
                    force_hf,
                    *model_args,
                    _process_group=process_group,
                    **kwargs,
                )
        except (NotImplementedError, RuntimeError) as e:
            _meta_err_msgs = (
                "Cannot copy out of meta tensor",
                "cannot be called on meta tensors",
                "aten::equal: attempted to run this operator with Meta tensors",
            )
            # if the error message contains any of the meta-tensor error messages, retry without meta-device init
            # When force_hf is True, we may still encounter the error, even tho is_meta_device is False
            # automodel /opt/Automodel/examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml is a good example
            if any(msg in str(e) for msg in _meta_err_msgs):
                logger.warning(
                    "Model init hit meta-tensor error (%s); retrying without meta-device init.",
                    type(e).__name__,
                )
                del model
                model = None
                gc.collect()
                is_meta_device = False
                with ContextManagers([no_init_weights(), no_hf_meta_device()]):
                    is_custom_model, model = _init_model(
                        cls,
                        pretrained_model_name_or_path_or_config,
                        attn_implementation,
                        torch_dtype,
                        quantization_config,
                        force_hf,
                        *model_args,
                        _process_group=process_group,
                        **kwargs,
                    )
            else:
                raise
        except ValueError as e:
            if "does not support" in str(e):
                del model
                attn_implementation = _get_next_fallback_attn(attn_implementation)
                logger.warning("Falling back to %s attention.", attn_implementation)
                return _retry(attn_implementation=attn_implementation)
            raise

        model = apply_model_runtime_patches(model, mesh)

        # Kernel patching
        try:
            if use_liger_kernel and not is_custom_model:
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logger.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        try:
            if use_sdpa_patching and (not is_custom_model or sdpa_method is not None):
                model = _patch_attention(model, sdpa_method)  # noqa: F821
        except Exception:
            logger.warning("Retrying without SDPA patching.")
            return _retry(use_sdpa_patching=False)

        # Resolve pretrained path for checkpoint loading
        is_pretrained = isinstance(pretrained_model_name_or_path_or_config, str)
        pretrained_path = (
            pretrained_model_name_or_path_or_config
            if is_pretrained
            else getattr(pretrained_model_name_or_path_or_config, "name_or_path", "")
        )

        if is_hf_model and attn_implementation != "magi":
            _verify_sdpa_support(model, mesh.cp_size)

        # HF from_pretrained on a real device loads (and potentially quantizes) weights
        # during init.  Custom models and meta-device initialization do not load weights
        # here; they rely on apply_model_infrastructure to load the checkpoint later.
        weights_already_loaded = not is_custom_model and not is_meta_device and load_base_model
        if load_base_model and pretrained_path and not os.path.isdir(pretrained_path) and _hf_config._commit_hash:
            # Give the checkpointer the selected snapshot, not a repo id whose
            # cache may also contain an unrelated (or newer) weight index.
            pretrained_path = snapshot_download(
                pretrained_path,
                revision=_hf_config._commit_hash,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            pretrained_path = os.path.join(pretrained_path, kwargs.get("subfolder", ""))

        from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate

        attach_capabilities_and_validate(model, mesh)

        model = apply_model_infrastructure(
            model=model,
            pretrained_model_name_or_path=pretrained_path,
            mesh=mesh,
            peft_config=peft_config,
            quantization_config=quantization_config,
            fp8_config=fp8_config,
            parallelize_fn=parallelize_fn,
            model_wrapper=model_wrapper,
            is_meta_device=is_meta_device,
            device=device,
            compile_config=compile_config,
            load_base_model=load_base_model,
            cache_dir=cache_dir,
            freeze_config=freeze_config,
            weights_already_loaded=weights_already_loaded,
            inject_te_attention=inject_te_attention,
        )

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: List[Union[SDPBackend, str]] | None = None,
        torch_dtype="auto",
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        quantization_config=None,
        force_hf: bool = False,
        distributed_setup: DistributedSetup | None = None,
        device_mesh: Optional["DeviceMesh"] = None,
        peft_config: dict | None = None,
        fp8_config: Optional["FP8Config"] = None,
        compile_config: Optional["CompileConfig"] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Instantiate and (optionally) patch a causal-language model.

        This is a light wrapper around
        `transformers.AutoModelForCausalLM.from_pretrained` that can
        automatically apply Liger and/or SDPA (scaled-dot-product
        attention) kernel optimizations, as well as PEFT, quantization,
        and distributed parallelism.

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Hugging Face
                hub repo ID or local path accepted by
                `AutoModelForCausalLM.from_pretrained`.
            *model_args: Positional arguments forwarded verbatim to
                `AutoModelForCausalLM.from_pretrained`.
            use_liger_kernel (bool, default=True): If `True`, try to patch
                the model with Liger kernels for faster inference/training.
            use_sdpa_patching (bool, default=True): If `True`, patch the
                model with SDPA-based attention optimizations.
            sdpa_method (list[SDPBackend | str] | None, optional): Explicit list of
                SDPA back-ends to consider when `use_sdpa_patching=True`.
                Accepts both SDPBackend enum values and string names (e.g.
                ``["flash_attention", "efficient_attention"]``). When ``None``,
                auto-selects based on CP and activation checkpointing.
            torch_dtype (str | torch.dtype):
                Deprecated alias for ``dtype``. Defaults to ``auto``.
            attn_implementation (str, optional):
                Specifies which attention implementation to use (e.g.,
                ``"flash_attention_2"``, ``"eager"``). Only applied when the
                base model supports this kwarg. Defaults to ``"flash_attention_2"``,
                if flash attention is not available, defaults to ``"sdpa"``.
            quantization_config (optional): BitsAndBytesConfig configuration object that
                specifies all quantization settings. If provided, quantization
                will be applied to the model.
            force_hf (bool, default=False): If `True`, force the use of HF model implementation.
                If `False`, the model will be loaded using the custom model implementation if available.
            distributed_setup (DistributedSetup | None, optional): Resolved distributed
                topology and policy object. Default: None.
            device_mesh (DeviceMesh | None, optional): Pre-created Hugging Face-style
                device mesh. NeMo wraps it in a topology-only ``DistributedSetup``
                internally. Use ``distributed_setup`` when passing NeMo-specific
                policies such as strategy, MoE, or activation checkpointing.
                Default: None.
            peft_config (dict | None, optional): PEFT/LoRA configuration dictionary.
                If provided, LoRA adapters will be applied to the model. Default: None.
            fp8_config (FP8Config | None, optional): FP8 quantization configuration.
                If provided, FP8 quantization will be applied. Default: None.
            compile_config (CompileConfig | None, optional): Configuration for torch.compile.
                If provided, the model will be compiled. Default: None.
            **kwargs: Additional keyword arguments. Notable ones include:
                - dtype (str | torch.dtype): Model storage dtype. Takes precedence
                  over ``torch_dtype`` when not ``None``; accepts ``auto``.
                - has_packed_sequence (bool): Whether using packed sequences. Default: False.
                - cache_dir (str): Cache directory for model weights.

        Returns:
            transformers.PreTrainedModel: The loaded (and possibly patched)
            model instance with all infrastructure applied.
        """
        _reject_separate_distributed_kwargs(kwargs)
        # Resolve HF's dtype alias before config overrides or model construction
        # can consume it independently of AutoModel's storage-dtype handling.
        if (dtype := kwargs.pop("dtype", None)) is not None:
            torch_dtype = dtype
        setup = _resolve_distributed_setup(
            distributed_setup=distributed_setup,
            device_mesh=device_mesh,
        )
        mesh = setup.mesh_context
        distributed_config = setup.strategy_config
        moe_parallel_config = setup.moe_parallel_config
        activation_checkpointing = setup.activation_checkpointing

        model_wrapper, parallelize_fn = instantiate_infrastructure(
            distributed_config=distributed_config,
            moe_parallel_config=moe_parallel_config,
            activation_checkpointing=activation_checkpointing,
            mesh=mesh,
        )

        try:
            hf_config = get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs)
        except Exception as e:
            if "does not support" in str(e):
                attn_implementation = _get_next_fallback_attn(attn_implementation)
                logger.warning("Config rejected attention implementation, falling back to %s.", attn_implementation)
                hf_config = get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs)
            else:
                raise
        # Keep config rereads, remote code, and weights on the same snapshot.
        if hf_config._commit_hash is not None:
            kwargs["revision"] = hf_config._commit_hash
            kwargs["_commit_hash"] = hf_config._commit_hash
        is_hf_model = get_is_hf_model(hf_config, force_hf)

        # Layer 2: reject loading a checkpoint with tie_word_embeddings flipped from the
        # value it was saved with (the class-level TieSupport policy cannot catch this).
        _maybe_reject_tie_word_embeddings_flip(pretrained_model_name_or_path, hf_config, kwargs)

        sdpa_method = resolve_sdpa_method(sdpa_method, mesh.device_mesh, activation_checkpointing)

        return cls._build_model(
            pretrained_model_name_or_path,
            *model_args,
            is_hf_model=is_hf_model,
            use_liger_kernel=use_liger_kernel,
            use_sdpa_patching=use_sdpa_patching,
            sdpa_method=sdpa_method,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            force_hf=force_hf,
            model_wrapper=model_wrapper,
            parallelize_fn=parallelize_fn,
            mesh=mesh,
            peft_config=peft_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            load_base_model=True,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config,
        *model_args,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: List[Union[SDPBackend, str]] | None = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        quantization_config=None,
        force_hf: bool = False,
        distributed_setup: DistributedSetup | None = None,
        device_mesh: Optional["DeviceMesh"] = None,
        peft_config: dict | None = None,
        fp8_config: Optional["FP8Config"] = None,
        compile_config: Optional["CompileConfig"] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Instantiate a model from a ``transformers.PretrainedConfig`` (no pretrained
        weights). Accepts the same infrastructure arguments as ``from_pretrained``.

        See ``from_pretrained`` for full parameter documentation.

        Args:
            config (transformers.PretrainedConfig | str):
                The configuration object used to build the model.
                If config is passed as a string (e.g., model-id / local checkpoint),
                it will create a config internally using AutoConfig.
            torch_dtype (str | torch.dtype):
                Deprecated alias for ``dtype``. Defaults to ``auto``, which selects ``torch.bfloat16``.
            **kwargs: Additional arguments documented in ``from_pretrained``.
                ``dtype`` takes precedence over ``torch_dtype`` when not ``None``.
        """
        _reject_separate_distributed_kwargs(kwargs)
        if (dtype := kwargs.pop("dtype", None)) is not None:
            torch_dtype = dtype
        setup = _resolve_distributed_setup(
            distributed_setup=distributed_setup,
            device_mesh=device_mesh,
        )
        mesh = setup.mesh_context
        distributed_config = setup.strategy_config
        moe_parallel_config = setup.moe_parallel_config
        activation_checkpointing = setup.activation_checkpointing

        # Only instantiate infrastructure when distributed_config is provided
        model_wrapper = parallelize_fn = None
        if distributed_config is not None:
            model_wrapper, parallelize_fn = instantiate_infrastructure(
                distributed_config=distributed_config,
                moe_parallel_config=moe_parallel_config,
                activation_checkpointing=activation_checkpointing,
                mesh=mesh,
            )

        torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch.bfloat16
        name_or_path = config if isinstance(config, str) else getattr(config, "name_or_path", None)
        kwargs["trust_remote_code"] = kwargs.get(
            "trust_remote_code", resolve_trust_remote_code(name_or_path) if name_or_path else False
        )
        if isinstance(config, str):
            try:
                config = get_hf_config(config, attn_implementation, **kwargs)
            except Exception as e:
                if "does not support" in str(e):
                    attn_implementation = _get_next_fallback_attn(attn_implementation)
                    logger.warning("Config rejected attention implementation, falling back to %s.", attn_implementation)
                    config = get_hf_config(config, attn_implementation, **kwargs)
                else:
                    raise
        _consume_config_overrides(config, kwargs)
        is_hf_model = get_is_hf_model(config, force_hf)

        sdpa_method = resolve_sdpa_method(sdpa_method, mesh.device_mesh, activation_checkpointing)

        return cls._build_model(
            config,
            *model_args,
            is_hf_model=is_hf_model,
            use_liger_kernel=use_liger_kernel,
            use_sdpa_patching=use_sdpa_patching,
            sdpa_method=sdpa_method,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            force_hf=force_hf,
            model_wrapper=model_wrapper,
            parallelize_fn=parallelize_fn,
            mesh=mesh,
            peft_config=peft_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            load_base_model=kwargs.pop("load_base_model", False),
            **kwargs,
        )


#  Concrete Auto-Model classes
class NeMoAutoModelForCausalLM(_BaseNeMoAutoModelClass, AutoModelForCausalLM):
    """
    Drop-in replacement for ``transformers.AutoModelForCausalLM`` that includes custom-kernels.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    Notes:
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.

    Examples:
    --------
    >>> model = NeMoAutoModelForCausalLM.from_pretrained("gpt2")            # try Liger
    >>> model = NeMoAutoModelForCausalLM.from_pretrained(
    ...     "gpt2", use_liger_kernel=False)                                 # skip Liger
    """

    pass


class NeMoAutoModelForImageTextToText(_BaseNeMoAutoModelClass, AutoModelForImageTextToText):
    """Drop-in replacement for ``transformers.AutoModelForImageTextToText`` with custom-kernels.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    Notes:
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.

    Examples:
    --------
    >>> model = NeMoAutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct") # try Liger
    >>> model = NeMoAutoModelForImageTextToText.from_pretrained(
    ...     "Qwen/Qwen2.5-VL-3B-Instruct", use_liger_kernel=False)                            # skip Liger
    """

    pass
