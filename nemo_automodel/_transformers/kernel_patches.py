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

"""Kernel, attention, and model runtime patching utilities.

Functions for SDPA, Liger-kernel, model runtime hooks, and
attention-implementation overrides. These are stateless helpers used during
model construction.
"""

import functools
import importlib
import inspect
import logging
import types

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.shared.import_utils import safe_import

# Check availability without fully importing (avoids CUDA init at import time).
# The actual ``liger_kernel.transformers`` module is imported lazily inside
# ``_patch_liger_kernel`` so that the import (which may trigger Triton JIT /
# CUDA context creation) only happens *after* ``torch.cuda.set_device`` has
# been called by the distributed init code.
HAS_LIGER_KERNEL = importlib.util.find_spec("liger_kernel") is not None
liger_kernel_trf = None  # lazily populated; tests may inject a stub here
HAS_FA, _ = safe_import("flash_attn")
# FA3 ships as dist "flash-attn-3" (module flash_attn_interface); FA4 ships as
# dist "flash-attn-4" but installs into the same "flash_attn" import namespace
# as FA2 (subpackage flash_attn.cute), so distribution-aware checks are needed
# rather than plain module imports. transformers >= 4.56 provides them.
try:
    from transformers.utils import is_flash_attn_3_available, is_flash_attn_4_available

    HAS_FA3 = is_flash_attn_3_available()
    HAS_FA4 = is_flash_attn_4_available()
except ImportError:  # older transformers without FA3/FA4 support
    HAS_FA3 = False
    HAS_FA4 = False


def _device_supports_fa3() -> bool:
    """FA3 kernels are compiled for SM90a only (Hopper ISA).

    They do not run on Blackwell (SM100+) or older architectures, and
    ``is_flash_attn_3_available()`` only checks the package, not the device.
    (FA4 JIT-compiles per-arch and supports SM90/SM100+, so no gate is needed there.)
    """
    return torch.cuda.get_device_capability()[0] == 9


# Gate on the actual device so the fallback ladder routes flash_attention_3
# requests to flash_attention_2/sdpa on non-Hopper GPUs instead of failing at
# kernel launch.
if HAS_FA3 and torch.cuda.is_available() and not _device_supports_fa3():
    HAS_FA3 = False
FLASH_ATTN_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if HAS_FA else "sdpa"

logger = logging.getLogger(__name__)

# Models build their CP/fp32-gate-aware modules at construction; no load-time
# runtime patching is registered here. (Qwen3.5 dense/MoE build the native
# CPAwareGatedDeltaNet + fp32 SSMGate in their model __init__.)
_MODEL_RUNTIME_PATCHES = {}


def _set_global_sdpa_backends(sdpa_method):
    """Apply resolved SDPA backend constraints process-wide for checkpoint recompute."""
    if sdpa_method is None:
        return

    enabled = {getattr(backend, "name", str(backend)) for backend in sdpa_method}
    backend_setters = {
        "FLASH_ATTENTION": "enable_flash_sdp",
        "EFFICIENT_ATTENTION": "enable_mem_efficient_sdp",
        "MATH": "enable_math_sdp",
        "CUDNN_ATTENTION": "enable_cudnn_sdp",
    }
    for backend_name, setter_name in backend_setters.items():
        setter = getattr(torch.backends.cuda, setter_name, None)
        if setter is not None:
            setter(backend_name in enabled)

    logger.info("Set global SDPA backends to %s", sdpa_method)


def _assert_same_signature(original, patched):
    """
    Raise AssertionError if the two call signatures differ.
    """
    sig_orig = inspect.signature(original)
    sig_patch = inspect.signature(patched)

    if sig_orig != sig_patch:
        raise AssertionError(f"Signature mismatch:\n  original: {sig_orig}\n  patched : {sig_patch}")


def _patch_attention(obj, sdpa_method=None):
    """
    Wrap the `forward` method of `obj` in an `sdap_kernel` context manager.

    Args:
        obj: Any object with a `.forward(*args, **kwargs)` method.
        sdpa_method (list[SDPBackend], optional): Ordered list of SDPBackend
            implementations to attempt. If None, defaults to
            [CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH].

    Returns:
        The same `obj` with its `.forward` method patched.
    """
    if sdpa_method is None:
        sdpa_method = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    else:
        _set_global_sdpa_backends(sdpa_method)
    orig_forward = obj.forward

    def patch_method(method):
        func = method.__func__

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with sdpa_kernel(sdpa_method):
                return func(self, *args, **kwargs)

        wrapper.__doc__ = "SDPA kernel patch\n" + inspect.getdoc(method)
        return types.MethodType(wrapper, method.__self__)  # re-bind

    obj.forward = patch_method(obj.forward)
    # runtime check
    _assert_same_signature(orig_forward, obj.forward)

    logger.info("Patched model with SDPA method= %s", sdpa_method)
    return obj


def _patch_liger_kernel(model):
    """
    Patches a model with liger-kernel and sdpa_kernel

    Args:
        model (nn.Module): the model to patch
        use_liger_kernel (bool): Applies liger-kernel to model Default True.
        use_sdpa_patching (bool): Enables model patching with SDPA kernel optim. Default True.
        sdpa_method (list[SDPBackend], optional): Ordered list of SDPBackend
            implementations to attempt. If None, defaults to
            [CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH].
    Returns:
        nn.Module: the patched model
    """
    if not HAS_LIGER_KERNEL:
        logger.warning("Asked to use Liger Kernel, but could not import")
        return model

    # Unit tests may pass lightweight mocks; skip patching in that case.
    # (The wrapper logic itself is tested separately by patching this function.)
    if not isinstance(model, torch.nn.Module):
        logger.warning("Skipping Liger Kernel patch for non-nn.Module model: %s", type(model))
        return model

    try:
        # Lazy import: liger_kernel.transformers may trigger Triton JIT / CUDA
        # init, so we must not import it until the correct CUDA device is set.
        global liger_kernel_trf
        if liger_kernel_trf is None:
            import liger_kernel.transformers

            liger_kernel_trf = liger_kernel.transformers

        liger_kernel_trf._apply_liger_kernel_to_instance(model=model)
        logger.info("Applied liger-kernel to model")
        return model
    except Exception:
        logger.warning("Failed to apply liger-kernels to model; falling back to eager")
        del model
        raise RuntimeError("Failed to patch model")


def _model_runtime_patch_keys(model):
    config = getattr(model, "config", None)
    keys = list(getattr(config, "architectures", None) or [])
    model_cls_name = type(model).__name__
    if model_cls_name not in keys:
        keys.append(model_cls_name)
    return keys


def apply_model_runtime_patches(model, mesh):
    """Apply registered architecture-specific runtime patches to a model."""
    seen_hooks = set()
    for key in _model_runtime_patch_keys(model):
        hook_spec = _MODEL_RUNTIME_PATCHES.get(key)
        if hook_spec is None or hook_spec in seen_hooks:
            continue
        seen_hooks.add(hook_spec)

        module_name, hook_name = hook_spec
        try:
            hook = getattr(importlib.import_module(module_name), hook_name)
        except ImportError:
            logger.debug("Runtime patch hook for %s is unavailable: %s.%s", key, module_name, hook_name)
            continue
        model = hook(model, mesh=mesh)
    return model


def _get_next_fallback_attn(attn_implementation: str) -> str:
    """
    Get the next attention implementation in the priority list, in reverse order.

    If a model does not support a given attention implementation, the next
    implementation in the priority list is returned.

    If the current attention implementation is not in the priority list, it uses eager.

    Args:
        attn_implementation (str): The current attention implementation.

    Returns:
        str: The next attention implementation in the priority list.
    """
    priorities = [
        "eager",
        "sdpa",
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
    ]
    if attn_implementation in priorities:
        pos = priorities.index(attn_implementation)
        return priorities[max(0, pos - 1)]
    else:
        return priorities[0]


def _apply_preload_overrides(tp_size, cp_size, has_packed_sequence, attn_implementation, use_liger_kernel):
    """
    Compute final attention implementation and liger-kernel flag based on TP/CP and packed sequence constraints.
    """
    if tp_size > 1 or cp_size > 1:
        logger.info("Disabling Liger kernel with TP ({}) or CP ({})".format(tp_size, cp_size))
        use_liger_kernel = False

    # MagiAttention runs its own context-parallel dispatch + masking (incl. packing),
    # so it must keep its registered backend rather than being forced to SDPA/flash here.
    if attn_implementation == "magi":
        return attn_implementation, use_liger_kernel

    if cp_size > 1:
        attn_implementation = "sdpa"
        logger.warning("Packed sequence is supported only with SDPA. Setting model's attn_implementation to sdpa")

    if has_packed_sequence:
        if cp_size == 1:
            # FA2/FA3/FA4 all support varlen packing via transformers'
            # flash-attention wrapper; keep an explicitly requested version.
            if attn_implementation not in FLASH_ATTN_IMPLEMENTATIONS:
                assert HAS_FA or HAS_FA3 or HAS_FA4, "Flash Attention is not available"
                attn_implementation = (
                    DEFAULT_ATTN_IMPLEMENTATION if HAS_FA else ("flash_attention_3" if HAS_FA3 else "flash_attention_4")
                )
                logger.warning(
                    "Packed sequence is supported only with Flash Attention. Setting model's attn_implementation to %s",
                    attn_implementation,
                )
        else:
            # TODO: support packed sequence with CP size > 1
            raise ValueError("Packed sequence is only supported with CP size 1")
    return attn_implementation, use_liger_kernel


def _verify_sdpa_support(model, cp_size):
    """
    Validate SDPA support when CP is enabled for HF models.
    """
    if cp_size > 1:
        if getattr(model, "_supports_sdpa", False) is False:
            raise ValueError("Model does not support SDPA required for context parallelism")
