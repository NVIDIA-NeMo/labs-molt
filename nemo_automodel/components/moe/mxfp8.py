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

"""torchao MXFP8 grouped-GEMM plumbing for the ``experts="torch_mm_mxfp8"`` path.

torchao exposes a drop-in differentiable replacement for ``torch._grouped_mm`` that
dynamically quantizes both operands to MXFP8 (e4m3 data + e8m0 block scales,
block_size=32). It mirrors ``torch._grouped_mm``'s contract exactly: 2D activations
(M*num_groups, K), 3D [E, K, N] stacked expert weights, int32 ``offs`` group
boundaries — so no transpose is needed for Automodel's gate_and_up_projs
([E, dim, up]) or down_projs ([E, inter, dim]).

torchao is unpinned and (when present) comes from the base image rather than the uv
lock, so the API generation is resolved defensively at runtime across known versions
and normalized to a uniform ``mxfp8_grouped_mm(A, B, offs)`` callable. If torchao is
missing entirely, the runtime gate falls back to ``torch._grouped_mm``.

Public entry: :func:`select_grouped_mm` returns the grouped-GEMM callable the expert
forward should use (mxfp8 with the contiguous-operand relayout when requested and
available, else plain ``torch._grouped_mm``).
"""

import warnings

import torch

_MXFP8_GROUPED_MM = None  # cached uniform callable (set on first resolve)
_MXFP8_RESOLVED = False  # whether the import ladder has run
_MXFP8_FALLBACK_WARNED = False  # one-time runtime fallback warning
_MXFP8_ACTIVE_ANNOUNCED = False  # one-time "mxfp8 active" confirmation


def _resolve_mxfp8_grouped_mm():
    """Resolve a torchao MXFP8 grouped-GEMM callable, normalizing across API generations.

    Returns a callable ``mxfp8_grouped_mm(A, B, offs)`` mirroring ``torch._grouped_mm``,
    or ``None`` if no supported torchao API is importable. The result is cached.
    """
    global _MXFP8_GROUPED_MM, _MXFP8_RESOLVED
    if _MXFP8_RESOLVED:
        return _MXFP8_GROUPED_MM
    _MXFP8_RESOLVED = True

    # (1) current-main / v0.17.0: _to_mxfp8_then_scaled_grouped_mm(A, B_t, offs=...).
    # wgrad_with_hp=True keeps the WEIGHT-GRADIENT GEMM in high precision instead of
    # re-quantizing grad_output to MXFP8 twice. torchao v0.17.0's e8m0 block-scale has
    # incomplete nan/inf handling on the backward grad-quant (acknowledged TODO); the
    # large first-step grad of gpt-oss (bias + clamped swiglu) saturates a block -> NaN
    # wgrad -> NaN weights -> iter-1 nan. wgrad_with_hp=True is torchao's documented combo
    # with MXTensor inputs and avoids that path. (Older gens lacking the kwarg fall back.)
    try:
        import inspect

        from torchao.prototype.moe_training import _to_mxfp8_then_scaled_grouped_mm

        # Resolve ONCE whether this torchao build accepts wgrad_with_hp (don't per-call
        # try/except, which would silently swallow unrelated TypeErrors).
        _has_wgrad_hp = "wgrad_with_hp" in inspect.signature(_to_mxfp8_then_scaled_grouped_mm).parameters

        if _has_wgrad_hp:

            def _impl(A, B, offs, _fn=_to_mxfp8_then_scaled_grouped_mm):
                return _fn(A, B, offs=offs, wgrad_with_hp=True)  # pragma: no cover
        else:

            def _impl(A, B, offs, _fn=_to_mxfp8_then_scaled_grouped_mm):
                return _fn(A, B, offs=offs)  # pragma: no cover

        _MXFP8_GROUPED_MM = _impl
        return _MXFP8_GROUPED_MM
    except ImportError:
        pass

    # Blog-era / intermediate generations take a MoEScalingType.MXFP8 argument.
    try:
        from torchao.prototype.moe_training.conversion_utils import MoEScalingType

        # (2) intermediate: _quantize_then_scaled_grouped_mm(A, B_t, offs=, scaling_type=)
        try:
            from torchao.prototype.moe_training.scaled_grouped_mm import _quantize_then_scaled_grouped_mm

            def _impl(A, B, offs, _fn=_quantize_then_scaled_grouped_mm, _st=MoEScalingType.MXFP8):
                return _fn(A, B, offs=offs, scaling_type=_st)  # pragma: no cover

            _MXFP8_GROUPED_MM = _impl
            return _MXFP8_GROUPED_MM
        except ImportError:
            pass

        # (3) v0.13-era: _scaled_grouped_mm(A, B_t, offs=, scaling_type=)
        from torchao.prototype.moe_training import _scaled_grouped_mm

        def _impl(A, B, offs, _fn=_scaled_grouped_mm, _st=MoEScalingType.MXFP8):
            return _fn(A, B, offs=offs, scaling_type=_st)  # pragma: no cover

        _MXFP8_GROUPED_MM = _impl
        return _MXFP8_GROUPED_MM
    except ImportError:
        pass

    return None


def _mxfp8_grouped_mm_or_none():
    """Return the MXFP8 grouped-GEMM callable iff it is usable on this device.

    Requires CUDA with compute capability >= 10 (GB200/sm_100+) AND a successful
    torchao import. Otherwise returns ``None`` (callers fall back to
    ``torch._grouped_mm``). Emits a one-time warning when MXFP8 was requested but is
    unavailable.
    """
    global _MXFP8_FALLBACK_WARNED, _MXFP8_ACTIVE_ANNOUNCED
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
        fn = _resolve_mxfp8_grouped_mm()
        if fn is not None:
            if not _MXFP8_ACTIVE_ANNOUNCED:
                _MXFP8_ACTIVE_ANNOUNCED = True
                # Positive confirmation so the e2e log unambiguously shows the MXFP8
                # path engaged (vs. silently falling back to torch._grouped_mm).
                warnings.warn(
                    "experts='torch_mm_mxfp8': MXFP8 grouped GEMM active "
                    "(routing expert GEMMs through torchao.prototype.moe_training).",
                    category=UserWarning,
                    stacklevel=2,
                )
            return fn
    if not _MXFP8_FALLBACK_WARNED:
        _MXFP8_FALLBACK_WARNED = True
        warnings.warn(
            "experts='torch_mm_mxfp8' requested but MXFP8 grouped GEMM is unavailable "
            "(requires CUDA compute capability >= 10 and an importable "
            "torchao.prototype.moe_training; note torchao may be absent from the base "
            "image). Falling back to torch._grouped_mm.",
            category=UserWarning,
            stacklevel=2,
        )
    return None


def _default_grouped_mm(A, B, offs):
    """Fallback grouped GEMM (plain ``torch._grouped_mm``) used when MXFP8 is off."""
    return torch._grouped_mm(A, B, offs=offs)


def _mxfp8_weight_relayout(B):
    """Lay the [E,K,N] expert weight out so its (-2,-1) transpose is contiguous.

    torchao's MXFP8 quantizer calls ``to_mx(B.transpose(-2,-1))`` and strictly asserts
    the input is contiguous, so the weight must be stored as [E,N,K]-contiguous (viewed
    as [E,K,N]) — also the column-major B_t layout torchao's grouped GEMM wants.
    """
    return B.transpose(-2, -1).contiguous().transpose(-2, -1)


def select_grouped_mm(use_mxfp8):
    """Return the grouped-GEMM callable ``grouped_mm(A, B, offs)`` for the expert GEMMs.

    When ``use_mxfp8`` and the torchao MXFP8 kernel is usable on this device, returns a
    wrapper that makes both operands contiguous in the layout torchao requires (A
    contiguous; B relaid out so its transpose is contiguous — see _mxfp8_weight_relayout)
    and routes through it. Otherwise returns the plain ``torch._grouped_mm`` fallback,
    leaving the bf16 path byte-identical. Shared by the no-bias helper and the inline
    bias paths so dispatch + relayout are defined once.
    """
    mxfp8_grouped_mm = _mxfp8_grouped_mm_or_none() if use_mxfp8 else None
    if mxfp8_grouped_mm is None:
        return _default_grouped_mm

    def grouped_mm(A, B, offs, _fn=mxfp8_grouped_mm):
        # Executes the torchao MXFP8 kernel — SM100-only, not reachable on CPU CI.
        return _fn(A.contiguous(), _mxfp8_weight_relayout(B), offs)  # pragma: no cover

    return grouped_mm
