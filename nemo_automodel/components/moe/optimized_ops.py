# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Memory-optimized MoE elementwise ops extracted from ``experts.py``.

Chunked custom-autograd router-weight fp32 multiply: computes the identical
fp32 math in row chunks and saves only low-precision inputs, removing the
full-size fp32 intermediates that otherwise pin ~7 GiB blocks per MoE layer
under activation checkpointing.

With ``BackendConfig.compile_router_weight`` the same Function runs each pass as
one ``torch.compile``d kernel over the whole tensor instead of the eager chunk
loop (see ``_compile_router_weight_cores``).
"""

from __future__ import annotations

from typing import Any

import torch

_RW_CHUNK_ROWS = 8192
# Engage the chunked path only for large dispatch tensors, where the memory
# saving matters; below this the backward recompute is a net compute tax.
_RW_CHUNK_THRESHOLD = 12288


def _rw_fwd_core(x: torch.Tensor, probs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """fp32 router-weight multiply: ``(x * probs)`` in fp32, cast to ``out_dtype``."""
    return (x.float() * probs.float()).to(out_dtype)


def _rw_bwd_x_core(grad_out: torch.Tensor, probs: torch.Tensor, x_dtype: torch.dtype) -> torch.Tensor:
    """Gradient w.r.t. ``x``: ``(grad_out * probs)`` in fp32, cast to ``x_dtype``."""
    return (grad_out.float() * probs.float()).to(x_dtype)


def _rw_bwd_probs_core(grad_out: torch.Tensor, x: torch.Tensor, probs_dtype: torch.dtype) -> torch.Tensor:
    """Gradient w.r.t. ``probs``: ``(grad_out * x).sum(-1)`` in fp32, cast to ``probs_dtype``."""
    return (grad_out.float() * x.float()).sum(dim=-1, keepdim=True).to(probs_dtype)


# Module-level dispatch targets. ``_compile_router_weight_cores`` swaps them for
# ``torch.compile`` wrappers once per process (``BackendConfig.compile_router_weight``);
# the Function below then runs each pass as one fused kernel over the whole tensor
# instead of the eager row-chunk loop (cast, multiply, cast, slice-assign per chunk).
_rw_fwd_dispatch = _rw_fwd_core
_rw_bwd_x_dispatch = _rw_bwd_x_core
_rw_bwd_probs_dispatch = _rw_bwd_probs_core
_RW_CORES_COMPILED = False


def _compile_router_weight_cores() -> None:
    """Wrap the router-weight multiply cores with ``torch.compile``.

    Runs once per process, same lazy pattern as Kimi K3's ``compile_situ``: the
    compiled functions replace the module-level dispatch targets, so every
    expert module shares one compiled kernel per pass and repeated model
    construction does not recompile. Fused kernels keep no fp32 intermediates,
    so the chunk loop (which only exists to bound eager fp32 transients) is
    skipped on this path. Numerics are allclose to eager (same fp32 math, one
    rounding to the output dtype), not guaranteed bitwise-identical.
    """
    global _rw_fwd_dispatch, _rw_bwd_x_dispatch, _rw_bwd_probs_dispatch, _RW_CORES_COMPILED
    if _RW_CORES_COMPILED:
        return
    _rw_fwd_dispatch = torch.compile(_rw_fwd_core, dynamic=True)
    _rw_bwd_x_dispatch = torch.compile(_rw_bwd_x_core, dynamic=True)
    _rw_bwd_probs_dispatch = torch.compile(_rw_bwd_probs_core, dynamic=True)
    _RW_CORES_COMPILED = True


class _RouterWeightMulFunction(torch.autograd.Function):
    """Chunked fp32 router-weight multiply that saves only the raw inputs.

    The plain ``(x.float() * probs.float()).to(dtype)`` lets autograd keep
    full-size fp32 [tokens, hidden] intermediates alive for backward (the
    upcast input, the product, and a recompute copy under activation
    checkpointing). This Function computes the same fp32 multiply in row
    chunks and saves only the low-precision inputs. The forward is
    bitwise-identical; the backward matches autograd's fp32 chain
    (``grad_x = (g_f32 * probs_f32).to(x.dtype)``,
    ``grad_probs = (g_f32 * x_f32).sum(-1, keepdim=True)`` cast to
    ``probs.dtype``).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        probs: torch.Tensor,
        out_dtype: torch.dtype,
        save_x: bool,
    ) -> torch.Tensor:
        """Multiply expert outputs by routing probabilities in fp32.

        Args:
            ctx: Autograd context.
            x: Expert outputs of shape [tokens, hidden].
            probs: Routing probabilities of shape [tokens, 1].
            out_dtype: Output dtype (the dispatcher's expected activation
                dtype, or fp32 for the scatter-add reduction path).
            save_x: Whether backward needs ``x``. ``x`` is only consumed by
                the ``probs`` gradient; when ``probs`` carries no grad (e.g.
                FakeBalancedGate emits constant weights), saving it would pin
                a full-size [tokens, hidden] tensor per MoE layer across the
                activation-checkpointing backward window for nothing. Callers
                pass ``probs.requires_grad``.

        Returns:
            Tensor of shape [tokens, hidden] and dtype ``out_dtype``.
        """
        if save_x:
            ctx.save_for_backward(x, probs)
        else:
            ctx.save_for_backward(probs)
        ctx.save_x = save_x
        ctx.x_dtype = x.dtype
        if _RW_CORES_COMPILED and x.shape[0] > 0:
            # One fused kernel: cast, multiply, cast to out_dtype -- no fp32 intermediate.
            return _rw_fwd_dispatch(x, probs, out_dtype)
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        for s in range(0, x.shape[0], _RW_CHUNK_ROWS):
            e = min(s + _RW_CHUNK_ROWS, x.shape[0])
            out[s:e] = (x[s:e].float() * probs[s:e].float()).to(out_dtype)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        """Compute chunked fp32 gradients for the router-weight multiply.

        Args:
            ctx: Autograd context holding the saved inputs.
            grad_out: Upstream gradient of shape [tokens, hidden].

        Returns:
            Tuple ``(grad_x, grad_probs, None, None)`` where ``grad_x`` has
            ``x``'s shape [tokens, hidden] and dtype, and ``grad_probs`` has
            ``probs``'s shape [tokens, 1] and dtype (``None`` when the
            respective input requires no gradient).
        """
        if ctx.save_x:
            x, probs = ctx.saved_tensors
        else:
            (probs,) = ctx.saved_tensors
            x = None
        grad_x = None
        grad_p = None
        if _RW_CORES_COMPILED and grad_out.shape[0] > 0:
            if ctx.needs_input_grad[0]:
                grad_x = _rw_bwd_x_dispatch(grad_out, probs, ctx.x_dtype)
            if ctx.needs_input_grad[1] and x is not None:
                grad_p = _rw_bwd_probs_dispatch(grad_out, x, probs.dtype)
            return grad_x, grad_p, None, None
        if ctx.needs_input_grad[0]:
            grad_x = torch.empty(grad_out.shape, dtype=ctx.x_dtype, device=grad_out.device)
        if ctx.needs_input_grad[1] and x is not None:
            grad_p = torch.empty_like(probs)
        for s in range(0, grad_out.shape[0], _RW_CHUNK_ROWS):
            e = min(s + _RW_CHUNK_ROWS, grad_out.shape[0])
            g = grad_out[s:e].float()
            if grad_x is not None:
                grad_x[s:e] = (g * probs[s:e].float()).to(grad_x.dtype)
            if grad_p is not None:
                grad_p[s:e] = (g * x[s:e].float()).sum(dim=-1, keepdim=True).to(probs.dtype)
        return grad_x, grad_p, None, None


def _apply_router_weight_fp32(
    output2: torch.Tensor,
    permuted_probs: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply routing probabilities to expert outputs with fp32 arithmetic.

    Large 2-D row-aligned inputs go through the chunked custom Function so
    autograd does not retain full-size fp32 intermediates; every other shape
    keeps the plain eager multiply (bitwise-identical result).

    Args:
        output2: Expert down-projection outputs of shape [tokens, hidden].
        permuted_probs: Routing probabilities broadcastable against
            ``output2``, typically of shape [tokens, 1].
        compute_dtype: Output dtype.

    Returns:
        ``(output2 * permuted_probs)`` computed in fp32 and cast to
        ``compute_dtype``, with ``output2``'s broadcast shape.
    """
    if (
        output2.dim() == 2
        and permuted_probs.dim() == 2
        and permuted_probs.shape[0] == output2.shape[0]
        and permuted_probs.shape[1] == 1
        and output2.shape[0] > _RW_CHUNK_THRESHOLD
    ):
        return _RouterWeightMulFunction.apply(output2, permuted_probs, compute_dtype, permuted_probs.requires_grad)
    return (output2.float() * permuted_probs.float()).to(compute_dtype)
