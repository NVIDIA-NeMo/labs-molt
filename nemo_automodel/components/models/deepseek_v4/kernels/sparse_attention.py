# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Autograd wrapper for vendored Miles DeepSeek V4 sparse-attention kernels.

Attribution:
* Upstream project: Miles, https://github.com/yueming-yuan/miles
* Upstream revision: e561465d0b9bbf06188b7a5e2020dc7fd691f732, deepseek-v4 branch
* Upstream license: Apache-2.0, copyright 2025 Zhipu AI
* Original source:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/attention_core.py
"""

from __future__ import annotations

import torch

from nemo_automodel.components.models.deepseek_v4.kernels import tilelang_sparse_mla_bwd as sparse_mla_bwd
from nemo_automodel.components.models.deepseek_v4.kernels import tilelang_sparse_mla_fwd as sparse_mla_fwd


class DeepSeekV4SparseAttention(torch.autograd.Function):
    """TileLang sparse MQA attention with custom backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sm_scale: float | None = None,
        reference_rounding: bool = False,
    ) -> torch.Tensor:
        """Run the vendored sparse attention forward kernel.

        Args:
            ctx: Autograd context retaining inputs, output and base-two LSE.
            q: Contiguous CUDA BF16 queries [batch, sequence, heads, head_dim].
            kv: Contiguous CUDA BF16 shared keys/values [batch, kv_sequence, head_dim].
            attn_sink: CUDA FP32 softmax denominator biases [heads].
            topk_idxs: Contiguous CUDA integer indices [batch, sequence, slots],
                with -1 for masked slots.
            sm_scale: Score multiplier, defaulting to head_dim**-0.5.
            reference_rounding: Use scaled-logit exponent arithmetic matching
                the released inference kernel.

        Returns:
            Independent CUDA BF16 output [batch, sequence, heads, head_dim].
        """
        output, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(
            q, kv, attn_sink, topk_idxs, sm_scale=sm_scale, reference_rounding=reference_rounding
        )
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, output, lse)
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Run the existing backward using the saved base-two LSE.

        Args:
            ctx: Saved forward tensors and score multiplier.
            grad_output: CUDA output gradient [batch, sequence, heads, head_dim].
                Non-contiguous gradients are copied before the kernel call.

        Returns:
            Query gradient [batch, sequence, heads, head_dim], shared-KV gradient
            [batch, kv_sequence, head_dim], and sink gradient [heads], followed
            by None for non-differentiable inputs. Tuple length matches the
            arguments actually passed to apply; gradients use their input dtypes.
        """
        q, kv, attn_sink, topk_idxs, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_q, grad_kv, grad_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q,
            kv,
            attn_sink,
            output.contiguous(),
            grad_output,
            topk_idxs,
            lse,
            sm_scale=ctx.sm_scale,
        )
        gradients = (grad_q, grad_kv, grad_attn_sink, None, None, None)
        # Direct .apply callers may omit optional arguments. Autograd requires
        # one entry per argument actually supplied, including optional flags.
        return gradients[: len(ctx.needs_input_grad)]


def sparse_attn_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
    reference_rounding: bool = False,
) -> torch.Tensor:
    """Run vendored Miles DeepSeek V4 TileLang sparse attention.

    Args:
        q: Contiguous CUDA BF16 queries [batch, sequence, heads, head_dim].
        kv: Contiguous CUDA BF16 shared keys/values [batch, kv_sequence, head_dim].
        attn_sink: CUDA FP32 denominator biases [heads].
        topk_idxs: Contiguous CUDA integer indices [batch, sequence, slots],
            with -1 for masked slots.
        sm_scale: Score multiplier, defaulting to head_dim**-0.5.
        reference_rounding: Select released-inference forward exponent arithmetic.

    Returns:
        Independent BF16 tensor [batch, sequence, heads, head_dim], with the
        existing custom backward for queries, shared KV and sink parameters.
    """
    return DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale, reference_rounding)


class DeepSeekV4SparseAttentionHeadChunked(torch.autograd.Function):
    """TileLang sparse attention with smaller head groups and fp32 KV-grad accumulation."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        max_heads_per_kernel: int,
        sm_scale: float | None = None,
        reference_rounding: bool = False,
    ) -> torch.Tensor:
        """Run forward over head chunks while preserving backward chunk size.

        Args:
            ctx: Autograd context retaining tensors and backward chunk size.
            q: CUDA BF16 queries [batch, sequence, heads, head_dim].
            kv: Contiguous CUDA BF16 shared keys/values [batch, kv_sequence, head_dim].
            attn_sink: CUDA FP32 denominator biases [heads].
            topk_idxs: Contiguous CUDA integer indices [batch, sequence, slots],
                with -1 for masked slots.
            max_heads_per_kernel: Maximum heads per backward kernel. Forward
                uses the same count unless reference mode requires at least 64.
            sm_scale: Score multiplier, defaulting to head_dim**-0.5.
            reference_rounding: Preserve the original 64-head forward layout
                and scaled-logit exponent arithmetic.

        Returns:
            Independent CUDA BF16 output [batch, sequence, heads, head_dim].
        """
        output = q.new_empty(q.shape)
        lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
        # Preserve the original 64-head forward reduction layout for its exact
        # rounding contract. Backward retains the smaller memory-bounded chunks.
        forward_heads = max(max_heads_per_kernel, 64) if reference_rounding else max_heads_per_kernel
        for start in range(0, q.shape[2], forward_heads):
            end = min(start + forward_heads, q.shape[2])
            chunk_output, chunk_lse = sparse_mla_fwd.sparse_mqa_fwd_interface(
                q[:, :, start:end, :].contiguous(),
                kv,
                attn_sink[start:end].contiguous(),
                topk_idxs,
                sm_scale=sm_scale,
                reference_rounding=reference_rounding,
            )
            output[:, :, start:end, :].copy_(chunk_output)
            lse[:, :, start:end].copy_(chunk_lse)
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, output, lse)
        ctx.max_heads_per_kernel = max_heads_per_kernel
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Run chunked backward and accumulate shared KV gradients in FP32.

        Args:
            ctx: Saved forward tensors, score multiplier and backward chunk size.
            grad_output: CUDA output gradient [batch, sequence, heads, head_dim].
                Non-contiguous gradients are copied before the kernel call.

        Returns:
            Query gradient [batch, sequence, heads, head_dim], shared-KV gradient
            [batch, kv_sequence, head_dim], and sink gradient [heads], in their
            input dtypes. Remaining entries are None, and tuple length matches
            the arguments actually supplied to apply.
        """
        q, kv, attn_sink, topk_idxs, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_q_full = torch.empty_like(q)
        grad_kv = torch.zeros_like(kv, dtype=torch.float32)
        grad_attn_sink_full = torch.empty_like(attn_sink)
        max_heads = ctx.max_heads_per_kernel
        for start in range(0, q.shape[2], max_heads):
            end = min(start + max_heads, q.shape[2])
            grad_q_chunk, grad_kv_chunk, grad_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
                q[:, :, start:end, :].contiguous(),
                kv,
                attn_sink[start:end].contiguous(),
                output[:, :, start:end, :].contiguous(),
                grad_output[:, :, start:end, :].contiguous(),
                topk_idxs,
                lse[:, :, start:end].contiguous(),
                sm_scale=ctx.sm_scale,
                return_dkv_accum_dtype=True,
            )
            grad_q_full[:, :, start:end, :].copy_(grad_q_chunk)
            grad_kv += grad_kv_chunk
            grad_attn_sink_full[start:end].copy_(grad_attn_sink)
        gradients = (
            grad_q_full,
            grad_kv.to(kv.dtype),
            grad_attn_sink_full,
            None,
            None,
            None,
            None,
        )
        return gradients[: len(ctx.needs_input_grad)]


def sparse_attn_tilelang_head_chunked(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    max_heads_per_kernel: int,
    sm_scale: float | None = None,
    reference_rounding: bool = False,
) -> torch.Tensor:
    """Run vendored sparse attention with bounded backward head chunks.

    Args:
        q: CUDA BF16 queries [batch, sequence, heads, head_dim].
        kv: Contiguous CUDA BF16 shared keys/values [batch, kv_sequence, head_dim].
        attn_sink: CUDA FP32 denominator biases [heads].
        topk_idxs: Contiguous CUDA integer indices [batch, sequence, slots],
            with -1 for masked slots.
        max_heads_per_kernel: Backward head chunk size; forward uses at least
            64 heads per chunk when reference_rounding is enabled.
        sm_scale: Score multiplier, defaulting to head_dim**-0.5.
        reference_rounding: Select original-inference forward rounding while
            retaining smaller backward chunks and FP32 shared-KV accumulation.

    Returns:
        Independent CUDA BF16 tensor [batch, sequence, heads, head_dim].
    """
    return DeepSeekV4SparseAttentionHeadChunked.apply(
        q, kv, attn_sink, topk_idxs, max_heads_per_kernel, sm_scale, reference_rounding
    )
