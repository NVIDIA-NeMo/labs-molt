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
#
# ruff: noqa
# Upstream attribution:
#   Project: Miles, https://github.com/yueming-yuan/miles
#   Revision: e561465d0b9bbf06188b7a5e2020dc7fd691f732, deepseek-v4 branch
#   License: Apache-2.0, copyright 2025 Zhipu AI
#   Original source:
# https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer.py
"""TileLang-based DSA Indexer for DeepSeek-V4.

Adapts GLM-5's lighting_indexer to V4's SBHD data layout and causal masking.
Provides both a low-level per-sample interface and a batched autograd Function.
"""

import torch

from .tilelang_indexer_bwd import batched_indexer_bwd
from .tilelang_indexer_fwd import _make_causal_cu_seqlens, batched_indexer_fwd


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = (topk_indices >= 0) & (topk_indices < logits.shape[dim])
    safe_indices = topk_indices.clamp(min=0, max=max(logits.shape[dim] - 1, 0)).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


class V4IndexerFunction(torch.autograd.Function):
    """Autograd function for V4 tilelang indexer.

    Inputs are in V4's native SBHD layout:
        q:       [seqlen, batch, heads, dim]  bf16
        k:       [seqlen_kv, batch, dim]      bf16
        weights: [seqlen, batch, heads]        fp32
    """

    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        compress_ratio: int,
        topk: int,
        topk_indices: torch.Tensor | None = None,
    ):
        seqlen_q = index_q.shape[0]
        seq_len_kv = index_k.shape[0]

        cu_seqlen_ks, cu_seqlen_ke = _make_causal_cu_seqlens(seqlen_q, seq_len_kv, compress_ratio, index_q.device)

        # [batch, seqlen, seqlen_kv]
        logits = batched_indexer_fwd(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke)

        if topk_indices is None:
            actual_topk = min(topk, seq_len_kv)
            index_score, topk_indices = torch.topk(logits, actual_topk, dim=-1)
            topk_indices = topk_indices.to(torch.int32)
            topk_indices = topk_indices.masked_fill(index_score == -torch.inf, -1)

        index_score = pytorch_extract_topk_scores(logits, topk_indices)

        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        ctx.compress_ratio = compress_ratio
        ctx.topk = topk
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors
        grad_q, grad_w, grad_k = batched_indexer_bwd(index_q, weights, index_k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None, None


def v4_lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_ratio: int,
    topk: int,
    topk_indices: torch.Tensor | None = None,
):
    """Main entry point for V4 tilelang indexer.

    Args:
        index_q:       [seqlen, batch, heads, dim]  bf16
        index_k:       [seqlen_kv, batch, dim]      bf16
        weights:       [seqlen, batch, heads]        fp32
        compress_ratio: compression ratio (4 for C4 layers)
        topk:          number of top-k indices to select
        topk_indices:  optional pre-computed topk indices [batch, seqlen, topk] int32

    Returns:
        index_score:  [batch, seqlen, topk] fp32
        topk_indices: [batch, seqlen, topk] int32
    """
    return V4IndexerFunction.apply(index_q, index_k, weights, compress_ratio, topk, topk_indices)
