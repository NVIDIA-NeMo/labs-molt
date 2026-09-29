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
# ruff: noqa: D101,D103

import torch

from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import indexer_fwd_interface

# 32M fp32 scores are 128 MiB; H100 torch.topk uses about 8x that as workspace here.
_TOPK_MAX_ELEMENTS_PER_CALL = 32 * 1024 * 1024


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


def _topk_in_row_chunks(logits: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound temporary memory while preserving independent per-row top-k selection."""
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError(f"top-k logits must have shape [queries, keys], got {tuple(logits.shape)}.")
    chunk_rows = max(1, _TOPK_MAX_ELEMENTS_PER_CALL // logits.shape[1])
    if chunk_rows >= logits.shape[0]:
        return torch.topk(logits, topk, dim=-1)

    score_chunks = []
    index_chunks = []
    for logits_chunk in logits.split(chunk_rows, dim=0):
        scores, indices = torch.topk(logits_chunk, topk, dim=-1)
        score_chunks.append(scores)
        index_chunks.append(indices)
    return torch.cat(score_chunks, dim=0), torch.cat(index_chunks, dim=0)


class IndexerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        topk_indices: torch.Tensor | None = None,
    ):
        _, head_num, _ = index_q.shape
        logits = indexer_fwd_interface(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)
        if topk_indices is None:
            # Pad logits with -inf when seq_len_kv < topk so torch.topk keeps a fixed k=topk
            # (static shape); the padded columns become the same -1 sentinel as causal masking.
            pad = topk - logits.shape[-1]
            if pad > 0:
                logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
            index_score, topk_indices = _topk_in_row_chunks(logits, topk)
            topk_indices = topk_indices.to(torch.int32)
            topk_indices = topk_indices.masked_fill(index_score == -torch.inf, -1)

        index_score = pytorch_extract_topk_scores(logits, topk_indices)

        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        ctx.topk = topk
        ctx.head_num = head_num
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors
        grad_q, grad_w, grad_k = indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None, None, None, None, None, None


def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
):
    return IndexerFunction.apply(index_q, index_k, weights.squeeze(-1), cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices)


def generate_varlen_mask_params(cu_seqlens):
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends
