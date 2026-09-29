# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Fused frozen V4.1 indexer scoring with the released BF16 rounding boundaries.

The tiled GEMM/head-reduction structure follows the vendored V4 TileLang indexer.
V4.1 rounds the GEMM and weighted head scores to BF16 before reducing in FP32.
Visibility is supplied by the existing attention metadata path, including packed
document boundaries and global CP positions; top-k/candidate selection stays there.
"""

from __future__ import annotations

import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang


@tilelang.jit
def _indexer_fwd(heads: int, index_dim: int):
    """Compile a query/key tile; per-head intermediates remain on chip."""
    block_q = max(128 // heads, 1)
    block_k = 128
    batch = T.dynamic("batch")
    sequence = T.dynamic("sequence")
    width = T.dynamic("width")

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, sequence * heads, index_dim), "bfloat16"),
        K: T.Tensor((batch, width, index_dim), "bfloat16"),
        W: T.Tensor((batch, sequence, heads), "bfloat16"),
        Allowed: T.Tensor((batch, sequence, width), "bool"),
        Scores: T.Tensor((batch, sequence, width), "bfloat16"),
    ):
        """Compute masked scores for local queries and global keys.

        Args:
            Q: Flattened BF16 queries [batch, local_sequence * heads, index_dim].
            K: BF16 keys [batch, global_compressed, index_dim].
            W: BF16 per-head weights [batch, local_sequence, heads].
            Allowed: Visibility [batch, local_sequence, global_compressed].
            Scores: Output storage [batch, local_sequence, global_compressed].
                Every valid output element is written; inputs are not mutated.
        """
        with T.Kernel(T.ceildiv(sequence, block_q), T.ceildiv(width, block_k), batch, threads=256) as (bx, by, bz):
            q_shared = T.alloc_shared((block_q * heads, index_dim), "bfloat16")
            k_shared = T.alloc_shared((block_k, index_dim), "bfloat16")
            weights = T.alloc_fragment((block_q, heads), "bfloat16")
            products = T.alloc_fragment((block_k, block_q * heads), "float32")
            per_head = T.reshape(products, (block_k, block_q, heads))
            logits = T.alloc_fragment((block_k, block_q), "float32")
            T.copy(Q[bz, bx * block_q * heads, 0], q_shared)
            T.copy(K[bz, by * block_k, 0], k_shared)
            T.copy(W[bz, bx * block_q, 0], weights)
            T.gemm(k_shared, q_shared, products, transpose_B=True, clear_accum=True)
            for ki, qi, hi in T.Parallel(block_k, block_q, heads):
                # Match einsum -> BF16, ReLU * BF16 weight -> BF16, then sum.
                per_head[ki, qi, hi] = T.cast(
                    T.cast(
                        T.max(T.cast(T.cast(per_head[ki, qi, hi], "bfloat16"), "float32"), 0)
                        * T.cast(weights[qi, hi], "float32"),
                        "bfloat16",
                    ),
                    "float32",
                )
            T.reduce_sum(per_head, logits, dim=-1, clear=True)
            for ki, qi in T.Parallel(block_k, block_q):
                if bx * block_q + qi < sequence and by * block_k + ki < width:
                    Scores[bz, bx * block_q + qi, by * block_k + ki] = T.if_then_else(
                        Allowed[bz, bx * block_q + qi, by * block_k + ki],
                        logits[ki, qi],
                        -T.infinity("float32"),
                    )

    return kernel


@torch.no_grad()
def indexer_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    """Fuse frozen indexer scoring without materializing per-head global scores.

    Args:
        queries: BF16 CUDA queries [batch, local_sequence, heads, index_dim].
            Heads must be a power of two no larger than 128; index_dim must
            be divisible by 16. Quantize/dequantize and RoPE are already applied.
        keys: BF16 CUDA keys [batch, global_compressed, index_dim], gathered
            across CP ranks before this call.
        weights: BF16 CUDA weights [batch, local_sequence, heads], including
            the released head/dimension scaling.
        allowed: Boolean CUDA visibility [batch, local_sequence, global_compressed],
            including global causality, valid compression groups and packed
            document isolation. All inputs must reside on the same CUDA device.

    Returns:
        Independent BF16 scores [batch, local_sequence, global_compressed],
        with negative infinity at disallowed positions. Inputs are not mutated.
        This frozen indexer does not build an autograd graph.
    """
    if queries.ndim != 4 or keys.ndim != 3:
        raise ValueError("V4.1 indexer expects queries [B,S,H,D] and keys [B,K,D]")
    batch, sequence, heads, index_dim = queries.shape
    width = keys.shape[1]
    if keys.shape != (batch, width, index_dim) or weights.shape != (batch, sequence, heads):
        raise ValueError("V4.1 indexer key/weight shapes do not match the queries")
    if allowed.shape != (batch, sequence, width) or allowed.dtype != torch.bool:
        raise ValueError("V4.1 indexer visibility must be boolean [B,S,K]")
    if any(t.device != queries.device for t in (keys, weights, allowed)) or not queries.is_cuda:
        raise ValueError("V4.1 TileLang indexer inputs must share a CUDA device")
    if any(t.dtype != torch.bfloat16 for t in (queries, keys, weights)):
        raise ValueError("V4.1 TileLang indexer requires BF16 queries, keys and weights")
    if heads < 1 or heads > 128 or heads & (heads - 1) or index_dim % 16:
        raise ValueError("V4.1 TileLang indexer requires power-of-two heads <= 128 and index_dim divisible by 16")
    scores = torch.empty((batch, sequence, width), device=queries.device, dtype=queries.dtype)
    if sequence and width:
        _indexer_fwd(heads, index_dim)(
            queries.contiguous().view(batch, sequence * heads, index_dim),
            keys.contiguous(),
            weights.contiguous(),
            allowed.contiguous(),
            scores,
        )
    return scores
