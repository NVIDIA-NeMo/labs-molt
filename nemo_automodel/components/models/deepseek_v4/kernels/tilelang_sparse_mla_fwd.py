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
# https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_fwd.py
# Adapted from miles_plugins/models/glm5/ops/tilelang_sparse_mla_fwd.py for DeepSeek-V4.
# Key differences from GLM-5:
#   - attn_sink: learnable per-head scalar added to softmax denominator
#   - Single-head KV: kv shape [B, S_kv, D] (no kv_group, no D/D_tail split)
#   - Index shape: [B, S, topk] (no kv_group dim)
#   - Output: [B, S, H, D] + LSE [B, S, H]
import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mqa_fwd(
    heads,
    dim,
    topk,
    sm_scale=None,
    block_I=64,
    num_stages=2,
    threads=256,
    reference_rounding=False,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"dim must be power of 2, got {dim}"
    assert topk % block_I == 0, f"topk ({topk}) must be divisible by block_I ({block_I})"
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    if not reference_rounding:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, topk]
    lse_shape = [batch, seq_len, heads]
    attn_sink_shape = [heads]
    indices_dtype = T.int32
    mask_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = heads
    padded_H = max(tilelang.math.next_power_of_2(heads), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim

    if heads > 64:
        assert heads % 64 == 0, "heads should be a multiple of 64"
        REPLICATE_H = heads // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        ValidMask: T.Tensor(indices_shape, mask_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -1e30 if reference_rounding else -(2**30))

            b_i = by
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = ValidMask[b_i, s_i, i_i * BI + bi_i] != 0

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, i_i * BI + bi_i], d_i]

                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], acc_s[h_i, bi_i], -T.infinity(acc_s.dtype))
                if reference_rounding:
                    # The original inference kernel rounds scaled logits before
                    # its maximum reduction and subtraction. Scaling raw logits
                    # separately inside exp2 changes BF16 probability rounding.
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] *= sm_scale
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                if reference_rounding:
                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp(m_i_prev[h_i] - m_i[h_i])
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp(acc_s[h_i, bi_i] - m_i[h_i])
                else:
                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # attn_sink: add exp(attn_sink[h] - max_scaled) to softmax denominator
            # attn_sink is a pre-scaled logit (same space as scores*sm_scale), so only convert to log2 base
            for h_i in T.Parallel(H_per_block):
                if reference_rounding:
                    sumexp[h_i] += T.exp(AttnSink[H0 + h_i] - m_i[h_i])
                else:
                    sumexp[h_i] += T.exp2(AttnSink[H0 + h_i] * 1.44269504 - m_i[h_i] * sm_scale)

            # Rescale output
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            # LSE = log2(sumexp) + m_i * sm_scale (in log2 space)
            for h_i in T.Parallel(H_per_block):
                if reference_rounding:
                    # Backward consumes base-two LSE in both arithmetic modes.
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * 1.44269504
                else:
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mqa_fwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
    block_I: int = 64,
    num_stages: int = 2,
    threads: int = 256,
    reference_rounding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward interface for V4 sparse MQA attention.

    Args:
        q: Contiguous CUDA BF16 queries [batch, sequence, heads, head_dim].
        kv: Contiguous CUDA BF16 shared keys/values [batch, kv_sequence, head_dim].
        attn_sink: CUDA FP32 denominator biases [heads].
        topk_idxs: Contiguous CUDA integer indices [batch, sequence, slots].
            Entries outside [0, kv_sequence) are masked; slots are internally
            padded to a multiple of block_I.
        sm_scale: Score multiplier, defaulting to head_dim**-0.5.
        block_I: Sparse-key slots processed by each kernel iteration.
        num_stages: Pipeline stages for the generated kernel.
        threads: CUDA threads per block.
        reference_rounding: Match the original inference kernel's scaled-logit
            FP32 arithmetic while retaining log2 LSE for the existing backward.

    Returns:
        Independent BF16 output [batch, sequence, heads, head_dim] and FP32
        base-two log-sum-exp [batch, sequence, heads], both on q's device.
    """
    assert q.is_contiguous() and kv.is_contiguous() and topk_idxs.is_contiguous()
    batch, seq_len, heads, dim = q.shape
    _, seq_len_kv, kv_dim = kv.shape
    assert kv_dim == dim
    _, _, topk = topk_idxs.shape

    # Pad topk to next multiple of block_I (kernel requires divisibility)
    padded_topk = (topk + block_I - 1) // block_I * block_I
    if padded_topk != topk:
        pad = torch.full((batch, seq_len, padded_topk - topk), -1, device=topk_idxs.device, dtype=topk_idxs.dtype)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1).contiguous()
        topk = padded_topk

    valid_mask = ((topk_idxs >= 0) & (topk_idxs < seq_len_kv)).to(torch.int32).contiguous()
    topk_idxs = topk_idxs.clamp(min=0, max=max(seq_len_kv - 1, 0)).to(torch.int32).contiguous()

    kernel = sparse_mqa_fwd(
        heads,
        dim,
        topk,
        sm_scale,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
        reference_rounding=reference_rounding,
    )
    out, lse = kernel(q, kv, attn_sink, topk_idxs, valid_mask)
    return out, lse
