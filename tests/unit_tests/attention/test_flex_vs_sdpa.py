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

import math

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from nemo_automodel.components.attention.flex_attention import FlexAttention

# Over the default 5s budget on purpose: CUDA FlexAttention compilation takes longer on a cold worker.
# Reduce cold compiler startup before lowering this further.
pytestmark = pytest.mark.timeout(60)


def test_flex_attention_without_sinks_matches_causal_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(371)
    query = torch.randn(2, 3, 8, 16)
    key = torch.randn(2, 3, 8, 16)
    value = torch.randn(2, 3, 8, 12)
    scale = 0.2

    monkeypatch.setattr(FlexAttention, "flex_attn", flex_attention)
    monkeypatch.setattr(FlexAttention, "block_masks", {})

    attention = FlexAttention()
    actual = attention(query, key, value, scale=scale)
    expected = F.scaled_dot_product_attention(query, key, value, scale=scale, is_causal=True)

    torch.testing.assert_close(actual, expected)
    assert len(FlexAttention.block_masks) == 1

    repeated = attention(query, key, value, scale=scale)
    torch.testing.assert_close(repeated, expected)
    assert len(FlexAttention.block_masks) == 1

    smaller_batch = attention(query[:1], key[:1], value[:1], scale=scale)
    smaller_batch_expected = F.scaled_dot_product_attention(query[:1], key[:1], value[:1], scale=scale, is_causal=True)
    torch.testing.assert_close(smaller_batch, smaller_batch_expected)
    assert len(FlexAttention.block_masks) == 2


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK.float(), dim=-1)
    W = W[..., :-1].to(V.dtype)
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


def _flex_forward_from_qkm(Q, K, V, S_vec, sm_scale, sliding_window=0, device=None, dtype=torch.bfloat16):
    # Q: [S, H_kv, q_mult, D], K/V: [S, H_kv, D]
    S, H_kv, q_mult, D = Q.shape
    H_q = H_kv * q_mult
    q = Q.permute(1, 2, 0, 3).contiguous().view(H_q, S, D).unsqueeze(0).to(device=device, dtype=dtype)
    k = K.permute(1, 0, 2).contiguous().unsqueeze(0).to(device=device, dtype=dtype)
    v = V.permute(1, 0, 2).contiguous().unsqueeze(0).to(device=device, dtype=dtype)

    sink_weights = S_vec.to(device=device, dtype=dtype)  # [H_q]

    attn = FlexAttention().to(device)
    # Map window semantics if non-zero to match <= vs < difference
    flex_window = int(sliding_window)
    out = attn(
        q,
        k,
        v,
        scale=sm_scale,
        sink_weights=sink_weights,
        sliding_window=flex_window,
        enable_gqa=True,
    )
    out = out.squeeze(0).permute(1, 0, 2).contiguous().view(S, -1)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flex_attention_matches_sdpa():
    torch.manual_seed(123)
    device = "cuda:0"
    dtype = torch.bfloat16

    S = 64
    H_kv = 4
    q_mult = 2
    D = 32
    H_q = H_kv * q_mult

    Q = torch.randn(S, H_kv, q_mult, D, device=device, dtype=dtype)
    K = torch.randn(S, H_kv, D, device=device, dtype=dtype)
    V = torch.randn(S, H_kv, D, device=device, dtype=dtype)
    S_vec = torch.randn(H_q, device=device, dtype=dtype) / math.sqrt(D)
    sm_scale = 1.0 / math.sqrt(D)

    for window in (8, 16):
        ref = sdpa(Q, K, V, S_vec, sm_scale, sliding_window=window)
        out = _flex_forward_from_qkm(Q, K, V, S_vec, sm_scale, sliding_window=window, device=device, dtype=dtype)

        assert ref.shape == out.shape == (S, H_q * D)
        atol = 1e-2 if dtype == torch.bfloat16 else 5e-4
        rtol = 1e-2 if dtype == torch.bfloat16 else 5e-4
        max_err = (ref - out).abs().max().item()
        assert torch.allclose(ref, out, rtol=rtol, atol=atol), f"Mismatch (window={window}); max_err={max_err:.4e}"
