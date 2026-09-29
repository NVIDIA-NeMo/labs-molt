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

"""1-GPU parity: varlen block-diagonal CP attention (flash AND te) == dense-mask SDPA.

The CP ranks are simulated in-process (each rank's local query chunk attends the
full K/V, exactly what the all-gather delivers), so this needs a single GPU. It
sweeps multi-document packs with padding tails, GQA head layouts, left-straddling
documents (the ``sk > sq`` boundary case), single documents spanning every rank,
and head_dim 256, for every simulated rank offset.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.distributed.blockdiag_cp import (
    attach_cp1_packed_varlen_hooks,
    disable_cp1_packed_varlen,
    enable_cp1_packed_varlen,
)
from nemo_automodel.components.distributed.blockdiag_cp import packed as bd_packed
from nemo_automodel.components.distributed.blockdiag_cp.kernels import (
    _cp_blockdiag_mask,
    _cp_blockdiag_varlen,
    precompute_blockdiag_varlen_meta,
)
from nemo_automodel.components.distributed.blockdiag_cp.runtime import _ORIGINAL_SDPA

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen parity requires CUDA")


def _dense_ref(query, key_full, value_full, doc_ids, row_offset):
    """The exact ops the dense CP path runs: GQA-repeat K/V + [B,1,L,S] mask + SDPA.

    Args:
        query: Local queries ``[B, Hq, L, D]``.
        key_full: Full-sequence keys ``[B, Hkv, S, D]``.
        value_full: Full-sequence values ``[B, Hkv, S, D]``.
        doc_ids: Per-position document ids ``[B, S]`` (0 == padding).
        row_offset: Global position of the first local query row.

    Returns:
        Reference attention output ``[B, Hq, L, D]``.
    """
    B, Hq, L, D = query.shape
    Hkv, S = key_full.shape[1], key_full.shape[2]
    k = key_full
    v = value_full
    if Hkv != Hq:
        n_rep = Hq // Hkv
        k = key_full.repeat_interleave(n_rep, dim=1)
        v = value_full.repeat_interleave(n_rep, dim=1)
    allow = _cp_blockdiag_mask(doc_ids, row_offset, L, S, B)  # [B,1,L,S] bool
    return F.scaled_dot_product_attention(query, k, v, attn_mask=allow)


def _make_doc_ids(seg_lens, pad, S, device):
    """Build ``[1, S]`` document ids: 1-based per-document runs + a zero pad tail."""
    ids = []
    for d, n in enumerate(seg_lens, start=1):
        ids += [d] * n
    ids += [0] * pad
    assert len(ids) == S, (len(ids), S)
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,S]


# (name, seg_lens, pad, S, cp, Hq, Hkv, D)
_CASES = [
    ("multidoc+pad cp2", [40, 30, 25], 33, 128, 2, 8, 8, 128),
    ("multidoc+pad cp4", [40, 30, 25], 33, 128, 4, 8, 8, 128),
    ("multidoc+pad cp8", [40, 30, 25], 33, 128, 8, 8, 8, 128),
    ("gqa cp4", [50, 40, 38], 0, 128, 4, 8, 2, 128),
    ("gqa cp8 +pad", [50, 40, 20], 18, 128, 8, 8, 2, 128),
    ("single-doc cp4", [128], 0, 128, 4, 8, 8, 128),
    ("straddle cp8", [70, 20, 20], 18, 128, 8, 8, 8, 128),
    ("hd256 gqa straddle cp8", [70, 20, 20], 18, 128, 8, 16, 2, 256),
    ("heavy-pad cp4", [20, 10], 34, 64, 4, 8, 8, 128),
]


def _backend_or_skip(backend: str) -> None:
    if backend == "flash":
        pytest.importorskip("flash_attn")
    else:
        pytest.importorskip("transformer_engine.pytorch")


@requires_cuda
@pytest.mark.parametrize("backend", ["flash", "te"])
@pytest.mark.parametrize(("name", "seg_lens", "pad", "S", "cp", "Hq", "Hkv", "D"), _CASES)
def test_varlen_blockdiag_parity(backend, name, seg_lens, pad, S, cp, Hq, Hkv, D):
    """Varlen output matches the dense-mask reference on every simulated CP rank."""
    _backend_or_skip(backend)
    device = "cuda"
    dtype = torch.bfloat16
    atol = 2e-2
    torch.manual_seed(0)
    B = 1
    doc_ids = _make_doc_ids(seg_lens, pad, S, device)
    key_full = torch.randn(B, Hkv, S, D, device=device, dtype=dtype)
    value_full = torch.randn(B, Hkv, S, D, device=device, dtype=dtype)
    L = S // cp
    for r in range(cp):
        off = r * L
        q = torch.randn(B, Hq, L, D, device=device, dtype=dtype)
        ref = _dense_ref(q, key_full, value_full, doc_ids, off)  # [B,Hq,L,D]
        got = _cp_blockdiag_varlen(q, key_full, value_full, doc_ids, off, backend=backend)
        assert got is not None, f"{backend} path returned None (import/dtype fallback)"
        # compare only REAL local rows (doc id > 0); padding rows are zeros by design
        local = doc_ids[0, off : off + L]
        real = local > 0
        if real.sum() == 0:
            # all-padding shard: output must be all zeros
            md = got.abs().max().item()
            assert md == 0.0, f"{name} r{r}: all-pad shard not zero ({md})"
            continue
        rr = real.nonzero().flatten()
        d = (ref[0, :, rr, :].float() - got[0, :, rr, :].float()).abs().max().item()
        assert d < atol, f"[{backend}] {name} r{r} off{off}: max|diff|={d:.4f} >= {atol}"


@requires_cuda
def test_flash_varlen_dropout_is_applied_and_backward_is_finite():
    """Nonzero Flash dropout changes varlen attention and preserves finite Q/K/V gradients."""
    _backend_or_skip("flash")
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(23)
    doc_ids = torch.ones(1, 64, dtype=torch.long, device=device)
    query = torch.randn(1, 4, 64, 64, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(1, 4, 64, 64, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(1, 4, 64, 64, device=device, dtype=dtype, requires_grad=True)

    without_dropout = _cp_blockdiag_varlen(query, key, value, doc_ids, 0, backend="flash")
    torch.manual_seed(29)
    with_dropout = _cp_blockdiag_varlen(query, key, value, doc_ids, 0, backend="flash", dropout_p=0.25)

    assert without_dropout is not None and with_dropout is not None
    assert not torch.equal(without_dropout, with_dropout)
    with_dropout.float().square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@requires_cuda
def test_te_varlen_dropout_uses_dense_block_diagonal_fallback():
    """TE THD dropout falls back before launch and exactly matches masked dense SDPA."""
    _backend_or_skip("te")
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(41)
    doc_ids = _make_doc_ids([20, 12], 0, 32, device)
    query = torch.randn(1, 4, 32, 64, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(1, 4, 32, 64, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(1, 4, 32, 64, device=device, dtype=dtype, requires_grad=True)
    allow = _cp_blockdiag_mask(doc_ids, 0, 32, 32, 1)

    enable_cp1_packed_varlen(doc_ids, "te")
    try:
        torch.manual_seed(43)
        got = bd_packed._packed_varlen_sdpa(query, key, value, dropout_p=0.25)
    finally:
        disable_cp1_packed_varlen()
    torch.manual_seed(43)
    ref = _ORIGINAL_SDPA(query, key, value, attn_mask=allow, dropout_p=0.25)

    torch.testing.assert_close(got, ref)
    got.float().square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@requires_cuda
@pytest.mark.parametrize("backend", ["flash"])
@pytest.mark.parametrize(("name", "seg_lens", "pad", "S", "cp", "Hq", "Hkv", "D"), _CASES)
def test_precomputed_meta_bit_identical(backend, name, seg_lens, pad, S, cp, Hq, Hkv, D):
    """Feeding a precomputed meta yields BIT-IDENTICAL output to the inline path.

    The segmentation depends only on (doc_ids, row_offset, local_len) -- all
    step-constant -- so the per-step precompute must reproduce the exact same
    kernel launch (same cu_seqlens) as the inline per-call segmentation.
    """
    _backend_or_skip(backend)
    device = "cuda"
    torch.manual_seed(0)
    doc_ids = _make_doc_ids(seg_lens, pad, S, device)
    key_full = torch.randn(1, Hkv, S, D, device=device, dtype=torch.bfloat16)
    value_full = torch.randn(1, Hkv, S, D, device=device, dtype=torch.bfloat16)
    L = S // cp
    for r in range(cp):
        off = r * L
        q = torch.randn(1, Hq, L, D, device=device, dtype=torch.bfloat16)
        base = _cp_blockdiag_varlen(q, key_full, value_full, doc_ids, off, backend=backend)
        meta = precompute_blockdiag_varlen_meta(doc_ids, off, L, device)
        fast = _cp_blockdiag_varlen(q, key_full, value_full, doc_ids, off, backend=backend, meta=meta)
        assert base is not None and fast is not None, f"{name} r{r}: None output"
        assert torch.equal(base, fast), (
            f"[{backend}] {name} r{r} off{off}: precomputed-meta output differs "
            f"max|diff|={(base.float() - fast.float()).abs().max().item()}"
        )


@requires_cuda
@pytest.mark.parametrize("backend", ["flash", "te"])
@pytest.mark.parametrize(
    ("seg_lens", "pad", "S", "Hq", "Hkv", "D"),
    [
        ([40, 30, 25], 33, 128, 8, 8, 128),
        ([50, 40, 20], 18, 128, 16, 2, 256),  # GQA 8:1, head_dim 256
    ],
)
def test_cp1_packed_hook_parity(backend, seg_lens, pad, S, Hq, Hkv, D):
    """cp1 packed path: F.sdpa under the scoped varlen hooks == dense block-diagonal SDPA.

    Exercises the full hook chain (attach_cp1_packed_varlen_hooks pre/post hooks ->
    _packed_varlen_sdpa -> _cp_blockdiag_varlen at row_offset=0), as used by packed
    cp_size==1 runs. The whole packed sequence is on one rank (q == k == full). The
    SDPA patch must be live only during the hooked attention forward and restored
    afterwards.
    """
    _backend_or_skip(backend)
    device = "cuda"
    dtype = torch.bfloat16
    atol = 2e-2
    torch.manual_seed(0)
    doc_ids = _make_doc_ids(seg_lens, pad, S, device)  # [1, S]
    q = torch.randn(1, Hq, S, D, device=device, dtype=dtype)
    k = torch.randn(1, Hkv, S, D, device=device, dtype=dtype)
    v = torch.randn(1, Hkv, S, D, device=device, dtype=dtype)
    ref = _dense_ref(q, k, v, doc_ids, 0)  # [1, Hq, S, D]

    class _SdpaSelfAttn(torch.nn.Module):
        def forward(self, q, k, v, scale, enable_gqa):
            """Stock-SDPA attention over ``q`` ``[B, Hq, S, D]`` / ``k``/``v`` ``[B, Hkv, S, D]``."""
            return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=enable_gqa)

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _SdpaSelfAttn()

        def forward(self, q, k, v, scale, enable_gqa):
            """Route ``q``/``k``/``v`` ``[B, H, S, D]`` through the hooked attention module."""
            return self.self_attn(q, k, v, scale, enable_gqa)

    model = _Block()
    attach_cp1_packed_varlen_hooks(model)
    enable_cp1_packed_varlen(doc_ids, backend)
    try:
        got = model(q, k, v, D**-0.5, Hkv != Hq)
    finally:
        disable_cp1_packed_varlen()
    assert F.scaled_dot_product_attention is _ORIGINAL_SDPA  # patch scoped to the hooked forward
    real = (doc_ids[0] > 0).nonzero().flatten()
    d = (ref[0, :, real, :].float() - got[0, :, real, :].float()).abs().max().item()
    assert d < atol, f"[cp1-hook {backend}] max|diff|={d:.4f} >= {atol}"
