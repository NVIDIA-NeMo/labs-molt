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

"""CPU unit tests for the block-diagonal context-parallelism core.

These are the regression guards for CP *numerical correctness*: the headline test
proves that the block-diagonal CP attention (per-rank local queries against
all-gathered K/V with a per-document causal mask) is equivalent to plain full
attention with a block-causal mask -- i.e. cp_size>1 does not change the math vs
cp_size=1. No GPU / no distributed init required: the all-gather is mocked to
identity (the test passes the full K/V) and the CP ranks are simulated
in-process.
"""

import sys
import types

import pytest
import torch

from nemo_automodel.components.distributed.blockdiag_cp import batch as bd_batch
from nemo_automodel.components.distributed.blockdiag_cp import exchange as bd_exchange
from nemo_automodel.components.distributed.blockdiag_cp import kernels as bd_kernels
from nemo_automodel.components.distributed.blockdiag_cp import packed as bd_packed
from nemo_automodel.components.distributed.blockdiag_cp import runtime as bd_runtime
from nemo_automodel.components.distributed.blockdiag_cp import state as bd_state


def _doc_ids():
    # 2 documents (positions 0-2 and 3-5) + 2 padding positions (6-7); length 8 so it
    # shards evenly for world in {2, 4}.
    return torch.tensor([[1, 1, 1, 2, 2, 2, 0, 0]], dtype=torch.long)


class _IdentityGather:
    """Stand-in for _AllGatherSeqDiff whose apply() returns its input unchanged, so each
    simulated CP rank sees the FULL (already-gathered) K/V we pass in."""

    @staticmethod
    def apply(x, group, seq_dim):
        return x


def _run_blockdiag_cp(Q, K, V, doc_ids, world, enable_gqa=False):
    """Simulate ``world`` CP ranks in-process; returns the concatenated local outputs.

    Args:
        Q: Full queries ``[B, Hq, S, D]`` (sliced per simulated rank).
        K: Full keys ``[B, Hkv, S, D]`` (the identity all-gather passes them through).
        V: Full values ``[B, Hkv, S, D]``.
        doc_ids: Per-position document ids ``[B, S]`` (0 == padding).
        world: Number of simulated CP ranks.
        enable_gqa: Forwarded to the SDPA under test.

    Returns:
        Concatenated per-rank outputs ``[B, Hq, S, D]``.
    """
    B, Hq, S, D = Q.shape
    assert S % world == 0
    L = S // world
    orig = bd_exchange._AllGatherSeqDiff
    bd_exchange._AllGatherSeqDiff = _IdentityGather  # identity all-gather: we pass full K/V below
    outs = []
    try:
        for r in range(world):
            state = {
                "group": None,
                "doc_ids": doc_ids,
                "row_offset": r * L,
                "seq_dim": 2,
            }
            token = bd_state._CP_BLOCKDIAG_STATE.set(state)
            try:
                q_local = Q[:, :, r * L : (r + 1) * L, :]
                outs.append(bd_runtime.cp_blockdiag_sdpa(q_local, K, V, enable_gqa=enable_gqa))
            finally:
                bd_state._CP_BLOCKDIAG_STATE.reset(token)
    finally:
        bd_exchange._AllGatherSeqDiff = orig
    return torch.cat(outs, dim=2)


def _full_attention(Q, K, V, doc_ids, enable_gqa=False):
    """Reference: plain full attention ``[B, Hq, S, D]`` with the block-causal (per-document) mask.

    Args:
        Q: Full queries ``[B, Hq, S, D]``.
        K: Full keys ``[B, Hkv, S, D]`` (repeat-interleaved when ``enable_gqa``).
        V: Full values ``[B, Hkv, S, D]``.
        doc_ids: Per-position document ids ``[B, S]`` (0 == padding).
        enable_gqa: Expand K/V heads to match Q before the reference SDPA.
    """
    B, Hq, S, D = Q.shape
    mask = bd_kernels._cp_blockdiag_mask(doc_ids, 0, S, S, B)  # [B, 1, S, S]
    K2, V2 = K, V
    if enable_gqa and K.shape[1] != Hq:
        n = Hq // K.shape[1]
        K2 = K.repeat_interleave(n, dim=1)
        V2 = V.repeat_interleave(n, dim=1)
    return bd_runtime._ORIGINAL_SDPA(Q, K2, V2, attn_mask=mask, is_causal=False)


def test_knob_normalization_synonyms_and_defaults():
    assert bd_state.normalize_attn_backend(None) == "flash"
    assert bd_state.normalize_attn_backend("AUTO") == "flash"
    assert bd_state.normalize_attn_backend("flash_attention_2") == "flash"
    assert bd_state.normalize_attn_backend("transformer_engine") == "te"
    assert bd_state.normalize_attn_backend("sdpa") == "dense"
    assert bd_state.normalize_kv_exchange(None) == "allgather"
    assert bd_state.normalize_kv_exchange("all-gather") == "allgather"
    assert bd_state.normalize_kv_exchange("needed_only") == "halo"
    assert bd_state.normalize_kv_exchange("all_to_all") == "a2a"
    with pytest.raises(ValueError, match="attention backend"):
        bd_state.normalize_attn_backend("bogus")
    with pytest.raises(ValueError, match="kv_exchange"):
        bd_state.normalize_kv_exchange("bogus")


def test_configure_cp_varlen_roundtrip():
    prev = bd_state.cp_varlen_runtime_config()
    try:
        bd_state.configure_cp_varlen(attn_backend="te", kv_exchange="halo")
        assert bd_state.cp_varlen_runtime_config() == {"attn_backend": "te", "kv_exchange": "halo"}
    finally:
        bd_state.configure_cp_varlen(**prev)


def test_blockdiag_mask_shards_concat_to_full():
    """Per-rank masks (query rows sliced) concatenated == the full block-causal mask."""
    doc_ids = _doc_ids()
    S = doc_ids.shape[1]
    full = bd_kernels._cp_blockdiag_mask(doc_ids, 0, S, S, 1)  # [1, 1, S, S]
    for world in (2, 4):
        L = S // world
        shards = [bd_kernels._cp_blockdiag_mask(doc_ids, r * L, L, S, 1) for r in range(world)]
        recombined = torch.cat(shards, dim=2)
        assert recombined.shape == full.shape
        assert torch.equal(recombined, full), f"mask mismatch at world={world}"


def test_blockdiag_mask_expected_matrix():
    doc_ids = _doc_ids()
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
    ).view(1, 1, 8, 8)

    got = bd_kernels._cp_blockdiag_mask(doc_ids, 0, 8, 8, 1)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("world", [2, 4])
def test_blockdiag_sdpa_parity_vs_full_attention(world):
    """cp=world block-diagonal attention == cp=1 full attention on identical inputs."""
    torch.manual_seed(0)
    B, H, S, D = 1, 4, 8, 16
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)
    doc_ids = _doc_ids()

    cp_out = _run_blockdiag_cp(Q, K, V, doc_ids, world=world, enable_gqa=False)
    full_out = _full_attention(Q, K, V, doc_ids, enable_gqa=False)

    assert cp_out.shape == full_out.shape == (B, H, S, D)
    max_diff = (cp_out - full_out).abs().max().item()
    assert max_diff < 1e-5, f"world={world} CP-vs-full SDPA max_diff={max_diff}"


@pytest.mark.parametrize("world", [2, 4])
def test_blockdiag_sdpa_parity_gqa(world):
    """GQA path (n_kv_heads < n_q_heads): CP repeat-interleave matches full attention."""
    torch.manual_seed(1)
    B, Hq, Hkv, S, D = 1, 4, 2, 8, 16
    Q = torch.randn(B, Hq, S, D, dtype=torch.float32)
    K = torch.randn(B, Hkv, S, D, dtype=torch.float32)
    V = torch.randn(B, Hkv, S, D, dtype=torch.float32)
    doc_ids = _doc_ids()

    cp_out = _run_blockdiag_cp(Q, K, V, doc_ids, world=world, enable_gqa=True)
    full_out = _full_attention(Q, K, V, doc_ids, enable_gqa=True)

    assert cp_out.shape == full_out.shape == (B, Hq, S, D)
    max_diff = (cp_out - full_out).abs().max().item()
    assert max_diff < 1e-5, f"GQA world={world} CP-vs-full SDPA max_diff={max_diff}"


def test_blockdiag_sdpa_noop_without_state():
    """With no CP state set, cp_blockdiag_sdpa is a plain pass-through to stock SDPA."""
    torch.manual_seed(2)
    Q = torch.randn(1, 2, 4, 8)
    out = bd_runtime.cp_blockdiag_sdpa(Q, Q, Q, is_causal=True)
    ref = bd_runtime._ORIGINAL_SDPA(Q, Q, Q, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-6)


def test_blockdiag_sdpa_forwards_dropout_to_varlen(monkeypatch):
    """The CP wrapper preserves the caller's dropout probability on the varlen path."""
    forwarded_dropout = []

    def fake_varlen(query, key, value, doc_ids, row_offset, scale, backend, *, dropout_p=0.0, meta=None):
        """Return ``query`` ``[B, Hq, L, D]`` while recording the routed dropout probability."""
        forwarded_dropout.append(dropout_p)
        return query

    monkeypatch.setattr(bd_exchange, "_AllGatherSeqDiff", _IdentityGather)
    monkeypatch.setattr(bd_kernels, "_cp_blockdiag_varlen", fake_varlen)
    state = {
        "group": None,
        "doc_ids": torch.tensor([[1, 1, 2, 2]], dtype=torch.long),
        "row_offset": 0,
        "attn_backend": "flash",
        "kv_exchange": "allgather",
        "varlen_meta": {"n_real": 4},
    }
    token = bd_state._CP_BLOCKDIAG_STATE.set(state)
    try:
        qkv = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        assert bd_runtime.cp_blockdiag_sdpa(qkv, qkv, qkv, dropout_p=0.25) is qkv
    finally:
        bd_state._CP_BLOCKDIAG_STATE.reset(token)

    assert forwarded_dropout == [0.25]


def test_blockdiag_batch_synthesizes_and_shards_padding_mask(monkeypatch):
    class _Mesh:
        def size(self):
            return 2

        def get_local_rank(self):
            return 1

        def get_group(self):
            return object()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    batch = {
        "inputs_embeds": torch.randn(1, 5, 3),
        "labels": torch.arange(5).view(1, 5),
        "_packed_seq_ids": torch.tensor([[1, 1, 2, 0, 0]], dtype=torch.long),
    }

    ctx, sharded, layout = bd_batch.make_cp_blockdiag_batch_and_ctx(_Mesh(), None, batch)

    # The shared CP contract pads to 2 * cp_size: length 5 -> 8, so rank 1 owns
    # rows [4, 8), which are the input pad token plus the synthesized pad tail.
    assert torch.equal(sharded["padding_mask"], torch.tensor([[True, True, True, True]]))
    assert torch.equal(sharded["labels"], torch.tensor([[4, -100, -100, -100]]))
    assert layout.original_seq_len == 5
    assert layout.padded_seq_len == 8
    with ctx():
        state = bd_state._CP_BLOCKDIAG_STATE.get()
        assert state is not None
        assert state["row_offset"] == 4
        assert torch.equal(state["doc_ids"], torch.tensor([[1, 1, 2, 0, 0, 0, 0, 0]]))
        assert torch.equal(state["packed_cu_seqlens"], torch.tensor([0, 2, 3, 8]))
        assert torch.equal(state["packed_cu_seqlens_cpu"], torch.tensor([0, 2, 3, 8]))
        assert state["varlen_meta"]["n_real"] == 0  # all-padding local chunk
        model_state = bd_state.current_blockdiag_cp_state()
        assert isinstance(model_state, bd_state.BlockdiagCpModelState)
        assert model_state.group is state["group"]
        assert model_state.packed_cu_seqlens is state["packed_cu_seqlens"]
        assert model_state.packed_cu_seqlens_cpu is state["packed_cu_seqlens_cpu"]
    assert bd_state._CP_BLOCKDIAG_STATE.get() is None  # ctx exit restores the slot


def test_blockdiag_batch_shards_loss_mask(monkeypatch):
    class _Mesh:
        def size(self):
            return 2

        def get_local_rank(self):
            return 0

        def get_group(self):
            return object()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    batch = {
        "inputs_embeds": torch.randn(1, 6, 3),
        "labels": torch.arange(6).view(1, 6),
        "_packed_seq_ids": torch.tensor([[1, 1, 1, 2, 2, 0]], dtype=torch.long),
    }
    loss_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)

    _, sharded, layout = bd_batch.make_cp_blockdiag_batch_and_ctx(_Mesh(), None, batch, loss_mask=loss_mask)

    assert torch.equal(sharded["loss_mask"], torch.tensor([[1, 1, 1, 1]]))
    assert sharded["inputs_embeds"].shape == (1, 4, 3)
    assert layout.original_seq_len == 6
    assert layout.padded_seq_len == 8


def test_blockdiag_batch_aux_only_keeps_primary_full(monkeypatch):
    """In-forward multimodal embedding keeps input_ids full while aux tensors shard."""

    class _Mesh:
        def size(self):
            return 2

        def get_local_rank(self):
            return 1

        def get_group(self):
            return object()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    input_ids = torch.arange(5).view(1, 5)
    batch = {
        "input_ids": input_ids,
        "labels": torch.arange(5).view(1, 5),
        "position_ids": torch.arange(15).view(3, 1, 5),
        "_packed_seq_ids": torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long),
    }

    _, sharded, layout = bd_batch.make_cp_blockdiag_batch_and_ctx(_Mesh(), None, batch, shard_primary=False)

    assert sharded["input_ids"] is input_ids
    assert input_ids.shape == (1, 5)
    assert torch.equal(sharded["labels"], torch.tensor([[4, -100, -100, -100]]))
    assert torch.equal(sharded["position_ids"], torch.tensor([[[4, 0, 0, 0]], [[9, 0, 0, 0]], [[14, 0, 0, 0]]]))
    assert layout.original_seq_len == 5
    assert layout.padded_seq_len == 8


def test_blockdiag_batch_rejects_batch_size_gt_one(monkeypatch):
    class _Mesh:
        def size(self):
            return 2

        def get_local_rank(self):
            return 0

        def get_group(self):
            return object()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    batch = {
        "inputs_embeds": torch.randn(2, 4, 3),
        "labels": torch.zeros(2, 4, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="local_batch_size=1"):
        bd_batch.make_cp_blockdiag_batch_and_ctx(_Mesh(), None, batch)


def test_blockdiag_doc_ids_from_2d_attention_mask():
    batch = {"attention_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)}
    got = bd_batch._cp_blockdiag_doc_ids(batch, seq_len=5, device=torch.device("cpu"), batch_size=1)
    assert torch.equal(got, torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long))


@pytest.mark.parametrize("world", [2, 3, 4])
def test_allgather_seqdiff_forward_backward(monkeypatch, world):
    """_AllGatherSeqDiff: forward concatenates shards over the group; backward
    reduce-scatters the summed gradient. Validates world>2 (the all-gather is generic)."""

    def fake_all_gather(out_list, x, group=None):
        for o in out_list:
            o.copy_(x)  # simulate every rank holding the same shard

    def fake_reduce_scatter(local, chunks, op=None, group=None):
        local.copy_(sum(chunks))

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "reduce_scatter", fake_reduce_scatter)

    x = torch.randn(1, 2, 4, 8, requires_grad=True)
    out = bd_exchange._AllGatherSeqDiff.apply(x, None, 2)
    assert out.shape == (1, 2, 4 * world, 8)
    assert torch.allclose(out[:, :, :4], x.detach())  # first shard == this rank's x

    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert torch.equal(x.grad, torch.full_like(x, world))


def test_varlen_all_padding_rank_keeps_grad_attachment():
    """An all-padding CP rank must keep its output attached to K/V.

    Otherwise that rank skips the all-gather's backward reduce_scatter while
    real-token ranks fire it -> collective desync -> NCCL hang in backward.
    """
    doc_ids = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    q = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16, requires_grad=True)

    out = bd_kernels._cp_blockdiag_varlen(q, k, v, doc_ids, row_offset=2, backend="te")

    assert out is not None
    assert out.requires_grad
    assert torch.equal(out, torch.zeros_like(out))
    out.float().sum().backward()
    assert k.grad is not None and v.grad is not None
    assert torch.equal(k.grad, torch.zeros_like(k.grad))
    assert torch.equal(v.grad, torch.zeros_like(v.grad))


def test_varlen_metadata_guard_accepts_valid_left_straddle():
    meta = {
        "n_real": 5,
        "s_first": 0,
        "real_end": 8,
        "cu_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_k": torch.tensor([0, 5, 8], dtype=torch.int32),
        "max_q": 3,
        "max_k": 5,
    }

    reason = bd_kernels._varlen_metadata_unavailable_reason(
        meta,
        query_len=8,
        key_len=8,
        device=torch.device("cpu"),
    )

    assert reason is None
    assert meta["_validation_cache"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cu_q", torch.tensor([0, 2, 6], dtype=torch.int32), "n_real"),
        ("cu_k", torch.tensor([0, 5, 9], dtype=torch.int32), "slice length"),
        ("max_k", 4, "max_seqlen mismatch"),
    ],
)
def test_varlen_metadata_guard_rejects_kernel_oob_inputs(field, value, message):
    meta = {
        "n_real": 5,
        "s_first": 0,
        "real_end": 8,
        "cu_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_k": torch.tensor([0, 5, 8], dtype=torch.int32),
        "max_q": 3,
        "max_k": 5,
    }
    meta[field] = value

    reason = bd_kernels._varlen_metadata_unavailable_reason(
        meta,
        query_len=8,
        key_len=8,
        device=torch.device("cpu"),
    )

    assert reason is not None
    assert message in reason


def test_precompute_meta_matches_mask_segmentation():
    """The precomputed cu_seqlens reproduce the per-rank straddle geometry."""
    # doc 1 spans [0, 6) so rank 1 (rows [4, 8)) has a 4-token left straddle.
    doc_ids = torch.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0], dtype=torch.long)
    meta = bd_kernels.precompute_blockdiag_varlen_meta(doc_ids, row_offset=4, local_len=4, device=torch.device("cpu"))
    assert meta["n_real"] == 4
    assert meta["s_first"] == 0  # doc 1 starts at position 0
    assert meta["real_end"] == 8
    assert torch.equal(meta["cu_q"], torch.tensor([0, 2, 4], dtype=torch.int32))
    assert torch.equal(meta["cu_k"], torch.tensor([0, 6, 8], dtype=torch.int32))
    assert meta["first_q"] == 2 and meta["first_k"] == 6

    # all-padding chunk -> sentinel meta
    pad_meta = bd_kernels.precompute_blockdiag_varlen_meta(
        doc_ids, row_offset=11, local_len=1, device=torch.device("cpu")
    )
    assert pad_meta == {"n_real": 0}


def test_kv_plan_halo_vs_a2a_decision():
    """Docs fitting in <=2 ranks -> halo; a doc spanning >2 ranks -> a2a fallback."""
    dev = torch.device("cpu")
    # world=4, local_len=4: doc boundaries at 6 and 12 -> every straddle <= 4.
    doc_ids = torch.tensor([1] * 6 + [2] * 6 + [3] * 4, dtype=torch.long)
    plan = bd_exchange._compute_blockdiag_kv_plan(doc_ids, world=4, local_len=4, dev=dev)
    assert plan["use_halo"]
    # rank 1 straddles doc 1 back to position 0 (back=4); rank 2 straddles doc 2 by 2.
    assert plan["recv"] == [0, 4, 2, 0]
    assert plan["send"] == [4, 2, 0, 0]
    assert plan["nreal"] == [4, 4, 4, 4]

    # one doc across the whole sequence: rank 3 straddles 12 > local_len -> no halo.
    single = torch.ones(16, dtype=torch.long)
    plan_single = bd_exchange._compute_blockdiag_kv_plan(single, world=4, local_len=4, dev=dev)
    assert not plan_single["use_halo"]
    assert plan_single["recv"] == [0, 4, 8, 12]
    assert plan_single["s_first"] == [0, 0, 0, 0]


def test_needed_kv_a2a_plan_split_symmetry():
    """Every rank's a2a send splits mirror the receivers' out splits and cover the needed ranges."""
    dev = torch.device("cpu")
    world, local_len = 4, 4
    doc_ids = torch.ones(world * local_len, dtype=torch.long)  # single 16-token doc
    plan = bd_exchange._compute_blockdiag_kv_plan(doc_ids, world=world, local_len=local_len, dev=dev)

    in_splits_all, out_splits_all = [], []
    for r in range(world):
        in_splits, out_splits, send_index = bd_exchange._needed_kv_a2a_plan(plan, r, world, local_len, dev)
        in_splits_all.append(in_splits)
        out_splits_all.append(out_splits)
        assert send_index.numel() == sum(in_splits)
        # each rank receives exactly its needed range [s_first, real_end)
        assert sum(out_splits) == plan["real_end"][r] - plan["s_first"][r]
    for src in range(world):
        for dst in range(world):
            assert in_splits_all[src][dst] == out_splits_all[dst][src]


def test_select_kv_exchange_path_downgrades_name_reasons():
    """The path selector names every downgrade to all-gather."""
    state = {"attn_backend": "flash", "kv_exchange": "allgather", "varlen_meta": {"n_real": 1}}
    path, plan, reason = bd_runtime._select_kv_exchange_path(
        state, None, torch.ones(1, 8, dtype=torch.long), 4, torch.device("cpu"), 0
    )
    assert path == "allgather" and plan is None and "mode=allgather" in reason

    state = {"attn_backend": "dense", "kv_exchange": "halo", "varlen_meta": {"n_real": 1}}
    path, _, reason = bd_runtime._select_kv_exchange_path(
        state, None, torch.ones(1, 8, dtype=torch.long), 4, torch.device("cpu"), 0
    )
    assert path == "allgather" and "flash/te" in reason

    state = {"attn_backend": "flash", "kv_exchange": "halo", "varlen_meta": None}
    path, _, reason = bd_runtime._select_kv_exchange_path(
        state, None, torch.ones(1, 8, dtype=torch.long), 4, torch.device("cpu"), 0
    )
    assert path == "allgather" and "varlen_meta" in reason

    state = {"attn_backend": "te", "kv_exchange": "halo", "varlen_meta": {"n_real": 1}}
    path, _, reason = bd_runtime._select_kv_exchange_path(
        state,
        None,
        torch.ones(1, 8, dtype=torch.long),
        4,
        torch.device("cpu"),
        0,
        dropout_p=0.1,
    )
    assert path == "allgather" and "dropout" in reason


def test_flash_long_prefix_guard_peels_only_boundary_segment(monkeypatch):
    calls = []

    def fake_fixed(q, k, v, **kwargs):
        calls.append(("fixed", q.shape[1], k.shape[1], kwargs["causal"], kwargs["dropout_p"]))
        return q + 1

    def fake_varlen(q, k, v, **kwargs):
        calls.append(
            (
                "varlen",
                q.shape[0],
                k.shape[0],
                kwargs["max_seqlen_q"],
                kwargs["max_seqlen_k"],
                kwargs["dropout_p"],
            )
        )
        return q + 2

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_func = fake_fixed
    flash_attn.flash_attn_varlen_func = fake_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    q = torch.zeros(6, 4, 8, dtype=torch.bfloat16)
    k = torch.zeros(11, 2, 8, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    cu_q = torch.tensor([0, 2, 6], dtype=torch.int32)
    cu_k = torch.tensor([0, 7, 11], dtype=torch.int32)
    out = bd_kernels._flash_varlen_with_long_prefix_guard(
        q,
        k,
        v,
        cu_q=cu_q,
        cu_k=cu_k,
        max_q=4,
        max_k=7,
        local_query_len=4,
        scale=0.5,
        dropout_p=0.25,
        meta={"first_q": 2, "first_k": 7, "max_tail": 4},
    )

    assert calls == [("fixed", 2, 7, True, 0.25), ("varlen", 4, 4, 4, 4, 0.25)]
    assert torch.equal(out[:2], torch.ones_like(out[:2]))
    assert torch.equal(out[2:], torch.full_like(out[2:], 2))


def test_flash_long_prefix_guard_keeps_normal_single_varlen_call(monkeypatch):
    calls = []

    def fake_varlen(q, k, v, **kwargs):
        calls.append((q.shape[0], k.shape[0], kwargs["dropout_p"]))
        return q

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_func = lambda *args, **kwargs: pytest.fail("fixed Flash must not run for a normal segment")
    flash_attn.flash_attn_varlen_func = fake_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    q = torch.zeros(6, 4, 8, dtype=torch.bfloat16)
    k = torch.zeros(6, 2, 8, dtype=torch.bfloat16)
    out = bd_kernels._flash_varlen_with_long_prefix_guard(
        q,
        k,
        k,
        cu_q=torch.tensor([0, 2, 6], dtype=torch.int32),
        cu_k=torch.tensor([0, 2, 6], dtype=torch.int32),
        max_q=4,
        max_k=4,
        local_query_len=4,
        scale=None,
        dropout_p=0.125,
        meta={},
    )

    assert calls == [(6, 6, 0.125)]
    assert out is q


def test_cp1_packed_precomputes_varlen_metadata_once_per_forward(monkeypatch):
    """Every CP1 attention layer reuses the metadata armed by the outer forward."""
    doc_ids = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    meta = {"n_real": 4, "sentinel": object()}
    precompute_calls = []
    forwarded = []

    def fake_precompute(got_doc_ids, row_offset, local_len, device):
        precompute_calls.append((got_doc_ids, row_offset, local_len, device))
        return meta

    def fake_varlen(
        query,
        key,
        value,
        doc_ids_arg,
        row_offset,
        scale,
        backend,
        *,
        dropout_p=0.0,
        meta=None,
    ):
        """Return ``query`` ``[B, Hq, S, D]`` while recording per-forward kernel options."""
        forwarded.append((meta, dropout_p))
        return query

    monkeypatch.setattr(bd_packed.kernels, "precompute_blockdiag_varlen_meta", fake_precompute)
    monkeypatch.setattr(bd_packed.kernels, "_cp_blockdiag_varlen", fake_varlen)

    bd_packed.enable_cp1_packed_varlen(doc_ids, "flash")
    try:
        qkv = torch.randn(1, 2, 5, 4)
        assert bd_packed._packed_varlen_sdpa(qkv, qkv, qkv, dropout_p=0.2) is qkv
        assert bd_packed._packed_varlen_sdpa(qkv, qkv, qkv, dropout_p=0.2) is qkv
    finally:
        bd_packed.disable_cp1_packed_varlen()

    assert len(precompute_calls) == 1
    got_doc_ids, row_offset, local_len, device = precompute_calls[0]
    assert got_doc_ids is doc_ids
    assert row_offset == 0
    assert local_len == doc_ids.shape[-1]
    assert device == doc_ids.device
    assert forwarded == [(meta, 0.2), (meta, 0.2)]


def test_cp1_packed_1d_doc_ids_use_block_diagonal_attention():
    """A documented ``[S]`` doc-id vector must not fall through to ordinary causal SDPA."""
    torch.manual_seed(7)
    doc_ids = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    qkv = torch.randn(1, 2, 4, 8)
    allow = bd_kernels._cp_blockdiag_mask(doc_ids, 0, 4, 4, 1)
    ref = bd_runtime._ORIGINAL_SDPA(qkv, qkv, qkv, attn_mask=allow)
    ordinary = bd_runtime._ORIGINAL_SDPA(qkv, qkv, qkv, is_causal=True)

    bd_packed.enable_cp1_packed_varlen(doc_ids, "flash")
    try:
        got = bd_packed._packed_varlen_sdpa(qkv, qkv, qkv, is_causal=True)
    finally:
        bd_packed.disable_cp1_packed_varlen()

    torch.testing.assert_close(got, ref)
    assert not torch.allclose(got, ordinary)


def test_cp1_packed_passes_through_non_matching_shapes():
    """SDPA calls that do not match the armed doc_ids (e.g. vision attention) pass through."""
    doc_ids = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    bd_packed.enable_cp1_packed_varlen(doc_ids, "flash")
    try:
        q = torch.randn(1, 2, 7, 4)  # seq len 7 != 5 -> pass-through
        out = bd_packed._packed_varlen_sdpa(q, q, q, is_causal=True)
        ref = bd_runtime._ORIGINAL_SDPA(q, q, q, is_causal=True)
        assert torch.allclose(out, ref, atol=1e-6)
    finally:
        bd_packed.disable_cp1_packed_varlen()


def test_cp1_packed_hooks_scope_patch_and_cover_ac_recompute(monkeypatch):
    """The SDPA patch is live only inside hooked attention forwards (incl. AC recompute).

    ``attach_cp1_packed_varlen_hooks`` must (1) leave the process-wide
    ``F.scaled_dot_product_attention`` untouched outside a hooked forward, (2) route
    matching SDPA calls inside the attention forward through the varlen path, and
    (3) fire again during activation-checkpointing recompute in backward (hooks sit
    on the checkpoint-wrapped inner module).
    """
    import torch.nn.functional as F
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    doc_ids = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    varlen_calls = []
    patched_at_forward_entry = []

    def fake_varlen(
        query,
        key,
        value,
        doc_ids_arg,
        row_offset,
        scale,
        backend,
        *,
        dropout_p=0.0,
        meta=None,
    ):
        """Return doubled ``query`` ``[B, Hq, S, D]`` for hook/recompute observability."""
        varlen_calls.append(backend)
        return query * 2.0

    monkeypatch.setattr(bd_packed.kernels, "_cp_blockdiag_varlen", fake_varlen)

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            """Project ``x`` ``[B, S, E=4]`` to ``[B, H=2, S, D=2]``, run SDPA, return ``[B, S, E]``."""
            # Record the patch state at forward entry: AC recompute may early-stop the
            # body once every saved tensor is rebuilt, but it always enters here.
            patched_at_forward_entry.append(F.scaled_dot_product_attention is bd_packed._packed_varlen_sdpa)
            B, S, _ = x.shape
            q = self.proj(x).view(B, S, 2, 2).transpose(1, 2)  # [B, H, S, D]
            o = F.scaled_dot_product_attention(q, q, q)
            return o.transpose(1, 2).reshape(B, S, 4)

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = checkpoint_wrapper(_Attn())

        def forward(self, x):
            """Run the checkpoint-wrapped attention on ``x`` ``[B, S, E]``."""
            return self.self_attn(x)

    model = _Block()
    bd_packed.attach_cp1_packed_varlen_hooks(model)
    bd_packed.enable_cp1_packed_varlen(doc_ids, "flash")
    try:
        # Arming state does NOT patch process-wide SDPA.
        assert F.scaled_dot_product_attention is bd_runtime._ORIGINAL_SDPA
        x = torch.randn(1, 5, 4, requires_grad=True)
        out = model(x)
        assert patched_at_forward_entry == [True]  # patch was live inside the attention forward
        assert varlen_calls == ["flash"]  # ...and routed the matching SDPA call to varlen
        assert F.scaled_dot_product_attention is bd_runtime._ORIGINAL_SDPA  # restored after forward
        out.sum().backward()  # AC recompute re-enters the inner forward -> hooks fire again
        assert patched_at_forward_entry == [True, True]
        assert x.grad is not None
        assert F.scaled_dot_product_attention is bd_runtime._ORIGINAL_SDPA
        # A doc_ids-shape-matching SDPA call OUTSIDE any hooked forward is untouched.
        q = torch.randn(1, 2, 5, 2)
        n_calls = len(varlen_calls)
        assert torch.allclose(F.scaled_dot_product_attention(q, q, q), bd_runtime._ORIGINAL_SDPA(q, q, q))
        assert len(varlen_calls) == n_calls
    finally:
        bd_packed.disable_cp1_packed_varlen()
