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

"""CPU unit tests for DeepSeek V4 model-owned context-parallel batch prep.

Covers the model-owned CP path that runs without a real process group:
``make_dsv4_contiguous_shard_cp_batch_and_ctx`` (the ``ContextParallelSharder.shard_batch``
callable), the scalar group helpers, ``dsv4_cp_local_seq_multiple``, and the
sharder-only ``DeepseekV4ForCausalLM`` CP-prep hook (``prepare_model_inputs_for_cp``).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4 import cp as cpmod
from nemo_automodel.components.models.deepseek_v4.cp import (
    dsv4_cp_enabled,
    dsv4_cp_local_seq_multiple,
    dsv4_cp_rank,
    dsv4_cp_size,
    make_dsv4_contiguous_shard_cp_batch_and_ctx,
)
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM


@pytest.fixture(autouse=True)
def _force_no_dist(monkeypatch):
    """Exercise the single-process (no process group) path deterministically.

    These tests drive the CP helpers/callable with a fake mesh whose ``get_group``
    returns a sentinel, not a real ProcessGroup. If another test in the same pytest
    worker left ``torch.distributed`` initialized, the helpers would otherwise call
    ``dist.get_rank(group=<sentinel>)`` and raise. Pin ``is_initialized`` to False so
    rank resolution falls back to ``cp_mesh.get_local_rank()``.
    """
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


class _FakeMesh:
    """Minimal stand-in for a CP device-mesh slice (no real process group)."""

    def __init__(self, size: int, local_rank: int = 0, group: object = "cp_group_sentinel"):
        self._size = size
        self._local_rank = local_rank
        self._group = group

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_group(self) -> object:
        return self._group


def _shard(batch, *, cp_size, local_rank, **kwargs):
    mesh = _FakeMesh(cp_size, local_rank)
    return make_dsv4_contiguous_shard_cp_batch_and_ctx(mesh, None, batch, **kwargs)


# --------------------------------------------------------------------------- #
# Scalar group helpers (no dist initialized -> degenerate single-rank values)  #
# --------------------------------------------------------------------------- #
def test_group_helpers_without_dist():
    assert dsv4_cp_enabled(None) is False
    assert dsv4_cp_enabled("anything") is False
    assert dsv4_cp_rank(None) == 0
    assert dsv4_cp_size(None) == 1


# --------------------------------------------------------------------------- #
# dsv4_cp_local_seq_multiple                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ratios,expected",
    [
        (None, 1),  # no ratios configured
        ([], 1),
        ([0, 0], 1),  # only non-positive ratios -> filtered out
        ([2], 2),  # plain ratio
        ([4], 8),  # ratio-4 layers need 2*ratio for the cross-window overlap
        ([0, 4, 128], 128),  # lcm(8, 128)
        ([2, 3], 6),  # lcm of plain ratios
    ],
)
def test_local_seq_multiple(ratios, expected):
    cfg = SimpleNamespace(compress_ratios=ratios)
    assert dsv4_cp_local_seq_multiple(cfg) == expected
    # also accepts an object carrying `.config`
    assert dsv4_cp_local_seq_multiple(SimpleNamespace(config=cfg)) == expected


def test_build_packed_seq_ids_handles_1d_padding_zero_and_truncation():
    seq_ids = cpmod.build_packed_seq_ids(
        torch.tensor([2, cpmod._SEQ_LENS_PADDING_VALUE, 0, 3]),
        seq_len=4,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(seq_ids, torch.tensor([[1, 1, 3, 3]]))


def test_packed_cp_mask_gathers_metadata_and_applies_padding(monkeypatch):
    gathered = iter(
        (
            torch.tensor([[0, 1, 0, 1]]),
            torch.tensor([[1, 1, 2, 2]]),
            torch.tensor([[False, False, False, True]]),
        )
    )
    monkeypatch.setattr(cpmod, "dsv4_cp_all_gather_metadata", lambda *args, **kwargs: next(gathered))

    mask = cpmod.build_dsv4_cp_packed_causal_padding_mask(
        position_ids=torch.tensor([0, 1]),
        packed_seq_ids=torch.tensor([2, 2]),
        dtype=torch.float32,
        device=torch.device("cpu"),
        cp_group=object(),
        padding_mask=torch.tensor([[False, False]]),
        sliding_window=2,
    )

    min_value = torch.finfo(torch.float32).min
    expected = torch.full((1, 1, 2, 4), min_value)
    expected[:, :, :, 2] = 0
    torch.testing.assert_close(mask, expected)


# --------------------------------------------------------------------------- #
# make_dsv4_contiguous_shard_cp_batch_and_ctx                                   #
# --------------------------------------------------------------------------- #
def test_contiguous_shard_basic_input_ids():
    seq = 8
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
    }
    ctx, out, _ = _shard(batch, cp_size=2, local_rank=1)
    # context manager factory is the nullcontext class (instantiated by the caller)
    assert ctx is contextlib.nullcontext
    # divisor = cp_size * max(pad_multiple or 2, 2) = 4; seq=8 already divisible -> no pad.
    # local_seq = 8 / 2 = 4; rank 1 owns [4:8].
    assert out["input_ids"].shape == (1, 4)
    torch.testing.assert_close(out["input_ids"], torch.arange(4, 8).view(1, 4))
    torch.testing.assert_close(out["labels"], torch.arange(4, 8).view(1, 4))
    # position_ids were synthesized over the global sequence then sharded.
    torch.testing.assert_close(out["position_ids"], torch.arange(4, 8).view(1, 4))
    # the CP process group is handed to the forward.
    assert out["_dsv4_cp_group"] == "cp_group_sentinel"


def test_contiguous_shard_rank0_slice():
    batch = {"input_ids": torch.arange(8).view(1, 8), "labels": torch.arange(8).view(1, 8)}
    _, out, _ = _shard(batch, cp_size=2, local_rank=0)
    torch.testing.assert_close(out["input_ids"], torch.arange(0, 4).view(1, 4))


def test_cp_size_one_native_thd_preserves_packed_batch_without_padding():
    batch = {
        "input_ids": torch.arange(8).view(1, 8),
        "labels": torch.arange(8).view(1, 8),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 1, 1, 0, 0]]),
        "seq_lens": torch.tensor([[3, 2]]),
        "seq_lens_padded": torch.tensor([[4, 4]]),
    }
    expected = {key: value.clone() for key, value in batch.items()}

    ctx, out, _ = _shard(batch, cp_size=1, local_rank=0, pad_multiple=8, padding_token_id=99)

    assert ctx is contextlib.nullcontext
    assert out["qkv_format"] == "thd"
    assert "_dsv4_cp_group" not in out
    assert "packed_seq_ids" not in out
    assert "padding_mask" not in out
    for key, value in expected.items():
        torch.testing.assert_close(out[key], value)


def test_contiguous_shard_pads_to_divisor():
    # seq=5, cp_size=2, pad_multiple=2 -> divisor=2*max(2,2)=4 -> pad to 8.
    batch = {"input_ids": torch.arange(5).view(1, 5), "labels": torch.arange(5).view(1, 5)}
    _, out, _ = _shard(batch, cp_size=2, local_rank=0, pad_multiple=2)
    assert out["input_ids"].shape == (1, 4)  # padded global 8 // cp_size 2
    # labels pad uses ignore_index -100
    batch2 = {"input_ids": torch.arange(5).view(1, 5), "labels": torch.arange(5).view(1, 5)}
    _, out2, _ = _shard(batch2, cp_size=2, local_rank=1, pad_multiple=2)
    assert (out2["labels"] == -100).any()


def test_contiguous_shard_pad_multiple_controls_shard_size():
    # pad_multiple=4 -> divisor=cp_size*max(4,2)=8; seq=8 already divisible.
    batch = {"input_ids": torch.arange(8).view(1, 8), "labels": torch.arange(8).view(1, 8)}
    _, out, _ = _shard(batch, cp_size=2, local_rank=0, pad_multiple=4)
    assert out["input_ids"].shape == (1, 4)
    # seq=8, pad_multiple=8 -> divisor=16 -> pad to 16, local=8.
    batch2 = {"input_ids": torch.arange(8).view(1, 8), "labels": torch.arange(8).view(1, 8)}
    _, out2, _ = _shard(batch2, cp_size=2, local_rank=0, pad_multiple=8)
    assert out2["input_ids"].shape == (1, 8)


def test_contiguous_shard_attention_mask_2d_to_padding_mask():
    seq = 8
    attn = torch.ones(1, seq, dtype=torch.long)
    attn[0, -2:] = 0  # last two are padding
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
        "attention_mask": attn,
    }
    _, out, _ = _shard(batch, cp_size=2, local_rank=1)
    assert "attention_mask" not in out  # consumed
    # rank 1 owns [4:8]; padding_mask True == pad on the last two positions.
    torch.testing.assert_close(out["padding_mask"], torch.tensor([[False, False, True, True]]))


def test_contiguous_shard_attention_mask_4d_to_padding_mask():
    seq = 4
    # 4D additive mask: diagonal 0 == attend, nonzero (e.g. -inf penalty) == padded.
    attn = torch.zeros(1, 1, seq, seq)
    attn[0, 0, range(seq), range(seq)] = torch.tensor([0.0, 0.0, -1e9, -1e9])
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
        "attention_mask": attn,
    }
    _, out, _ = _shard(batch, cp_size=2, local_rank=1)
    torch.testing.assert_close(out["padding_mask"], torch.tensor([[True, True]]))


def test_contiguous_shard_attention_mask_4d_bool():
    seq = 4
    # 4D boolean mask: diagonal True == attend, so logical_not() == padded.
    attn = torch.zeros(1, 1, seq, seq, dtype=torch.bool)
    attn[0, 0, range(seq), range(seq)] = torch.tensor([True, True, True, False])
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
        "attention_mask": attn,
    }
    _, out, _ = _shard(batch, cp_size=2, local_rank=1)
    torch.testing.assert_close(out["padding_mask"], torch.tensor([[False, True]]))


def test_contiguous_shard_inputs_embeds_path():
    seq, hidden = 8, 3
    batch = {
        "inputs_embeds": torch.randn(1, seq, hidden),
        "labels": torch.arange(seq).view(1, seq),
    }
    _, out, _ = _shard(batch, cp_size=2, local_rank=0)
    assert "inputs_embeds" in out and out["inputs_embeds"].shape == (1, 4, hidden)
    assert "input_ids" not in out


def test_contiguous_shard_loss_mask_becomes_labels_when_labels_absent():
    seq = 8
    batch = {"input_ids": torch.arange(seq).view(1, seq)}
    loss_mask = torch.ones(1, seq)
    _, out, _ = _shard(batch, cp_size=2, local_rank=0, loss_mask=loss_mask)
    # loss_mask was promoted to labels then sharded.
    assert out["labels"].shape == (1, 4)


def test_contiguous_shard_loss_mask_kept_alongside_labels():
    seq = 8
    batch = {"input_ids": torch.arange(seq).view(1, seq), "labels": torch.arange(seq).view(1, seq)}
    loss_mask = torch.ones(1, seq)
    _, out, _ = _shard(batch, cp_size=2, local_rank=1, loss_mask=loss_mask)
    assert out["loss_mask"].shape == (1, 4)


def test_contiguous_shard_position_ids_3d():
    seq = 8
    pos = torch.arange(seq).view(1, 1, seq).expand(1, 3, seq).contiguous()
    batch = {"input_ids": torch.arange(seq).view(1, seq), "labels": torch.arange(seq).view(1, seq), "position_ids": pos}
    _, out, _ = _shard(batch, cp_size=2, local_rank=1)
    assert out["position_ids"].shape == (1, 3, 4)
    torch.testing.assert_close(out["position_ids"][0, 0], torch.arange(4, 8))


def test_contiguous_shard_packed_sequence_guards():
    base = {"input_ids": torch.arange(8).view(1, 8), "labels": torch.arange(8).view(1, 8)}
    with pytest.raises(NotImplementedError, match="pre-flattened `cu_seqlens`"):
        _shard({**base, "cu_seqlens": torch.tensor([0, 8])}, cp_size=2, local_rank=0)
    with pytest.raises(KeyError, match="requires `seq_lens`"):
        _shard({**base, "qkv_format": "thd"}, cp_size=2, local_rank=0)


def test_contiguous_shard_packed_sequence_pads_each_doc_before_cp_slice():
    batch = {
        "input_ids": torch.arange(8).view(1, 8),
        "labels": torch.arange(8).view(1, 8),
        "qkv_format": "thd",
        "seq_lens": torch.tensor([[3, 2]]),
        "seq_lens_padded": torch.tensor([[3, 5]]),
    }

    _, rank0, _ = _shard(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()},
        cp_size=2,
        local_rank=0,
        pad_multiple=4,
        padding_token_id=99,
    )
    _, rank1, _ = _shard(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()},
        cp_size=2,
        local_rank=1,
        pad_multiple=4,
        padding_token_id=99,
    )

    torch.testing.assert_close(rank0["input_ids"], torch.tensor([[0, 1, 2, 99]]))
    torch.testing.assert_close(rank0["labels"], torch.tensor([[0, 1, 2, -100]]))
    torch.testing.assert_close(rank0["position_ids"], torch.tensor([[0, 1, 2, 3]]))
    torch.testing.assert_close(rank0["padding_mask"], torch.tensor([[False, False, False, True]]))
    torch.testing.assert_close(rank0["packed_seq_ids"], torch.tensor([[1, 1, 1, 1]]))

    torch.testing.assert_close(rank1["input_ids"], torch.tensor([[3, 4, 99, 99]]))
    torch.testing.assert_close(rank1["labels"], torch.tensor([[3, 4, -100, -100]]))
    torch.testing.assert_close(rank1["position_ids"], torch.tensor([[0, 1, 2, 3]]))
    torch.testing.assert_close(rank1["padding_mask"], torch.tensor([[False, False, True, True]]))
    torch.testing.assert_close(rank1["packed_seq_ids"], torch.tensor([[2, 2, 2, 2]]))

    torch.testing.assert_close(rank0["seq_lens"], torch.tensor([[3, 2]]))
    torch.testing.assert_close(rank0["seq_lens_padded"], torch.tensor([[4, 4]]))
    assert rank0["qkv_format"] == "thd"


def test_repad_packed_inputs_embeds_and_loss_mask_with_1d_lengths():
    inputs_embeds = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
    batch = {
        "inputs_embeds": inputs_embeds,
        "labels": torch.tensor([[10, 11, 12]]),
        "seq_lens": torch.tensor([2, 1]),
        "seq_lens_padded": torch.tensor([2]),
    }

    out, loss_mask, _ = cpmod._repad_dsv4_packed_batch(
        batch,
        cp_size=2,
        pad_multiple=2,
        padding_token_id=99,
        loss_mask=torch.ones(1, 3),
    )

    assert out["inputs_embeds"].shape == (1, 4, 2)
    torch.testing.assert_close(out["inputs_embeds"][0, :3], inputs_embeds[0])
    torch.testing.assert_close(out["inputs_embeds"][0, 3], torch.zeros(2))
    torch.testing.assert_close(out["labels"], torch.tensor([[10, 11, 12, -100]]))
    torch.testing.assert_close(out["seq_lens"], torch.tensor([[2, 1]]))
    torch.testing.assert_close(out["seq_lens_padded"], torch.tensor([[2, 2]]))
    torch.testing.assert_close(loss_mask, torch.tensor([[1.0, 1.0, 1.0, 0.0]]))
    torch.testing.assert_close(
        cpmod._pad_1d([7], 3),
        torch.tensor([7, cpmod._SEQ_LENS_PADDING_VALUE, cpmod._SEQ_LENS_PADDING_VALUE]),
    )


def test_repad_packed_batch_returns_input_position_map():
    """The repad emits the input->rebuilt-row position map: real tokens point at
    their new columns, dropped input pad slots stay -1."""
    # Two docs in one row: [d0 d0 P | d1] with old per-doc pad after doc 0.
    batch = {
        "input_ids": torch.tensor([[10, 11, 99, 20]]),
        "labels": torch.tensor([[10, 11, -100, 20]]),
        "seq_lens": torch.tensor([[2, 1]]),
        "seq_lens_padded": torch.tensor([[3, 1]]),
    }
    out, _, input_positions = cpmod._repad_dsv4_packed_batch(
        batch,
        cp_size=1,
        pad_multiple=2,
        padding_token_id=99,
    )
    # Rebuilt row: doc0 padded to 2 -> [0:2]; doc1 padded to 2 -> [2:4].
    torch.testing.assert_close(input_positions, torch.tensor([[0, 1, -1, 2]]))
    # Sanity: the map really points at the rebuilt token positions.
    rebuilt = out["input_ids"][0]
    assert rebuilt[input_positions[0, 0]].item() == 10
    assert rebuilt[input_positions[0, 3]].item() == 20


def test_repad_packed_batch_validates_labels_and_metadata_extent():
    with pytest.raises(KeyError, match="labels"):
        cpmod._repad_dsv4_packed_batch(
            {"input_ids": torch.arange(2).view(1, 2), "seq_lens": torch.tensor([[2]])},
            cp_size=1,
            pad_multiple=2,
            padding_token_id=0,
        )

    with pytest.raises(ValueError, match="metadata exceeds token row length"):
        cpmod._repad_dsv4_packed_batch(
            {
                "input_ids": torch.arange(2).view(1, 2),
                "labels": torch.arange(2).view(1, 2),
                "seq_lens": torch.tensor([[3]]),
            },
            cp_size=1,
            pad_multiple=2,
            padding_token_id=0,
        )


def test_contiguous_shard_syncs_packed_length_for_hybridep(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    def _all_reduce_max(length, op):
        assert op == torch.distributed.ReduceOp.MAX
        length.fill_(16)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce_max)
    batch = {
        "input_ids": torch.arange(8).view(1, 8),
        "labels": torch.arange(8).view(1, 8),
        "qkv_format": "thd",
        "seq_lens": torch.tensor([[3, 2]]),
        "seq_lens_padded": torch.tensor([[3, 5]]),
    }

    _, out, _ = _shard(
        batch,
        cp_size=2,
        local_rank=0,
        pad_multiple=4,
        padding_token_id=99,
        sync_packed_length=True,
    )

    assert out["input_ids"].shape == (1, 8)
    torch.testing.assert_close(out["input_ids"], torch.tensor([[0, 1, 2, 99, 3, 4, 99, 99]]))
    torch.testing.assert_close(out["labels"], torch.tensor([[0, 1, 2, -100, 3, 4, -100, -100]]))

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    batch = {
        "input_ids": torch.arange(8).view(1, 8),
        "labels": torch.arange(8).view(1, 8),
        "qkv_format": "thd",
        "seq_lens": torch.tensor([[3, 2]]),
        "seq_lens_padded": torch.tensor([[3, 5]]),
    }
    _, out, _ = _shard(
        batch,
        cp_size=2,
        local_rank=1,
        pad_multiple=4,
        padding_token_id=99,
        sync_packed_length=True,
    )

    torch.testing.assert_close(out["input_ids"], torch.full((1, 8), 99))
    torch.testing.assert_close(out["labels"], torch.full((1, 8), -100))
    torch.testing.assert_close(out["padding_mask"], torch.ones((1, 8), dtype=torch.bool))
    torch.testing.assert_close(out["packed_seq_ids"], torch.zeros((1, 8), dtype=torch.long))


def test_contiguous_shard_requires_exactly_one_primary_key():
    # both input_ids and inputs_embeds -> assertion
    with pytest.raises(AssertionError):
        _shard(
            {
                "input_ids": torch.arange(8).view(1, 8),
                "inputs_embeds": torch.randn(1, 8, 2),
                "labels": torch.arange(8).view(1, 8),
            },
            cp_size=2,
            local_rank=0,
        )
    # neither -> assertion
    with pytest.raises(AssertionError):
        _shard({"labels": torch.arange(8).view(1, 8)}, cp_size=2, local_rank=0)


def test_contiguous_shard_requires_labels():
    with pytest.raises(KeyError, match="labels"):
        _shard({"input_ids": torch.arange(8).view(1, 8)}, cp_size=2, local_rank=0)


# --------------------------------------------------------------------------- #
# DeepseekV4ForCausalLM CP-prep hook                                           #
# --------------------------------------------------------------------------- #
def test_prepare_model_inputs_for_cp_returns_sharder():
    # The method only reads self.config, so a lightweight stand-in suffices.
    cfg = SimpleNamespace(compress_ratios=[0, 4, 128])
    fake_self = SimpleNamespace(config=cfg, backend=SimpleNamespace(dispatcher="hybridep"))
    prepared = DeepseekV4ForCausalLM.prepare_model_inputs_for_cp(fake_self, {"input_ids": torch.arange(8).view(1, 8)})

    sharder = prepared["cp_sharder"]
    from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices

    assert sharder.local_token_global_indices is contiguous_local_indices
    fn = sharder.shard_batch
    # the partial binds the config-derived per-rank multiple (lcm(8,128) == 128)
    assert fn.keywords["pad_multiple"] == 128
    assert fn.keywords["sync_packed_length"] is True
    assert fn.func is make_dsv4_contiguous_shard_cp_batch_and_ctx

    # the bound fn shards a batch end-to-end with a real (fake-mesh) divisor.
    batch = {"input_ids": torch.arange(256).view(1, 256), "labels": torch.arange(256).view(1, 256)}
    _, out, _ = fn(_FakeMesh(2, 0), None, batch)
    assert out["input_ids"].shape == (1, 128)


def test_prepare_model_inputs_for_cp_binds_shard_multiple():
    # The sharder-only CP hook binds the config-derived per-rank shard multiple;
    # a fake self exercises it without a model build (it touches no weights).
    cfg = SimpleNamespace(compress_ratios=[4])
    fake_self = SimpleNamespace(config=cfg, backend=SimpleNamespace(dispatcher="deepep"))
    out = DeepseekV4ForCausalLM.prepare_model_inputs_for_cp(fake_self, {"input_ids": torch.arange(8).view(1, 8)})
    assert out["cp_sharder"].shard_batch.keywords["pad_multiple"] == 8
    assert out["cp_sharder"].shard_batch.keywords["sync_packed_length"] is False


def test_setup_cp_attention_stores_group():
    from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4Attention

    fake_attn = SimpleNamespace()
    DeepseekV4Attention.setup_cp_attention(fake_attn, _FakeMesh(2, 0, group="grp"))
    assert fake_attn._cp_group == "grp"


def test_module_exposes_pad_helper_noops():
    from nemo_automodel.components.distributed.context_parallel import sharder as cp_sharder

    # pad_len <= 0 is a no-op (returns the same tensor object) for both pad helpers.
    t = torch.arange(6).view(1, 6)
    assert cp_sharder._pad_tensor_seq_dim_(t, 1, 0, 0) is t
    pos = torch.arange(6).view(1, 6)
    assert cp_sharder._pad_position_ids_seq_dim_(pos, 1, 0) is pos
