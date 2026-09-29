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

"""Targeted unit tests for context-parallel helper paths in context_parallel.utils.

These exercise the CP attention SDPA-swap hooks, the linear-attn position hook,
the classic DTensor _cp_sdpa path (DTensor mocked), padding/prepare helpers, and
the manual contiguous-shard batch builder branches that the broader suite leaves
uncovered. No GPU or real process group is required.
"""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.distributed.context_parallel import sharder as cm
from nemo_automodel.components.distributed.context_parallel import utils as cu


@pytest.fixture(autouse=True)
def _force_no_dist(monkeypatch):
    """Pin rank resolution to the dummy mesh's local rank.

    These tests drive CP helpers with fake meshes whose ``get_group`` returns a
    sentinel, not a real ProcessGroup. If another test in the same pytest worker
    left ``torch.distributed`` initialized (e.g. a TP correctness test), rank
    resolution would go through ``dist.get_rank`` instead of
    ``mesh.get_local_rank`` and shard the wrong slice.
    """
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


class _FakeMesh:
    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_group(self):
        return object()

    def get_local_rank(self):
        return self._rank


class _Attn(torch.nn.Module):
    def forward(self, query, key, value, attention_mask=None, **kwargs):
        return F.scaled_dot_product_attention(query, key, value)


class _Wrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


def _qkv(seq=4, d=8, heads=2):
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, d)
    k = torch.randn(1, heads, seq, d)
    v = torch.randn(1, heads, seq, d)
    return q, k, v


# ---------------------------------------------------------------------------
# attach_cp_sdpa_hooks: classic DTensor SDPA path. (Model-owned CP attention,
# e.g. Gemma4's p2p ring, is installed by the model via setup_cp_attention and is
# covered under tests/unit_tests/models/gemma4.)
# ---------------------------------------------------------------------------
def test_cp_attention_hooks_dtensor_sdpa_path():
    attn = _Attn()  # no run_cp_manual_attention -> classic DTensor _cp_sdpa
    model = _Wrapper(attn)
    q, k, v = _qkv()

    class _FakeDTensor:
        @staticmethod
        def from_local(t, device_mesh=None, placements=None):
            return t  # identity: treat local tensor as already-distributed

    # attach_cp_sdpa_hooks imports DTensor/Shard at attach time, so patch
    # before attaching.
    with (
        mock.patch("torch.distributed.tensor.DTensor", _FakeDTensor),
        mock.patch("torch.distributed.tensor.Shard", lambda d: ("shard", d)),
    ):
        cu.attach_cp_sdpa_hooks(model, _FakeMesh(size=2))
        out = attn(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref)


def test_cp_attention_hooks_restores_sdpa_on_exception():
    class _BoomAttn(torch.nn.Module):
        def forward(self, query, key, value, **kwargs):
            F.scaled_dot_product_attention  # trigger nothing; just raise after swap
            raise RuntimeError("boom")

    original = F.scaled_dot_product_attention
    attn = _BoomAttn()
    model = _Wrapper(attn)
    cu.attach_cp_sdpa_hooks(model, _FakeMesh(size=2))
    q, k, v = _qkv()
    with pytest.raises(RuntimeError, match="boom"):
        attn(q, k, v)
    # always_call post-hook restored the real SDPA
    assert F.scaled_dot_product_attention is original


# ---------------------------------------------------------------------------
# _pad_tensor_seq_dim_ / _pad_position_ids_seq_dim_ pad_len<=0 no-ops
# ---------------------------------------------------------------------------
def test_pad_tensor_noop_when_pad_len_zero():
    t = torch.randn(2, 4)
    assert cm._pad_tensor_seq_dim_(t, 1, 0) is t


def test_pad_position_ids_noop_when_pad_len_zero():
    p = torch.arange(4).unsqueeze(0)
    assert cm._pad_position_ids_seq_dim_(p, 1, 0) is p


def test_pad_position_ids_extends_monotonically():
    p = torch.tensor([[0, 1, 2, 3]])
    out = cm._pad_position_ids_seq_dim_(p, 1, 2)
    assert out.tolist() == [[0, 1, 2, 3, 4, 5]]


# ---------------------------------------------------------------------------
# shard_batch_contiguous branches (model-owned CP prep + pad + slice)
# ---------------------------------------------------------------------------
def test_shard_batch_contiguous_builds_padding_mask_from_4d_mask():
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    batch = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "labels": torch.zeros(1, 4, dtype=torch.long),
        "attention_mask": mask,
    }
    cm.shard_batch_contiguous(_FakeMesh(size=2, rank=0), None, batch)
    assert "attention_mask" not in batch
    assert "padding_mask" in batch and batch["padding_mask"].shape == (1, 2)


def test_shard_batch_contiguous_labels_from_loss_mask():
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    loss_mask = torch.arange(4).unsqueeze(0)
    _, out, _ = cm.shard_batch_contiguous(_FakeMesh(size=2, rank=1), None, batch, loss_mask=loss_mask)
    # promoted to labels (rank 1 slice) and NOT also carried as loss_mask
    assert torch.equal(out["labels"], torch.tensor([[2, 3]]))
    assert "loss_mask" not in out


def test_shard_batch_contiguous_raises_without_labels():
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    with pytest.raises(KeyError, match="labels"):
        cm.shard_batch_contiguous(_FakeMesh(size=2, rank=0), None, batch)


def test_shard_batch_contiguous_pads_and_slices_all_keys():
    cp_size = 2
    seq = 6  # pad_len = (-6) % 4 = 2 -> padded to 8, local 4
    d = 8
    batch = {
        "inputs_embeds": torch.randn(1, seq, d),
        "labels": torch.zeros(1, seq, dtype=torch.long),
        "position_ids": torch.arange(seq).unsqueeze(0),
        "mm_token_type_ids": torch.zeros(1, seq, dtype=torch.long),
        "per_layer_inputs": torch.randn(1, seq, d),
        "padding_mask": torch.zeros(1, seq, dtype=torch.bool),
        "vision_extra": torch.zeros(1, seq, dtype=torch.long),
    }
    loss_mask = torch.ones(1, seq, dtype=torch.long)

    _ctx, out, layout = cm.shard_batch_contiguous(
        _FakeMesh(size=cp_size, rank=0),
        None,
        batch,
        loss_mask=loss_mask,
        # model-specific per-token keys arrive as explicit arguments
        extra_seq_keys={"vision_extra": 1, "mm_token_type_ids": 1, "per_layer_inputs": 1},
        extra_pad_values={"vision_extra": 0, "mm_token_type_ids": 0, "per_layer_inputs": 0},
    )
    # rank 0 keeps the first local_seq_len=4 positions
    assert out["inputs_embeds"].shape == (1, 4, d)
    assert out["labels"].shape == (1, 4)
    assert out["per_layer_inputs"].shape == (1, 4, d)
    assert out["padding_mask"].shape == (1, 4)
    assert out["vision_extra"].shape == (1, 4)
    assert out["loss_mask"].shape == (1, 4)
    assert (layout.original_seq_len, layout.padded_seq_len) == (6, 8)


def test_shard_batch_contiguous_uses_dist_rank_when_initialized():
    batch = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "labels": torch.zeros(1, 4, dtype=torch.long),
    }
    with (
        mock.patch("torch.distributed.is_available", return_value=True),
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_rank", return_value=0),
    ):
        _ctx, out, _ = cm.shard_batch_contiguous(_FakeMesh(size=2, rank=0), None, batch)
    assert out["input_ids"].shape == (1, 2)


def test_shard_batch_contiguous_raises_when_not_divisible(monkeypatch):
    # Force padding to be a no-op so the post-pad divisibility check trips.
    monkeypatch.setattr(cm, "_pad_tensor_seq_dim_", lambda t, *a, **k: t)
    monkeypatch.setattr(cm, "_pad_position_ids_seq_dim_", lambda p, *a, **k: p)
    batch = {
        "input_ids": torch.zeros(1, 6, dtype=torch.long),
        "labels": torch.zeros(1, 6, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="divisible by cp_size"):
        cm.shard_batch_contiguous(_FakeMesh(size=4, rank=0), None, batch)


# ---------------------------------------------------------------------------
# _make_cp_batch_and_ctx use_te routing
# ---------------------------------------------------------------------------
