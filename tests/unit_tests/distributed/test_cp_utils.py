# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for :pyfile:`nemo_automodel/components/distributed/context_parallel/utils.py`.

The real implementation relies heavily on ``torch.distributed`` and GPU-specific
behavior.  These unit-tests therefore *mock* the heavyweight distributed pieces
so they can run quickly on CPU-only CI systems while still verifying the public
contract of the helper utilities.
"""

from __future__ import annotations

import contextlib
from functools import partial
from unittest import mock

import pytest
import torch

# Import module under test
from nemo_automodel.components.distributed.context_parallel import utils as _cu
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_batch_contiguous,
)
from nemo_automodel.components.models.gemma4_moe import cp_batch as _cm


# ContextParallelSharder used by the model-owned dispatch tests below (passed as an explicit
# _make_cp_batch_and_ctx parameter; the batch itself stays pure tensors). Exercises the public
# contiguous shard (the production entry DSV4/Gemma4 wrap) on the model-provided per-token keys.
def _contiguous_sharder():
    return ContextParallelSharder(
        shard_batch=partial(
            shard_batch_contiguous,
            extra_seq_keys={"per_layer_inputs": 1, "_packed_seq_ids": 1, "mm_token_type_ids": 1},
            extra_pad_values={"per_layer_inputs": 0, "_packed_seq_ids": 0, "mm_token_type_ids": 0},
        ),
        local_token_global_indices=contiguous_local_indices,
    )


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


class _DummySubMesh:
    """A minimal stub emulating ``torch.distributed.device_mesh.DeviceMesh`` slices."""

    def __init__(self, size: int, local_rank: int = 0):
        self._size = size
        self._local_rank = local_rank

    def size(self) -> int:  # noqa: D401  (simple method)
        return self._size

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_group(self):  # noqa: D401  (simple method)
        """Return None to simulate no distributed process group."""
        return None


class _DummyDeviceMesh(dict):
    """Dictionary-like container expected by :pyfunc:`_make_cp_batch_and_ctx`."""

    def __init__(self, cp_size: int, tp_size: int, cp_rank: int = 0):
        super().__init__()
        self["cp"] = _DummySubMesh(cp_size, cp_rank)
        self["tp"] = _DummySubMesh(tp_size)
        self.mesh_dim_names = ["cp", "tp"]


def _construct_strategy_sharder(strategy, device_mesh):
    """Construct a mesh-configured sharder from a resolved strategy."""
    return ContextParallelSharder(
        device_mesh=device_mesh,
        shard_batch=strategy.shard_batch,
        local_token_global_indices=strategy.local_token_global_indices,
        shard_layout=strategy.shard_layout,
    )


# ============================================================================
# Tests for attach_context_parallel_hooks
# ============================================================================


class _FakeSelfAttn(torch.nn.Module):
    """Minimal module that records the kwargs it receives."""

    def forward(self, hidden_states, **kwargs):
        self.last_kwargs = kwargs
        return hidden_states


class _FakeTransformerBlock(torch.nn.Module):
    """A toy model with a ``self_attn`` sub-module to test hook attachment."""

    def __init__(self):
        super().__init__()
        self.self_attn = _FakeSelfAttn()


class _FakeModel(torch.nn.Module):
    """Two-layer model with ``self_attn`` sub-modules."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeTransformerBlock(), _FakeTransformerBlock()])


def test_attach_context_parallel_hooks_registers_on_self_attn():
    """Hooks should be registered on every module whose name ends with 'self_attn'."""
    model = _FakeModel()

    # Count hooks before
    hooks_before = {
        name: len(mod._forward_pre_hooks) for name, mod in model.named_modules() if name.endswith("self_attn")
    }

    _cu.attach_context_parallel_hooks(model)

    for name, mod in model.named_modules():
        if name.endswith("self_attn"):
            assert len(mod._forward_pre_hooks) == hooks_before[name] + 1


def test_attach_context_parallel_hooks_strips_attention_mask():
    """The hook should replace attention_mask with None and set is_causal=True."""
    model = _FakeModel()
    _cu.attach_context_parallel_hooks(model)

    dummy_input = torch.randn(1, 4, 8)
    attn_mask = torch.ones(1, 1, 4, 4)

    model.layers[0].self_attn(dummy_input, attention_mask=attn_mask)

    kwargs = model.layers[0].self_attn.last_kwargs
    assert kwargs["attention_mask"] is None, "attention_mask should be set to None by the hook"
    assert kwargs["is_causal"] is True, "is_causal should be set to True by the hook"


def test_attach_context_parallel_hooks_no_mask_passthrough():
    """When no attention_mask kwarg is passed, the hook should be a no-op."""
    model = _FakeModel()
    _cu.attach_context_parallel_hooks(model)

    dummy_input = torch.randn(1, 4, 8)
    model.layers[0].self_attn(dummy_input, some_other_kwarg=42)

    kwargs = model.layers[0].self_attn.last_kwargs
    assert "attention_mask" not in kwargs
    assert "is_causal" not in kwargs
    assert kwargs["some_other_kwarg"] == 42


def test_attach_context_parallel_hooks_skips_non_self_attn():
    """Modules not ending with 'self_attn' should have no hooks added."""
    model = _FakeModel()
    _cu.attach_context_parallel_hooks(model)

    # The top-level model and the layers list should not get hooks
    assert len(model._forward_pre_hooks) == 0
    assert len(model.layers._forward_pre_hooks) == 0
    for layer in model.layers:
        assert len(layer._forward_pre_hooks) == 0


def test_attach_te_context_parallel_configures_full_and_sliding_attention(monkeypatch):
    """TE setup must configure TP independently and choose the CP communication mode."""

    class _FakeDotProductAttention:
        def __init__(self):
            self.calls = []
            self.tp_calls = []
            self.num_attention_heads = 8
            self.num_gqa_groups = 4
            self.tp_size = 1
            self.num_gqa_groups_per_partition = 4

        def set_context_parallel_group(self, group, ranks, stream, *, cp_comm_type):
            self.calls.append((group, ranks, stream, cp_comm_type))

        def set_tensor_parallel_group(self, group):
            self.tp_calls.append(group)

    class _Attention(torch.nn.Module):
        def __init__(self, sliding_window):
            super().__init__()
            self.attn_module = _FakeDotProductAttention()
            self.sliding_window = sliding_window

    class _Block(torch.nn.Module):
        def __init__(self, sliding_window):
            super().__init__()
            self.self_attn = _Attention(sliding_window)

    model = torch.nn.ModuleList([_Block(None), _Block(128)])
    group = object()
    stream = object()
    cp_mesh = mock.MagicMock()
    cp_mesh.size.return_value = 2
    cp_mesh.get_group.return_value = group
    tp_group = object()
    tp_mesh = mock.MagicMock()
    tp_mesh.size.return_value = 2
    tp_mesh.get_group.return_value = tp_group

    monkeypatch.setattr(
        "nemo_automodel.shared.import_utils.safe_import_from",
        lambda *_args: (True, _FakeDotProductAttention),
    )
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _group: [0, 1])
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)

    configured = _cu.attach_te_context_parallel(model, cp_mesh, tp_mesh)

    assert configured == 2
    assert model[0].self_attn.attn_module.calls == [(group, [0, 1], stream, "p2p")]
    assert model[1].self_attn.attn_module.calls == [(group, [0, 1], stream, "all_gather")]
    for block in model:
        assert block.self_attn.attn_module.tp_calls == [tp_group]
        assert block.self_attn.attn_module.tp_size == 2
        assert block.self_attn.attn_module.num_gqa_groups_per_partition == 2

    tp_only_model = torch.nn.ModuleList([_Block(None)])
    configured = _cu.attach_te_context_parallel(tp_only_model, tp_mesh=tp_mesh)

    assert configured == 1
    assert tp_only_model[0].self_attn.attn_module.calls == []
    assert tp_only_model[0].self_attn.attn_module.tp_calls == [tp_group]
    assert tp_only_model[0].self_attn.attn_module.tp_size == 2
    assert tp_only_model[0].self_attn.attn_module.num_gqa_groups_per_partition == 2


# ============================================================================
# Tests for make_cp_batch_for_te
# ============================================================================


def test_make_cp_batch_for_te_basic(monkeypatch):
    """Test make_cp_batch_for_te with basic input."""
    cp_mesh = _DummySubMesh(size=2)

    # Create simple batch in BSHD format
    # 2 sequences: [1,2,3,4] and [5,6,7,8]
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
    position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    seq_lens = torch.tensor([[4], [4]])  # Both sequences have length 4
    seq_lens_padded = torch.tensor([[4], [4]])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
        "pixel_values": torch.randn(1, 3, 8, 12),
        "_global_vision_mask": input_ids == 7,
    }

    def mock_get_rank(group=None):
        return 0

    # Mock tex.thd_get_partitioned_indices to return all indices (simplified)
    def mock_thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
        # For simplicity, just return all indices
        return torch.arange(total_tokens)

    # Mock transformer_engine_torch module
    class MockTex:
        @staticmethod
        def thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
            return mock_thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank)

    # Mock at the module level where it's imported
    import sys

    sys.modules["transformer_engine_torch"] = MockTex

    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)

    result = _cu.make_cp_batch_for_te(
        cp_mesh=cp_mesh,
        batch=batch,
    )

    # Should return processed batch with correct keys
    assert "input_ids" in result
    assert "labels" in result
    assert "position_ids" in result
    assert "cu_seqlens" in result
    assert "max_seqlen" in result
    assert "qkv_format" in result
    assert "padding_mask" in result

    # Verify format
    assert result["qkv_format"] == "thd"

    # Verify cu_seqlens are properly formatted
    assert result["cu_seqlens"].dtype == torch.int32
    assert result["pixel_values"] is batch["pixel_values"]
    assert result["_global_vision_mask"] is batch["_global_vision_mask"]


def test_make_cp_batch_for_te_multi_chunk(monkeypatch):
    """The num_chunks > 1 path shards and stacks every pipeline chunk.

    Covers the per-chunk shard call, which the single-chunk test does not reach.
    """
    cp_mesh = _DummySubMesh(size=2)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "labels": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
        "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        "seq_lens": torch.tensor([[4], [4]]),
        "seq_lens_padded": torch.tensor([[4], [4]]),
    }

    class MockTex:
        @staticmethod
        def thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
            return torch.arange(total_tokens)

    import sys

    sys.modules["transformer_engine_torch"] = MockTex
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    result = _cu.make_cp_batch_for_te(cp_mesh=cp_mesh, batch=batch, num_chunks=2)

    assert result["qkv_format"] == "thd"
    assert result["input_ids"].shape[0] == 2
    assert result["padding_mask"].shape == result["input_ids"].shape


def test_shard_thd_chunk_skips_missing_padding_mask(monkeypatch):
    """Test that _shard_thd_chunk_for_te handles missing padding_mask gracefully."""
    cp_mesh = _DummySubMesh(size=2)

    def mock_get_rank(group=None):
        return 0

    class MockTex:
        @staticmethod
        def thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
            return torch.arange(total_tokens)

    import sys

    sys.modules["transformer_engine_torch"] = MockTex

    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)

    # Batch without padding_mask — should not raise KeyError
    batch = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([10, 20, 30, 40]),
        "position_ids": torch.tensor([0, 1, 2, 3]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 4], dtype=torch.int32),
    }

    result, local_indices = _cu._shard_thd_chunk_for_te(batch, cp_mesh, "thd", -1000, 0)

    assert "input_ids" in result
    assert "attention_mask" not in result
    assert "cu_seqlens_padded" not in result
    # the partition IS the local-token global index map (mock returns arange)
    assert torch.equal(local_indices, torch.arange(4))


def test_make_cp_batch_for_te_unsupported_format():
    """Test that unsupported qvk_format raises ValueError."""
    cp_mesh = _DummySubMesh(size=2)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[10, 20, 30, 40]])
    seq_lens = torch.tensor([[4]])
    seq_lens_padded = torch.tensor([[4]])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }

    with pytest.raises(ValueError, match="Currently only 'thd' format is supported"):
        _cu.make_cp_batch_for_te(
            cp_mesh=cp_mesh,
            batch=batch,
            qkv_format="bshd",
        )


def test_make_cp_batch_for_te_requires_seqlens():
    """Test that make_cp_batch_for_te raises error when seq_lens and seq_lens_padded are not provided."""
    cp_mesh = _DummySubMesh(size=1)

    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[10, 20, 30]])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": torch.tensor([[0, 1, 2]]),
    }

    with pytest.raises(KeyError, match="seq_lens"):
        _cu.make_cp_batch_for_te(
            cp_mesh=cp_mesh,
            batch=batch,
        )


def test_synthesize_single_document_seq_ids_from_padding_mask():
    # A single sequence has no collate-emitted `_packed_seq_ids`; the manual CP
    # path synthesizes the trivial one-document map (1 = real token, 0 = pad)
    # from `padding_mask` so the all-gather attention mask builder has boundaries.
    batch = {
        "input_ids": torch.zeros(1, 6, dtype=torch.long),
        "padding_mask": torch.tensor([[False, False, False, False, True, True]]),
    }
    _cm._synthesize_single_document_seq_ids(batch, 6)
    assert torch.equal(batch["_packed_seq_ids"], torch.tensor([[1, 1, 1, 1, 0, 0]]))


def test_synthesize_single_document_seq_ids_all_ones_without_padding_mask():
    # No padding info -> single document spanning the whole sequence.
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    _cm._synthesize_single_document_seq_ids(batch, 4)
    assert torch.equal(batch["_packed_seq_ids"], torch.tensor([[1, 1, 1, 1]]))


def test_synthesize_single_document_seq_ids_noop_when_present():
    # Genuinely packed input already carries `_packed_seq_ids`; leave it untouched.
    existing = torch.tensor([[1, 1, 2, 2, 0, 0]])
    batch = {"input_ids": torch.zeros(1, 6, dtype=torch.long), "_packed_seq_ids": existing}
    _cm._synthesize_single_document_seq_ids(batch, 6)
    assert torch.equal(batch["_packed_seq_ids"], existing)


def test_sharder_constructor_derives_te_from_model_and_thd_from_batch(monkeypatch):
    """A TE model and THD batch resolve a sharder without recipe-owned flags."""
    seen = {}
    local_indices = torch.tensor([1, 0])

    def fake_make_cp_batch_for_te(
        cp_mesh, batch, *, padding_token_id, qkv_format, num_chunks, seq_lens_padding_value, return_local_indices=False
    ):
        seen.update(
            cp_mesh=cp_mesh, pad=padding_token_id, fmt=qkv_format, chunks=num_chunks, sent=seq_lens_padding_value
        )
        return ({"thd": True}, local_indices) if return_local_indices else {"thd": True}

    monkeypatch.setattr(_cu, "make_cp_batch_for_te", fake_make_cp_batch_for_te)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    model = type("_Model", (), {"backend": type("_Backend", (), {"attn": "te"})()})()
    batch = {"input_ids": torch.tensor([[1, 2]]), "qkv_format": "thd"}
    sharder = ContextParallelSharder(
        model,
        device_mesh,
        batch,
        padding_token_id=7,
        num_chunks=3,
    )
    assert not seen
    ctx, batch = sharder.shard(batch)
    assert ctx is contextlib.nullcontext
    assert batch["thd"] is True
    assert torch.equal(batch["_thd_local_indices"], local_indices)
    assert seen["cp_mesh"] is device_mesh["cp"]
    assert (seen["pad"], seen["fmt"], seen["chunks"], seen["sent"]) == (7, "thd", 3, -1000)


def test_sharder_constructor_does_not_infer_te_from_batch_alone(monkeypatch):
    """A THD-origin batch does not force TE preparation on a non-TE model."""
    monkeypatch.setattr(
        _cu,
        "make_cp_batch_for_te",
        lambda *args, **kwargs: pytest.fail("TE batch preparation should not run"),
    )
    model = type("_Model", (), {"backend": type("_Backend", (), {"attn": "sdpa"})()})()
    batch = {"input_ids": torch.tensor([[1, 2]]), "qkv_format": "thd"}
    sharder = ContextParallelSharder(model, _DummyDeviceMesh(cp_size=1, tp_size=1), batch)
    ctx, out = sharder.shard(batch)
    assert ctx is contextlib.nullcontext
    assert out is batch


def test_sharder_constructor_merges_model_hook_batch_updates(monkeypatch):
    """Model-owned hooks may return batch metadata in addition to the sharder."""

    cp_context_kwargs = {}

    def fake_create_context_parallel_ctx(**kwargs):
        cp_context_kwargs.update(kwargs)
        return "cp_ctx"

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", fake_create_context_parallel_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: contextlib.nullcontext)

    position_ids = torch.arange(3 * 1 * 4).view(3, 1, 4)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    class _Model:
        def prepare_model_inputs_for_cp(self, batch, *, num_chunks):
            assert num_chunks == 3
            assert batch["mm_token_type_ids"].shape == (1, 4)
            return {
                "cp_sharder": ContextParallelSharder(
                    shard_batch=shard_batch_aux_only,
                    local_token_global_indices=round_robin_local_indices,
                ),
                "position_ids": position_ids,
                "mm_token_type_ids": None,
                "image_grid_thw": image_grid_thw,
                "image_grid_hws": None,
            }

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[10, 20, 30, 40]]),
        "mm_token_type_ids": torch.ones(1, 4, dtype=torch.long),
        "image_grid_hws": torch.tensor([[2, 2]]),
    }

    sharder = ContextParallelSharder(_Model(), _DummyDeviceMesh(cp_size=2, tp_size=1), batch, num_chunks=3)

    assert "cp_sharder" not in batch
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 3, 4]]))
    assert torch.equal(batch["labels"], torch.tensor([[10, 20, 30, 40]]))
    assert batch["position_ids"] is position_ids
    assert batch["mm_token_type_ids"] is None
    assert batch["image_grid_thw"] is image_grid_thw
    assert batch["image_grid_hws"] is None
    assert cp_context_kwargs == {}
    ctx, out = sharder.shard(batch)
    assert ctx is contextlib.nullcontext
    assert out is batch
    assert cp_context_kwargs["cp_buffers"][1] is position_ids
    assert cp_context_kwargs["cp_seq_dims"] == [1, 2]
    assert sharder.shard_layout.original_seq_len == 4
    assert sharder.shard_layout.padded_seq_len == 4


class _FakeMagiState:
    """Fake MagiState whose dispatch returns a fixed local index map."""

    enabled = True
    domain = "llm"
    cp_size = 2

    def __init__(self, local_indices):
        self._local_indices = local_indices

    def make_cp_batch(self, cp_mesh, batch, *, return_local_indices=False, **kwargs):
        prepped = {"prepared": True}
        return (prepped, self._local_indices) if return_local_indices else prepped


def test_make_cp_batch_for_te_identity_indices_without_cp():
    """At cp<=1 the THD stream is unsharded, so the index map is the identity."""
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[10, 20, 30, 40]]),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
        "seq_lens": torch.tensor([[4]]),
        "seq_lens_padded": torch.tensor([[4]]),
    }
    out, local_indices = _cu.make_cp_batch_for_te(None, batch, return_local_indices=True)
    assert torch.equal(local_indices, torch.arange(out["input_ids"].shape[-1]))


