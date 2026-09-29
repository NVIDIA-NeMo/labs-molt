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

"""Unit tests for CPAwareGatedDeltaNet (cp_linear_attn.py).

Tests cover:
  - _extract_local_seq_index: various tensor shapes and fallback behavior
  - _build_dual_chunk_local_positions: DualChunkSwap layout derivation
  - _undo_attention_load_balancing / _redo_attention_load_balancing: correctness
  - _AllGatherConcatFn: forward in a single-rank mock scenario
  - CPAwareGatedDeltaNet.forward: fast path delegation when CP is disabled
  - _conv1d_with_cp: boundary token exchange logic
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")
pytest.importorskip("fla")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
    CPAwareGatedDeltaNet,
    _AllGatherConcatFn,
)

# Over the default 5s budget on purpose: CUDA gated-delta kernels take about 72s to compile on a cold worker.
# Reduce cold compiler startup before lowering this further.
pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


@pytest.fixture
def text_config():
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        layer_types=["full_attention", "linear_attention"],
    )


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@pytest.fixture
def module(text_config, device):
    """Create a CPAwareGatedDeltaNet on device with no CP mesh."""
    m = CPAwareGatedDeltaNet(text_config, layer_idx=1)
    m = m.to(device)
    return m


# -- helpers for mocking dist.all_gather in a CP world_size=2 scenario -------


def _make_fake_all_gather(rank0_pos, rank1_pos, rank0_hidden, rank1_hidden, device):
    """Return a fake all_gather that fills gathered lists for a 2-rank CP setup."""

    def fake_all_gather(gathered, tensor, group=None):
        if tensor.ndim == 1:
            # position tensor (1-D)
            gathered[0].copy_(rank0_pos.to(device))
            gathered[1].copy_(rank1_pos.to(device))
        else:
            # hidden states (B, S, D)
            gathered[0].copy_(rank0_hidden.to(device) if tensor.shape == rank0_hidden.shape else tensor)
            gathered[1].copy_(
                rank1_hidden.to(device) if tensor.shape == rank1_hidden.shape else torch.randn_like(tensor)
            )

    return fake_all_gather


import contextlib


@contextlib.contextmanager
def _patch_dist_for_cp(rank=0, world_size=2):
    """Context manager that patches dist rank/world_size for CP testing."""
    with (
        patch(
            "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=world_size
        ),
        patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=rank),
    ):
        yield


# ============================================================================
# _extract_local_seq_index
# ============================================================================


class TestExtractLocalSeqIndex:
    def test_1d_seq_index(self, module, device):
        seq_index = torch.tensor([3, 1, 0, 2], device=device)
        result = module._extract_local_seq_index(seq_index, seq_len=4)
        assert result is not None
        assert torch.equal(result, seq_index.long())

    def test_2d_seq_index_takes_first_row(self, module, device):
        seq_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
        result = module._extract_local_seq_index(seq_index, seq_len=4)
        assert result is not None
        assert torch.equal(result, seq_index[0].long())

    def test_casts_to_long(self, module, device):
        seq_index = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
        result = module._extract_local_seq_index(seq_index, seq_len=4)
        assert result.dtype == torch.long

    @pytest.mark.parametrize(
        "seq_index, seq_len",
        [
            (None, 4),  # None -> None (falls back to DualChunkSwap derivation)
            (torch.tensor([0, 1, 2]), 4),  # length mismatch
            (torch.arange(4).reshape(1, 1, 4), 4),  # 3D tensor skipped
        ],
        ids=["none", "length_mismatch", "3d_skipped"],
    )
    def test_returns_none_for_invalid_inputs(self, module, device, seq_index, seq_len):
        if seq_index is not None:
            seq_index = seq_index.to(device)
        result = module._extract_local_seq_index(seq_index, seq_len=seq_len)
        assert result is None


# ============================================================================
# _build_dual_chunk_local_positions
# ============================================================================


class TestBuildDualChunkLocalPositions:
    def test_rank0_of_2_takes_first_and_last_chunks(self, module, device):
        # cp_size=2 -> 4 chunks indexed 0..3 across ranks. rank 0 owns chunk 0
        # (first) and chunk 2*2-1-0=3 (last); seq_len=4 -> chunk_len=2.
        result = module._build_dual_chunk_local_positions(seq_len=4, cp_size=2, cp_rank=0, device=device)
        assert result.tolist() == [0, 1, 6, 7]

    def test_rank1_of_2_takes_inner_chunks(self, module, device):
        # rank 1 owns chunk 1 and chunk 2*2-1-1=2.
        result = module._build_dual_chunk_local_positions(seq_len=4, cp_size=2, cp_rank=1, device=device)
        assert result.tolist() == [2, 3, 4, 5]

    def test_result_dtype_and_device(self, module, device):
        result = module._build_dual_chunk_local_positions(seq_len=8, cp_size=4, cp_rank=2, device=device)
        assert result.dtype == torch.long
        assert result.device.type == device.type
        assert result.shape == (8,)

    def test_raises_on_odd_seq_len(self, module, device):
        with pytest.raises(RuntimeError, match="even local sequence length"):
            module._build_dual_chunk_local_positions(seq_len=5, cp_size=2, cp_rank=0, device=device)


# ============================================================================
# _undo_attention_load_balancing
# ============================================================================


class TestUndoAttentionLoadBalancing:
    """Test load-balancing undo using mocked dist calls (simulating CP world_size=2)."""

    def test_reorders_to_dense(self, module, device):
        """Tokens in load-balanced order should be sorted to dense 0..S-1 order."""
        B, S_local, D = 1, 4, module.hidden_size
        hidden = torch.randn(B, S_local, D, device=device)
        positions = torch.tensor([0, 3, 4, 7], device=device, dtype=torch.long)

        rank1_positions = torch.tensor([1, 2, 5, 6], dtype=torch.long)
        rank1_hidden = torch.randn(B, S_local, D)

        fake_ag = _make_fake_all_gather(positions, rank1_positions, hidden, rank1_hidden, device)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_ag),
            _patch_dist_for_cp(rank=0, world_size=2),
        ):
            result_hidden, sorted_pos = module._undo_attention_load_balancing(hidden, positions, MagicMock())

        # sorted_pos should be 0..7
        assert torch.equal(sorted_pos, torch.arange(8, device=device, dtype=torch.long))
        # result_hidden is rank 0's chunk of the dense order (positions 0..3)
        assert result_hidden.shape == (B, S_local, D)

    def test_raises_on_non_dense_positions(self, module, device):
        """Should raise if gathered positions don't form a dense 0..S-1 sequence."""
        B, S_local, D = 1, 4, module.hidden_size
        hidden = torch.randn(B, S_local, D, device=device)
        positions = torch.tensor([0, 2, 4, 8], device=device, dtype=torch.long)

        rank1_positions = torch.tensor([1, 3, 5, 9], dtype=torch.long)  # gap at 6,7
        rank1_hidden = torch.randn(B, S_local, D)

        fake_ag = _make_fake_all_gather(positions, rank1_positions, hidden, rank1_hidden, device)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_ag),
            _patch_dist_for_cp(rank=0, world_size=2),
        ):
            with pytest.raises(RuntimeError, match="dense global token positions"):
                module._undo_attention_load_balancing(hidden, positions, MagicMock())


# ============================================================================
# _redo_attention_load_balancing
# ============================================================================


class TestRedoAttentionLoadBalancing:
    """Test that _redo restores the original load-balanced CP layout."""

    def test_restores_original_layout(self, module, device):
        """Output gathered in dense order should be scattered back to load-balanced order."""
        B, S_local, D = 1, 4, module.hidden_size

        # Dense-order output from the attention computation
        output = (
            torch.arange(S_local, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(-1).expand(B, S_local, D)
        )

        # Rank 0 originally held positions [0, 3, 4, 7]
        original_positions = torch.tensor([0, 3, 4, 7], device=device, dtype=torch.long)
        sorted_positions = torch.arange(8, device=device, dtype=torch.long)

        rank1_output = (
            torch.arange(S_local, S_local * 2, device=device, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(B, S_local, D)
        )

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)
            gathered[1].copy_(rank1_output)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
            _patch_dist_for_cp(rank=0, world_size=2),
        ):
            result = module._redo_attention_load_balancing(output, original_positions, sorted_positions, MagicMock())

        # Result should have the same shape as input
        assert result.shape == (B, S_local, D)
        # The tokens at positions [0,3,4,7] should be selected from the full dense output
        expected_indices = original_positions
        for i, pos in enumerate(expected_indices):
            assert result[0, i, 0].item() == pos.item()


# ============================================================================
# forward fast path (no CP)
# ============================================================================


class TestForwardFastPath:
    def test_no_cp_mesh_delegates_to_forward_no_cp(self, module, device):
        """When _cp_mesh is None, forward should delegate to _forward_no_cp."""
        assert module._cp_mesh is None
        B, S, D = 1, 8, module.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        with patch.object(module, "_forward_no_cp", return_value=torch.randn(B, S, D, device=device)) as mock_no_cp_fwd:
            module.forward(hidden)
            mock_no_cp_fwd.assert_called_once()

    def test_cp_mesh_size_1_delegates_to_forward_no_cp(self, module, device):
        """When _cp_mesh.size() == 1, forward should delegate to _forward_no_cp."""
        mesh = MagicMock()
        mesh.size.return_value = 1
        module._cp_mesh = mesh

        B, S, D = 1, 8, module.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        with patch.object(module, "_forward_no_cp", return_value=torch.randn(B, S, D, device=device)) as mock_no_cp_fwd:
            module.forward(hidden)
            mock_no_cp_fwd.assert_called_once()

    def test_no_cp_does_not_forward_cache_position(self, module, device):
        """cache_position should not be forwarded to _forward_no_cp (removed in transformers>=5.5)."""
        assert module._cp_mesh is None
        B, S, D = 1, 8, module.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        with patch.object(module, "_forward_no_cp", return_value=torch.randn(B, S, D, device=device)) as mock_no_cp_fwd:
            module.forward(hidden, cache_position=torch.arange(S, device=device))
            mock_no_cp_fwd.assert_called_once()
            _, kwargs = mock_no_cp_fwd.call_args
            assert "cache_position" not in kwargs

    def test_cp_mesh_gt_1_calls_forward_with_cp(self, module, device):
        """When _cp_mesh.size() > 1, forward should call _forward_with_cp."""
        mesh = MagicMock()
        mesh.size.return_value = 2
        module._cp_mesh = mesh

        B, S, D = 1, 8, module.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        with patch.object(module, "_forward_with_cp", return_value=torch.randn(B, S, D, device=device)) as mock_cp_fwd:
            module.forward(hidden, position_ids=torch.arange(S, device=device).unsqueeze(0))
            mock_cp_fwd.assert_called_once()


class TestForwardNoCpV55CacheAPI:
    """_forward_no_cp must use the transformers v5.5 per-layer cache API.

    v5.5 renamed ``has_previous_state`` to a method taking ``layer_idx``, moved
    states under ``cache.layers[layer_idx]``, and exposes ``update_conv_state`` /
    ``update_recurrent_state`` methods instead of direct dict assignment. A plain
    ``DynamicCache`` (no pre-existing state, as in training) has no top-level
    ``conv_states`` attribute — the pre-v5.5 pattern raised ``AttributeError``.
    """

    def test_training_cache_no_previous_state_runs(self, module, text_config, device):
        """Training-style forward with a fresh DynamicCache (no previous state) must not raise."""
        from transformers import DynamicCache

        B, S, D = 1, 8, module.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        out = module._forward_no_cp(hidden, cache_params=DynamicCache(config=text_config))
        assert out.shape == (B, S, D)

    def test_no_cache_path_still_works(self, module, device):
        """When cache_params is None, _forward_no_cp runs the pure compute path."""
        B, S, D = 1, 8, module.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        out = module._forward_no_cp(hidden, cache_params=None)
        assert out.shape == (B, S, D)

    def test_updates_conv_state_via_method(self, module, text_config, device):
        """Prefill writes the conv state via ``update_conv_state(state, layer_idx)``."""
        from transformers import DynamicCache

        B, S, D = 1, 8, module.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        cache = DynamicCache(config=text_config)
        with (
            patch.object(cache, "update_conv_state", wraps=cache.update_conv_state) as mock_update_conv,
            patch.object(cache, "update_recurrent_state", wraps=cache.update_recurrent_state) as mock_update_rec,
        ):
            module._forward_no_cp(hidden, cache_params=cache)
        mock_update_conv.assert_called_once()
        # Written at the layer_idx owned by the module.
        args, _ = mock_update_conv.call_args
        assert args[1] == module.layer_idx
        mock_update_rec.assert_called_once()

    def test_has_previous_state_called_as_method_with_layer_idx(self, module, text_config, device):
        """v5.5 ``has_previous_state`` is a method that takes ``layer_idx``."""
        from transformers import DynamicCache

        B, S, D = 1, 8, module.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        cache = DynamicCache(config=text_config)
        with patch.object(cache, "has_previous_state", wraps=cache.has_previous_state) as mock_hps:
            module._forward_no_cp(hidden, cache_params=cache)
        mock_hps.assert_called()
        # At least one call must pass the module's layer_idx.
        layer_idx_seen = any(
            (call.args and call.args[0] == module.layer_idx) or call.kwargs.get("layer_idx") == module.layer_idx
            for call in mock_hps.call_args_list
        )
        assert layer_idx_seen


# ============================================================================
# _conv1d_with_cp
# ============================================================================


class TestConv1dWithCP:
    def test_output_shape_matches_input(self, module, device):
        """Conv1d output should preserve [B, D, S_local] shape."""
        B = 1
        conv_dim = module.conv1d.weight.shape[0]
        S_local = 8
        mixed_qkv = torch.randn(B, conv_dim, S_local, device=device)

        def fake_causal_conv1d(x, weight, bias, activation, cp_context):
            assert x.shape == (1, S_local, conv_dim)
            return x, None

        with patch("fla.modules.convolution.causal_conv1d", side_effect=fake_causal_conv1d):
            result = module._conv1d_with_cp(mixed_qkv, MagicMock())

        assert result.shape == (B, conv_dim, S_local)
        assert torch.equal(result, mixed_qkv)

    def test_invokes_fla_cp_conv_once_per_batch_item(self, module, device):
        """FLA CP conv only supports batch=1, so the wrapper should loop over batch items."""
        B = 3
        conv_dim = module.conv1d.weight.shape[0]
        S_local = 8
        mixed_qkv = torch.randn(B, conv_dim, S_local, device=device)

        def fake_causal_conv1d(x, weight, bias, activation, cp_context):
            return x + 1, None

        with patch("fla.modules.convolution.causal_conv1d", side_effect=fake_causal_conv1d) as mock_conv:
            result = module._conv1d_with_cp(mixed_qkv, MagicMock())

        assert mock_conv.call_count == B
        assert result.shape == (B, conv_dim, S_local)
        assert torch.equal(result, mixed_qkv + 1)

    def test_passes_cp_context_to_fla_conv(self, module, device):
        """The wrapper should forward the built cp_context into FLA's conv path."""
        conv_dim = module.conv1d.weight.shape[0]
        S_local = 8
        mixed_qkv = torch.randn(1, conv_dim, S_local, device=device)
        cp_context = MagicMock()

        def fake_causal_conv1d(x, weight, bias, activation, cp_context):
            assert cp_context is not None
            return x, None

        with patch("fla.modules.convolution.causal_conv1d", side_effect=fake_causal_conv1d):
            result = module._conv1d_with_cp(mixed_qkv, cp_context)

        assert result.shape == mixed_qkv.shape


# ============================================================================
# _AllGatherConcatFn
# ============================================================================


class TestAllGatherConcatFn:
    def test_forward_concatenates_gathered_shards(self, device):
        """Forward should gather and concatenate along the specified dim."""
        local = torch.tensor([[1.0, 2.0]], device=device)
        group = MagicMock()

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)
            gathered[1].copy_(tensor * 2)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=2),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=0),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
        ):
            result = _AllGatherConcatFn.apply(local, group, 1)

        expected = torch.tensor([[1.0, 2.0, 2.0, 4.0]], device=device)
        assert torch.equal(result, expected)

    def test_forward_dim0(self, device):
        """Forward should work along dim=0."""
        local = torch.tensor([[1.0], [2.0]], device=device)
        group = MagicMock()

        def fake_all_gather(gathered, tensor, group=None):
            gathered[0].copy_(tensor)
            gathered[1].copy_(tensor + 10)

        with (
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_world_size", return_value=2),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.get_rank", return_value=0),
            patch("nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.dist.all_gather", fake_all_gather),
        ):
            result = _AllGatherConcatFn.apply(local, group, 0)

        expected = torch.tensor([[1.0], [2.0], [11.0], [12.0]], device=device)
        assert torch.equal(result, expected)
