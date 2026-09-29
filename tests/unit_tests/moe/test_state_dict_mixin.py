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

from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")

from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin, get_world_size_safe

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


def test_get_world_size_safe_uses_initialized_process_group():
    with (
        patch("nemo_automodel.components.moe.state_dict_mixin.torch.distributed.is_initialized", return_value=True),
        patch("nemo_automodel.components.moe.state_dict_mixin.torch.distributed.get_world_size", return_value=8),
    ):
        assert get_world_size_safe() == 8


def test_get_world_size_safe_uses_preinit_environment():
    with (
        patch("nemo_automodel.components.moe.state_dict_mixin.torch.distributed.is_initialized", return_value=False),
        patch.dict("os.environ", {"WORLD_SIZE": "8"}),
    ):
        assert get_world_size_safe() == 8


class MockMoEConfig:
    def __init__(self, n_routed_experts=8, moe_inter_dim=512, expert_activation="swiglu", expert_bias=False):
        self.n_routed_experts = n_routed_experts
        self.moe_inter_dim = moe_inter_dim
        self.expert_activation = expert_activation
        self.expert_bias = expert_bias


class MockConfig:
    def __init__(self):
        pass


class MockBackend:
    def __init__(self):
        self.experts = "gmm"
        self.dispatcher = "torch"


class MockMoEStateDictMixin(MoESplitExpertsStateDictMixin):
    _supports_low_memory_dcp_load = True

    def __init__(
        self,
        n_experts=8,
        inter_dim=512,
        dtype=torch.float32,
        uses_model_prefix=True,
        expert_activation="swiglu",
    ):
        self.moe_config = MockMoEConfig(n_experts, inter_dim, expert_activation)
        self.config = MockConfig()
        self.backend = MockBackend()
        self.dtype = dtype
        self._uses_model_prefix = uses_model_prefix
        self._last_expert_ids = []


def _run_ep_free_dtensor_split(rank: int, world_size: int, init_file: str) -> None:
    """Verify non-EP DTensors split without collecting full expert weights."""
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        mesh = DeviceMesh("cpu", torch.arange(world_size), mesh_dim_names=("dp_shard_cp",))
        full_weight = torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4)
        mixin = MockMoEStateDictMixin(n_experts=4, inter_dim=2)

        expert_sharded_weight = distribute_tensor(full_weight, mesh, [Shard(0)])
        splits = mixin._split_experts_weights(expert_sharded_weight, 4)

        local_expert_ids = [rank * 2, rank * 2 + 1]
        assert mixin._last_expert_ids == local_expert_ids
        assert len(splits) == 2
        for expert_id, expert_weight in zip(local_expert_ids, splits):
            assert not hasattr(expert_weight, "placements")
            torch.testing.assert_close(expert_weight, full_weight[expert_id])

        inner_sharded_weight = distribute_tensor(full_weight, mesh, [Shard(1)])
        splits = mixin._split_experts_weights(inner_sharded_weight, 4)

        assert mixin._last_expert_ids == [0, 1, 2, 3]
        assert len(splits) == 4
        for expert_id, expert_weight in enumerate(splits):
            assert expert_weight.placements == (Shard(0),)
            torch.testing.assert_close(expert_weight.full_tensor(), full_weight[expert_id])

        converted = dict(
            mixin._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.gate_and_up_projs",
                expert_sharded_weight,
                for_checkpoint_load=True,
            )
        )
        assert len(converted) == 4
        for expert_id in local_expert_ids:
            gate_key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            up_key = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
            torch.testing.assert_close(converted[gate_key], full_weight[expert_id, :, :2].T)
            torch.testing.assert_close(converted[up_key], full_weight[expert_id, :, 2:].T)

        full_down_weight = torch.arange(4 * 2 * 6, dtype=torch.float32).reshape(4, 2, 6)
        down_weight = distribute_tensor(full_down_weight, mesh, [Shard(0)])
        converted.update(
            mixin._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.down_projs",
                down_weight,
                for_checkpoint_load=True,
            )
        )

        restored = mixin._from_hf_w_merged_experts(converted)
        assert restored == {}
        assert mixin.view_loaded_native_keys == {
            "model.layers.0.mlp.experts.gate_and_up_projs",
            "model.layers.0.mlp.experts.down_projs",
        }
        assert mixin._inplace_loaded_native_keys == set()
    finally:
        dist.destroy_process_group()


class TestValidateExpertAvailability:
    def test_no_expert_weights_in_state_dict(self):
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {"layers.0.attention.weight": torch.randn(10, 10)}

        mixin._validate_expert_availability(hf_state_dict, 8)

    def test_all_experts_available_no_device_mesh(self):
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        for layer in range(2):
            for expert in range(8):
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        mixin._validate_expert_availability(hf_state_dict, 8)

    def test_missing_experts_no_device_mesh(self):
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        # Only add experts 0-6, missing expert 7
        for layer in range(2):
            for expert in range(7):  # Missing expert 7
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        with pytest.raises(RuntimeError, match="Expert weights missing from checkpoint"):
            mixin._validate_expert_availability(hf_state_dict, 8)

    def test_inplace_loaded_experts_are_not_reported_missing(self):
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        for expert in [0, 1]:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"model.layers.0.mlp.experts.{expert}.{proj}.weight"
                hf_state_dict[key] = torch.randn(512, 1024)

        mixin._inplace_loaded_native_keys = {
            "model.layers.0.mlp.experts.gate_and_up_projs",
            "model.layers.0.mlp.experts.down_projs",
        }
        mixin._last_expert_ids = [0, 1]

        mixin._validate_expert_availability(hf_state_dict, 8)

    def test_inplace_loaded_gate_up_still_validates_down_projection(self):
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 1024),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(512, 1024),
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(512, 1024),
        }
        mixin._inplace_loaded_native_keys = {"model.layers.0.mlp.experts.gate_and_up_projs"}
        mixin._last_expert_ids = [0]

        with pytest.raises(RuntimeError, match=r"model\.layers\.0\.mlp\.experts\.1\.down_proj\.weight"):
            mixin._validate_expert_availability(hf_state_dict, 8)

    def test_without_model_prefix(self):
        mixin = MockMoEStateDictMixin(uses_model_prefix=False)
        hf_state_dict = {}

        # Add experts without "model." prefix
        for layer in range(2):
            for expert in range(8):
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        mixin._validate_expert_availability(hf_state_dict, 8)

    def test_with_language_model_prefix(self):
        """Test validation with model.language_model. prefix (VLM models)."""
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        # Add experts with "model.language_model." prefix (VLM style)
        for layer in range(2):
            for expert in range(8):
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        mixin._validate_expert_availability(hf_state_dict, 8)

    def test_missing_experts_with_language_model_prefix(self):
        """Test validation fails when experts missing with language_model prefix."""
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        # Only add experts 0-6, missing expert 7 with language_model prefix
        for layer in range(2):
            for expert in range(7):  # Missing expert 7
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        with pytest.raises(RuntimeError, match="Expert weights missing from checkpoint"):
            mixin._validate_expert_availability(hf_state_dict, 8)

    @skip_if_no_gpu
    @patch("nemo_automodel.components.moe.state_dict_mixin.get_expert_range_for_rank_from_mesh")
    @patch("nemo_automodel.components.moe.state_dict_mixin.get_submesh")
    def test_with_device_mesh(self, mock_get_submesh, mock_get_expert_range):
        mock_get_expert_range.return_value = (2, 4)  # Only need experts 2-3

        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep"]

        mock_ep_mesh = Mock()
        mock_ep_mesh.get_rank.return_value = 1
        mock_get_submesh.return_value = mock_ep_mesh

        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        # Only add experts 2-3 (required for this rank)
        for layer in range(2):
            for expert in [2, 3]:
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        mixin._validate_expert_availability(hf_state_dict, 8, mock_device_mesh)


class TestSplitExpertsWeights:
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_regular_tensor(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin()
        weight = torch.randn(8, 512, 1024)

        result = mixin._split_experts_weights(weight, 8)

        assert len(result) == 8
        assert len(mixin._last_expert_ids) == 8
        assert mixin._last_expert_ids == list(range(8))
        for i, expert_weight in enumerate(result):
            assert expert_weight.shape == (512, 1024)
            assert torch.equal(expert_weight, weight[i])

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_shape_mismatch(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin()
        weight = torch.randn(6, 512, 1024)  # Wrong number of experts

        with pytest.raises(ValueError, match="Expected first dimension to be 8, got 6"):
            mixin._split_experts_weights(weight, 8)

    @patch("nemo_automodel.components.moe.state_dict_mixin.split_experts_weights_dtensor_aware")
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_dtensor(self, mock_is_dtensor, mock_split_dtensor):
        mock_is_dtensor.return_value = True
        mock_split_dtensor.return_value = ([torch.randn(512, 1024), torch.randn(512, 1024)], [2, 3])

        mixin = MockMoEStateDictMixin()
        mock_weight = Mock()
        mock_weight.device_mesh.mesh_dim_names = ("ep",)

        result = mixin._split_experts_weights(mock_weight, 8)

        assert len(result) == 2
        assert mixin._last_expert_ids == [2, 3]
        mock_split_dtensor.assert_called_once_with(mock_weight, 8)

    def test_dtensor_without_ep_mesh_avoids_collecting_experts(self, tmp_path):
        torch.multiprocessing.spawn(
            _run_ep_free_dtensor_split,
            args=(2, str(tmp_path / "ep_free_dtensor_split")),
            nprocs=2,
            join=True,
        )


class TestConcatenateExpertWeights:
    def test_complete_experts_available(self):
        mixin = MockMoEStateDictMixin()

        expert_weights_by_layer = {
            "0": {
                "abstract_key": {
                    0: torch.randn(512, 1024),
                    1: torch.randn(512, 1024),
                    2: torch.randn(512, 1024),
                    3: torch.randn(512, 1024),
                }
            }
        }

        result = mixin._concatenate_expert_weights(expert_weights_by_layer, 4)

        assert result is not None
        assert result.shape == (4, 512, 1024)
        assert "0" not in expert_weights_by_layer  # Should be cleaned up

    def test_incomplete_experts(self):
        mixin = MockMoEStateDictMixin()

        expert_weights_by_layer = {
            "0": {
                "abstract_key": {
                    0: torch.randn(512, 1024),
                    1: torch.randn(512, 1024),
                    # Missing experts 2 and 3
                }
            }
        }

        result = mixin._concatenate_expert_weights(expert_weights_by_layer, 4)

        assert result is None
        assert "0" in expert_weights_by_layer  # Should not be cleaned up

    def test_multiple_layers_first_complete(self):
        mixin = MockMoEStateDictMixin()

        expert_weights_by_layer = {
            "0": {
                "abstract_key1": {
                    0: torch.randn(512, 1024),
                    1: torch.randn(512, 1024),
                },
                "abstract_key2": {
                    0: torch.randn(512, 1024),
                    1: torch.randn(512, 1024),
                },
            }
        }

        result = mixin._concatenate_expert_weights(expert_weights_by_layer, 2)

        assert result is not None
        assert result.shape == (2, 512, 1024)


class TestToHfWSplitExperts:
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_gate_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=4)

        # DeepEP input: gate_and_up_projs [n_experts, dim, 2*inter_dim]
        state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 1024, 1024),
            "other_weight": torch.randn(10, 10),
        }

        result = mixin._to_hf_w_split_experts(state_dict)

        # Check that gate_proj and up_proj weights were created
        for expert_id in range(4):
            gate_key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            up_key = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
            assert gate_key in result
            assert up_key in result

        # Check that other weights are preserved
        assert "other_weight" in result

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_up_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=4)

        # DeepEP input for layer 1
        state_dict = {
            "model.layers.1.mlp.experts.gate_and_up_projs": torch.randn(4, 1024, 1024),
        }

        result = mixin._to_hf_w_split_experts(state_dict)

        for expert_id in range(4):
            up_key = f"model.layers.1.mlp.experts.{expert_id}.up_proj.weight"
            assert up_key in result

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_down_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=4)

        # DeepEP down_projs: [n_experts, inter_dim, dim]
        state_dict = {
            "model.layers.2.mlp.experts.down_projs": torch.randn(4, 512, 1024),
        }

        result = mixin._to_hf_w_split_experts(state_dict)

        for expert_id in range(4):
            down_key = f"model.layers.2.mlp.experts.{expert_id}.down_proj.weight"
            assert down_key in result

    @patch("nemo_automodel.components.moe.state_dict_utils.validate_dtensor_expert_sharding")
    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    def test_dtensor_validation(self, mock_is_dtensor, mock_validate):
        mock_is_dtensor.return_value = True

        mixin = MockMoEStateDictMixin(n_experts=4)

        # Mock split to avoid depending on dtensor internals
        combined_weights = [torch.randn(1024, 1024) for _ in range(4)]
        mixin._split_experts_weights = Mock(return_value=combined_weights)
        mixin._last_expert_ids = [0, 1, 2, 3]

        mock_dtensor = Mock()
        state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": mock_dtensor,
        }

        mixin._to_hf_w_split_experts(state_dict)

        mock_validate.assert_called_once_with(mock_dtensor, 4, "gate_and_up_projs layer 0")

    def test_without_model_prefix(self):
        mixin = MockMoEStateDictMixin(n_experts=4, uses_model_prefix=False)

        with patch.object(mixin, "_split_experts_weights") as mock_split:
            gate_and_up_weights = [torch.randn(1024, 1024) for _ in range(4)]
            mock_split.return_value = gate_and_up_weights
            mixin._last_expert_ids = [0, 1, 2, 3]

            state_dict = {
                "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 1024, 1024),
            }

        with patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor", return_value=False):
            result = mixin._to_hf_w_split_experts(state_dict)

            # Without model prefix, keys should not have "model."
            for expert_id in range(4):
                expected_key = f"layers.0.mlp.experts.{expert_id}.gate_proj.weight"
                assert expected_key in result

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_gate_and_up_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        # Create mock gate_and_up tensor [n_experts, dim, 2*inter_dim]
        gate_and_up_weights = [torch.randn(1024, 1024) for _ in range(2)]  # [dim, 2*inter_dim]
        mixin._split_experts_weights = Mock(return_value=gate_and_up_weights)
        mixin._last_expert_ids = [0, 1]

        state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(2, 1024, 1024),
        }

        result = mixin._to_hf_w_split_experts(state_dict)

        # Check that gate_proj and up_proj weights were created
        for expert_id in range(2):
            gate_key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            up_key = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
            assert gate_key in result
            assert up_key in result
            assert result[gate_key].shape == (512, 1024)  # [inter_dim, dim]
            assert result[up_key].shape == (512, 1024)  # [inter_dim, dim]

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_down_projs_conversion_n2(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        # Create mock down tensor [n_experts, inter_dim, dim]
        down_weights = [torch.randn(512, 1024) for _ in range(2)]  # [inter_dim, dim]
        mixin._split_experts_weights = Mock(return_value=down_weights)
        mixin._last_expert_ids = [0, 1]

        state_dict = {
            "model.layers.0.mlp.experts.down_projs": torch.randn(2, 512, 1024),
        }

        result = mixin._to_hf_w_split_experts(state_dict)

        # Check that down_proj weights were transposed correctly
        for expert_id in range(2):
            down_key = f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"
            assert down_key in result
            assert result[down_key].shape == (1024, 512)  # [dim, inter_dim]

    @patch("nemo_automodel.components.moe.state_dict_utils.validate_dtensor_expert_sharding")
    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    def test_dtensor_validation_n2(self, mock_is_dtensor, mock_validate):
        mock_is_dtensor.return_value = True

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        gate_and_up_weights = [torch.randn(1024, 1024) for _ in range(2)]
        mixin._split_experts_weights = Mock(return_value=gate_and_up_weights)
        mixin._last_expert_ids = [0, 1]

        mock_dtensor = Mock()
        state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": mock_dtensor,
        }

        mixin._to_hf_w_split_experts(state_dict)

        mock_validate.assert_called_once_with(mock_dtensor, 2, "gate_and_up_projs layer 0")

    # Tests merged into TestToHfWSplitExperts


class TestFromHfWMergedExperts:
    def test_direct_fill_gated_experts_preserves_values_dtype_and_layout(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=2, dtype=torch.bfloat16)
        hf_state_dict = {}
        for expert_id, gate_start, up_start in ((0, 0, 10), (1, 20, 30)):
            gate_weight = torch.arange(gate_start, gate_start + 6, dtype=torch.float32).reshape(3, 2).T
            up_weight = torch.arange(up_start, up_start + 6, dtype=torch.float32).reshape(3, 2).T
            assert not gate_weight.is_contiguous()
            assert not up_weight.is_contiguous()
            hf_state_dict[f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"] = gate_weight
            hf_state_dict[f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"] = up_weight

        with (
            patch.object(mixin, "_validate_expert_availability"),
            patch("nemo_automodel.components.moe.state_dict_mixin.gc.collect") as collect,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            result = mixin._from_hf_w_merged_experts(hf_state_dict)

        grouped = result["model.layers.0.mlp.experts.gate_and_up_projs"]
        expected = torch.tensor(
            [
                [[0, 1, 10, 11], [2, 3, 12, 13], [4, 5, 14, 15]],
                [[20, 21, 30, 31], [22, 23, 32, 33], [24, 25, 34, 35]],
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(grouped, expected, rtol=0, atol=0)
        assert grouped.dtype == torch.bfloat16
        assert grouped.is_contiguous()
        collect.assert_called_once_with(0)
        empty_cache.assert_not_called()

    def test_direct_fill_non_gated_and_down_experts_preserves_values(self):
        mixin = MockMoEStateDictMixin(
            n_experts=2,
            inter_dim=2,
            dtype=torch.float32,
            expert_activation="relu2",
        )
        hf_state_dict = {}
        for expert_id, up_start, down_start in ((0, 0, 10), (1, 20, 30)):
            hf_state_dict[f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"] = (
                torch.arange(up_start, up_start + 6, dtype=torch.float64).reshape(3, 2).T
            )
            hf_state_dict[f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"] = (
                torch.arange(down_start, down_start + 6, dtype=torch.float64).reshape(2, 3).T
            )

        with patch.object(mixin, "_validate_expert_availability"):
            result = mixin._from_hf_w_merged_experts(hf_state_dict)

        up = result["model.layers.0.mlp.experts.gate_and_up_projs"]
        down = result["model.layers.0.mlp.experts.down_projs"]
        expected_up = torch.tensor(
            [
                [[0, 1], [2, 3], [4, 5]],
                [[20, 21], [22, 23], [24, 25]],
            ],
            dtype=torch.float32,
        )
        expected_down = torch.tensor(
            [
                [[10, 11, 12], [13, 14, 15]],
                [[30, 31, 32], [33, 34, 35]],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(up, expected_up, rtol=0, atol=0)
        torch.testing.assert_close(down, expected_down, rtol=0, atol=0)
        assert up.is_contiguous()
        assert down.is_contiguous()

    def test_direct_fill_merge_kernels_write_into_final_storage(self):
        mixin = MockMoEStateDictMixin(dtype=torch.float32)
        gated_parts = [
            (torch.zeros(3, 2), torch.ones(3, 2)),
            (torch.full((3, 2), 2.0), torch.full((3, 2), 3.0)),
        ]

        with patch("torch.cat", wraps=torch.cat) as cat:
            grouped = mixin._direct_fill_grouped_expert_tensor(gated_parts)

        grouped_ptr = grouped.untyped_storage().data_ptr()
        assert cat.call_count == 2
        assert all(call.kwargs["out"].untyped_storage().data_ptr() == grouped_ptr for call in cat.call_args_list)

        single_parts = [(torch.zeros(3, 2),), (torch.ones(3, 2),)]
        with patch("torch.stack", wraps=torch.stack) as stack:
            grouped = mixin._direct_fill_grouped_expert_tensor(single_parts)

        assert stack.call_count == 1
        assert stack.call_args.kwargs["out"] is grouped

    @skip_if_no_gpu
    @pytest.mark.run_only_on("GPU")
    @pytest.mark.torch_memory_limit(cuda_mb=40)
    def test_direct_fill_cuda_peak_is_inputs_plus_one_output(self):
        """Catch reintroducing per-expert concatenation buffers before the final stack."""
        mixin = MockMoEStateDictMixin(n_experts=8, inter_dim=512, dtype=torch.bfloat16)
        # Inputs occupy 16 MiB and the grouped output occupies 16 MiB. The old cat-then-stack implementation retained
        # another 16 MiB of per-expert concatenations and therefore exceeded this 40 MiB budget.
        expert_parts = [
            (
                torch.empty((1024, 512), dtype=torch.bfloat16, device="cuda"),
                torch.empty((1024, 512), dtype=torch.bfloat16, device="cuda"),
            )
            for _ in range(8)
        ]

        grouped = mixin._direct_fill_grouped_expert_tensor(expert_parts)

        assert grouped.shape == (8, 1024, 1024)
        assert grouped.is_contiguous()

    @patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local")
    @patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank")
    def test_basic_conversion(self, mock_should_load, mock_create_dtensor):
        mock_should_load.return_value = True
        mock_create_dtensor.side_effect = lambda x, *args: x  # Return local tensor as-is

        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {}
        # Add gate_proj and up_proj weights for 2 experts in layer 0
        for expert_id in range(2):
            key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.randn(512, 1024)
            key_up = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
            hf_state_dict[key_up] = torch.randn(512, 1024)

        with patch.object(mixin, "_validate_expert_availability"):
            result = mixin._from_hf_w_merged_experts(hf_state_dict)

        # Check that gate_and_up_projs tensor was created
        expected_key = "model.layers.0.mlp.experts.gate_and_up_projs"
        assert expected_key in result
        assert result[expected_key].shape == (2, 1024, 1024)

    def test_partial_expert_loading(self):
        # Test that the method respects should_load_expert_for_rank filtering
        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {}
        for expert_id in range(2):
            key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.randn(512, 1024)
            key_up = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
            hf_state_dict[key_up] = torch.randn(512, 1024)

        with patch.object(mixin, "_validate_expert_availability"):
            with patch(
                "nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank"
            ) as mock_should_load:
                mock_should_load.side_effect = lambda expert_id, *args: expert_id == 1  # Only load expert 1
                with patch(
                    "nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local",
                    side_effect=lambda x, *args: x,
                ):
                    result = mixin._from_hf_w_merged_experts(hf_state_dict)

        # When only partial experts are loaded, no tensor should be created until all are available
        # This is the expected behavior based on the code logic
        expected_key = "model.layers.0.mlp.experts.gate_and_up_projs"
        assert expected_key not in result  # No tensor created because we don't have all expected experts

    def test_without_model_prefix(self):
        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {}
        # Add weights without "model." prefix
        for expert_id in range(2):
            key = f"layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.randn(512, 1024)
            key_up = f"layers.0.mlp.experts.{expert_id}.up_proj.weight"
            hf_state_dict[key_up] = torch.randn(512, 1024)

        with patch.object(mixin, "_validate_expert_availability"):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch(
                    "nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local",
                    side_effect=lambda x, *args: x,
                ):
                    result = mixin._from_hf_w_merged_experts(hf_state_dict)

        # Result key preserves the empty prefix from input
        expected_key = "layers.0.mlp.experts.gate_and_up_projs"
        assert expected_key in result

    def test_with_language_model_prefix(self):
        """Test conversion with model.language_model. prefix (VLM models)."""
        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {}
        # Add weights with "model.language_model." prefix (VLM style)
        for expert_id in range(2):
            key = f"model.language_model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.randn(512, 1024)
            key_up = f"model.language_model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
            hf_state_dict[key_up] = torch.randn(512, 1024)

        with patch.object(mixin, "_validate_expert_availability"):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch(
                    "nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local",
                    side_effect=lambda x, *args: x,
                ):
                    result = mixin._from_hf_w_merged_experts(hf_state_dict)

        # Result key should preserve the language_model prefix
        expected_key = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"
        assert expected_key in result
        assert result[expected_key].shape == (2, 1024, 1024)

    def test_with_language_model_prefix_down_proj(self):
        """Test down_proj conversion with model.language_model. prefix."""
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512, dtype=torch.float32)

        hf_state_dict = {}
        for expert_id in range(2):
            key = f"model.language_model.layers.0.mlp.experts.{expert_id}.down_proj.weight"
            hf_state_dict[key] = torch.randn(1024, 512)  # [dim, inter_dim]

        with patch.object(mixin, "_validate_expert_availability"):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch(
                    "nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local",
                    side_effect=lambda x, *args: x,
                ):
                    result = mixin._from_hf_w_merged_experts(hf_state_dict)

        # Result key should preserve the language_model prefix
        expected_key = "model.language_model.layers.0.mlp.experts.down_projs"
        assert expected_key in result
        assert result[expected_key].shape == (2, 512, 1024)  # [n_experts, inter_dim, dim]

    @skip_if_no_gpu
    @patch("nemo_automodel.components.moe.state_dict_mixin.get_expert_range_for_rank_from_mesh")
    @patch("nemo_automodel.components.moe.state_dict_mixin.get_submesh")
    def test_with_device_mesh(self, mock_get_submesh, mock_get_expert_range):
        mock_get_expert_range.return_value = (0, 1)  # Only expert 0 for this rank

        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep"]

        mock_ep_mesh = Mock()
        mock_ep_mesh.get_rank.return_value = 0
        mock_get_submesh.return_value = mock_ep_mesh

        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 1024),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(512, 1024),
        }

        with patch.object(mixin, "_validate_expert_availability"):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch(
                    "nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local",
                    side_effect=lambda x, *args: x,
                ):
                    result = mixin._from_hf_w_merged_experts(hf_state_dict, mock_device_mesh)

        expected_key = "model.layers.0.mlp.experts.gate_and_up_projs"
        assert expected_key in result
        assert result[expected_key].shape == (1, 1024, 1024)

    @patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local")
    @patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank")
    def test_gate_and_up_combination(self, mock_should_load, mock_create_dtensor):
        mock_should_load.return_value = True
        mock_create_dtensor.side_effect = lambda x, *args: x

        mixin = MockMoEStateDictMixin(n_experts=1, inter_dim=512, dtype=torch.float32)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 1024),  # [inter_dim, dim]
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(512, 1024),  # [inter_dim, dim]
        }

        with patch.object(mixin, "_validate_expert_availability"):
            result = mixin._from_hf_w_merged_experts(hf_state_dict)

        # Should create gate_and_up_projs tensor
        expected_key = "model.layers.0.mlp.experts.gate_and_up_projs"
        assert expected_key in result
        assert result[expected_key].shape == (1, 1024, 1024)  # [n_experts, dim, 2*inter_dim]

    @patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local")
    @patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank")
    def test_down_proj_transpose(self, mock_should_load, mock_create_dtensor):
        mock_should_load.return_value = True
        mock_create_dtensor.side_effect = lambda x, *args: x

        mixin = MockMoEStateDictMixin(n_experts=1, inter_dim=512, dtype=torch.float32)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(1024, 512),  # [dim, inter_dim]
        }

        with patch.object(mixin, "_validate_expert_availability"):
            result = mixin._from_hf_w_merged_experts(hf_state_dict)

        # Should create transposed down_projs tensor
        expected_key = "model.layers.0.mlp.experts.down_projs"
        assert expected_key in result
        assert result[expected_key].shape == (1, 512, 1024)  # [n_experts, inter_dim, dim]

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_dtensor_input_handling(self, mock_is_dtensor):
        # Test when input tensors are already DTensors
        mock_is_dtensor.return_value = True

        mixin = MockMoEStateDictMixin(n_experts=1, inter_dim=512, dtype=torch.float32)

        # Mock DTensor inputs
        mock_gate_dtensor = Mock()
        mock_gate_dtensor.to_local.return_value = torch.randn(512, 1024)
        mock_up_dtensor = Mock()
        mock_up_dtensor.to_local.return_value = torch.randn(512, 1024)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": mock_gate_dtensor,
            "model.layers.0.mlp.experts.0.up_proj.weight": mock_up_dtensor,
        }

        with patch.object(mixin, "_validate_expert_availability"):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch(
                    "nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local",
                    side_effect=lambda x, *args: x,
                ):
                    mixin._from_hf_w_merged_experts(hf_state_dict)

        # Verify to_local was called on DTensor inputs
        mock_gate_dtensor.to_local.assert_called_once()
        mock_up_dtensor.to_local.assert_called_once()

    def test_skip_scale_inv_keys(self):
        mixin = MockMoEStateDictMixin()

        hf_state_dict = {
            "some_weight": torch.randn(10, 10),
            "some_weight_scale_inv": torch.randn(10),  # Should be skipped
        }

        with patch.object(mixin, "_validate_expert_availability"):
            result = mixin._from_hf_w_merged_experts(hf_state_dict)

        assert "some_weight" in result
        assert "some_weight_scale_inv" not in result

    # Tests merged into TestFromHfWMergedExperts


class TestConvertSingleMergedExpertToHfSplitExperts:
    def test_allocating_cuda_conversions_use_generation_zero_collection(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=3)
        mixin.backend.experts = "te"
        mixin.backend.dispatcher = "deepep"
        gate_up_tensor = Mock(spec=torch.Tensor, is_meta=False, is_cuda=True)
        down_tensor = Mock(spec=torch.Tensor, ndim=3, shape=(2, 3, 4), is_meta=False, is_cuda=True)
        gate_up_splits = [
            torch.arange(24, dtype=torch.float32).reshape(4, 6) + 24 * expert_id for expert_id in range(2)
        ]
        down_splits = [torch.arange(12, dtype=torch.float32).reshape(3, 4) + 12 * expert_id for expert_id in range(2)]

        with (
            patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=2),
            patch.object(mixin, "_split_experts_weights", side_effect=[gate_up_splits, down_splits]),
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_automodel.components.moe.state_dict_mixin.gc.collect") as collect,
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            mixin._last_expert_ids = [0, 1]
            gate_up_result = mixin._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.gate_and_up_projs", gate_up_tensor
            )
            down_result = mixin._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.down_projs", down_tensor
            )

        assert gate_up_result is not None and down_result is not None
        assert [gc_call.args for gc_call in collect.call_args_list] == [(0,), (0,)]
        assert empty_cache.call_count == 2
        converted = dict(gate_up_result + down_result)
        for expert_id, (gate_up_split, down_split) in enumerate(zip(gate_up_splits, down_splits)):
            torch.testing.assert_close(
                converted[f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"], gate_up_split[:, :3].T
            )
            torch.testing.assert_close(
                converted[f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"], gate_up_split[:, 3:].T
            )
            torch.testing.assert_close(
                converted[f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"], down_split.T
            )

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_gate_and_up_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        # Create gate_and_up_projs tensor [n_experts, dim, 2*inter_dim]
        tensor = torch.randn(2, 1024, 1024)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        result = mixin._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)

        assert result is not None
        assert len(result) == 4  # 2 experts * 2 projections (gate + up)

        # Check gate_proj and up_proj for each expert
        for expert_id in range(2):
            gate_key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            up_key = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"

            gate_found = any(k == gate_key for k, _ in result)
            up_found = any(k == up_key for k, _ in result)

            assert gate_found, f"Expected {gate_key} in result"
            assert up_found, f"Expected {up_key} in result"

            # Check shapes
            for k, v in result:
                if k == gate_key or k == up_key:
                    assert v.shape == (512, 1024)  # [inter_dim, dim]

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_down_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        # Create down_projs tensor [n_experts, inter_dim, dim]
        tensor = torch.randn(2, 512, 1024)
        fqn = "model.layers.1.mlp.experts.down_projs"

        result = mixin._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)

        assert result is not None
        assert len(result) == 2  # 2 experts

        for expert_id in range(2):
            down_key = f"model.layers.1.mlp.experts.{expert_id}.down_proj.weight"
            down_found = any(k == down_key for k, _ in result)
            assert down_found, f"Expected {down_key} in result"

            for k, v in result:
                if k == down_key:
                    assert v.shape == (1024, 512)  # [dim, inter_dim] - transposed

    def test_non_expert_tensor_returns_none(self):
        mixin = MockMoEStateDictMixin()

        # Regular weight tensor
        tensor = torch.randn(512, 512)
        fqn = "model.layers.0.attention.weight"

        result = mixin._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)

        assert result is None

    def test_without_model_prefix(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512, uses_model_prefix=False)

        with patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor", return_value=False):
            tensor = torch.randn(2, 1024, 1024)
            fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

            result = mixin._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)

            assert result is not None
            # Keys should not have "model." prefix
            for key, _ in result:
                assert key.startswith("layers."), f"Expected key to start with 'layers.', got {key}"
                assert not key.startswith("model."), f"Key should not have 'model.' prefix: {key}"

    @patch("nemo_automodel.components.moe.state_dict_utils.validate_dtensor_expert_sharding")
    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    def test_dtensor_validation_called(self, mock_is_dtensor, mock_validate):
        mock_is_dtensor.return_value = True

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        # Mock split to avoid depending on dtensor internals
        weights = [torch.randn(1024, 1024) for _ in range(2)]
        mixin._split_experts_weights = Mock(return_value=weights)
        mixin._last_expert_ids = [0, 1]

        mock_dtensor = Mock()
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        result = mixin._convert_single_merged_expert_to_hf_split_experts(fqn, mock_dtensor)

        mock_validate.assert_called_once_with(mock_dtensor, 2, "gate_and_up_projs layer 0")
        assert result is not None


class TestInplaceLoadViews:
    """to_hf returns non-contiguous views into the model's grouped tensor's
    local storage whenever the source is a model DTensor with plain
    (non-DTensor) per-expert splits. DCP writes safetensors data through the
    views into model storage, and ``_from_hf_w_merged_experts`` skips the
    rebuild for those native keys (the model already holds the data). Save
    callers must materialize the views to contiguous before serializing —
    see ``_materialize_to_hf_views_for_save`` in checkpointing.

    The mixin re-imports ``is_dtensor`` from ``state_dict_utils`` inside the
    conversion function, so patches must target that module path.
    """


    def test_grouped_expert_bias_native_roundtrip_preserves_grouped_keys(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=3)
        mixin.moe_config.expert_bias = True
        native = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(2, 4, 6),
            "model.layers.0.mlp.experts.down_projs": torch.randn(2, 3, 4),
            "model.layers.0.mlp.experts.gate_up_proj_bias": torch.randn(2, 6),
            "model.layers.0.mlp.experts.down_proj_bias": torch.randn(2, 4),
        }

        assert mixin.supports_low_memory_dcp_load is False
        hf_state = mixin._to_hf_w_split_experts(dict(native))
        assert (
            hf_state["model.layers.0.mlp.experts.gate_up_proj_bias"]
            is native["model.layers.0.mlp.experts.gate_up_proj_bias"]
        )
        assert (
            hf_state["model.layers.0.mlp.experts.down_proj_bias"] is native["model.layers.0.mlp.experts.down_proj_bias"]
        )

        restored = mixin._from_hf_w_merged_experts(dict(hf_state))
        assert set(restored) == set(native)
        for key in native:
            torch.testing.assert_close(restored[key], native[key])

    def test_hf_per_expert_bias_load_is_rejected(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=3)
        mixin.moe_config.expert_bias = True

        with pytest.raises(NotImplementedError, match="per-expert bias"):
            mixin._from_hf_w_merged_experts({"model.layers.0.mlp.experts.0.gate_proj.bias": torch.randn(3)})

    def _run_inplace_conversion(self, mixin, fqn, mock_dtensor, splits, *, for_checkpoint_load=False):
        mixin._split_experts_weights = Mock(return_value=splits)
        mixin._last_expert_ids = list(range(len(splits)))

        with (
            patch(
                "nemo_automodel.components.moe.state_dict_utils.is_dtensor",
                side_effect=lambda x: x is mock_dtensor,
            ),
            patch("nemo_automodel.components.moe.state_dict_utils.validate_dtensor_expert_sharding"),
        ):
            return mixin._convert_single_merged_expert_to_hf_split_experts(
                fqn, mock_dtensor, for_checkpoint_load=for_checkpoint_load
            )

    def test_inplace_load_gate_and_up_returns_views(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)
        # local[i] for gated has shape (dim=1024, 2*inter=1024).
        local_storage = torch.randn(2, 1024, 1024)
        splits = [local_storage[i] for i in range(2)]
        mock_dtensor = Mock()

        result = self._run_inplace_conversion(
            mixin,
            "model.layers.0.mlp.experts.gate_and_up_projs",
            mock_dtensor,
            splits,
            for_checkpoint_load=True,
        )

        assert result is not None
        src_ptr = local_storage.untyped_storage().data_ptr()
        for k, v in result:
            assert v.untyped_storage().data_ptr() == src_ptr, f"in-place view for {k} should alias model storage"
            assert not v.is_contiguous(), f"in-place view for {k} must be the strided transpose, not a copy"
        assert "model.layers.0.mlp.experts.gate_and_up_projs" in mixin._inplace_loaded_native_keys

    @pytest.mark.parametrize("projection", ["gate_and_up_projs", "down_projs"])
    def test_export_materializes_expert_tensors_without_registering_loaded_keys(self, projection):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=3)
        shape = (2, 4, 6) if projection == "gate_and_up_projs" else (2, 3, 4)
        local_storage = torch.arange(48 if projection == "gate_and_up_projs" else 24).reshape(shape).float()
        mock_dtensor = Mock(spec=["ndim", "shape", "is_meta"])
        mock_dtensor.ndim = 3
        mock_dtensor.shape = shape
        mock_dtensor.is_meta = False

        result = self._run_inplace_conversion(
            mixin,
            f"model.layers.0.mlp.experts.{projection}",
            mock_dtensor,
            list(local_storage.unbind()),
            for_checkpoint_load=False,
        )

        assert result is not None
        for key, tensor in result:
            assert tensor.is_contiguous()
            assert tensor.untyped_storage().data_ptr() != local_storage.untyped_storage().data_ptr()
            expert_id = int(key.split(".")[-3])
            expected = local_storage[expert_id]
            if projection == "gate_and_up_projs":
                expected = expected[:, :3] if key.endswith("gate_proj.weight") else expected[:, 3:]
            torch.testing.assert_close(tensor, expected.T)
        assert not getattr(mixin, "_inplace_loaded_native_keys", set())

    def test_inplace_load_down_projs_returns_views(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)
        # local[i] for down has shape (inter=512, dim=1024).
        local_storage = torch.randn(2, 512, 1024)
        splits = [local_storage[i] for i in range(2)]

        # The down branch dispatches via ``tensor.shape[1] == inter_dim``, so
        # the mock must answer that check before splits are computed.
        mock_dtensor = Mock(spec=["ndim", "shape", "is_meta"])
        mock_dtensor.ndim = 3
        mock_dtensor.shape = (2, 512, 1024)
        mock_dtensor.is_meta = False

        result = self._run_inplace_conversion(
            mixin,
            "model.layers.3.mlp.experts.down_projs",
            mock_dtensor,
            splits,
            for_checkpoint_load=True,
        )

        assert result is not None and len(result) == 2
        src_ptr = local_storage.untyped_storage().data_ptr()
        for k, v in result:
            assert v.untyped_storage().data_ptr() == src_ptr, f"in-place view for {k} should alias model storage"
        assert "model.layers.3.mlp.experts.down_projs" in mixin._inplace_loaded_native_keys

    def test_inplace_load_skips_when_source_not_dtensor(self):
        # When tensor is a plain CPU tensor (not from the model), the in-place
        # path must not engage — there is no model storage to alias and
        # contiguous copies are the correct fallback.
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)
        tensor = torch.randn(2, 1024, 1024)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        with patch(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor",
            return_value=False,
        ):
            result = mixin._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)

        assert result is not None
        for _, v in result:
            assert v.is_contiguous(), "non-DTensor source should emit contiguous copies"
        assert not hasattr(
            mixin, "_inplace_loaded_native_keys"
        ) or "model.layers.0.mlp.experts.gate_and_up_projs" not in (mixin._inplace_loaded_native_keys or set())

    def test_inplace_load_writes_through_to_model_storage(self):
        # Simulate DCP-style copy_ on the emitted views and verify the model's
        # underlying storage is updated at the correct slice.
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)
        local_storage = torch.zeros(2, 1024, 1024)
        splits = [local_storage[i] for i in range(2)]
        mock_dtensor = Mock()

        result = self._run_inplace_conversion(
            mixin,
            "model.layers.0.mlp.experts.gate_and_up_projs",
            mock_dtensor,
            splits,
            for_checkpoint_load=True,
        )

        gate0 = next(v for k, v in result if k.endswith("0.gate_proj.weight"))
        gate0.copy_(torch.full_like(gate0, 7.0))
        # Gated layout: local[0, :, :inter] holds gate, local[0, :, inter:] holds up.
        assert torch.allclose(local_storage[0, :, :512], torch.full((1024, 512), 7.0))
        assert torch.all(local_storage[0, :, 512:] == 0)
        assert torch.all(local_storage[1] == 0)

    def test_from_hf_skips_rebuild_for_inplace_loaded_keys(self):
        # When _inplace_loaded_native_keys contains a layer's grouped key, the
        # per-expert HF keys for that layer must NOT be merged back into a
        # native key in the output state_dict.
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)
        mixin._inplace_loaded_native_keys = {
            "model.layers.0.mlp.experts.gate_and_up_projs",
            "model.layers.0.mlp.experts.down_projs",
        }
        hf_state_dict = {}
        for expert_id in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                hf_state_dict[f"model.layers.0.mlp.experts.{expert_id}.{proj}.weight"] = torch.randn(512, 1024)

        out = mixin._from_hf_w_merged_experts(hf_state_dict)

        assert "model.layers.0.mlp.experts.gate_and_up_projs" not in out
        assert "model.layers.0.mlp.experts.down_projs" not in out
        assert mixin._inplace_loaded_native_keys == set()

    def test_save_conversion_does_not_poison_later_from_hf(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=3)
        grouped_gate_up = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
        grouped_down = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        gate_up_dtensor = Mock()
        down_dtensor = Mock(spec=["ndim", "shape", "is_meta"])
        down_dtensor.ndim = 3
        down_dtensor.shape = (2, 3, 4)
        down_dtensor.is_meta = False

        exported = self._run_inplace_conversion(
            mixin,
            "model.layers.0.mlp.experts.gate_and_up_projs",
            gate_up_dtensor,
            [grouped_gate_up[expert_id] for expert_id in range(2)],
            for_checkpoint_load=False,
        )
        exported += self._run_inplace_conversion(
            mixin,
            "model.layers.0.mlp.experts.down_projs",
            down_dtensor,
            [grouped_down[expert_id] for expert_id in range(2)],
            for_checkpoint_load=False,
        )

        assert not hasattr(mixin, "_inplace_loaded_native_keys")
        restored = mixin._from_hf_w_merged_experts(dict(exported))
        torch.testing.assert_close(restored["model.layers.0.mlp.experts.gate_and_up_projs"], grouped_gate_up)
        torch.testing.assert_close(restored["model.layers.0.mlp.experts.down_projs"], grouped_down)

    def test_inplace_load_skips_when_backend_experts_is_te_gate_and_up(self):
        # GroupedExpertsTE (backend.experts == "te") exposes gate_and_up_projs as a
        # torch.stack copy of per-expert weights that does not alias the model's grouped
        # storage. Even for a DTensor source the in-place path must not engage, otherwise
        # the copy_ would write the throwaway and the experts would never be loaded.
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)
        mixin.backend.experts = "te"
        mixin.backend.dispatcher = "deepep"
        local_storage = torch.randn(2, 1024, 1024)
        splits = [local_storage[i] for i in range(2)]
        mock_dtensor = Mock()

        with patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=8):
            result = self._run_inplace_conversion(
                mixin, "model.layers.0.mlp.experts.gate_and_up_projs", mock_dtensor, splits
            )

        assert result is not None
        converted = dict(result)
        for _, v in result:
            assert v.is_contiguous(), "experts=='te' must emit contiguous copies, not in-place views"
        for expert_id in range(2):
            torch.testing.assert_close(
                converted[f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"],
                local_storage[expert_id, :, :512].T,
            )
            torch.testing.assert_close(
                converted[f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"],
                local_storage[expert_id, :, 512:].T,
            )
        assert not hasattr(mixin, "_inplace_loaded_native_keys") or (
            "model.layers.0.mlp.experts.gate_and_up_projs" not in (mixin._inplace_loaded_native_keys or set())
        )

    def test_mok_noncontiguous_checkpoint_views_are_reused(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=3)
        mixin.backend.experts = "gmm"
        mixin.backend.dispatcher = "mok"
        initialized = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)

        with patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor", return_value=False):
            result = mixin._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.gate_and_up_projs",
                initialized,
                for_checkpoint_load=True,
            )

        assert result is not None and len(result) == 4
        source_ptr = initialized.untyped_storage().data_ptr()
        for _, destination in result:
            assert not destination.is_contiguous()
            assert destination.untyped_storage().data_ptr() == source_ptr

    def test_te_checkpoint_layout_views_are_reused(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=3)
        mixin.backend.experts = "te"
        mixin.backend.dispatcher = "deepep"
        gate_up_storage = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        down_storage = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

        # GroupedExpertsTE exposes stack(per_expert_weights).transpose(-1, -2).
        # The adapter transposes each expert back, yielding contiguous checkpoint-layout views.
        virtual_gate_up = gate_up_storage.transpose(-1, -2)
        virtual_down = down_storage.transpose(-1, -2)

        with (
            patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=8),
            patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor", return_value=False),
            patch("torch.empty_like") as empty_like,
        ):
            destinations = dict(
                mixin._convert_single_merged_expert_to_hf_split_experts(
                    "model.layers.0.mlp.experts.gate_and_up_projs",
                    virtual_gate_up,
                    for_checkpoint_load=True,
                )
            )
            destinations.update(
                mixin._convert_single_merged_expert_to_hf_split_experts(
                    "model.layers.0.mlp.experts.down_projs",
                    virtual_down,
                    for_checkpoint_load=True,
                )
            )

        empty_like.assert_not_called()
        for expert_id in range(2):
            gate = destinations[f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"]
            up = destinations[f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"]
            down = destinations[f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"]
            assert gate.is_contiguous() and up.is_contiguous() and down.is_contiguous()
            assert gate.untyped_storage().data_ptr() == gate_up_storage.untyped_storage().data_ptr()
            assert up.untyped_storage().data_ptr() == gate_up_storage.untyped_storage().data_ptr()
            assert down.untyped_storage().data_ptr() == down_storage.untyped_storage().data_ptr()
            torch.testing.assert_close(gate, gate_up_storage[expert_id, :3])
            torch.testing.assert_close(up, gate_up_storage[expert_id, 3:])
            torch.testing.assert_close(down, down_storage[expert_id])

    @skip_if_no_gpu
    @pytest.mark.run_only_on("GPU")
    @pytest.mark.torch_memory_limit(cuda_mb=70)
    def test_te_checkpoint_layout_cuda_peak_stays_within_one_buffer(self):
        """Catch allocating a second model-sized destination for TE checkpoint views."""
        mixin = MockMoEStateDictMixin(n_experts=8, inter_dim=1024, dtype=torch.bfloat16)
        mixin.backend.experts = "te"
        mixin.backend.dispatcher = "deepep"
        # The TE stack is 64 MiB. Reusing its checkpoint-layout views fits this 70 MiB budget; allocating blank
        # destinations of the same total size would double live allocation to 128 MiB.
        gate_up_storage = torch.empty((8, 2048, 2048), dtype=torch.bfloat16, device="cuda")
        virtual_gate_up = gate_up_storage.transpose(-1, -2)

        with (
            patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=8),
            patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor", return_value=False),
            patch("nemo_automodel.components.moe.state_dict_mixin.gc.collect") as collect,
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            destinations = mixin._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.gate_and_up_projs",
                virtual_gate_up,
                for_checkpoint_load=True,
            )

        assert destinations is not None and len(destinations) == 16
        source_ptr = gate_up_storage.untyped_storage().data_ptr()
        assert all(value.untyped_storage().data_ptr() == source_ptr for _, value in destinations)
        collect.assert_not_called()
        empty_cache.assert_not_called()

    def test_temporary_checkpoint_views_round_trip_loaded_values(self):
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=3)
        mixin.backend.experts = "te"
        mixin.backend.dispatcher = "deepep"
        initialized_gate_up = torch.full((2, 4, 6), 99.0)
        initialized_down = torch.full((2, 3, 4), 99.0)

        with (
            patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=8),
            patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor", return_value=False),
        ):
            destinations = dict(
                mixin._convert_single_merged_expert_to_hf_split_experts(
                    "model.layers.0.mlp.experts.gate_and_up_projs",
                    initialized_gate_up,
                    for_checkpoint_load=True,
                )
            )
            destinations.update(
                mixin._convert_single_merged_expert_to_hf_split_experts(
                    "model.layers.0.mlp.experts.down_projs",
                    initialized_down,
                    for_checkpoint_load=True,
                )
            )

        expected_gate_up = torch.empty_like(initialized_gate_up)
        expected_down = torch.empty_like(initialized_down)
        for expert_id in range(2):
            gate = torch.full((3, 4), 10.0 + expert_id)
            up = torch.full((3, 4), 20.0 + expert_id)
            down = torch.full((4, 3), 30.0 + expert_id)
            destinations[f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"].copy_(gate)
            destinations[f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"].copy_(up)
            destinations[f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"].copy_(down)
            expected_gate_up[expert_id] = torch.cat((gate.T, up.T), dim=-1)
            expected_down[expert_id] = down.T

        restored = mixin._from_hf_w_merged_experts(destinations)

        torch.testing.assert_close(
            restored["model.layers.0.mlp.experts.gate_and_up_projs"],
            expected_gate_up,
        )
        torch.testing.assert_close(
            restored["model.layers.0.mlp.experts.down_projs"],
            expected_down,
        )

    def test_inplace_load_skips_when_backend_experts_is_te_down_projs(self):
        # Same non-aliasing reason as the gate_and_up case, for the down_projs branch.
        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)
        mixin.backend.experts = "te"
        mixin.backend.dispatcher = "deepep"
        local_storage = torch.randn(2, 512, 1024)
        splits = [local_storage[i] for i in range(2)]
        mock_dtensor = Mock(spec=["ndim", "shape", "is_meta"])
        mock_dtensor.ndim = 3
        mock_dtensor.shape = (2, 512, 1024)
        mock_dtensor.is_meta = False

        with patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=8):
            result = self._run_inplace_conversion(mixin, "model.layers.3.mlp.experts.down_projs", mock_dtensor, splits)

        assert result is not None and len(result) == 2
        for _, v in result:
            assert v.is_contiguous(), "experts=='te' must emit contiguous copies, not in-place views"
        assert not hasattr(mixin, "_inplace_loaded_native_keys") or (
            "model.layers.3.mlp.experts.down_projs" not in (mixin._inplace_loaded_native_keys or set())
        )

