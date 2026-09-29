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

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")

from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_expert_slice_for_rank,
    is_dtensor,
    should_load_expert_for_rank,
    split_experts_weights_dtensor_aware,
    validate_dtensor_expert_sharding,
)


class TestIsDtensor:
    def test_regular_tensor(self):
        tensor = torch.randn(4, 8)
        assert not is_dtensor(tensor)

    def test_dtensor_mock(self):
        from torch.distributed._tensor import DTensor

        with patch("nemo_automodel.components.moe.state_dict_utils.DTensor", DTensor):
            mock_tensor = Mock(spec=DTensor)
            mock_tensor.__class__ = DTensor
            assert is_dtensor(mock_tensor)


class TestGetExpertSliceForRank:
    def test_regular_tensor(self):
        tensor = torch.randn(8, 16, 32)
        n_experts = 8

        local_tensor, start_expert, end_expert = get_expert_slice_for_rank(tensor, n_experts)

        assert torch.equal(local_tensor, tensor)
        assert start_expert == 0
        assert end_expert == 8

    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    @patch("nemo_automodel.components.moe.state_dict_utils.get_submesh")
    def test_dtensor_sharded_expert_dimension(self, mock_get_submesh, mock_is_dtensor):
        mock_is_dtensor.return_value = True

        # Create mock DTensor
        mock_dtensor = Mock()
        mock_local_tensor = torch.randn(2, 16, 32)
        mock_dtensor.to_local.return_value = mock_local_tensor

        # Mock device mesh
        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep", "dp"]
        mock_dtensor.device_mesh = mock_device_mesh

        # Mock EP submesh
        mock_ep_mesh = Mock()
        mock_ep_mesh.get_local_rank.return_value = 1
        mock_ep_mesh.size.return_value = 4
        mock_get_submesh.return_value = mock_ep_mesh

        # Mock placement - sharded on dim 0
        from torch.distributed._tensor.placement_types import Shard

        mock_placement = Shard(0)
        mock_dtensor.placements = [mock_placement, Mock()]

        local_tensor, start_expert, end_expert = get_expert_slice_for_rank(mock_dtensor, 8)

        assert torch.equal(local_tensor, mock_local_tensor)
        assert start_expert == 2
        assert end_expert == 4

    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    @patch("nemo_automodel.components.moe.state_dict_utils.get_submesh")
    def test_dtensor_replicated(self, mock_get_submesh, mock_is_dtensor):
        mock_is_dtensor.return_value = True

        mock_dtensor = Mock()
        mock_local_tensor = torch.randn(8, 16, 32)
        mock_dtensor.to_local.return_value = mock_local_tensor

        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep"]
        mock_dtensor.device_mesh = mock_device_mesh

        mock_ep_mesh = Mock()
        mock_get_submesh.return_value = mock_ep_mesh

        # Mock placement - replicated
        from torch.distributed._tensor.placement_types import Replicate

        mock_placement = Replicate()
        mock_dtensor.placements = [mock_placement]

        local_tensor, start_expert, end_expert = get_expert_slice_for_rank(mock_dtensor, 8)

        assert torch.equal(local_tensor, mock_local_tensor)
        assert start_expert == 0
        assert end_expert == 8

    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    @patch("nemo_automodel.components.moe.state_dict_utils.get_submesh")
    def test_ep_free_fsdp_sharded_expert_dimension(self, mock_get_submesh, mock_is_dtensor):
        from torch.distributed._tensor.placement_types import Shard

        mock_is_dtensor.return_value = True

        mock_dtensor = Mock()
        mock_local_tensor = torch.randn(2, 16, 32)
        mock_dtensor.to_local.return_value = mock_local_tensor
        mock_dtensor.device_mesh.mesh_dim_names = ("dp_shard_cp",)
        mock_dtensor.placements = (Shard(0),)

        mock_dp_mesh = Mock()
        mock_dp_mesh.get_local_rank.return_value = 2
        mock_dp_mesh.size.return_value = 4
        mock_get_submesh.return_value = mock_dp_mesh

        local_tensor, start_expert, end_expert = get_expert_slice_for_rank(mock_dtensor, 8)

        assert torch.equal(local_tensor, mock_local_tensor)
        assert start_expert == 4
        assert end_expert == 6


class TestSplitExpertsWeightsDtensorAware:
    def test_regular_tensor(self):
        weight = torch.randn(4, 16, 32)
        n_experts = 4

        split_weights, expert_ids = split_experts_weights_dtensor_aware(weight, n_experts)

        assert len(split_weights) == 4
        assert expert_ids == [0, 1, 2, 3]
        for i, w in enumerate(split_weights):
            assert w.shape == (16, 32)
            assert torch.equal(w, weight[i])

    def test_shape_mismatch_error(self):
        weight = torch.randn(3, 16, 32)
        n_experts = 4

        with pytest.raises(ValueError, match="Expected local tensor first dimension to be 4"):
            split_experts_weights_dtensor_aware(weight, n_experts)

    @patch("nemo_automodel.components.moe.state_dict_utils.get_submesh")
    @patch("nemo_automodel.components.moe.state_dict_utils.get_expert_slice_for_rank")
    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    def test_dtensor_expert_splitting(self, mock_is_dtensor, mock_get_expert_slice, mock_get_submesh):
        mock_is_dtensor.return_value = True

        # Mock DTensor
        mock_weight = Mock()
        mock_weight.device_mesh = Mock()
        mock_weight.device_mesh.mesh_dim_names = ["ep", "dp"]
        mock_weight.placements = [Mock(), Mock()]

        # Mock local tensor slice
        mock_local_tensor = torch.randn(2, 16, 32)
        mock_get_expert_slice.return_value = (mock_local_tensor, 2, 4)

        # Mock submesh calls
        mock_submesh = Mock()
        mock_submesh.size.return_value = 1  # Make it seem like dp dimension is not used
        mock_get_submesh.return_value = mock_submesh

        split_weights, expert_ids = split_experts_weights_dtensor_aware(mock_weight, 8)

        assert len(split_weights) == 2
        assert expert_ids == [2, 3]


class TestValidateDtensorExpertSharding:
    def test_regular_tensor_valid(self):
        tensor = torch.randn(8, 16, 32)
        assert validate_dtensor_expert_sharding(tensor, 8, "test_tensor")

    def test_regular_tensor_invalid_shape(self):
        tensor = torch.randn(6, 16, 32)
        with pytest.raises(ValueError, match="test_tensor has shape 6 experts, expected 8"):
            validate_dtensor_expert_sharding(tensor, 8, "test_tensor")

    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    def test_dtensor_valid_sharded(self, mock_is_dtensor):
        mock_is_dtensor.return_value = True

        mock_dtensor = Mock()
        mock_dtensor.shape = [8, 16, 32]

        from torch.distributed._tensor.placement_types import Shard

        mock_placement = Shard(0)
        mock_dtensor.placements = [mock_placement]

        assert validate_dtensor_expert_sharding(mock_dtensor, 8, "test_tensor")

    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    def test_dtensor_invalid_shape(self, mock_is_dtensor):
        mock_is_dtensor.return_value = True

        mock_dtensor = Mock()
        mock_dtensor.shape = [6, 16, 32]

        with pytest.raises(ValueError, match="test_tensor global shape has 6 experts, expected 8"):
            validate_dtensor_expert_sharding(mock_dtensor, 8, "test_tensor")

    @patch("nemo_automodel.components.moe.state_dict_utils.is_dtensor")
    def test_dtensor_unsupported_placement(self, mock_is_dtensor):
        mock_is_dtensor.return_value = True

        mock_dtensor = Mock()
        mock_dtensor.shape = [8, 16, 32]

        # Create a placement that's neither Shard nor Replicate
        mock_placement = Mock()
        mock_placement.__class__ = type("MockPlacement", (), {})
        mock_dtensor.placements = [mock_placement]

        with pytest.raises(ValueError, match="test_tensor has unsupported DTensor placement"):
            validate_dtensor_expert_sharding(mock_dtensor, 8, "test_tensor")


class TestCreateDtensorFromLocal:
    def test_no_device_mesh(self):
        local_tensor = torch.randn(4, 16, 32)
        result = create_dtensor_from_local(local_tensor, None)
        assert torch.equal(result, local_tensor)

    @skip_if_no_gpu
    @patch("nemo_automodel.components.moe.state_dict_utils.get_submesh")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.current_device")
    def test_with_device_mesh_cuda(self, mock_current_device, mock_cuda_available, mock_get_submesh):
        mock_cuda_available.return_value = True
        mock_current_device.return_value = 0

        local_tensor = torch.randn(4, 16, 32)
        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep"]

        mock_submesh = Mock()
        mock_get_submesh.return_value = mock_submesh

        with patch("nemo_automodel.components.moe.state_dict_utils.DTensor") as mock_dtensor_class:
            mock_dtensor_instance = Mock()
            mock_dtensor_class.from_local.return_value = mock_dtensor_instance

            result = create_dtensor_from_local(local_tensor, mock_device_mesh, rank=0)

            assert result == mock_dtensor_instance

    @patch("nemo_automodel.components.moe.state_dict_utils.get_submesh")
    def test_with_complex_mesh_dimensions(self, mock_get_submesh):
        local_tensor = torch.randn(4, 16, 32)
        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep", "ep_shard", "ep_replicate"]

        def mock_get_submesh_side_effect(device_mesh, dims):
            mock_submesh = Mock()
            if "ep_shard" in dims or "ep_replicate" in dims:
                mock_submesh.size.return_value = 2
            else:
                mock_submesh.size.return_value = 1
            return mock_submesh

        mock_get_submesh.side_effect = mock_get_submesh_side_effect

        with patch("nemo_automodel.components.moe.state_dict_utils.DTensor") as mock_dtensor_class:
            mock_dtensor_instance = Mock()
            mock_dtensor_class.from_local.return_value = mock_dtensor_instance

            result = create_dtensor_from_local(local_tensor, mock_device_mesh)

            assert result == mock_dtensor_instance


class TestGetExpertRangeForRankFromMesh:
    def test_no_device_mesh(self):
        start, end = get_expert_range_for_rank_from_mesh(None, 8)
        assert start == 0
        assert end == 8

    @patch("nemo_automodel.components.moe.state_dict_utils.get_submesh")
    def test_with_device_mesh_even_distribution(self, mock_get_submesh):
        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep"]

        mock_ep_mesh = Mock()
        mock_ep_mesh.size.return_value = 4
        mock_ep_mesh.get_local_rank.return_value = 1
        mock_get_submesh.return_value = mock_ep_mesh

        start, end = get_expert_range_for_rank_from_mesh(mock_device_mesh, 8)

        assert start == 2
        assert end == 4

    @patch("nemo_automodel.components.moe.state_dict_utils.get_submesh")
    def test_with_device_mesh_uneven_distribution(self, mock_get_submesh):
        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep"]

        mock_ep_mesh = Mock()
        mock_ep_mesh.size.return_value = 3
        mock_ep_mesh.get_local_rank.return_value = 0  # First rank gets extra expert
        mock_get_submesh.return_value = mock_ep_mesh

        start, end = get_expert_range_for_rank_from_mesh(mock_device_mesh, 8)  # 8 experts, 3 ranks -> 3,3,2

        assert start == 0
        assert end == 3

    def test_without_ep_dimension(self):
        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["dp"]
        mock_device_mesh.size.return_value = 4
        mock_device_mesh.get_local_rank.return_value = 2

        start, end = get_expert_range_for_rank_from_mesh(mock_device_mesh, 8)

        assert start == 4
        assert end == 6


class TestShouldLoadExpertForRank:
    @patch("nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh")
    def test_expert_in_range(self, mock_get_expert_range):
        mock_get_expert_range.return_value = (2, 4)

        assert should_load_expert_for_rank(3, Mock(), 8)
        mock_get_expert_range.assert_called_once()

    @patch("nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh")
    def test_expert_not_in_range(self, mock_get_expert_range):
        mock_get_expert_range.return_value = (2, 4)

        assert not should_load_expert_for_rank(5, Mock(), 8)
        mock_get_expert_range.assert_called_once()

    @patch("nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh")
    def test_expert_at_boundary(self, mock_get_expert_range):
        mock_get_expert_range.return_value = (2, 4)

        assert should_load_expert_for_rank(2, Mock(), 8)  # Start is inclusive
        assert not should_load_expert_for_rank(4, Mock(), 8)  # End is exclusive
