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

"""Tests for parallel_styles.py - LoRA-aware tensor parallel strategies."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)

from nemo_automodel.components.distributed.parallel_styles import (
    ColwiseParallelLora,
    RowwiseParallelLora,
    SequenceParallelLora,
    _distribute_param,
    translate_to_lora,
)

# ==================== Fixtures ====================


@pytest.fixture
def mock_device_mesh():
    """Create a mock DeviceMesh for testing."""
    mesh = MagicMock(spec=DeviceMesh)
    mesh.device_type = "cuda"
    mesh.size = MagicMock(return_value=2)
    return mesh


@pytest.fixture
def mock_lora_linear_module():
    """Create a mock linear module with LoRA adapters."""

    class MockLoRALinear(nn.Module):
        def __init__(self, in_features=10, out_features=20, lora_dim=4):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.randn(out_features))

            # LoRA adapters
            self.lora_A = nn.Linear(in_features, lora_dim, bias=False)
            self.lora_B = nn.Linear(lora_dim, out_features, bias=False)

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight, self.bias) + self.lora_B(self.lora_A(x))

    return MockLoRALinear()


@pytest.fixture
def mock_linear_module():
    """Create a simple linear module without LoRA."""
    return nn.Linear(10, 20)


@pytest.fixture
def mock_embedding_module():
    """Create an embedding module."""
    return nn.Embedding(100, 50)


# ==================== Tests for _distribute_param ====================


class TestDistributeParam:
    """Tests for the _distribute_param utility function."""

    def test_distribute_param_basic(self, mock_device_mesh):
        """Test basic parameter distribution."""
        module = nn.Linear(10, 20)
        original_param = module.weight.clone()
        original_requires_grad = module.weight.requires_grad

        with patch("nemo_automodel.components.distributed.parallel_styles.distribute_tensor") as mock_distribute:
            # Mock distribute_tensor to return a tensor-like object
            mock_dtensor = torch.randn_like(original_param)
            mock_distribute.return_value = mock_dtensor

            _distribute_param(module, "weight", mock_device_mesh, src_data_rank=0, placements=[Shard(0)])

            # Verify distribute_tensor was called with correct args
            mock_distribute.assert_called_once()
            call_args = mock_distribute.call_args
            assert torch.allclose(call_args[0][0], original_param)
            assert call_args[0][1] == mock_device_mesh
            assert call_args[1]["src_data_rank"] == 0

            # Verify the parameter was updated
            assert isinstance(module.weight, nn.Parameter)

    def test_distribute_param_preserves_requires_grad_true(self, mock_device_mesh):
        """Test that requires_grad=True is preserved."""
        module = nn.Linear(10, 20)
        module.weight.requires_grad = True

        with patch("nemo_automodel.components.distributed.parallel_styles.distribute_tensor") as mock_distribute:
            mock_dtensor = torch.randn_like(module.weight)
            mock_distribute.return_value = mock_dtensor

            _distribute_param(module, "weight", mock_device_mesh, src_data_rank=0, placements=[Shard(0)])

            # Check that the parameter still has requires_grad=True
            assert module.weight.requires_grad is True

    def test_distribute_param_preserves_requires_grad_false(self, mock_device_mesh):
        """Test that requires_grad=False is preserved."""
        module = nn.Linear(10, 20)
        module.weight.requires_grad = False

        with patch("nemo_automodel.components.distributed.parallel_styles.distribute_tensor") as mock_distribute:
            mock_dtensor = torch.randn_like(module.weight)
            mock_distribute.return_value = mock_dtensor

            _distribute_param(module, "weight", mock_device_mesh, src_data_rank=0, placements=[Shard(0)])

            # Check that the parameter still has requires_grad=False
            assert module.weight.requires_grad is False

    def test_distribute_param_with_bias(self, mock_device_mesh):
        """Test distributing bias parameter."""
        module = nn.Linear(10, 20)

        with patch("nemo_automodel.components.distributed.parallel_styles.distribute_tensor") as mock_distribute:
            mock_dtensor = torch.randn_like(module.bias)
            mock_distribute.return_value = mock_dtensor

            _distribute_param(module, "bias", mock_device_mesh, src_data_rank=0, placements=[Replicate()])

            mock_distribute.assert_called_once()
            # Verify the bias was updated
            assert isinstance(module.bias, nn.Parameter)


# ==================== Tests for ColwiseParallelLora ====================


class TestColwiseParallelLora:
    """Tests for the ColwiseParallelLora class."""

    def test_inherits_from_colwise_parallel(self):
        """Test that ColwiseParallelLora inherits from ColwiseParallel."""
        assert issubclass(ColwiseParallelLora, ColwiseParallel)

    def test_partition_linear_fn_with_lora(self, mock_lora_linear_module, mock_device_mesh):
        """Test partitioning a linear module with LoRA adapters."""
        colwise_lora = ColwiseParallelLora()
        colwise_lora.src_data_rank = 0

        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            colwise_lora._partition_linear_fn("test_linear", mock_lora_linear_module, mock_device_mesh)

            # Should be called for weight, bias, lora_A.weight, and lora_B.weight
            assert mock_dist.call_count == 4

            # Check that all parameters were distributed with Shard(0)
            for call_item in mock_dist.call_args_list:
                assert call_item[0][2] == mock_device_mesh
                assert call_item[0][3] == 0
                assert call_item[0][4] == [Shard(0)]

    def test_partition_linear_fn_without_lora(self, mock_linear_module, mock_device_mesh):
        """Test partitioning a linear module without LoRA adapters."""
        colwise_lora = ColwiseParallelLora()
        colwise_lora.src_data_rank = 0

        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            colwise_lora._partition_linear_fn("test_linear", mock_linear_module, mock_device_mesh)

            # Should be called for weight and bias only
            assert mock_dist.call_count == 2

    def test_get_module_and_name_lora_a(self, mock_lora_linear_module):
        """Test that _get_module_and_name correctly handles lora_A.weight."""
        colwise_lora = ColwiseParallelLora()
        colwise_lora.src_data_rank = 0

        # Access the internal method through partition_linear_fn
        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            mock_device_mesh = MagicMock()
            colwise_lora._partition_linear_fn("test", mock_lora_linear_module, mock_device_mesh)

            # Find the call for lora_A.weight
            lora_a_calls = [c for c in mock_dist.call_args_list if c[0][0] == mock_lora_linear_module.lora_A]
            assert len(lora_a_calls) == 1
            assert lora_a_calls[0][0][1] == "weight"

    def test_get_module_and_name_lora_b(self, mock_lora_linear_module):
        """Test that _get_module_and_name correctly handles lora_B.weight."""
        colwise_lora = ColwiseParallelLora()
        colwise_lora.src_data_rank = 0

        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            mock_device_mesh = MagicMock()
            colwise_lora._partition_linear_fn("test", mock_lora_linear_module, mock_device_mesh)

            # Find the call for lora_B.weight
            lora_b_calls = [c for c in mock_dist.call_args_list if c[0][0] == mock_lora_linear_module.lora_B]
            assert len(lora_b_calls) == 1
            assert lora_b_calls[0][0][1] == "weight"

    def test_partition_embedding_fn(self, mock_embedding_module, mock_device_mesh):
        """Test partitioning an embedding module."""
        colwise_lora = ColwiseParallelLora()
        colwise_lora.src_data_rank = 0

        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            colwise_lora._partition_embedding_fn("test_embedding", mock_embedding_module, mock_device_mesh)

            # Should be called once for embedding weight
            mock_dist.assert_called_once()
            # Embedding should use Shard(1) for colwise
            assert mock_dist.call_args[0][4] == [Shard(1)]


# ==================== Tests for RowwiseParallelLora ====================


class TestRowwiseParallelLora:
    """Tests for the RowwiseParallelLora class."""

    def test_inherits_from_rowwise_parallel(self):
        """Test that RowwiseParallelLora inherits from RowwiseParallel."""
        assert issubclass(RowwiseParallelLora, RowwiseParallel)

    def test_partition_linear_fn_with_lora(self, mock_lora_linear_module, mock_device_mesh):
        """Test partitioning a linear module with LoRA adapters."""
        rowwise_lora = RowwiseParallelLora()
        rowwise_lora.src_data_rank = 0

        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            rowwise_lora._partition_linear_fn("test_linear", mock_lora_linear_module, mock_device_mesh)

            # Should be called for weight, bias, lora_A.weight, and lora_B.weight
            assert mock_dist.call_count == 4

            # Check weight is Shard(1)
            weight_call = mock_dist.call_args_list[0]
            assert weight_call[0][1] == "weight"
            assert weight_call[0][4] == [Shard(1)]

            # Check bias is Replicate()
            bias_call = mock_dist.call_args_list[1]
            assert bias_call[0][1] == "bias"
            assert bias_call[0][4] == [Replicate()]

            # Check LoRA adapters are Shard(1)
            lora_a_call = mock_dist.call_args_list[2]
            assert lora_a_call[0][4] == [Shard(1)]
            lora_b_call = mock_dist.call_args_list[3]
            assert lora_b_call[0][4] == [Shard(1)]

    def test_partition_linear_fn_without_lora(self, mock_linear_module, mock_device_mesh):
        """Test partitioning a linear module without LoRA adapters."""
        rowwise_lora = RowwiseParallelLora()
        rowwise_lora.src_data_rank = 0

        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            rowwise_lora._partition_linear_fn("test_linear", mock_linear_module, mock_device_mesh)

            # Should be called for weight and bias only
            assert mock_dist.call_count == 2

    def test_partition_linear_fn_without_bias(self, mock_device_mesh):
        """Test partitioning a linear module without bias."""
        module = nn.Linear(10, 20, bias=False)
        rowwise_lora = RowwiseParallelLora()
        rowwise_lora.src_data_rank = 0

        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            rowwise_lora._partition_linear_fn("test_linear", module, mock_device_mesh)

            # Should only be called for weight
            mock_dist.assert_called_once()
            assert mock_dist.call_args[0][1] == "weight"

    def test_partition_embedding_fn(self, mock_embedding_module, mock_device_mesh):
        """Test partitioning an embedding module."""
        rowwise_lora = RowwiseParallelLora()
        rowwise_lora.src_data_rank = 0

        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param") as mock_dist:
            rowwise_lora._partition_embedding_fn("test_embedding", mock_embedding_module, mock_device_mesh)

            # Should be called once for embedding weight
            mock_dist.assert_called_once()
            # Embedding should use Shard(0) for rowwise
            assert mock_dist.call_args[0][4] == [Shard(0)]


# ==================== Tests for SequenceParallelLora ====================


class TestSequenceParallelLora:
    """Tests for the SequenceParallelLora class."""

    def test_inherits_from_sequence_parallel(self):
        """Test that SequenceParallelLora inherits from SequenceParallel."""
        assert issubclass(SequenceParallelLora, SequenceParallel)

    def test_replicate_module_fn_simple_module(self, mock_linear_module, mock_device_mesh):
        """Test replicating a simple module."""
        seq_lora = SequenceParallelLora()

        with patch("nemo_automodel.components.distributed.parallel_styles.DTensor") as mock_dtensor:
            # Setup mock DTensor.from_local to return actual tensors
            def mock_from_local(tensor, mesh, placements, run_check=False):
                # Return a tensor with the same shape as input
                return torch.randn_like(tensor)

            mock_dtensor.from_local.side_effect = mock_from_local

            seq_lora._replicate_module_fn("test_linear", mock_linear_module, mock_device_mesh)

            # Should be called for each parameter (weight and bias)
            assert mock_dtensor.from_local.call_count == 2

            # Check that Replicate() placement was used
            for call_item in mock_dtensor.from_local.call_args_list:
                placements = call_item[0][2]
                assert len(placements) == 1
                assert isinstance(placements[0], Replicate)

            # Verify parameters were updated
            assert isinstance(mock_linear_module.weight, nn.Parameter)
            assert isinstance(mock_linear_module.bias, nn.Parameter)

    def test_replicate_module_fn_preserves_requires_grad(self, mock_linear_module, mock_device_mesh):
        """Test that requires_grad is preserved during replication."""
        seq_lora = SequenceParallelLora()

        # Set different requires_grad values
        mock_linear_module.weight.requires_grad = True
        mock_linear_module.bias.requires_grad = False

        with patch("nemo_automodel.components.distributed.parallel_styles.DTensor") as mock_dtensor:
            # Setup mock DTensor.from_local to return actual tensors
            def mock_from_local(tensor, mesh, placements, run_check=False):
                return torch.randn_like(tensor)

            mock_dtensor.from_local.side_effect = mock_from_local

            seq_lora._replicate_module_fn("test", mock_linear_module, mock_device_mesh)

            # Check that requires_grad was preserved
            assert mock_linear_module.weight.requires_grad is True
            assert mock_linear_module.bias.requires_grad is False


# ==================== Tests for translate_to_lora ====================


class TestTranslateToLora:
    """Tests for the translate_to_lora function."""

    def test_translate_colwise_parallel(self):
        """Test translating ColwiseParallel to ColwiseParallelLora."""
        plan = ColwiseParallel()
        result = translate_to_lora(plan)

        assert isinstance(result, ColwiseParallelLora)
        assert result.__class__ == ColwiseParallelLora

    def test_translate_rowwise_parallel(self):
        """Test translating RowwiseParallel to RowwiseParallelLora."""
        plan = RowwiseParallel()
        result = translate_to_lora(plan)

        assert isinstance(result, RowwiseParallelLora)
        assert result.__class__ == RowwiseParallelLora

    def test_translate_sequence_parallel(self):
        """Test translating SequenceParallel to SequenceParallelLora."""
        plan = SequenceParallel()
        result = translate_to_lora(plan)

        assert isinstance(result, SequenceParallelLora)
        assert result.__class__ == SequenceParallelLora

    def test_translate_unknown_type(self):
        """Test that unknown types remain unchanged."""

        class UnknownParallel:
            pass

        plan = UnknownParallel()
        result = translate_to_lora(plan)

        # Should remain as UnknownParallel
        assert isinstance(result, UnknownParallel)
        assert result.__class__ == UnknownParallel

    def test_translate_preserves_attributes(self):
        """Test that translation preserves plan attributes."""
        plan = ColwiseParallel()
        plan.custom_attr = "test_value"
        plan.src_data_rank = 5

        result = translate_to_lora(plan)

        # Check that attributes are preserved
        assert hasattr(result, "custom_attr")
        assert result.custom_attr == "test_value"
        assert result.src_data_rank == 5

    def test_translate_returns_same_object(self):
        """Test that translate_to_lora modifies the object in place."""
        plan = ColwiseParallel()
        original_id = id(plan)

        result = translate_to_lora(plan)

        # Should be the same object (modified in place)
        assert id(result) == original_id

    def test_translate_idempotent(self):
        """Test that translating twice doesn't cause issues."""
        plan = ColwiseParallel()

        result1 = translate_to_lora(plan)
        result2 = translate_to_lora(result1)

        # Should still be ColwiseParallelLora
        assert isinstance(result2, ColwiseParallelLora)
        assert result2.__class__ == ColwiseParallelLora


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for parallel styles with LoRA."""

    def test_colwise_and_rowwise_compatibility(self, mock_lora_linear_module, mock_device_mesh):
        """Test that ColwiseParallelLora and RowwiseParallelLora can work together."""
        colwise = ColwiseParallelLora()
        colwise.src_data_rank = 0

        rowwise = RowwiseParallelLora()
        rowwise.src_data_rank = 0

        # Both should be able to partition the same module
        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param"):
            colwise._partition_linear_fn("test", mock_lora_linear_module, mock_device_mesh)
            rowwise._partition_linear_fn("test", mock_lora_linear_module, mock_device_mesh)

    def test_translate_chain(self):
        """Test translating multiple plans in sequence."""
        plans = [
            ColwiseParallel(),
            RowwiseParallel(),
            SequenceParallel(),
        ]

        translated = [translate_to_lora(p) for p in plans]

        assert isinstance(translated[0], ColwiseParallelLora)
        assert isinstance(translated[1], RowwiseParallelLora)
        assert isinstance(translated[2], SequenceParallelLora)

    def test_module_without_lora_attributes(self, mock_linear_module, mock_device_mesh):
        """Test that modules without LoRA attributes don't cause errors."""
        colwise = ColwiseParallelLora()
        colwise.src_data_rank = 0

        # Should not raise an error
        with patch("nemo_automodel.components.distributed.parallel_styles._distribute_param"):
            colwise._partition_linear_fn("test", mock_linear_module, mock_device_mesh)
