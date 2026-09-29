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

"""Integration tests to verify the strategy pattern maintains backward compatibility."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from nemo_automodel.components.distributed.parallelizer import (
    DefaultParallelizationStrategy,
    NemotronHParallelizationStrategy,
    fsdp2_strategy_parallelize,
    get_parallelization_strategy,
)


class MockStandardModel(nn.Module):
    """Mock standard model (Llama-like)."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=8,
            num_key_value_heads=8,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])


class MockNemotronModel(nn.Module):
    """Mock NemotronH model."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=8,
            num_key_value_heads=8,
        )
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([self._create_layer() for _ in range(2)])
        self.supports = SimpleNamespace(
            mtp_enabled=False,
            supports_mtp_cp=False,
            supports_mtp_cp_pp=False,
        )
        self.__class__.__name__ = "NemotronHForCausalLM"

    def _create_layer(self):
        layer = nn.Module()
        layer.block_type = "mlp"
        return layer


@pytest.fixture
def mock_device_mesh():
    """Create a mock device mesh."""
    mesh = MagicMock()
    mesh.device_type = "cuda"
    mesh.mesh_dim_names = ("dp_replicate", "dp_shard", "cp", "tp")

    # Mock submeshes
    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 1
    dp_mesh = MagicMock()
    dp_mesh.size.return_value = 2
    dp_rep_mesh = MagicMock()
    dp_rep_mesh.size.return_value = 1
    dp_shard_cp_mesh = MagicMock()
    dp_shard_cp_mesh.size.return_value = 2

    # Support get_flat_mesh / get_submesh lookups
    mesh._get_root_mesh = MagicMock(return_value=mesh)
    mesh._flatten_mapping = {"dp_shard_cp": dp_shard_cp_mesh}

    mesh.__getitem__.side_effect = lambda key: {
        "tp": tp_mesh,
        "dp_replicate": dp_rep_mesh,
        ("dp_replicate", "dp_shard_cp"): dp_mesh,
    }.get(key, dp_mesh)

    return mesh


def test_strategy_selection_standard_model():
    """Test that standard models use DefaultParallelizationStrategy."""
    model = MockStandardModel()
    strategy = get_parallelization_strategy(model)

    assert isinstance(strategy, DefaultParallelizationStrategy)
    assert not isinstance(strategy, NemotronHParallelizationStrategy)


def test_strategy_selection_nemotron_model():
    """Test that NemotronH models use NemotronHParallelizationStrategy."""
    model = MockNemotronModel()
    strategy = get_parallelization_strategy(model)

    assert isinstance(strategy, NemotronHParallelizationStrategy)
    assert not isinstance(strategy, DefaultParallelizationStrategy)


@patch("torch.distributed.get_process_group_ranks", return_value=[0])
@patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
@patch("nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively")
@patch("nemo_automodel.components.distributed.parallelizer._extract_model_layer_groups")
@patch("nemo_automodel.components.distributed.parallelizer._get_parallel_plan")
def test_backward_compatibility_standard_model(
    mock_get_plan, mock_extract_layer_groups, mock_apply_fsdp, mock_fully_shard, mock_gpgr, mock_device_mesh
):
    """Test that the refactored code maintains backward compatibility for standard models."""
    mock_fully_shard.side_effect = lambda model, **kwargs: model
    mock_extract_layer_groups.return_value = {"language": []}
    mock_get_plan.return_value = {}

    model = MockStandardModel()

    result = fsdp2_strategy_parallelize(
        model=model,
        device_mesh=mock_device_mesh,
        sequence_parallel=False,
        activation_checkpointing=False,
    )

    # Should return the model unchanged
    assert result is model

    # Should have called the expected functions for standard flow
    mock_extract_layer_groups.assert_called_once_with(model)
    mock_apply_fsdp.assert_called_once()
    mock_fully_shard.assert_called()


@patch("torch.distributed.get_process_group_ranks", return_value=[0])
@patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
@patch("nemo_automodel.components.distributed.parallelizer.parallelize_module")
def test_backward_compatibility_nemotron_model(mock_parallelize_module, mock_fully_shard, mock_gpgr, mock_device_mesh):
    """Test that the refactored code maintains backward compatibility for NemotronH models."""
    mock_fully_shard.side_effect = lambda model, **kwargs: model

    model = MockNemotronModel()

    result = fsdp2_strategy_parallelize(
        model=model,
        device_mesh=mock_device_mesh,
        sequence_parallel=False,
        activation_checkpointing=False,
    )

    # Should return the model unchanged
    assert result is model

    # Should have called NemotronH-specific functions
    if mock_device_mesh["tp"].size() > 1:
        mock_parallelize_module.assert_called()  # For TP plans
    else:
        mock_parallelize_module.assert_not_called()
    mock_fully_shard.assert_called()  # For FSDP


def test_function_signature_preserved():
    """Test that the main function signature is preserved."""
    import inspect

    sig = inspect.signature(fsdp2_strategy_parallelize)

    # All original parameters should be present
    expected_params = {
        "model",
        "device_mesh",
        "mp_policy",
        "offload_policy",
        "sequence_parallel",
        "activation_checkpointing",
        "activation_checkpointing_scope",
        "tp_shard_plan",
        "dp_replicate_mesh_name",
        "dp_shard_cp_mesh_name",
        "tp_mesh_name",
    }

    actual_params = set(sig.parameters.keys())
    assert expected_params <= actual_params  # All expected params are present

    # Check that key defaults are preserved
    assert sig.parameters["sequence_parallel"].default is False
    assert sig.parameters["activation_checkpointing"].default is False


def test_no_runtime_errors_with_different_model_types(mock_device_mesh):
    """Test that both model types can be processed without runtime errors."""
    with (
        patch("torch.distributed.get_process_group_ranks", return_value=[0]),
        patch(
            "nemo_automodel.components.distributed.parallelizer.fully_shard", side_effect=lambda model, **kwargs: model
        ),
        patch("nemo_automodel.components.distributed.parallelizer.parallelize_module"),
    ):
        with patch("nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively"):
            with patch(
                "nemo_automodel.components.distributed.parallelizer._extract_model_layer_groups",
                return_value={"language": []},
            ):
                with patch("nemo_automodel.components.distributed.parallelizer._get_parallel_plan", return_value={}):
                    # Test standard model
                    standard_model = MockStandardModel()
                    result1 = fsdp2_strategy_parallelize(
                        model=standard_model,
                        device_mesh=mock_device_mesh,
                    )
                    assert result1 is standard_model

                    # Test NemotronH model
                    nemotron_model = MockNemotronModel()
                    result2 = fsdp2_strategy_parallelize(
                        model=nemotron_model,
                        device_mesh=mock_device_mesh,
                    )
                    assert result2 is nemotron_model
