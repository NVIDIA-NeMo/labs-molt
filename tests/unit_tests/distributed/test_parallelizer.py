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

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

import nemo_automodel.components.distributed.parallelizer as parallelizer
from nemo_automodel.components.distributed.optimized_tp_plans import _get_class_qualname
from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce
from nemo_automodel.components.distributed.parallelizer import _attention_is_head_sharded, _extract_model_layer_groups, _extract_model_layers, _filter_layer_groups_for_activation_checkpointing, _get_parallel_plan, _update_attention_head_counts_for_tp, apply_fsdp2_sharding_recursively, get_hf_tp_shard_plan, import_class_from_path


def test_fsdp_accumulated_grad_guard_only_handles_missing_unsharded_param(monkeypatch):
    """The FSDP2 guard only handles the exact missing lazy unsharded tensor case."""
    calls = {"count": 0}

    class FakeFSDPParam:
        def __init__(self, mode="ok"):
            self.mode = mode

        def to_accumulated_grad_if_needed(self):
            calls["count"] += 1
            if self.mode == "missing":
                return self._unsharded_param.grad
            if self.mode == "other":
                raise AttributeError("different attribute")
            return "called"

    fake_module = SimpleNamespace(FSDPParam=FakeFSDPParam)
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp._fully_shard._fsdp_param", fake_module)

    parallelizer._patch_fsdp_accumulated_grad_guard()

    param = FakeFSDPParam()
    assert param.to_accumulated_grad_if_needed() == "called"
    assert calls["count"] == 1

    param.mode = "missing"
    assert param.to_accumulated_grad_if_needed() is None
    assert calls["count"] == 2

    param.mode = "other"
    with pytest.raises(AttributeError, match="different attribute"):
        param.to_accumulated_grad_if_needed()
    assert calls["count"] == 3

    wrapped = FakeFSDPParam.to_accumulated_grad_if_needed
    parallelizer._patch_fsdp_accumulated_grad_guard()
    assert FakeFSDPParam.to_accumulated_grad_if_needed is wrapped


class MockModel(nn.Module):
    """Mock model for testing purposes."""

    def __init__(self, model_type="llama", num_attention_heads=8, num_key_value_heads=8):
        super().__init__()
        if model_type == "baichuan2":
            self.config = SimpleNamespace(
                num_attention_heads=num_attention_heads,
            )
        else:
            self.config = SimpleNamespace(
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )

        # Create mock model as a proper nn.Module so it gets picked up by named_children()
        class MockInnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockModel._create_mock_layer() for _ in range(2)])

        self.model = MockInnerModel()

        if model_type == "gemma3":
            self.language_model = SimpleNamespace()
            self.language_model.layers = self.model.layers
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                )
            )

    @staticmethod
    def _create_mock_layer():
        """Create a mock transformer layer."""
        layer = nn.Module()
        layer.mlp = nn.Linear(10, 10)  # Simple MLP for testing
        return layer

    def forward(self, x):
        return x


class MockGemma3Model(nn.Module):
    """Mock Gemma3 model that simulates Gemma3ForConditionalGeneration."""

    def __init__(self, num_attention_heads=8, num_key_value_heads=8):
        # Explicitly call nn.Module.__init__() to avoid MRO issues with multiple inheritance
        nn.Module.__init__(self)

        # Set up config structure for Gemma3 with both top-level and nested structure
        self.config = SimpleNamespace(
            # Top-level attributes for regular model compatibility
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            # Nested structure for Gemma3
            text_config=SimpleNamespace(
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            ),
        )

        # Create mock model as a proper nn.Module so it gets picked up by named_children()
        class MockInnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockGemma3Model._create_mock_layer() for _ in range(2)])

        self.model = MockInnerModel()

        # Create language_model structure expected by Gemma3 as a proper PyTorch module
        class LanguageModel(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers

        self.language_model = LanguageModel(self.model.layers)

    @staticmethod
    def _create_mock_layer():
        """Create a mock transformer layer."""
        layer = nn.Module()
        layer.mlp = nn.Linear(10, 10)  # Simple MLP for testing
        return layer

    def forward(self, x):
        return x


def create_gemma3_mock():
    """Factory function to create a mock that passes Gemma3 type checks."""

    # Create a simple hybrid class like in the functional test
    class MockGemma3ModelWithTypeCheck(MockGemma3Model, Gemma3ForConditionalGeneration):
        """Mock Gemma3 model that properly inherits from Gemma3ForConditionalGeneration."""

        def __init__(self, num_attention_heads=8, num_key_value_heads=8):
            # Explicitly call only MockGemma3Model.__init__ to avoid MRO issues
            MockGemma3Model.__init__(self, num_attention_heads, num_key_value_heads)

    # Create an instance of the hybrid class
    mock = MockGemma3ModelWithTypeCheck()
    return mock


class _CheckpointWrapped(nn.Module):
    """Minimal checkpoint wrapper used by BAGEL activation-checkpointing tests."""

    def __init__(self, inner, **kwargs):
        super().__init__()
        self._checkpoint_wrapped_module = inner
        self.kwargs = kwargs

    def forward(self, x):
        return self._checkpoint_wrapped_module(x)


def _make_bagel_model(num_language_layers: int = 2, num_vision_layers: int = 3):
    """Build the nested layer containers used by BAGEL without importing BAGEL."""

    class BagelForUnifiedMultimodal(nn.Module):
        """Stand-in with the exact class name used by the production mapper."""

        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.model = nn.Module()
            self.model.language_model.model.layers = nn.ModuleList([_FakeLayer() for _ in range(num_language_layers)])
            self.model.vit_model = nn.Module()
            self.model.vit_model.vision_model = nn.Module()
            self.model.vit_model.vision_model.encoder = nn.Module()
            self.model.vit_model.vision_model.encoder.layers = nn.ModuleList(
                [_FakeLayer() for _ in range(num_vision_layers)]
            )

    return BagelForUnifiedMultimodal()


@pytest.fixture
def mock_device_mesh_fsdp2():
    """Create a mock device mesh."""
    mesh = MagicMock(spec=DeviceMesh)

    # Mock device_type to return a valid string
    mesh.device_type = "cuda"

    # Mock submeshes
    dp_replicate_mesh = MagicMock()
    dp_shard_mesh = MagicMock()
    cp_mesh = MagicMock()
    tp_mesh = MagicMock()

    dp_replicate_mesh.size.return_value = 1
    dp_shard_mesh.size.return_value = 2
    tp_mesh.size.return_value = 1
    cp_mesh.size.return_value = 1

    dp_replicate_mesh.ndim = 1
    dp_shard_mesh.ndim = 1
    tp_mesh.ndim = 1
    cp_mesh.ndim = 1

    # Configure mesh access
    mesh.__getitem__.side_effect = lambda key: {
        "dp_replicate": dp_replicate_mesh,
        "dp_shard": dp_shard_mesh,
        "tp": tp_mesh,
        "cp": cp_mesh,
    }[key]

    return mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh, cp_mesh


@pytest.fixture
def mock_device_mesh_megatron_fsdp():
    """Create a mock device mesh."""
    mesh = MagicMock(spec=DeviceMesh)

    # Mock device_type to return a valid string
    mesh.device_type = "cuda"

    # Mock submeshes
    dp_mesh = MagicMock()
    cp_mesh = MagicMock()
    tp_mesh = MagicMock()

    dp_mesh.size.return_value = 2
    tp_mesh.size.return_value = 1
    cp_mesh.size.return_value = 1

    dp_mesh.ndim = 1
    tp_mesh.ndim = 1
    cp_mesh.ndim = 1

    # Configure mesh access
    mesh.__getitem__.side_effect = lambda key: {
        "dp": dp_mesh,
        "tp": tp_mesh,
        "cp": cp_mesh,
        "dp_cp": dp_mesh,
    }[key]

    return mesh, dp_mesh, tp_mesh, cp_mesh


@pytest.fixture
def mock_distributed_env(monkeypatch):
    """Mock the distributed environment."""
    # Mock torch.distributed
    dist_mock = SimpleNamespace()
    dist_mock.is_initialized = lambda: True
    dist_mock.get_rank = lambda: 0
    dist_mock.get_world_size = lambda: 2

    # Add device_mesh structure to dist_mock
    device_mesh_mock = SimpleNamespace()
    dist_mock.device_mesh = device_mesh_mock

    # Mock device mesh resources
    mesh_resources_mock = SimpleNamespace()
    mesh_resources_mock.root_to_flatten_mapping = MagicMock()
    mesh_resources_mock.root_to_flatten_mapping.get.return_value = {}
    device_mesh_mock._mesh_resources = mesh_resources_mock

    # Add FSDP structure to dist_mock
    fsdp_mock = SimpleNamespace()
    fsdp_mock.MixedPrecisionPolicy = MagicMock()
    fsdp_mock.CPUOffloadPolicy = MagicMock()
    fsdp_mock.fully_shard = MagicMock(side_effect=lambda model, **kwargs: model)
    dist_mock.fsdp = fsdp_mock

    # Add algorithms structure to dist_mock
    checkpoint_wrapper_mock = SimpleNamespace()
    checkpoint_wrapper_mock.checkpoint_wrapper = MagicMock(side_effect=lambda x: x)

    # Add tensor parallel structure to dist_mock
    tp_parallel_mock = SimpleNamespace()
    tp_parallel_mock.parallelize_module = MagicMock()
    tp_parallel_mock.checkpoint_wrapper = checkpoint_wrapper_mock.checkpoint_wrapper

    tensor_mock = SimpleNamespace()
    tensor_mock.parallel = tp_parallel_mock
    dist_mock.tensor = tensor_mock

    checkpoint_mock = SimpleNamespace()
    checkpoint_mock.checkpoint_wrapper = checkpoint_wrapper_mock

    algorithms_mock = SimpleNamespace()
    algorithms_mock._checkpoint = checkpoint_mock
    dist_mock.algorithms = algorithms_mock

    # Apply patches
    monkeypatch.setattr("torch.distributed", dist_mock, raising=False)
    # Patch the imported functions directly in the parallelizer module
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.fully_shard", fsdp_mock.fully_shard, raising=False
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.parallelize_module",
        tp_parallel_mock.parallelize_module,
        raising=False,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.checkpoint_wrapper",
        checkpoint_wrapper_mock.checkpoint_wrapper,
        raising=False,
    )
    # Whole-block/sub-module wrapping now lives in activation_checkpointing.py and
    # uses that module's checkpoint_wrapper, so patch it there too.
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.activation_checkpointing.checkpoint_wrapper",
        checkpoint_wrapper_mock.checkpoint_wrapper,
        raising=False,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer._mesh_resources", mesh_resources_mock, raising=False
    )

    return {
        "dist": dist_mock,
        "mesh_resources": mesh_resources_mock,
        "fsdp": fsdp_mock,
        "tensor_parallel": tp_parallel_mock,
    }


@pytest.fixture
def mock_optimized_tp_plans(monkeypatch):
    """Mock the PARALLELIZE_FUNCTIONS dictionary."""
    mock_plans = {}

    def mock_llama_plan(model, sequence_parallel=False):
        return {"model.layers.0.self_attn.q_proj": ColwiseParallel()}

    def mock_gemma3_plan(model, sequence_parallel=False):
        return {"language_model.layers.0.self_attn.q_proj": ColwiseParallel()}

    # Mock the import to avoid actual dependency
    with patch("nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS", mock_plans):
        # Add mock functions for different model types
        mock_plans[type(MockModel())] = mock_llama_plan
        mock_plans[type(create_gemma3_mock())] = mock_gemma3_plan
        yield mock_plans


class FakeMegatronFSDPMixedPrecisionPolicy:
    """Stand-in for megatron_fsdp.MixedPrecisionPolicy (megatron-fsdp==0.5.0)."""

    def __init__(self, *, main_params_dtype, main_grads_dtype, grad_comm_dtype):
        self.main_params_dtype = main_params_dtype
        self.main_grads_dtype = main_grads_dtype
        self.grad_comm_dtype = grad_comm_dtype


class TestMegatronFSDPStrategyParallelize:
    """Test suite for megatron_fsdp_strategy_parallelize function."""

    @pytest.fixture
    def mock_megatron_fsdp_env(self, monkeypatch):
        """Mock Megatron FSDP environment and dependencies (megatron-fsdp==0.5.0 API)."""

        def fully_shard_050(
            *,
            module,
            optimizer,
            fsdp_unit_modules,
            device_mesh,
            dp_shard_dim,
            tp_dim,
            zero_dp_strategy,
            init_model_with_meta_device,
            mixed_precision_policy,
            overlap_grad_reduce,
            overlap_param_gather,
            report_nan_in_param_grad,
            average_in_collective,
            disable_bucketing,
            calculate_per_token_loss,
            keep_fp8_transpose_cache,
            nccl_ub,
            fsdp_double_buffer,
        ):
            del (
                module,
                optimizer,
                fsdp_unit_modules,
                device_mesh,
                dp_shard_dim,
                tp_dim,
                zero_dp_strategy,
                init_model_with_meta_device,
                mixed_precision_policy,
                overlap_grad_reduce,
                overlap_param_gather,
                report_nan_in_param_grad,
                average_in_collective,
                disable_bucketing,
                calculate_per_token_loss,
                keep_fp8_transpose_cache,
                nccl_ub,
                fsdp_double_buffer,
            )

        # Mock megatron_fsdp module
        megatron_fsdp_mock = SimpleNamespace()
        megatron_fsdp_mock.fully_shard = create_autospec(
            fully_shard_050,
            return_value=(MagicMock(), None),
        )

        # Mock HAVE_MEGATRON_FSDP flag
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.HAVE_MEGATRON_FSDP", True, raising=False
        )
        monkeypatch.setattr(
            parallelizer,
            "MegatronFSDPMixedPrecisionPolicy",
            FakeMegatronFSDPMixedPrecisionPolicy,
            raising=True,
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.megatron_fsdp_fully_shard",
            megatron_fsdp_mock.fully_shard,
            raising=False,
        )

        # Mock parallelize_module
        parallelize_module_mock = MagicMock()
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.parallelize_module",
            parallelize_module_mock,
            raising=False,
        )

        # Mock import_classes_from_paths
        import_classes_mock = MagicMock(return_value=[])
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.import_classes_from_paths",
            import_classes_mock,
            raising=False,
        )
        return {
            "megatron_fsdp": megatron_fsdp_mock,
            "parallelize_module": parallelize_module_mock,
            "import_classes": import_classes_mock,
        }


class TestUtilityFunctions:
    """Test utility functions used by fsdp2_strategy_parallelize."""

    def test_import_class_from_path_success(self):
        """Test successful import of class from path."""
        # Test importing a real class
        cls = import_class_from_path("torch.nn.Linear")
        assert cls is torch.nn.Linear

    def test_import_class_from_path_error(self):
        """Test error handling in import_class_from_path."""
        with pytest.raises(Exception):
            import_class_from_path("nonexistent.module.Class")


class TestGetHfTpShardPlan:
    """Test suite for get_hf_tp_shard_plan function."""

    def test_standard_model_with_class_tp_plan(self):
        """Test standard model with TP plan defined on model class."""
        model = MockModel()
        model_cls = type(model)

        # Add TP plan to model class
        model_cls._tp_plan = {
            "layers.0.self_attn.q_proj": "colwise",
            "layers.0.self_attn.k_proj": "colwise",
            "layers.0.mlp.gate_proj": "colwise",
        }

        # Mock config for tied embeddings test
        model.config.tie_word_embeddings = True

        try:
            result = get_hf_tp_shard_plan(model)

            # Verify TP plan was applied correctly
            assert len(result) > 0
            assert "layers.0.self_attn.q_proj" in result
            assert isinstance(result["layers.0.self_attn.q_proj"], ColwiseParallel)

        finally:
            # Clean up class attribute
            if hasattr(model_cls, "_tp_plan"):
                delattr(model_cls, "_tp_plan")

    def test_standard_model_with_instance_tp_plan(self):
        """Test standard model with TP plan defined on model instance."""
        model = MockModel()

        # Add TP plan to model instance
        model._tp_plan = {
            "layers.0.self_attn.q_proj": "rowwise",
            "layers.0.mlp.down_proj": "rowwise",
        }
        model.config.tie_word_embeddings = False

        result = get_hf_tp_shard_plan(model)

        # Verify TP plan was applied correctly
        assert len(result) > 0
        assert "layers.0.self_attn.q_proj" in result
        assert isinstance(result["layers.0.self_attn.q_proj"], RowwiseParallel)

        # Should add embed_tokens since tie_word_embeddings=False
        assert "model.embed_tokens" in result
        assert isinstance(result["model.embed_tokens"], RowwiseParallel)

    def test_standard_model_with_inner_model_tp_plan(self):
        """Test standard model with TP plan defined on inner model."""
        model = MockModel()

        # Add TP plan to inner model
        model.model._tp_plan = {
            "layers.0.self_attn.v_proj": "colwise_rep",
            "layers.0.self_attn.o_proj": "rowwise_rep",
        }
        model.config.tie_word_embeddings = False

        result = get_hf_tp_shard_plan(model)

        # Verify TP plan was applied correctly with model prefix
        assert len(result) > 0
        assert "model.layers.0.self_attn.v_proj" in result
        assert isinstance(result["model.layers.0.self_attn.v_proj"], ColwiseParallel)
        assert "model.layers.0.self_attn.o_proj" in result
        assert isinstance(result["model.layers.0.self_attn.o_proj"], RowwiseParallel)

    def test_multiple_tp_plan_sources_precedence(self):
        """Test precedence when TP plans exist in multiple places."""
        model = MockModel()
        model_cls = type(model)

        # Add TP plans to all possible sources
        model_cls._tp_plan = {"layers.0.self_attn.q_proj": "colwise"}
        model._tp_plan = {"layers.0.self_attn.k_proj": "rowwise"}
        model.model._tp_plan = {"layers.0.self_attn.v_proj": "colwise_rep"}
        model.config.tie_word_embeddings = True

        try:
            result = get_hf_tp_shard_plan(model)

            # All plans should be merged
            assert "layers.0.self_attn.q_proj" in result  # from class
            assert "layers.0.self_attn.k_proj" in result  # from instance
            assert "model.layers.0.self_attn.v_proj" in result  # from inner model with prefix

            # Instance plan should take precedence over class plan if same key exists
            assert isinstance(result["layers.0.self_attn.q_proj"], ColwiseParallel)
        finally:
            # Clean up class attribute
            if hasattr(model_cls, "_tp_plan"):
                delattr(model_cls, "_tp_plan")

    def test_lm_head_optimization(self):
        """Test special optimization for lm_head with colwise_rep."""
        model = MockModel()

        model._tp_plan = {
            "lm_head": "colwise_rep",
            "layers.0.self_attn.q_proj": "colwise",
        }
        model.config.tie_word_embeddings = False

        result = get_hf_tp_shard_plan(model)

        # Verify lm_head gets special optimization
        assert "lm_head" in result
        lm_head_parallel = result["lm_head"]
        assert isinstance(lm_head_parallel, ColwiseParallel)
        # The optimization should set output_layouts=Shard(-1) and use_local_output=False
        assert not lm_head_parallel.use_local_output

    def test_lm_head_no_optimization_when_tied(self):
        """Test lm_head doesn't get optimization when embeddings are tied."""
        model = MockModel()

        model._tp_plan = {
            "lm_head": "colwise_rep",
            "layers.0.self_attn.q_proj": "colwise",
        }
        model.config.tie_word_embeddings = True

        result = get_hf_tp_shard_plan(model)

        # Verify lm_head gets standard translation, not optimization
        assert "lm_head" in result
        lm_head_parallel = result["lm_head"]
        assert isinstance(lm_head_parallel, ColwiseParallel)

    def test_embed_tokens_added_when_not_tied(self):
        """Test embed_tokens is added when tie_word_embeddings=False."""
        model = MockModel()

        model._tp_plan = {"layers.0.self_attn.q_proj": "colwise"}
        model.config.tie_word_embeddings = False

        result = get_hf_tp_shard_plan(model)

        assert "model.embed_tokens" in result
        assert isinstance(result["model.embed_tokens"], RowwiseParallel)

    def test_parallel_style_translations(self):
        """Test all parallel style string translations."""
        model = MockModel()

        model._tp_plan = {
            "layer1": "colwise",
            "layer2": "rowwise",
            "layer3": "colwise_rep",
            "layer4": "rowwise_rep",
            "layer5": "sequence_parallel",
            "layer6": "replicated_with_grad_allreduce",
        }
        model.config.tie_word_embeddings = True

        result = get_hf_tp_shard_plan(model)

        assert isinstance(result["layer1"], ColwiseParallel)
        assert isinstance(result["layer2"], RowwiseParallel)
        assert isinstance(result["layer3"], ColwiseParallel)
        assert isinstance(result["layer4"], RowwiseParallel)
        assert isinstance(result["layer5"], SequenceParallel)
        assert isinstance(result["layer6"], ReplicatedWithGradAllReduce)

    def test_no_tp_plan_error(self):
        """Test error when no TP plan is found."""
        model = MockModel()
        model.config.tie_word_embeddings = True

        with pytest.raises(AssertionError, match="Hugging Face tp plan is not supported"):
            get_hf_tp_shard_plan(model)

    def test_invalid_parallel_style_error(self):
        """Test error for invalid parallel style string."""
        model = MockModel()

        model._tp_plan = {"layers.0.self_attn.q_proj": "invalid_style"}
        model.config.tie_word_embeddings = True

        with pytest.raises(ValueError, match="Unknown parallel style"):
            get_hf_tp_shard_plan(model)

    @staticmethod
    def _bare_gemma3():
        """Gemma3 instance with exact class identity but no HF ``__init__``."""
        model = Gemma3ForConditionalGeneration.__new__(Gemma3ForConditionalGeneration)
        nn.Module.__init__(model)
        return model

    def test_gemma3_pre_standardization_tree_uses_language_model_prefix(self):
        """Old Gemma3 (transformers <= 4.51) hangs the text tower off a top-level
        ``language_model``; the prefix must follow the registered child module,
        not a transformers version gate.
        """
        model = self._bare_gemma3()
        language_model = nn.Module()
        language_model._tp_plan = {"model.layers.0.self_attn.q_proj": "colwise"}
        model.language_model = language_model

        result = get_hf_tp_shard_plan(model)

        assert isinstance(result["language_model.model.layers.0.self_attn.q_proj"], ColwiseParallel)
        assert "language_model.embed_tokens" in result

    def test_gemma3_standardized_tree_uses_model_prefix(self):
        """Standardized Gemma3 (transformers >= 4.52, incl. v5) nests everything
        under ``model``; the prefix must resolve structurally to ``model``.
        """
        model = self._bare_gemma3()
        inner = nn.Module()
        inner._tp_plan = {"language_model.layers.0.self_attn.q_proj": "colwise"}
        model.model = inner

        result = get_hf_tp_shard_plan(model)

        assert isinstance(result["model.language_model.layers.0.self_attn.q_proj"], ColwiseParallel)
        assert "model.embed_tokens" in result


class TestApplyFsdpShardingRecursively:
    """Test class for apply_fsdp2_sharding_recursively utility function."""

    @pytest.fixture
    def mock_module_list(self):
        """Create a mock ModuleList with transformer blocks."""
        module_list = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        return module_list

    @pytest.fixture
    def mock_single_module(self):
        """Create a mock module with child modules."""

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 10)
                self.nested = nn.ModuleList([nn.Linear(5, 5)])

        return TestModule()

    @pytest.fixture
    def mock_mesh(self):
        """Create a mock device mesh."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.mesh_dim_names = ("dp", "tp")
        return mesh

    @pytest.fixture
    def mock_mp_policy(self):
        """Create a mock mixed precision policy."""
        from torch.distributed.fsdp import MixedPrecisionPolicy

        mp_policy = MagicMock(spec=MixedPrecisionPolicy)
        return mp_policy

    @pytest.fixture
    def mock_offload_policy(self):
        """Create a mock offload policy."""
        from torch.distributed.fsdp import CPUOffloadPolicy

        offload_policy = MagicMock(spec=CPUOffloadPolicy)
        return offload_policy

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_module_list(
        self, mock_fully_shard, mock_module_list, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with a ModuleList."""

        # Set up mock return values - add FSDP2 prefetch methods that fully_shard normally provides
        def mock_shard(x, **kwargs):
            x.set_modules_to_forward_prefetch = MagicMock()
            x.set_modules_to_backward_prefetch = MagicMock()
            return x

        mock_fully_shard.side_effect = mock_shard

        # Call the function
        apply_fsdp2_sharding_recursively(
            module=mock_module_list, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # Verify fully_shard was called for each layer in the ModuleList
        assert mock_fully_shard.call_count == 3

        # Verify the call parameters for each layer
        calls = mock_fully_shard.call_args_list
        for i, call in enumerate(calls):
            args, kwargs = call
            assert args[0] is mock_module_list[i]  # The transformer block
            assert kwargs["mesh"] is mock_mesh
            assert kwargs["mp_policy"] is mock_mp_policy
            assert kwargs["offload_policy"] is mock_offload_policy

            # Check reshard_after_forward optimization (last layer should be False)
            expected_reshard = i < len(mock_module_list) - 1
            assert kwargs["reshard_after_forward"] == expected_reshard

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_module_list_without_offload_policy(
        self, mock_fully_shard, mock_module_list, mock_mesh, mock_mp_policy
    ):
        """Test apply_fsdp2_sharding_recursively with a ModuleList and no offload policy."""

        # Set up mock return values - add FSDP2 prefetch methods that fully_shard normally provides
        def mock_shard(x, **kwargs):
            x.set_modules_to_forward_prefetch = MagicMock()
            x.set_modules_to_backward_prefetch = MagicMock()
            return x

        mock_fully_shard.side_effect = mock_shard

        # Call the function without offload_policy
        apply_fsdp2_sharding_recursively(module=mock_module_list, mesh=mock_mesh, mp_policy=mock_mp_policy)

        # Verify fully_shard was called with None offload_policy
        calls = mock_fully_shard.call_args_list
        for call in calls:
            args, kwargs = call
            assert kwargs["offload_policy"] is None

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_module_list_respects_explicit_reshard_override(
        self, mock_fully_shard, mock_module_list, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with an explicit reshard override."""
        mock_mesh.mesh_dim_names = ("dp", "tp")

        def mock_shard(x, **kwargs):
            x.set_modules_to_forward_prefetch = MagicMock()
            x.set_modules_to_backward_prefetch = MagicMock()
            return x

        mock_fully_shard.side_effect = mock_shard

        apply_fsdp2_sharding_recursively(
            module=mock_module_list,
            mesh=mock_mesh,
            mp_policy=mock_mp_policy,
            offload_policy=mock_offload_policy,
            reshard_after_forward=False,
        )

        assert mock_fully_shard.call_count == 3
        for call in mock_fully_shard.call_args_list:
            assert call.kwargs["reshard_after_forward"] is False

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_regular_module(
        self, mock_fully_shard, mock_single_module, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with a regular module (not ModuleList)."""
        # Set up mock return values
        mock_fully_shard.side_effect = lambda x, **kwargs: x

        # Call the function
        apply_fsdp2_sharding_recursively(
            module=mock_single_module, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # For regular modules, it should recursively call on children
        # It should call itself recursively for the nested ModuleList
        # The nested ModuleList should get fully_shard called on its children
        assert mock_fully_shard.call_count == 1  # Just the nested ModuleList's single layer

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_replicates_frozen_multimodal_module(
        self, mock_fully_shard, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Frozen multimodal modules are skipped and collected when replication is requested."""
        mock_mesh.mesh_dim_names = ("dp",)

        class VisionTower(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10)])

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = VisionTower()
                self.text_layers = nn.ModuleList([nn.Linear(10, 10)])

        model = TestModule()
        for param in model.vision_tower.parameters():
            param.requires_grad = False

        mock_fully_shard.side_effect = lambda x, **kwargs: x
        ignored_params = set()

        apply_fsdp2_sharding_recursively(
            module=model,
            mesh=mock_mesh,
            mp_policy=mock_mp_policy,
            offload_policy=mock_offload_policy,
            frozen_multimodal_sharding="replicate",
            ignored_multimodal_params=ignored_params,
        )

        assert ignored_params == set(model.vision_tower.parameters())
        sharded_modules = [call.args[0] for call in mock_fully_shard.call_args_list]
        assert model.vision_tower.layers[0] not in sharded_modules
        assert model.text_layers[0] in sharded_modules

    @pytest.mark.parametrize(
        "frozen_multimodal_sharding, expected_towers_sharded",
        [("root", False), ("per_layer", True)],
    )
    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_applies_frozen_multimodal_policy(
        self,
        mock_fully_shard,
        mock_mesh,
        mock_mp_policy,
        mock_offload_policy,
        frozen_multimodal_sharding,
        expected_towers_sharded,
    ):
        """Root owns frozen towers while per-layer shards both modalities uniformly."""
        mock_mesh.mesh_dim_names = ("dp",)

        class AudioTower(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10)])

        class VisionTower(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10)])

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_tower = AudioTower()
                self.vision_tower = VisionTower()
                self.text_layers = nn.ModuleList([nn.Linear(10, 10)])

        model = TestModule()
        for tower in (model.audio_tower, model.vision_tower):
            for param in tower.parameters():
                param.requires_grad = False

        mock_fully_shard.side_effect = lambda x, **kwargs: x

        apply_fsdp2_sharding_recursively(
            module=model,
            mesh=mock_mesh,
            mp_policy=mock_mp_policy,
            offload_policy=mock_offload_policy,
            frozen_multimodal_sharding=frozen_multimodal_sharding,
        )

        sharded_modules = [call.args[0] for call in mock_fully_shard.call_args_list]
        assert (model.audio_tower.layers[0] in sharded_modules) is expected_towers_sharded
        assert (model.vision_tower.layers[0] in sharded_modules) is expected_towers_sharded
        assert model.text_layers[0] in sharded_modules

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_empty_module_list(
        self, mock_fully_shard, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with an empty ModuleList."""
        empty_module_list = nn.ModuleList([])

        # Call the function
        apply_fsdp2_sharding_recursively(
            module=empty_module_list, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # Should not call fully_shard for empty ModuleList
        assert mock_fully_shard.call_count == 0

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_single_item_module_list(
        self, mock_fully_shard, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with a single-item ModuleList."""
        single_module_list = nn.ModuleList([nn.Linear(10, 10)])
        mock_fully_shard.side_effect = lambda x, **kwargs: x

        # Call the function
        apply_fsdp2_sharding_recursively(
            module=single_module_list, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # Should call fully_shard once
        assert mock_fully_shard.call_count == 1

        # For single item, reshard_after_forward should be False (optimization)
        call_args = mock_fully_shard.call_args_list[0]
        assert call_args[1]["reshard_after_forward"] is False

    def test_apply_fsdp_sharding_no_children(self, mock_mesh, mock_mp_policy, mock_offload_policy):
        """Test apply_fsdp2_sharding_recursively with a module that has no children."""
        leaf_module = nn.Linear(10, 10)

        # This should complete without error (no children to recurse on)
        apply_fsdp2_sharding_recursively(
            module=leaf_module, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )


def test_default_parallelization_replicated_frozen_multimodal_params_are_ignored_by_root(monkeypatch):
    """The dense root FSDP unit must ignore frozen multimodal params when replication is requested."""
    from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

    class InnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])
            self.vision_tower = nn.Module()
            self.vision_tower.layers = nn.ModuleList([nn.Linear(10, 10)])

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = InnerModel()
            self.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8)

    model = TestModel()
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False

    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 1
    device_mesh = MagicMock(spec=DeviceMesh)
    device_mesh.__getitem__.side_effect = lambda key: tp_mesh

    dp_mesh = MagicMock()
    dp_mesh.mesh_dim_names = ("dp",)
    monkeypatch.setattr(parallelizer, "get_fsdp_dp_mesh", lambda *args, **kwargs: dp_mesh)
    monkeypatch.setattr(parallelizer, "_patch_fsdp_accumulated_grad_guard", lambda: None)

    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        module.set_modules_to_forward_prefetch = MagicMock()
        module.set_modules_to_backward_prefetch = MagicMock()
        return module

    result = DefaultParallelizationStrategy().parallelize(
        model=model,
        device_mesh=device_mesh,
        activation_checkpointing=False,
        frozen_multimodal_sharding="replicate",
        fully_shard_fn=fake_fully_shard,
    )

    assert result is model
    root_kwargs = next(kwargs for module, kwargs in calls if module is model)
    assert root_kwargs["ignored_params"] == set(model.model.vision_tower.parameters())


def test_default_parallelization_warns_for_per_layer_frozen_multimodal_policy(monkeypatch, caplog):
    """The expert per-layer policy makes its rank-uniform collective contract visible."""
    from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

    class InnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])
            self.vision_tower = nn.Module()
            self.vision_tower.layers = nn.ModuleList([nn.Linear(10, 10)])

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = InnerModel()
            self.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8)

    model = TestModel()
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False

    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 1
    device_mesh = MagicMock(spec=DeviceMesh)
    device_mesh.__getitem__.side_effect = lambda key: tp_mesh
    dp_mesh = MagicMock()
    dp_mesh.mesh_dim_names = ("dp",)
    monkeypatch.setattr(parallelizer, "get_fsdp_dp_mesh", lambda *args, **kwargs: dp_mesh)
    monkeypatch.setattr(parallelizer, "_patch_fsdp_accumulated_grad_guard", lambda: None)

    def fake_fully_shard(module, **kwargs):
        module.set_modules_to_forward_prefetch = MagicMock()
        module.set_modules_to_backward_prefetch = MagicMock()
        return module

    with caplog.at_level("WARNING"):
        DefaultParallelizationStrategy().parallelize(
            model=model,
            device_mesh=device_mesh,
            activation_checkpointing=False,
            frozen_multimodal_sharding="per_layer",
            fully_shard_fn=fake_fully_shard,
        )

    assert "rank-asymmetric modality execution can hang" in caplog.text


class TestGetParallelPlanClassNameFallback:
    """Test that _get_parallel_plan matches by qualified class name (module.qualname)."""

    def test_identity_match(self):
        """Exact class qualname in PARALLELIZE_FUNCTIONS is found."""
        sentinel_plan = {"layer": ColwiseParallel()}
        model = MockModel()

        with patch(
            "nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS",
            {_get_class_qualname(type(model)): lambda m, sp: sentinel_plan},
        ):
            plan = _get_parallel_plan(model, sequence_parallel=False, tp_shard_plan=None)
        assert plan is sentinel_plan

    def test_class_name_fallback(self):
        """A different class object with the same module.qualname still matches.

        With the old class-object-keyed dict, identity was required. With the new
        string-keyed dict, two distinct class objects that share ``__module__`` and
        ``__qualname__`` resolve to the same key and both match — which is exactly
        the NeMo-RL wrapping scenario this fix targets.
        """
        sentinel_plan = {"layer": ColwiseParallel()}

        # Create a *different* class object with the same name (and therefore the same
        # module.qualname since both are defined in this test module).
        DuplicateMockModel = type("MockModel", (nn.Module,), {"forward": lambda self, x: x})
        assert DuplicateMockModel is not MockModel
        assert _get_class_qualname(DuplicateMockModel) == _get_class_qualname(MockModel)

        model = MockModel()
        model.__class__ = DuplicateMockModel  # model's type is the duplicate

        with patch(
            "nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS",
            {_get_class_qualname(MockModel): lambda m, sp: sentinel_plan},
        ):
            plan = _get_parallel_plan(model, sequence_parallel=False, tp_shard_plan=None)
        # Matches because module.qualname is the same, even though the class object differs
        assert plan is sentinel_plan

    def test_nemo_rl_wrapped_class_match(self):
        """A different class object with the same module and qualname still matches.

        This simulates the NeMo-RL scenario: _get_mixin_wrapped_class() creates a new
        class via type(...) that preserves __module__ and __qualname__ from the original.
        Both the original and the wrapper resolve to the same _get_class_qualname() key.
        """
        sentinel_plan = {"layer": ColwiseParallel()}
        original_cls = type(MockModel())

        # Simulate _get_mixin_wrapped_class: create a *new* class object that copies
        # __module__ and __qualname__ from the original (same qualname, different object)
        WrappedCls = type(
            original_cls.__name__,
            (nn.Module,),
            {
                "forward": lambda self, x: x,
                "__module__": original_cls.__module__,
                "__qualname__": original_cls.__qualname__,
            },
        )
        assert WrappedCls is not original_cls
        assert _get_class_qualname(WrappedCls) == _get_class_qualname(original_cls)

        model = MockModel()
        model.__class__ = WrappedCls  # model's type is the wrapper

        with patch(
            "nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS",
            {_get_class_qualname(original_cls): lambda m, sp: sentinel_plan},
        ):
            plan = _get_parallel_plan(model, sequence_parallel=False, tp_shard_plan=None)
        assert plan is sentinel_plan

    def test_no_match_falls_through_to_default(self):
        """Completely unknown class qualname falls through to the default plan."""
        model = MockModel()
        model.__class__ = type("UnknownModel", (nn.Module,), {"forward": lambda self, x: x})

        with patch(
            "nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS",
            {_get_class_qualname(MockModel): lambda m, sp: {"x": ColwiseParallel()}},
        ):
            plan = _get_parallel_plan(model, sequence_parallel=False, tp_shard_plan=None)
        # Should get the default Llama3-style plan (has q_proj, k_proj, etc.)
        assert "model.layers.*.self_attn.q_proj" in plan


class TestUpdateAttentionHeadCountsForTP:
    """Tests for _update_attention_head_counts_for_tp."""

    @staticmethod
    def _make_model(num_heads=64, num_kv_heads=8, hidden_size=8192, architectures=None, model_type=None):
        model = nn.Module()
        cfg = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
        )
        if architectures is not None:
            cfg.architectures = architectures
        if model_type is not None:
            cfg.model_type = model_type
        model.config = cfg

        inner = nn.Module()
        layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            attn = nn.Module()
            attn.num_heads = num_heads
            attn.num_key_value_heads = num_kv_heads
            layer.self_attn = attn
            layers.append(layer)
        inner.layers = layers
        model.model = inner
        return model

    def test_noop_for_tp_size_1(self):
        model = self._make_model()
        _update_attention_head_counts_for_tp(model, tp_size=1)
        assert model.config.num_attention_heads == 64
        assert model.config.num_key_value_heads == 8

    def test_preserves_config_and_updates_layer_attrs(self):
        model = self._make_model(num_heads=64, num_kv_heads=8, hidden_size=8192)
        _update_attention_head_counts_for_tp(model, tp_size=2)
        assert model.config.num_attention_heads == 64
        assert model.config.num_key_value_heads == 8
        assert model.config.head_dim == 128
        for layer in model.model.layers:
            assert layer.self_attn.num_heads == 32
            assert layer.self_attn.num_key_value_heads == 4

    def test_preserves_existing_head_dim(self):
        model = self._make_model(num_heads=64, num_kv_heads=8, hidden_size=8192)
        model.config.head_dim = 128
        _update_attention_head_counts_for_tp(model, tp_size=2)
        assert model.config.head_dim == 128

    def test_computes_head_dim_when_missing(self):
        model = self._make_model(num_heads=32, num_kv_heads=8, hidden_size=4096)
        _update_attention_head_counts_for_tp(model, tp_size=2)
        assert model.config.head_dim == 128  # 4096 // 32


    def test_derives_kv_heads_from_num_key_value_groups(self):
        """When config.num_key_value_heads is None, fall back to num_key_value_groups."""
        model = self._make_model(num_heads=64, num_kv_heads=8, hidden_size=8192)
        model.config.num_key_value_heads = None
        for layer in model.model.layers:
            layer.self_attn.num_key_value_groups = 8
        _update_attention_head_counts_for_tp(model, tp_size=2)
        for layer in model.model.layers:
            assert layer.self_attn.num_heads == 32
            assert layer.self_attn.num_key_value_heads == 4  # 32 // 8

    def test_kv_heads_defaults_to_num_heads_without_groups(self):
        """When config.num_key_value_heads is None and no num_key_value_groups attr."""
        model = self._make_model(num_heads=64, num_kv_heads=8, hidden_size=8192)
        model.config.num_key_value_heads = None
        # num_key_value_groups is never set in _make_model, so already absent
        _update_attention_head_counts_for_tp(model, tp_size=2)
        for layer in model.model.layers:
            assert layer.self_attn.num_key_value_heads == 32  # same as local_num_attention_heads

    def test_language_model_inner_path(self):
        """Layers under model.language_model are found when model.model has no layers."""
        model = nn.Module()
        model.config = SimpleNamespace(
            num_attention_heads=64,
            num_key_value_heads=8,
            hidden_size=8192,
        )
        lang = nn.Module()
        layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            attn = nn.Module()
            attn.num_heads = 64
            attn.num_key_value_heads = 8
            layer.self_attn = attn
            layers.append(layer)
        lang.layers = layers
        model.language_model = lang
        _update_attention_head_counts_for_tp(model, tp_size=2)
        for layer in lang.layers:
            assert layer.self_attn.num_heads == 32
            assert layer.self_attn.num_key_value_heads == 4

    def test_noop_without_config(self):
        model = nn.Module()
        _update_attention_head_counts_for_tp(model, tp_size=2)

    def test_noop_without_layers(self):
        model = nn.Module()
        model.config = SimpleNamespace(num_attention_heads=8, hidden_size=64)
        _update_attention_head_counts_for_tp(model, tp_size=2)


class TestAttentionIsHeadSharded:
    """Tests for _attention_is_head_sharded."""

    def test_colwise_default_is_sharded(self):
        """ColwiseParallel() with default output (Shard) → heads are sharded."""
        plan = {
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        }
        assert _attention_is_head_sharded(plan) is True

    def test_colwise_explicit_shard_is_sharded(self):
        plan = {
            "model.layers.*.self_attn.q_proj": ColwiseParallel(output_layouts=Shard(-1)),
        }
        assert _attention_is_head_sharded(plan) is True

    def test_rowwise_replicate_is_not_sharded(self):
        """Phi-3 style: RowwiseParallel with Replicate output → not sharded."""
        plan = {
            "model.layers.*.self_attn.qkv_proj": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
            "model.layers.*.self_attn.o_proj": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
        }
        assert _attention_is_head_sharded(plan) is False

    def test_colwise_replicate_output_is_not_sharded(self):
        """ColwiseParallel with explicit Replicate output → not sharded."""
        plan = {
            "model.layers.*.self_attn.q_proj": ColwiseParallel(output_layouts=Replicate()),
        }
        assert _attention_is_head_sharded(plan) is False

    def test_no_attn_keys_is_not_sharded(self):
        """Plan with only MLP entries → not sharded."""
        plan = {
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
        }
        assert _attention_is_head_sharded(plan) is False

    def test_empty_plan_is_not_sharded(self):
        assert _attention_is_head_sharded({}) is False


# ---------------------------------------------------------------------------
# Activation checkpointing + KV-sharing tests
# ---------------------------------------------------------------------------


class _FakeLayer(nn.Module):
    """Minimal transformer layer with mlp, self_attn, and layernorms."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.mlp = nn.Linear(dim, dim)
        self.self_attn = nn.Linear(dim, dim)
        self.input_layernorm = nn.Linear(dim, dim)
        self.post_attention_layernorm = nn.Linear(dim, dim)

    def forward(self, x):
        return x


def _make_model_for_ac(
    num_layers: int = 2,
    dim: int = 16,
    use_cache: bool = True,
    num_kv_shared_layers: int = 0,
    text_config_nested: bool = True,
):
    """Build a minimal model with configurable KV-sharing for activation-checkpointing tests.

    Args:
        text_config_nested: If True, place ``num_kv_shared_layers`` under
            ``config.text_config`` (VLM pattern).  If False, place it directly
            on ``config`` (flat LLM pattern).
    """

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_FakeLayer(dim) for _ in range(num_layers)])

    model = nn.Module()
    model.model = _Inner()  # type: ignore[attr-defined]

    if text_config_nested:
        text_cfg = SimpleNamespace(num_kv_shared_layers=num_kv_shared_layers)
        model.config = SimpleNamespace(use_cache=use_cache, text_config=text_cfg)  # type: ignore[attr-defined]
    else:
        model.config = SimpleNamespace(  # type: ignore[attr-defined]
            use_cache=use_cache,
            num_kv_shared_layers=num_kv_shared_layers,
        )
    model.forward = lambda x: x  # type: ignore[attr-defined]
    return model


class TestActivationCheckpointingKVSharing:
    """Tests for the KV-sharing–aware activation-checkpointing guards
    in ``DefaultParallelizationStrategy.parallelize``.
    """

    @pytest.fixture(autouse=True)
    def _patch_parallelizer(self, monkeypatch):
        """Patch heavy distributed primitives so we can call ``parallelize``
        without a real GPU mesh.  ``checkpoint_wrapper`` is replaced with a
        lightweight wrapper that records which module was wrapped.
        """

        class _Wrapped(nn.Module):
            """Sentinel wrapper so we can assert which sub-modules were checkpointed.

            Must inherit from ``nn.Module`` because PyTorch's ``__setattr__``
            rejects non-Module values when replacing a registered child module.
            """

            def __init__(self, inner, **kwargs):
                super().__init__()
                self._inner = inner
                self.kwargs = kwargs

            @property
            def _checkpoint_wrapped_module(self):
                return self._inner

            def forward(self, x):
                return self._inner(x)

        self._Wrapped = _Wrapped

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.checkpoint_wrapper",
            lambda module, **kwargs: _Wrapped(module, **kwargs),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.activation_checkpointing.checkpoint_wrapper",
            _Wrapped,
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.fully_shard",
            lambda model, **kw: model,
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.get_fsdp_dp_mesh",
            lambda mesh, *a, **kw: MagicMock(),
        )

    def _run_parallelize(
        self,
        model,
        activation_checkpointing=True,
        activation_checkpointing_scope="all",
        enable_compile=False,
    ):
        """Invoke the strategy under test and return the model."""
        from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

        strategy = DefaultParallelizationStrategy()
        mesh = MagicMock(spec=DeviceMesh)
        tp_mesh = MagicMock()
        tp_mesh.size.return_value = 1  # no TP
        mesh.__getitem__ = lambda self_, key: tp_mesh
        return strategy.parallelize(
            model=model,
            device_mesh=mesh,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_scope=activation_checkpointing_scope,
            enable_compile=enable_compile,
        )

    # ------------------------------------------------------------------ #
    # use_cache preservation / disabling
    # ------------------------------------------------------------------ #

    def test_use_cache_preserved_when_kv_sharing(self):
        """Models with num_kv_shared_layers > 0 must keep use_cache=True."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=20)
        self._run_parallelize(model)
        assert model.config.use_cache is True

    def test_use_cache_disabled_without_kv_sharing(self):
        """Standard models (num_kv_shared_layers=0) get use_cache=False."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=0)
        self._run_parallelize(model)
        assert model.config.use_cache is False

    def test_use_cache_preserved_flat_config(self):
        """KV-sharing detected through a flat config (no text_config nesting)."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=10, text_config_nested=False)
        self._run_parallelize(model)
        assert model.config.use_cache is True

    def test_use_cache_disabled_flat_config_no_sharing(self):
        """Flat config without KV sharing still disables cache."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=0, text_config_nested=False)
        self._run_parallelize(model)
        assert model.config.use_cache is False

    def test_use_cache_noop_when_already_false(self):
        """If use_cache is already False and no KV sharing, code path is a no-op."""
        model = _make_model_for_ac(use_cache=False, num_kv_shared_layers=0)
        self._run_parallelize(model)
        assert model.config.use_cache is False

    def test_no_config_does_not_crash(self, monkeypatch):
        """Model without a config attribute must not raise."""
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer._extract_model_layer_groups",
            lambda m: {},
        )
        model = nn.Module()
        model.forward = lambda x: x  # type: ignore[attr-defined]
        # no model.config at all
        self._run_parallelize(model)  # should not raise

    # ------------------------------------------------------------------ #
    # self_attn checkpoint wrapping
    # ------------------------------------------------------------------ #

    def test_self_attn_not_wrapped_when_kv_sharing(self):
        """KV-shared models: self_attn must NOT be wrapped (would corrupt cache)."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=20)
        self._run_parallelize(model)
        for layer in model.model.layers:
            assert not isinstance(layer.self_attn, self._Wrapped), (
                "self_attn should NOT be checkpoint-wrapped for KV-shared models"
            )

    def test_self_attn_wrapped_without_kv_sharing(self):
        """Standard models: self_attn IS wrapped."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=0)
        self._run_parallelize(model)
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, self._Wrapped), (
                "self_attn should be checkpoint-wrapped for standard models"
            )

    def test_linear_attn_wrapped_with_compile(self):
        """Compile-compatible checkpointing wraps hybrid blocks' ``linear_attn`` mixers."""

        class _LinearAttentionLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = nn.Linear(16, 16)
                self.mlp = nn.Linear(16, 16)

        model = _make_model_for_ac(num_kv_shared_layers=0)
        model.model.layers = nn.ModuleList([_LinearAttentionLayer() for _ in range(2)])

        self._run_parallelize(model, enable_compile=True)

        for layer in model.model.layers:
            assert isinstance(layer.linear_attn, self._Wrapped)
            assert isinstance(layer.mlp, self._Wrapped)

    def test_mlp_always_wrapped(self):
        """MLP is checkpoint-wrapped regardless of KV sharing."""
        for kv_shared in (0, 20):
            model = _make_model_for_ac(num_kv_shared_layers=kv_shared)
            self._run_parallelize(model)
            for layer in model.model.layers:
                assert isinstance(layer.mlp, self._Wrapped), (
                    f"mlp should always be wrapped (num_kv_shared_layers={kv_shared})"
                )

    def test_layernorms_always_wrapped(self):
        """Layernorms are checkpoint-wrapped regardless of KV sharing."""
        for kv_shared in (0, 20):
            model = _make_model_for_ac(num_kv_shared_layers=kv_shared)
            self._run_parallelize(model)
            for layer in model.model.layers:
                assert isinstance(layer.input_layernorm, self._Wrapped)
                assert isinstance(layer.post_attention_layernorm, self._Wrapped)

    def test_vision_style_child_names_are_wrapped(self):
        """Vision/Ministral-style blocks use ``attention`` and ``feed_forward`` names."""

        class _VisionStyleLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.Linear(16, 16)
                self.feed_forward = nn.Linear(16, 16)
                self.attention_norm = nn.Linear(16, 16)
                self.ffn_norm = nn.Linear(16, 16)

            def forward(self, x):
                return x

        model = _make_model_for_ac(num_kv_shared_layers=0)
        model.model.layers = nn.ModuleList([_VisionStyleLayer() for _ in range(2)])

        self._run_parallelize(model)

        for layer in model.model.layers:
            assert isinstance(layer.attention, self._Wrapped)
            assert isinstance(layer.feed_forward, self._Wrapped)
            assert isinstance(layer.attention_norm, self._Wrapped)
            assert isinstance(layer.ffn_norm, self._Wrapped)

    def test_qwen_clip_style_vision_child_names_are_wrapped(self):
        """Qwen/SigLIP/CLIP-style vision blocks use ``attn`` or layer/norm pairs."""

        class _QwenStyleLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(16, 16)
                self.mlp = nn.Linear(16, 16)
                self.norm1 = nn.Linear(16, 16)
                self.norm2 = nn.Linear(16, 16)

            def forward(self, x):
                return x

        class _ClipStyleLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(16, 16)
                self.mlp = nn.Linear(16, 16)
                self.layer_norm1 = nn.Linear(16, 16)
                self.layer_norm2 = nn.Linear(16, 16)

            def forward(self, x):
                return x

        model = _make_model_for_ac(num_kv_shared_layers=0)
        model.model.layers = nn.ModuleList([_QwenStyleLayer(), _ClipStyleLayer()])

        self._run_parallelize(model)

        qwen_layer = model.model.layers[0]
        assert isinstance(qwen_layer.attn, self._Wrapped)
        assert isinstance(qwen_layer.mlp, self._Wrapped)
        assert isinstance(qwen_layer.norm1, self._Wrapped)
        assert isinstance(qwen_layer.norm2, self._Wrapped)

        clip_layer = model.model.layers[1]
        assert isinstance(clip_layer.self_attn, self._Wrapped)
        assert isinstance(clip_layer.mlp, self._Wrapped)
        assert isinstance(clip_layer.layer_norm1, self._Wrapped)
        assert isinstance(clip_layer.layer_norm2, self._Wrapped)


    def test_no_wrapping_without_activation_checkpointing(self):
        """When activation_checkpointing=False, nothing is wrapped."""
        model = _make_model_for_ac(num_kv_shared_layers=0)
        self._run_parallelize(model, activation_checkpointing=False)
        for layer in model.model.layers:
            assert not isinstance(layer.mlp, self._Wrapped)
            assert not isinstance(layer.self_attn, self._Wrapped)
        assert model.config.use_cache is True  # untouched

    def test_selective_checkpointing_wraps_whole_layers(self, monkeypatch):
        """Selective activation checkpointing wraps full transformer blocks."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

        calls = []

        def fake_checkpoint_wrapper(module, **kwargs):
            calls.append((module, kwargs))
            return self._Wrapped(module)

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.activation_checkpointing.checkpoint_wrapper",
            fake_checkpoint_wrapper,
        )

        from nemo_automodel.components.distributed.activation_checkpointing import SELECTIVE_AC_WRAPPER_FLAG

        model = _make_model_for_ac(num_kv_shared_layers=0)
        self._run_parallelize(model, activation_checkpointing="selective")

        assert len(calls) == len(model.model.layers)
        for layer in model.model.layers:
            assert isinstance(layer, self._Wrapped)
            # Wrapper is tagged so the per-layer compile step compiles it OUTER.
            assert getattr(layer, SELECTIVE_AC_WRAPPER_FLAG, False) is True
        for _, kwargs in calls:
            assert kwargs["checkpoint_impl"] == CheckpointImpl.NO_REENTRANT
            assert kwargs["preserve_rng_state"] is True
            assert callable(kwargs["context_fn"])

    def test_selective_checkpointing_kv_sharing_falls_back_to_submodule_wrapping(self):
        """KV-shared models cannot checkpoint the whole block because attention mutates cache."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=20)
        self._run_parallelize(model, activation_checkpointing="selective")

        for layer in model.model.layers:
            assert not isinstance(layer, self._Wrapped)
            assert isinstance(layer.mlp, self._Wrapped)
            assert not isinstance(layer.self_attn, self._Wrapped)
            assert isinstance(layer.input_layernorm, self._Wrapped)
            assert isinstance(layer.post_attention_layernorm, self._Wrapped)

    def test_bagel_parallelize_uses_full_layer_checkpointing(self):
        """BAGEL wraps whole Qwen/SigLIP layers through the special AC path."""
        model = _make_bagel_model(num_language_layers=2, num_vision_layers=2)

        self._run_parallelize(model, activation_checkpointing=True)

        language_layers = model.model.language_model.model.layers
        vision_layers = model.model.vit_model.vision_model.encoder.layers
        assert all(isinstance(layer, self._Wrapped) for layer in language_layers)
        assert all(isinstance(layer, self._Wrapped) for layer in vision_layers)

    # ------------------------------------------------------------------ #
    # HF native gradient-checkpointing path
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Exception / edge-case branches
    # ------------------------------------------------------------------ #

    def test_frozen_config_use_cache_except_branch(self):
        """When ``model.config.use_cache = False`` raises, the except branch runs."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=0)

        class _FrozenConfig:
            use_cache = True
            text_config = SimpleNamespace(num_kv_shared_layers=0)

            def __setattr__(self, name, value):
                raise AttributeError("frozen")

        model.config = _FrozenConfig()  # type: ignore[attr-defined]
        self._run_parallelize(model)
        # use_cache stays True because the assignment raised and was caught
        assert model.config.use_cache is True

    def test_no_config_with_layers_does_not_crash(self):
        """Model without ``config`` but with extractable layers does not crash."""

        class _Bare(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([_FakeLayer() for _ in range(2)])  # type: ignore[attr-defined]

            def forward(self, x):
                return x

        model = _Bare()
        # no model.config → hasattr(model, "config") is False
        self._run_parallelize(model)
        # mlp should still be wrapped (activation_checkpointing still applies)
        for layer in model.model.layers:
            assert isinstance(layer.mlp, self._Wrapped)

    def test_layer_missing_self_attn(self):
        """Layers without ``self_attn`` are skipped gracefully."""

        class _MlpOnlyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Linear(16, 16)

            def forward(self, x):
                return x

        model = _make_model_for_ac(num_kv_shared_layers=0)
        model.model.layers = nn.ModuleList([_MlpOnlyLayer() for _ in range(2)])
        self._run_parallelize(model)
        for layer in model.model.layers:
            assert isinstance(layer.mlp, self._Wrapped)
            assert not hasattr(layer, "self_attn")

    def test_layer_missing_mlp(self):
        """Layers without ``mlp`` are skipped gracefully."""

        class _AttnOnlyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(16, 16)

            def forward(self, x):
                return x

        model = _make_model_for_ac(num_kv_shared_layers=0)
        model.model.layers = nn.ModuleList([_AttnOnlyLayer() for _ in range(2)])
        self._run_parallelize(model)
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, self._Wrapped)
            assert not hasattr(layer, "mlp")

    # ------------------------------------------------------------------ #
    # HF-native gradient-checkpointing candidates
    # ------------------------------------------------------------------ #

    @staticmethod
    def _setup_hf_native_model(monkeypatch, num_kv_shared_layers, replay_safe_kv_sharing=False):
        """Helper: configure a model + fake transformers module for the HF native path.

        Args:
            monkeypatch: pytest monkeypatch fixture.
            num_kv_shared_layers: Value placed on the text config; > 0 marks the
                model as KV-shared.
            replay_safe_kv_sharing: Whether the model declares its shared-K/V store
                safe under whole-block checkpoint replay, as Gemma4 E2B/E4B do.
        """
        import types

        class _FakeGradLayer(_FakeLayer):
            pass

        _FakeGradLayer.__module__ = "transformers.models.gemma4.modeling_gemma4"

        fake_module = types.ModuleType("transformers.modeling_layers")
        fake_module.GradientCheckpointingLayer = _FakeGradLayer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers.modeling_layers", fake_module)

        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=num_kv_shared_layers)
        for i in range(len(model.model.layers)):
            model.model.layers[i] = _FakeGradLayer()
        model.supports_gradient_checkpointing = True  # type: ignore[attr-defined]
        model.gradient_checkpointing_enable = MagicMock()  # type: ignore[attr-defined]
        if replay_safe_kv_sharing:
            model.kv_sharing_survives_checkpoint_replay = True  # type: ignore[attr-defined]
        return model

    def test_hf_native_candidate_with_replay_safe_kv_sharing_uses_full_layer_checkpointing(self, monkeypatch):
        """A model that declares its shared-K/V store replay-safe keeps whole-block wrapping.

        ``apply_submodule_checkpointing`` leaves ``self_attn`` unwrapped for
        KV-shared models, so routing Gemma4 E2B/E4B there drops attention
        activations from checkpointing and inflates peak memory. The cache
        contract is unchanged: ``use_cache`` still stays on.
        """
        model = self._setup_hf_native_model(monkeypatch, num_kv_shared_layers=20, replay_safe_kv_sharing=True)
        self._run_parallelize(model)

        assert model.config.use_cache is True
        model.gradient_checkpointing_enable.assert_not_called()
        assert all(isinstance(layer, self._Wrapped) for layer in model.model.layers)
        # Whole-block wrapping, not the sub-module fallback that skips self_attn.
        inner_layers = [layer._checkpoint_wrapped_module for layer in model.model.layers]
        assert all(not isinstance(inner.mlp, self._Wrapped) for inner in inner_layers)
        assert all(not isinstance(inner.self_attn, self._Wrapped) for inner in inner_layers)

    def test_hf_native_candidate_with_plain_kv_sharing_uses_submodule_checkpointing(self, monkeypatch):
        """A KV-shared model that does not opt in stays off whole-block checkpointing.

        Native HF KV-shared models (e.g. ``Gemma3nForCausalLM``) keep an
        accumulating ``DynamicCache`` while ``use_cache=True``. Replaying a whole
        block calls ``Cache.update()`` a second time and backward dies with a
        ``CheckpointError`` about changed K/V metadata, so they must stay on the
        sub-module path that leaves ``self_attn`` unwrapped.
        """
        model = self._setup_hf_native_model(monkeypatch, num_kv_shared_layers=20)
        self._run_parallelize(model)

        assert model.config.use_cache is True
        model.gradient_checkpointing_enable.assert_not_called()
        assert all(not isinstance(layer, self._Wrapped) for layer in model.model.layers)
        assert all(isinstance(layer.mlp, self._Wrapped) for layer in model.model.layers)
        assert all(not isinstance(layer.self_attn, self._Wrapped) for layer in model.model.layers)

    def test_hf_native_candidate_uses_non_reentrant_full_layer_checkpointing(self, monkeypatch):
        """HF-native candidates use full-layer checkpoint wrappers instead of the HF API."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

        model = self._setup_hf_native_model(monkeypatch, num_kv_shared_layers=0)
        self._run_parallelize(model)

        assert model.config.use_cache is False
        model.gradient_checkpointing_enable.assert_not_called()
        assert all(isinstance(layer, self._Wrapped) for layer in model.model.layers)
        assert all(layer.kwargs["checkpoint_impl"] is CheckpointImpl.NO_REENTRANT for layer in model.model.layers)
        assert all(layer.kwargs["preserve_rng_state"] is True for layer in model.model.layers)

    def test_hf_native_grad_ckpt_skips_frozen_layers(self, monkeypatch):
        """Frozen layers force scoped submodule wrapping instead of whole-model HF native GC."""
        model = self._setup_hf_native_model(monkeypatch, num_kv_shared_layers=0)
        model.model.layers[0].requires_grad_(False)

        self._run_parallelize(model)

        model.gradient_checkpointing_enable.assert_not_called()
        assert not isinstance(model.model.layers[0].mlp, self._Wrapped)
        assert isinstance(model.model.layers[1].mlp, self._Wrapped)


class TestSelectiveCheckpointNumerics:
    """Real forward/backward parity for the selective-AC op policy.

    These tests use the *real* torch checkpoint primitives (no mocked
    ``checkpoint_wrapper``) so they exercise the policy returned by
    ``make_selective_checkpoint_context_fn``. The policy saves every other
    matmul, so the per-pass matmul counter must be keyed on
    ``ctx.is_recompute``. A single shared counter continues from the forward
    count into recompute and flips the save/recompute parity whenever a region
    has an odd number of matmuls, silently corrupting gradients.
    """

    class _MatmulBlock(nn.Module):
        def __init__(self, dim: int, num_linears: int):
            super().__init__()
            self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_linears)])

        def forward(self, x):
            for linear in self.linears:
                x = torch.relu(linear(x))
            return x

    def _assert_grads_match_baseline(self, num_linears: int):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )

        from nemo_automodel.components.distributed.activation_checkpointing import (
            make_selective_checkpoint_context_fn,
        )

        torch.manual_seed(0)
        dim = 8
        baseline = self._MatmulBlock(dim, num_linears)

        wrapped_inner = self._MatmulBlock(dim, num_linears)
        wrapped_inner.load_state_dict(baseline.state_dict())
        wrapped = checkpoint_wrapper(
            wrapped_inner,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            context_fn=make_selective_checkpoint_context_fn(),
            preserve_rng_state=True,
        )

        x = torch.randn(4, dim)

        out_base = baseline(x.clone())
        out_base.sum().backward()

        out_wrapped = wrapped(x.clone())
        out_wrapped.sum().backward()

        torch.testing.assert_close(out_wrapped, out_base)
        for (name, p_base), (_, p_wrap) in zip(baseline.named_parameters(), wrapped_inner.named_parameters()):
            torch.testing.assert_close(p_wrap.grad, p_base.grad, msg=f"grad mismatch for {name}")

    def test_gradients_match_with_odd_matmul_count(self):
        """Regression: an odd number of matmuls must not flip recompute parity."""
        self._assert_grads_match_baseline(num_linears=3)

    def test_gradients_match_with_even_matmul_count(self):
        """Even matmul count is the easy case and must also stay correct."""
        self._assert_grads_match_baseline(num_linears=4)


class TestSelectiveCheckpointCompile:
    """``_apply_per_layer_compile`` must compile selective-AC wrappers OUTER.

    Selective AC wraps the whole block, so torch.compile must compile the
    wrapper (not the unwrapped inner layer) for the partitioner to honor the
    SAC recompute tags. Non-selective layer-level wrappers (PP path) are still
    unwrapped and the decoder layer is compiled directly.
    """

    class _Block(nn.Module):
        def __init__(self, dim: int = 8):
            super().__init__()
            self.fc = nn.Linear(dim, dim)

        def forward(self, x):
            return self.fc(x)

    def _build_model(self, *, tag: bool):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )

        from nemo_automodel.components.distributed.activation_checkpointing import SELECTIVE_AC_WRAPPER_FLAG

        layers = nn.ModuleList()
        for _ in range(2):
            wrapper = checkpoint_wrapper(self._Block(), checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            if tag:
                setattr(wrapper, SELECTIVE_AC_WRAPPER_FLAG, True)
            layers.append(wrapper)
        inner = nn.Module()
        inner.layers = layers
        model = nn.Module()
        model.model = inner
        return model

    def _run_compile(self, model, monkeypatch):
        import nemo_automodel.components.distributed.parallelizer as parallelizer

        monkeypatch.setattr(parallelizer, "_patch_dtensor_spec_hash_for_symint", lambda: None)
        compiled = []
        monkeypatch.setattr(torch.nn.Module, "compile", lambda self, *a, **k: compiled.append(self))
        parallelizer._apply_per_layer_compile(model)
        return compiled

    def test_tagged_selective_wrapper_compiled_outer(self, monkeypatch):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        model = self._build_model(tag=True)
        compiled = self._run_compile(model, monkeypatch)

        assert len(compiled) == 2
        for m in compiled:
            assert isinstance(m, CheckpointWrapper), "selective wrapper must be compiled, not unwrapped"

    def test_untagged_wrapper_compiled_inner(self, monkeypatch):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        model = self._build_model(tag=False)
        compiled = self._run_compile(model, monkeypatch)

        assert len(compiled) == 2
        for m in compiled:
            assert not isinstance(m, CheckpointWrapper)
            assert isinstance(m, self._Block), "non-selective wrapper must be unwrapped before compile"

    def test_disable_dynamo_lru_cache_is_best_effort(self, monkeypatch):
        """Missing private dynamo API must not raise."""
        import nemo_automodel.components.distributed.activation_checkpointing as ac

        monkeypatch.delattr(torch._C._dynamo.eval_frame, "_set_lru_cache", raising=False)
        ac._disable_dynamo_lru_cache()  # should not raise


class TestSingleGpuActivationCheckpointing:
    """FSDP2Manager single-GPU (world_size==1) activation-checkpointing behavior."""

    def _make_manager(self, monkeypatch, activation_checkpointing):
        import nemo_automodel.components.distributed.fsdp2 as fsdp2_mod
        from nemo_automodel.components.distributed.config import FSDP2Config

        monkeypatch.setattr(fsdp2_mod, "get_world_size_safe", lambda: 1)
        config = FSDP2Config(activation_checkpointing=activation_checkpointing)
        return fsdp2_mod.FSDP2Manager(config, device_mesh=MagicMock())

    def test_selective_wraps_layers_on_single_gpu(self, monkeypatch):
        """Selective AC is honored on a single GPU (not silently full-checkpointed)."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        from nemo_automodel.components.distributed.activation_checkpointing import SELECTIVE_AC_WRAPPER_FLAG

        manager = self._make_manager(monkeypatch, "selective")
        model = _make_model_for_ac(num_kv_shared_layers=0)
        manager.parallelize(model)

        for layer in model.model.layers:
            assert isinstance(layer, CheckpointWrapper)
            assert getattr(layer, SELECTIVE_AC_WRAPPER_FLAG, False) is True
        assert model.config.use_cache is False

    def test_selective_kv_sharing_falls_back_on_single_gpu(self, monkeypatch):
        """KV-shared models fall back to sub-module checkpointing, not whole-block."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        manager = self._make_manager(monkeypatch, "selective")
        model = _make_model_for_ac(num_kv_shared_layers=20)
        manager.parallelize(model)

        for layer in model.model.layers:
            assert not isinstance(layer, CheckpointWrapper)
            assert isinstance(layer.mlp, CheckpointWrapper)
            assert not isinstance(layer.self_attn, CheckpointWrapper)

    def test_full_wraps_layers_on_single_gpu_without_hf_native(self, monkeypatch):
        """Non-selective AC wraps layers on single GPU when the model is not an HF native GC candidate."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        manager = self._make_manager(monkeypatch, True)
        model = _make_model_for_ac(num_kv_shared_layers=0)
        model.gradient_checkpointing_enable = MagicMock()
        manager.parallelize(model)

        model.gradient_checkpointing_enable.assert_not_called()
        for layer in model.model.layers:
            assert not isinstance(layer, CheckpointWrapper)
            assert isinstance(layer.mlp, CheckpointWrapper)
            assert isinstance(layer.self_attn, CheckpointWrapper)


class TestFsdp2ShardingEnabled:
    """`fsdp2_sharding_enabled` reports whether fully_shard — and its mixed-precision casts — apply."""

    def test_disabled_on_single_rank_world(self, monkeypatch):
        import nemo_automodel.components.distributed.fsdp2 as fsdp2_mod

        monkeypatch.setattr(fsdp2_mod, "get_world_size_safe", lambda: 1)

        # The mesh is never inspected once the world is single-rank.
        assert fsdp2_mod.fsdp2_sharding_enabled(MagicMock()) is False

    def test_disabled_on_single_element_mesh(self, monkeypatch):
        import nemo_automodel.components.distributed.fsdp2 as fsdp2_mod

        monkeypatch.setattr(fsdp2_mod, "get_world_size_safe", lambda: 4)

        assert fsdp2_mod.fsdp2_sharding_enabled(SimpleNamespace(size=lambda: 1)) is False

    def test_enabled_on_multi_rank_mesh(self, monkeypatch):
        import nemo_automodel.components.distributed.fsdp2 as fsdp2_mod

        monkeypatch.setattr(fsdp2_mod, "get_world_size_safe", lambda: 4)

        assert fsdp2_mod.fsdp2_sharding_enabled(SimpleNamespace(size=lambda: 4)) is True


class TestSelectiveCheckpointSaveOps:
    """Tests for the TorchTitan-style save-op set used by selective AC."""

    def test_matmul_ops_alternate_mm_linear_and_grouped_mm(self):
        """mm/linear and the MoE grouped-GEMM variants alternate; addmm/bmm are always saved."""
        from nemo_automodel.components.distributed.activation_checkpointing import _SELECTIVE_AC_MATMUL_OPS

        # mm and linear always exist; grouped-GEMM variants are version-dependent.
        assert torch.ops.aten.mm.default in _SELECTIVE_AC_MATMUL_OPS
        assert torch.ops.aten.linear.default in _SELECTIVE_AC_MATMUL_OPS
        # addmm/bmm must NOT alternate (they are always-saved).
        assert torch.ops.aten.addmm.default not in _SELECTIVE_AC_MATMUL_OPS
        assert torch.ops.aten.bmm.default not in _SELECTIVE_AC_MATMUL_OPS
        # When available, the grouped-GEMM op (expert compute) alternates so it is
        # not unconditionally recomputed under EP.
        grouped_mm = getattr(torch.ops.aten, "_grouped_mm", None)
        if grouped_mm is not None:
            assert grouped_mm.default in _SELECTIVE_AC_MATMUL_OPS

    def test_save_ops_include_compute_and_comm_ops(self):
        """The save-set covers matmuls, attention, and communication collectives."""
        from nemo_automodel.components.distributed.activation_checkpointing import _SELECTIVE_AC_MUST_SAVE_OPS

        expected = [
            torch.ops.aten.mm.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.linear.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            torch.ops._c10d_functional.all_to_all_single.default,
        ]
        for op in expected:
            assert op in _SELECTIVE_AC_MUST_SAVE_OPS, f"{op} missing from save-op set"

    def test_save_ops_seeded_from_partitioner(self):
        """The set is seeded from PyTorch's compute-intensive op list, not hardcoded."""
        from nemo_automodel.components.distributed.activation_checkpointing import _default_compute_intensive_ops

        seeded = _default_compute_intensive_ops()
        assert isinstance(seeded, tuple)
        # mm is compute-intensive in every supported torch version.
        assert torch.ops.aten.mm.default in seeded

    def test_build_save_ops_falls_back_without_partitioner(self, monkeypatch):
        """If the private partitioner API is unavailable, the curated supplement still applies."""
        import nemo_automodel.components.distributed.activation_checkpointing as ac

        monkeypatch.setattr(ac, "_default_compute_intensive_ops", lambda: ())
        save_ops = ac._build_selective_ac_save_ops()
        # Curated supplement still provides the core matmul + attention ops.
        assert torch.ops.aten.mm.default in save_ops
        assert torch.ops.aten.addmm.default in save_ops
        assert torch.ops.aten.bmm.default in save_ops
        assert torch.ops.aten._scaled_dot_product_flash_attention.default in save_ops

    def test_resolve_op_attr_returns_none_for_missing(self):
        """Optional/absent ops resolve to None instead of raising."""
        from nemo_automodel.components.distributed.activation_checkpointing import _resolve_op_attr

        assert _resolve_op_attr(torch.ops, "definitely_not_a_namespace.foo.default") is None
        assert _resolve_op_attr(torch, "_higher_order_ops.flex_attention") is not None

    def test_trace_logs_each_op_once_with_verdict(self, caplog):
        """The opt-in policy trace logs each unique op a single time with its verdict."""
        import logging as _logging

        from torch.utils.checkpoint import CheckpointPolicy

        import nemo_automodel.components.distributed.activation_checkpointing as ac

        ac._SELECTIVE_AC_TRACE_SEEN.clear()
        with patch.object(ac, "_SELECTIVE_AC_TRACE", True):
            with caplog.at_level(_logging.INFO, logger=ac.__name__):
                ac._maybe_trace_selective_ac_decision(
                    torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE, True, is_recompute=False
                )
                # Duplicate of the same op must not log a second time.
                ac._maybe_trace_selective_ac_decision(
                    torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE, True, is_recompute=False
                )
                ac._maybe_trace_selective_ac_decision(
                    torch.ops._c10d_functional.all_to_all_single.default,
                    CheckpointPolicy.MUST_SAVE,
                    False,
                    is_recompute=False,
                )
                ac._maybe_trace_selective_ac_decision(
                    torch.ops.aten.add.Tensor, CheckpointPolicy.PREFER_RECOMPUTE, False, is_recompute=False
                )

        lines = [r.getMessage() for r in caplog.records if "[selective-ac]" in r.getMessage()]
        assert len(lines) == 3  # mm logged once (dedup), all_to_all, add
        assert any("ALTERNATE" in ln and "mm" in ln for ln in lines)
        assert any("SAVE" in ln and "all_to_all_single" in ln for ln in lines)
        assert any("RECOMPUTE" in ln and "add" in ln for ln in lines)


class TestExtractModelLayers:
    """Tests for ``_extract_model_layers`` flattening of ModuleList results.

    Covers the PR that replaced ``layers.extend(_reduce_attrs(...))`` with a
    helper that flattens ModuleList elements so each decoder layer ends up as
    its own list entry (what AC wrapping expects). PP splitting represents kept
    layer subsets as ModuleDicts, and those layer containers should be flattened
    the same way.
    """

    def _make_layers(self, n: int) -> nn.ModuleList:
        return nn.ModuleList([_FakeLayer() for _ in range(n)])

    @staticmethod
    def _bare_instance(cls):
        """Instantiate an HF model class without running HF ``__init__``.

        Needed because ``MODEL_CLS_TO_LAYERS`` is keyed by exact class identity
        (no subclass match), but the real classes require a config to
        construct. ``__new__`` + manual ``nn.Module.__init__`` gives us an
        instance where ``type(model) is cls`` while skipping the expensive
        construction path.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        return obj

    def test_class_keyed_single_fqn_flattens_modulelist(self):
        """GPT2LMHeadModel entry ``["transformer.h"]`` → individual layers.

        Before the fix, ``layers.extend(_reduce_attrs(...))`` put the ModuleList
        itself into ``layers`` as one element; hasattr(layer, 'mlp') then failed
        and AC silently skipped every layer. Flattening must restore the
        per-layer elements so the AC loop can wrap them.
        """
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        model = self._bare_instance(GPT2LMHeadModel)
        transformer = nn.Module()
        layers = self._make_layers(3)
        transformer.h = layers
        model.transformer = transformer

        result = _extract_model_layers(model)

        assert len(result) == 3, f"expected 3 flat layers, got {len(result)}"
        assert all(r is layers[i] for i, r in enumerate(result)), (
            "result should be the individual decoder layer objects, not a ModuleList container"
        )
        assert not any(isinstance(r, nn.ModuleList) for r in result)

    def test_string_keyed_arm_flattens_modulelist(self):
        """``NemotronHForCausalLM`` is string-keyed in MODEL_CLS_TO_LAYERS.

        Hits the ``model_cls.__name__ in MODEL_CLS_TO_LAYERS`` branch.
        """

        class NemotronHForCausalLM(nn.Module):
            def __init__(self, layers):
                super().__init__()
                backbone = nn.Module()
                backbone.layers = layers
                self.backbone = backbone

        layers = self._make_layers(4)
        result = _extract_model_layers(NemotronHForCausalLM(layers))

        assert len(result) == 4
        assert all(r is layers[i] for i, r in enumerate(result))

    def _attach_qwen_vl_towers(self, model, lang, vis, *, nested):
        """Attach Qwen2-VL-style towers for one historical tree shape.

        ``nested=True`` builds the standardized tree (transformers >= 4.52,
        incl. v5): ``model.language_model.layers`` + ``model.visual.blocks``.
        ``nested=False`` builds the pre-standardization tree (<= 4.51):
        ``model.layers`` + top-level ``visual.blocks``.
        """
        visual = nn.Module()
        visual.blocks = vis
        if nested:
            language_model = nn.Module()
            language_model.layers = lang
            inner = nn.Module()
            inner.language_model = language_model
            inner.visual = visual
            model.model = inner
        else:
            text_model = nn.Module()
            text_model.layers = lang
            model.model = text_model
            model.visual = visual

    @staticmethod
    def _attach_language_vision_towers(model, lang, vis, *, shape):
        """Attach Gemma3/Llava-style towers for one historical tree ``shape``.

        ``"pre_standardization"`` (<= 4.51): ``language_model.model.layers`` +
        ``vision_tower.vision_model.encoder.layers``.
        ``"standardized_v4"`` (4.52-4.x): ``model.language_model.layers`` +
        ``model.vision_tower.vision_model.encoder.layers``.
        ``"v5"`` (>= 5.0): ``model.language_model.layers`` +
        ``model.vision_tower.encoder.layers`` (flattened tower).
        """
        if shape == "pre_standardization":
            language_model = nn.Module()
            language_model.model = nn.Module()
            language_model.model.layers = lang
            vision_tower = nn.Module()
            vision_tower.vision_model = nn.Module()
            vision_tower.vision_model.encoder = nn.Module()
            vision_tower.vision_model.encoder.layers = vis
            model.language_model = language_model
            model.vision_tower = vision_tower
            return
        inner = nn.Module()
        inner.language_model = nn.Module()
        inner.language_model.layers = lang
        inner.vision_tower = nn.Module()
        if shape == "standardized_v4":
            inner.vision_tower.vision_model = nn.Module()
            inner.vision_tower.vision_model.encoder = nn.Module()
            inner.vision_tower.vision_model.encoder.layers = vis
        else:  # v5: flattened tower, no inner `vision_model`.
            inner.vision_tower.encoder = nn.Module()
            inner.vision_tower.encoder.layers = vis
        model.model = inner

    def test_multi_fqn_flattens_each_modulelist(self):
        """Qwen2.5-VL pre-standardization tree (``model.layers`` + ``visual.blocks``).

        Both groups resolve to ModuleLists; both must be flattened so all
        decoder and vision blocks appear as individual elements in the final
        list.
        """
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )

        model = self._bare_instance(Qwen2_5_VLForConditionalGeneration)
        lang = self._make_layers(5)
        vis = self._make_layers(2)
        self._attach_qwen_vl_towers(model, lang, vis, nested=False)

        result = _extract_model_layers(model)

        assert len(result) == 7
        assert [id(r) for r in result[:5]] == [id(item) for item in lang]
        assert [id(r) for r in result[5:]] == [id(item) for item in vis]
        assert not any(isinstance(r, nn.ModuleList) for r in result)

    @pytest.mark.parametrize("nested", [False, True], ids=["pre_standardization", "standardized"])
    def test_qwen2_vl_tree_shapes_extract_language_and_vision_groups(self, nested):
        """Every historical Qwen2-VL tree must resolve structurally, no version gate.

        Pre-standardization (verified transformers 4.51.3): ``model.layers`` +
        ``visual.blocks``. Standardized (verified 4.57.1/5.8.1/5.12.1):
        ``model.language_model.layers`` + ``model.visual.blocks``. Version-gated
        FQNs picked the wrong tree for parts of the 4.x line and silently
        disabled activation checkpointing.
        """
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLForConditionalGeneration,
        )

        for cls in (Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration):
            model = self._bare_instance(cls)
            lang = self._make_layers(3)
            vis = self._make_layers(2)
            self._attach_qwen_vl_towers(model, lang, vis, nested=nested)

            groups = _extract_model_layer_groups(model)

            assert set(groups) == {"language", "vision"}, cls.__name__
            assert [id(m) for m in groups["language"]] == [id(item) for item in lang]
            assert [id(m) for m in groups["vision"]] == [id(item) for item in vis]

    def test_qwen2_vl_deprecation_aliases_do_not_double_count_layers(self):
        """Standardized 4.x keeps deprecated top-level ``visual``/``language_model``
        aliases for the nested towers; first-match resolution must count each
        layer exactly once even when the historical FQN also resolves.
        """
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLForConditionalGeneration,
        )

        model = self._bare_instance(Qwen2VLForConditionalGeneration)
        lang = self._make_layers(3)
        vis = self._make_layers(2)
        self._attach_qwen_vl_towers(model, lang, vis, nested=True)
        # Simulate the transformers 4.52-4.x deprecation aliases: the same
        # towers are reachable at the historical top-level paths too.
        model.visual = model.model.visual
        model.language_model = model.model.language_model

        groups = _extract_model_layer_groups(model)

        assert [id(m) for m in groups["language"]] == [id(item) for item in lang]
        assert [id(m) for m in groups["vision"]] == [id(item) for item in vis]

    @pytest.mark.parametrize("shape", ["pre_standardization", "standardized_v4", "v5"])
    def test_gemma3_tree_shapes_extract_language_and_vision_groups(self, shape):
        """Every historical Gemma3 tree must resolve structurally, no version gate.

        Shapes verified by meta-instantiation: ``language_model.model.layers`` +
        ``vision_tower.vision_model.encoder.layers`` on 4.51.3,
        ``model.language_model.layers`` +
        ``model.vision_tower.vision_model.encoder.layers`` on 4.57.1, and
        ``model.language_model.layers`` + ``model.vision_tower.encoder.layers``
        on 5.8.1/5.12.1.
        """
        model = self._bare_instance(Gemma3ForConditionalGeneration)
        lang = self._make_layers(3)
        vis = self._make_layers(2)
        self._attach_language_vision_towers(model, lang, vis, shape=shape)

        groups = _extract_model_layer_groups(model)

        assert set(groups) == {"language", "vision"}
        assert [id(m) for m in groups["language"]] == [id(item) for item in lang]
        assert [id(m) for m in groups["vision"]] == [id(item) for item in vis]

    @pytest.mark.parametrize("shape", ["pre_standardization", "standardized_v4", "v5"])
    def test_llava_tree_shapes_extract_language_and_vision_groups(self, shape):
        """Llava-family trees share the Gemma3 lineage (CLIP instead of SigLIP);
        each historical shape must resolve structurally for every Llava class.
        """
        from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
        from transformers.models.llava_next.modeling_llava_next import (
            LlavaNextForConditionalGeneration,
        )
        from transformers.models.llava_next_video.modeling_llava_next_video import (
            LlavaNextVideoForConditionalGeneration,
        )
        from transformers.models.llava_onevision.modeling_llava_onevision import (
            LlavaOnevisionForConditionalGeneration,
        )

        for cls in (
            LlavaForConditionalGeneration,
            LlavaNextForConditionalGeneration,
            LlavaNextVideoForConditionalGeneration,
            LlavaOnevisionForConditionalGeneration,
        ):
            model = self._bare_instance(cls)
            lang = self._make_layers(3)
            vis = self._make_layers(2)
            self._attach_language_vision_towers(model, lang, vis, shape=shape)

            groups = _extract_model_layer_groups(model)

            assert set(groups) == {"language", "vision"}, cls.__name__
            assert [id(m) for m in groups["language"]] == [id(item) for item in lang], cls.__name__
            assert [id(m) for m in groups["vision"]] == [id(item) for item in vis], cls.__name__

    def test_spec_resolving_no_modules_warns_and_returns_empty(self, caplog):
        """A mapped model class whose spec FQNs all fail to resolve must warn.

        This is the transformers-version-drift failure mode: extraction used to
        return ``{}`` silently and activation checkpointing became a no-op.
        """
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLForConditionalGeneration,
        )

        # A tree that matches no known Qwen2-VL shape: top-level
        # `language_model.layers` (never a registered module path for this
        # class; it only ever existed as a deprecation alias).
        model = self._bare_instance(Qwen2VLForConditionalGeneration)
        language_model = nn.Module()
        language_model.layers = self._make_layers(2)
        model.language_model = language_model

        with caplog.at_level("WARNING", logger=parallelizer.logger.name):
            groups = _extract_model_layer_groups(model)

        assert groups == {}
        assert "Qwen2VLForConditionalGeneration" in caplog.text
        assert "model.language_model.layers" in caplog.text
        assert "model.visual.blocks" in caplog.text

    def test_moduledict_layer_container_flattens(self):
        """PP post-split: ``_reduce_attrs`` returns a ModuleDict.

        The pipeline splitter replaces a ModuleList with a numeric-key
        ModuleDict. ``_extract_model_layers`` must still return individual
        layers so AC, TP follow-up logic, and FSDP layer handling see the same
        shape as the unsplit path.
        """
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        model = self._bare_instance(GPT2LMHeadModel)
        layer_dict = nn.ModuleDict({"0": _FakeLayer(), "1": _FakeLayer()})
        transformer = nn.Module()
        transformer.h = layer_dict
        model.transformer = transformer

        result = _extract_model_layers(model)

        assert len(result) == 2
        assert [id(r) for r in result] == [id(v) for v in layer_dict.values()]

    def test_fallback_branch_still_handles_modulelist(self):
        """Non-MODEL_CLS_TO_LAYERS models hit the ``hasattr(model.model, 'layers')``
        fallback, which is unchanged by the PR. Guard against accidental regression.
        """

        class GenericCausalLM(nn.Module):
            def __init__(self, layers):
                super().__init__()
                inner = nn.Module()
                inner.layers = layers
                self.model = inner

        layers = self._make_layers(2)
        result = _extract_model_layers(GenericCausalLM(layers))
        assert len(result) == 2
        assert all(r is layers[i] for i, r in enumerate(result))

    def test_fallback_branch_handles_moduledict(self):
        """Fallback branch already normalises ModuleDict via ``.values()``."""

        class GenericCausalLM(nn.Module):
            def __init__(self, layer_dict):
                super().__init__()
                inner = nn.Module()
                inner.layers = layer_dict
                self.model = inner

        layer_dict = nn.ModuleDict({"0": _FakeLayer(), "1": _FakeLayer(), "2": _FakeLayer()})
        result = _extract_model_layers(GenericCausalLM(layer_dict))
        assert len(result) == 3
        assert [id(r) for r in result] == [id(v) for v in layer_dict.values()]

    def test_heuristic_ignores_named_moduledict(self):
        """The unknown-model heuristic should not treat arbitrary ModuleDicts as layers."""

        class UnknownWithAdapterRegistry(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapters = nn.ModuleDict({"default": nn.Linear(4, 4)})

        with pytest.raises(ValueError, match="no ModuleList or ModuleDict found"):
            _extract_model_layers(UnknownWithAdapterRegistry())

    def test_string_keyed_mistral3_fp8_vlm(self):
        """The ``"Mistral3FP8VLMForConditionalGeneration"`` string-key entry
        catches the runtime class produced by ``_get_mixin_wrapped_class``
        (``HFCheckpointingMixin``), which has the same ``__name__`` as our
        custom class but a distinct identity. Without this entry, the model
        falls through to the largest-ModuleList heuristic and crashes.

        Validates the elif ``model_cls.__name__ in MODEL_CLS_TO_LAYERS`` branch.
        """

        class Mistral3FP8VLMForConditionalGeneration(nn.Module):
            """Stand-in named exactly like the registered string key —
            mirrors the wrapper class that NeMo Auto creates at runtime."""

            def __init__(self):
                super().__init__()
                inner = nn.Module()
                lang = nn.Module()
                lang.layers = self._mklayers(3)
                inner.language_model = lang
                vt = nn.Module()
                tx = nn.Module()
                tx.layers = self._mklayers(2)
                vt.transformer = tx
                inner.vision_tower = vt
                self.model = inner

            @staticmethod
            def _mklayers(n):
                return nn.ModuleList([_FakeLayer() for _ in range(n)])

        model = Mistral3FP8VLMForConditionalGeneration()
        result = _extract_model_layers(model)

        # 3 text-decoder + 2 vision tower layers, all flattened.
        assert len(result) == 5
        assert not any(isinstance(r, nn.ModuleList) for r in result)


    def test_activation_checkpointing_scope_filtering(self):
        language = [_FakeLayer(), _FakeLayer()]
        vision = [_FakeLayer()]
        audio = [_FakeLayer()]
        vision[0].requires_grad_(False)

        groups = {"language": language, "vision": vision, "audio": audio}

        selected, scopes = _filter_layer_groups_for_activation_checkpointing(groups, "language")
        assert scopes == ("language",)
        assert selected == language

        selected, scopes = _filter_layer_groups_for_activation_checkpointing(groups, "multimodal")
        assert scopes == ("multimodal",)
        assert selected == audio

        selected, scopes = _filter_layer_groups_for_activation_checkpointing(groups, ["language", "vision"])
        assert scopes == ("language", "vision")
        assert selected == language

        selected, scopes = _filter_layer_groups_for_activation_checkpointing(groups, "all")
        assert scopes == ("all",)
        assert selected == language + audio

    def test_activation_checkpointing_scope_filtering_warns_when_scope_has_only_frozen_layers(self, caplog):
        vision = [_FakeLayer()]
        vision[0].requires_grad_(False)

        selected, scopes = _filter_layer_groups_for_activation_checkpointing({"vision": vision}, "vision")

        assert scopes == ("vision",)
        assert selected == []
        assert "selected no layers" in caplog.text

    def test_string_keyed_new_vlm_families_extract_language_and_vision_layers(self):
        """Native VLM families in examples should not fall back to the largest-layer heuristic."""

        def _layers(count):
            return nn.ModuleList([_FakeLayer() for _ in range(count)])

        def _assert_counts(model, language_count, vision_count):
            groups = _extract_model_layer_groups(model)
            result = _extract_model_layers(model)
            assert set(groups) == {"language", "vision"}
            assert len(groups["language"]) == language_count
            assert len(groups["vision"]) == vision_count
            assert result == groups["language"] + groups["vision"]

        class KimiVLForConditionalGeneration(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.layers = _layers(3)
                self.model.vision_tower = nn.Module()
                self.model.vision_tower.encoder = nn.Module()
                self.model.vision_tower.encoder.blocks = _layers(2)

        class KimiK25VLForConditionalGeneration(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.layers = _layers(4)
                self.model.vision_tower = nn.Module()
                self.model.vision_tower.encoder = nn.Module()
                self.model.vision_tower.encoder.blocks = _layers(2)

        class MiniMaxM3SparseForConditionalGeneration(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleDict({str(i): _FakeLayer() for i in range(5)})
                self.vision_tower = nn.Module()
                self.vision_tower.vision_model = nn.Module()
                self.vision_tower.vision_model.encoder = nn.Module()
                self.vision_tower.vision_model.encoder.layers = _layers(2)

        class Qwen3_5MoeForConditionalGeneration(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.layers = _layers(6)
                self.model.visual = nn.Module()
                self.model.visual.blocks = _layers(3)

        class Qwen3VLMoeForConditionalGeneration(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.layers = _layers(8)
                self.model.visual = nn.Module()
                self.model.visual.blocks = _layers(3)

        class Step3p7ForConditionalGeneration(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.layers = _layers(7)
                self.model.vision_model = nn.Module()
                self.model.vision_model.transformer = nn.Module()
                self.model.vision_model.transformer.resblocks = _layers(2)

        _assert_counts(KimiVLForConditionalGeneration(), 3, 2)
        _assert_counts(KimiK25VLForConditionalGeneration(), 4, 2)
        _assert_counts(MiniMaxM3SparseForConditionalGeneration(), 5, 2)
        _assert_counts(Qwen3_5MoeForConditionalGeneration(), 6, 3)
        _assert_counts(Qwen3VLMoeForConditionalGeneration(), 8, 3)
        _assert_counts(Step3p7ForConditionalGeneration(), 7, 2)

    def test_string_keyed_bagel_extracts_language_and_vision_layers(self):
        """BAGEL exposes Qwen decoder layers and SigLIP encoder layers."""
        model = _make_bagel_model(num_language_layers=2, num_vision_layers=3)

        result = _extract_model_layers(model)

        language_layers = model.model.language_model.model.layers
        vision_layers = model.model.vit_model.vision_model.encoder.layers
        assert len(result) == 5
        assert [id(r) for r in result[:2]] == [id(layer) for layer in language_layers]
        assert [id(r) for r in result[2:]] == [id(layer) for layer in vision_layers]


class TestBagelFullLayerActivationCheckpointing:
    """Tests for native BAGEL-style whole-layer activation checkpointing."""

    def test_get_module_by_fqn_resolves_nested_module_and_missing_path(self):
        """Nested FQN lookup returns the module or None for missing paths."""
        model = _make_bagel_model()

        result = parallelizer._get_module_by_fqn(model, "model.vit_model.vision_model.encoder.layers")

        assert result is model.model.vit_model.vision_model.encoder.layers
        assert parallelizer._get_module_by_fqn(model, "model.missing.layers") is None

    def test_apply_bagel_full_layer_activation_checkpointing_wraps_each_layer(self, monkeypatch):
        """BAGEL wraps Qwen and SigLIP layers once and skips already wrapped layers."""
        model = _make_bagel_model(num_language_layers=2, num_vision_layers=3)
        wrap_calls = []

        def _fake_checkpoint_wrapper(module, **kwargs):
            wrap_calls.append((module, kwargs))
            return _CheckpointWrapped(module, **kwargs)

        monkeypatch.setattr(parallelizer, "checkpoint_wrapper", _fake_checkpoint_wrapper)

        assert parallelizer._apply_bagel_full_layer_activation_checkpointing(model) is True

        language_layers = model.model.language_model.model.layers
        vision_layers = model.model.vit_model.vision_model.encoder.layers
        wrapped_layers = list(language_layers) + list(vision_layers)
        assert len(wrap_calls) == 5
        assert all(isinstance(layer, _CheckpointWrapped) for layer in wrapped_layers)
        assert all(call_kwargs["checkpoint_impl"].name == "NO_REENTRANT" for _, call_kwargs in wrap_calls)

        assert parallelizer._apply_bagel_full_layer_activation_checkpointing(model) is False
        assert len(wrap_calls) == 5

    def test_apply_bagel_full_layer_activation_checkpointing_ignores_other_models(self):
        """Non-BAGEL models continue through the generic checkpointing path."""
        assert parallelizer._apply_bagel_full_layer_activation_checkpointing(nn.Module()) is False
