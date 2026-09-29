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

import importlib.util
import sys
import types
from unittest.mock import Mock, patch

import pytest
import torch

try:
    import fast_hadamard_transform  # noqa: F401
except ImportError:
    if "fast_hadamard_transform" not in sys.modules:
        mock_hadamard = types.ModuleType("fast_hadamard_transform")
        mock_hadamard.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
        mock_hadamard.hadamard_transform = lambda x, scale: x
        sys.modules["fast_hadamard_transform"] = mock_hadamard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm4_moe.state_dict_adapter import Glm4MoeStateDictAdapter
from nemo_automodel.components.models.glm_moe_dsa.state_dict_adapter import (
    _GLM_TRITON_AVAILABLE,
    GlmMoeDsaStateDictAdapter,
    _dequantize_glm_fp8,
    _dequantize_glm_with_torch_offsets,
    _dequantize_glm_with_triton_offsets,
    _slice_glm_scale_for_dtensor,
    should_quantize_key,
)
from nemo_automodel.components.moe.config import MoEConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_experts = 4
    cfg.quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    return cfg


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def adapter(config, moe_config, backend_config):
    return GlmMoeDsaStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32)


class TestGlmMoeDsaStateDictAdapterInheritance:
    def test_inherits_from_glm4_moe_adapter(self):
        assert issubclass(GlmMoeDsaStateDictAdapter, Glm4MoeStateDictAdapter)

    def test_has_indexer_non_quantized_keys(self):
        expected = [
            "indexer.k_norm.weight",
            "indexer.k_norm.bias",
            "indexer.weights_proj.weight",
        ]
        assert GlmMoeDsaStateDictAdapter._indexer_non_quantized_keys == expected


class TestConvertSingleTensorToHf:
    def test_expert_tensor_conversion(self, adapter):
        tensor = torch.randn(4, 64, 128)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts") as mock_convert:
            mock_convert.return_value = [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(64, 64)),
                ("model.layers.0.mlp.experts.0.up_proj.weight", torch.randn(64, 64)),
            ]

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            mock_convert.assert_called_once_with(fqn, tensor)
            assert len(result) == 2
            assert result[0][0] == "model.layers.0.mlp.experts.0.gate_proj.weight"
            assert result[1][0] == "model.layers.0.mlp.experts.0.up_proj.weight"

    def test_non_expert_tensor_conversion(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.attention.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts") as mock_convert:
            mock_convert.return_value = None

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert torch.equal(result[0][1], tensor)

    def test_preserves_tensor_identity_for_non_experts(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1] is tensor

    def test_exclude_key_regex(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "exclude_this.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

            assert len(result) == 0

    def test_expert_tensor_with_exclude_regex(self, adapter):
        tensor = torch.randn(4, 64, 128)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts") as mock_convert:
            mock_convert.return_value = [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(64, 64)),
                ("exclude_me.weight", torch.randn(64, 64)),
            ]

            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

            assert len(result) == 1
            assert result[0][0] == "model.layers.0.mlp.experts.0.gate_proj.weight"
            assert "exclude_me.weight" not in [k for k, _ in result]

    def test_exclude_key_regex_no_match(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r".*kv_proj.*")

            assert len(result) == 1
            assert result[0][0] == fqn


class TestConvertSingleTensorToHfQuantization:
    def test_quantization_normal_weight(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_a_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 2
            assert result[0][0] == fqn
            assert result[0][1].dtype == torch.float8_e4m3fn
            assert result[1][0] == fqn + "_scale_inv"
            assert result[1][1].shape == (1, 1)

    def test_quantization_skips_non_weight_keys(self, adapter):
        tensor = torch.randn(64)
        fqn = "model.layers.0.self_attn.q_proj.bias"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_skips_indexer_k_norm_weight(self, adapter):
        tensor = torch.randn(64)
        fqn = "model.layers.0.self_attn.indexer.k_norm.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_skips_indexer_k_norm_bias(self, adapter):
        tensor = torch.randn(64)
        fqn = "model.layers.0.self_attn.indexer.k_norm.bias"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_skips_indexer_weights_proj(self, adapter):
        tensor = torch.randn(64, 128)
        fqn = "model.layers.0.self_attn.indexer.weights_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_applies_to_indexer_linear_weights(self, adapter):
        tensor = torch.randn(64, 128)
        fqn = "model.layers.0.self_attn.indexer.wq_b.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 2
            assert result[0][0] == fqn
            assert result[0][1].dtype == torch.float8_e4m3fn
            assert result[1][0] == fqn + "_scale_inv"

    @pytest.mark.parametrize(
        "fqn",
        [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_a_layernorm.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.78.eh_proj.weight",
            "model.layers.78.enorm.weight",
            "lm_head.weight",
        ],
    )
    def test_quantization_skips_unscaled_glm53_weights(self, adapter, fqn):
        tensor = torch.randn(64, 64)

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

        assert len(result) == 1
        assert result[0][0] == fqn
        assert result[0][1] is tensor

    def test_quantization_is_not_enabled_for_bf16_checkpoint(self, adapter):
        adapter.config.quantization_config = None
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_a_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

        assert len(result) == 1
        assert result[0][0] == fqn
        assert result[0][1] is tensor

    def test_without_quantization_preserves_dtype(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_a_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=False)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_quantization_with_exclude_regex(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_a_proj.weight"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            result = adapter.convert_single_tensor_to_hf(
                fqn, tensor, quantization=True, exclude_key_regex=r".*q_a_proj.*"
            )

            assert len(result) == 0


class TestGlm53Fp8Dequantization:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("model.layers.0.self_attn.q_a_proj.weight", True),
            ("model.layers.10.mlp.experts.3.down_proj.weight", True),
            ("model.layers.0.self_attn.indexer.wq_b.weight", True),
            ("model.layers.0.input_layernorm.weight", False),
            ("model.layers.0.self_attn.indexer.k_norm.weight", False),
            ("model.layers.0.self_attn.indexer.weights_proj.weight", False),
            ("model.layers.10.mlp.gate.weight", False),
            ("model.layers.78.eh_proj.weight", False),
            ("model.layers.78.enorm.weight", False),
            ("model.embed_tokens.weight", False),
            ("lm_head.weight", False),
        ],
    )
    def test_should_quantize_key_matches_glm53_checkpoint(self, key, expected):
        assert should_quantize_key(key) is expected

    def test_dequantize_applies_scale_and_removes_scale_tensor(self, adapter):
        weight = torch.full((128, 128), 2.0).to(torch.float8_e4m3fn)
        scale_inv = torch.full((1, 1), 0.25)
        state_dict = {
            "model.layers.0.self_attn.q_a_proj.weight": weight,
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv": scale_inv,
            "model.layers.0.input_layernorm.weight": torch.ones(128),
        }

        result = adapter._dequantize(state_dict)

        assert result["model.layers.0.self_attn.q_a_proj.weight"].dtype == torch.float32
        assert torch.all(result["model.layers.0.self_attn.q_a_proj.weight"] == 0.5)
        assert "model.layers.0.self_attn.q_a_proj.weight_scale_inv" not in result
        assert "model.layers.0.input_layernorm.weight" in result

    def test_unaligned_shard_slices_scales_from_checkpoint_metadata(self):
        """A 72-row shard starting at row 72 spans two global 128-row blocks."""
        from torch.distributed.checkpoint.metadata import ChunkStorageMetadata

        class FakeDTensor:
            def __create_chunk_list__(self):
                return [ChunkStorageMetadata(offsets=torch.Size((72, 0)), sizes=torch.Size((72, 128)))]

        scale_inv = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        weight_local = torch.ones((72, 128), dtype=torch.float8_e4m3fn)

        result = _slice_glm_scale_for_dtensor(scale_inv, FakeDTensor(), weight_local)

        torch.testing.assert_close(result, scale_inv[:2])

    def test_torch_dequant_preserves_global_block_boundaries(self):
        """Rows 72..143 use 56 values from block 0 and 16 from block 1."""
        weight = torch.ones((72, 128), dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor([[2.0], [4.0]], dtype=torch.float32)

        result = _dequantize_glm_with_torch_offsets(
            weight,
            scale_inv,
            torch.float32,
            128,
            offsets_within_first_block=(72, 0),
        )

        assert torch.all(result[:56] == 2.0)
        assert torch.all(result[56:] == 4.0)

    def test_full_dtensor_path_uses_global_block_offsets(self):
        local_weight = torch.ones((72, 128), dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
        mock_weight = Mock()
        mock_weight.to_local.return_value = local_weight
        mock_weight.device_mesh = Mock()
        mock_weight.placements = [Mock()]

        with (
            patch(
                "nemo_automodel.components.models.glm_moe_dsa.state_dict_adapter.is_dtensor",
                side_effect=lambda value: value is mock_weight,
            ),
            patch(
                "nemo_automodel.components.models.glm_moe_dsa.state_dict_adapter._glm_dtensor_local_offsets",
                return_value=(72, 0),
            ),
            patch(
                "torch.distributed._tensor.DTensor.from_local",
                side_effect=lambda local, *_args: local,
            ),
        ):
            result = _dequantize_glm_fp8(mock_weight, global_scale, dtype=torch.float32)

        assert torch.all(result[:56] == 2.0)
        assert torch.all(result[56:] == 4.0)

    @pytest.mark.skipif(not _GLM_TRITON_AVAILABLE, reason="Triton is required")
    def test_triton_offset_dequant_matches_torch(self):
        weight = torch.ones((72, 128), dtype=torch.float8_e4m3fn, device="cuda")
        scale_inv = torch.tensor([[2.0], [4.0]], dtype=torch.float32, device="cuda")
        offsets = (72, 0)

        expected = _dequantize_glm_with_torch_offsets(weight, scale_inv, torch.float32, 128, offsets)
        actual = _dequantize_glm_with_triton_offsets(weight, scale_inv, torch.float32, 128, offsets)

        torch.testing.assert_close(actual, expected)
