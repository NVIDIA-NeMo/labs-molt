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
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.common import BackendConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def config():
    cfg = Mock(spec=Qwen3MoeConfig)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
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
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
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
    return Qwen3MoeStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32)


class TestInitialization:
    def test_sets_expected_attributes(self, config, moe_config, backend_config):
        adapter = Qwen3MoeStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float16)

        assert adapter.config is config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend_config
        assert adapter.dtype == torch.float16
        assert adapter._uses_model_prefix is True


class TestToHF:
    def test_merges_experts_into_individual_tensors(self, adapter):
        state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 128),
            "model.layers.0.mlp.experts.down_projs": torch.randn(4, 64, 64),
        }

        out = adapter.to_hf(state_dict)

        keys = set(out.keys())
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" in keys
        assert "model.layers.0.mlp.experts.0.up_proj.weight" in keys
        assert "model.layers.0.mlp.experts.0.down_proj.weight" in keys
        assert not any(k.endswith("gate_and_up_projs") or k.endswith("down_projs") for k in keys)

    def test_respects_exclude_regex(self, adapter):
        state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 128),
            "exclude.me": torch.randn(1),
        }

        out = adapter.to_hf(state_dict, exclude_key_regex=r"^exclude")

        assert "exclude.me" not in out


class TestFromHF:
    def test_detects_model_prefix(self, adapter):
        hf_state = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(64, 32),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(64, 32),
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(32, 64),
            "model.layers.0.mlp.experts.1.gate_proj.weight": torch.randn(64, 32),
            "model.layers.0.mlp.experts.1.up_proj.weight": torch.randn(64, 32),
            "model.layers.0.mlp.experts.1.down_proj.weight": torch.randn(32, 64),
            "model.layers.0.mlp.experts.2.gate_proj.weight": torch.randn(64, 32),
            "model.layers.0.mlp.experts.2.up_proj.weight": torch.randn(64, 32),
            "model.layers.0.mlp.experts.2.down_proj.weight": torch.randn(32, 64),
            "model.layers.0.mlp.experts.3.gate_proj.weight": torch.randn(64, 32),
            "model.layers.0.mlp.experts.3.up_proj.weight": torch.randn(64, 32),
            "model.layers.0.mlp.experts.3.down_proj.weight": torch.randn(32, 64),
        }

        adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is True

    def test_converts_router_weights_when_present(self, adapter):
        hf_state = {
            "model.layers.0.mlp.router.weight": torch.randn(64, 64),
            "model.layers.0.mlp.router.bias": torch.randn(64),
        }

        out = adapter.from_hf(hf_state)

        # Router keys remain untouched because Qwen3 MoE uses pure MoE layers
        assert "model.layers.0.mlp.router.weight" in out
        assert "model.layers.0.mlp.router.bias" in out

    def test_combines_expert_weights_when_all_available(self, adapter):
        hf_state = {
            f"model.layers.0.mlp.experts.{expert}.{proj}.weight": torch.randn(64, 32 if proj != "down_proj" else 64)
            for expert in range(4)
            for proj in ["gate_proj", "up_proj", "down_proj"]
        }

        out = adapter.from_hf(hf_state)

        assert "model.layers.0.mlp.experts.gate_and_up_projs" in out
        assert "model.layers.0.mlp.experts.down_projs" in out

    def test_handles_quantized_weights_gracefully(self, adapter):
        hf_state = {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.ones(1),
            "model.layers.0.mlp.experts.gate_up_proj_scales": torch.ones(1),
        }

        out = adapter.from_hf(hf_state)

        assert "model.layers.0.mlp.experts.gate_up_proj_blocks" in out
        assert "model.layers.0.mlp.experts.gate_up_proj_scales" in out


class TestConvertSingleTensorToHf:
    def test_expert_tensor_conversion(self, adapter):
        tensor = torch.randn(4, 64, 128)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts') as mock_convert:
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

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts') as mock_convert:
            mock_convert.return_value = None

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert torch.equal(result[0][1], tensor)

    def test_exclude_key_regex(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "exclude_this.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

            assert len(result) == 0

    def test_expert_tensor_with_exclude_regex(self, adapter):
        tensor = torch.randn(4, 64, 128)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts') as mock_convert:
            mock_convert.return_value = [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(64, 64)),
                ("exclude_me.weight", torch.randn(64, 64)),
            ]

            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

            assert len(result) == 1
            assert result[0][0] == "model.layers.0.mlp.experts.0.gate_proj.weight"
            assert "exclude_me.weight" not in [k for k, _ in result]

    def test_preserves_tensor_identity_for_non_experts(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.layers.0.self_attn.q_proj.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1] is tensor
