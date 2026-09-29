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

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_vl_moe.state_dict_adapter import Qwen3VLMoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


@pytest.fixture
def config():
    cfg = Mock()
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
    return Qwen3VLMoeStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32)


class TestInitialization:
    def test_sets_expected_attributes(self, config, moe_config, backend_config):
        adapter = Qwen3VLMoeStateDictAdapter(
            config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float16
        )

        assert adapter.config is config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend_config
        assert adapter.dtype == torch.float16
        assert adapter._uses_model_prefix is True


class TestToHF:
    def test_renames_expert_keys(self, adapter):
        gate = torch.randn(4, 64, 128)
        down = torch.randn(4, 128, 64)
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": gate,
            "model.language_model.layers.0.mlp.experts.down_projs": down,
        }

        out = adapter.to_hf(state_dict)

        assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in out
        assert "model.language_model.layers.0.mlp.experts.down_proj" in out

    def test_passes_tensors_through_unchanged(self, adapter):
        """to_hf is a pure rename — same tensor objects, no copy, no dtype cast."""
        gate = torch.randn(4, 64, 128, dtype=torch.float16)
        down = torch.randn(4, 128, 64, dtype=torch.float16)
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": gate,
            "model.language_model.layers.0.mlp.experts.down_projs": down,
        }

        out = adapter.to_hf(state_dict)

        assert out["model.language_model.layers.0.mlp.experts.gate_up_proj"] is gate
        assert out["model.language_model.layers.0.mlp.experts.down_proj"] is down

    def test_non_expert_keys_pass_through(self, adapter):
        tensor = torch.randn(16, 16)
        state_dict = {
            "model.language_model.layers.0.self_attn.q_proj.weight": tensor,
        }

        out = adapter.to_hf(state_dict)

        assert out["model.language_model.layers.0.self_attn.q_proj.weight"] is tensor

    def test_respects_exclude_regex(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 128),
            "exclude.me": torch.randn(1),
        }

        out = adapter.to_hf(state_dict, exclude_key_regex=r"^exclude")

        assert "exclude.me" not in out

    def test_device_mesh_kwarg_ignored(self, adapter):
        """to_hf no longer uses device_mesh — it should be silently ignored."""
        gate = torch.randn(4, 64, 128)
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": gate,
        }

        out = adapter.to_hf(state_dict, device_mesh=Mock())

        assert out["model.language_model.layers.0.mlp.experts.gate_up_proj"] is gate


class TestFromHF:
    def test_detects_model_prefix(self, adapter):
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
        }

        adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is True

    def test_handles_missing_model_prefix(self, adapter):
        hf_state = {
            "language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
        }

        out = adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is False
        assert "language_model.layers.0.mlp.experts.gate_and_up_projs" in out
        assert "language_model.layers.0.mlp.experts.down_projs" in out

    def test_renames_expert_keys_without_mesh(self, adapter):
        gate_up = torch.randn(4, 32, 64, dtype=torch.float16)
        down = torch.randn(4, 64, 32, dtype=torch.float16)

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        }

        out = adapter.from_hf(hf_state)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.language_model.layers.0.mlp.experts.down_projs"

        assert gate_key in out
        assert down_key in out
        # Without device_mesh, slices full range and casts to adapter.dtype
        torch.testing.assert_close(out[gate_key], gate_up.to(adapter.dtype))
        torch.testing.assert_close(out[down_key], down.to(adapter.dtype))

    def test_dtensor_passthrough_on_dcp_path(self, monkeypatch, adapter):
        """When values are DTensors (DCP path), from_hf just renames — no slicing, no create_dtensor."""
        gate_up = torch.randn(4, 16, 32)
        down = torch.randn(4, 32, 16)

        class FakeDTensor:
            def __init__(self, data):
                self._data = data

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor",
            lambda tensor: isinstance(tensor, FakeDTensor),
        )

        fake_gate = FakeDTensor(gate_up)
        fake_down = FakeDTensor(down)

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": fake_gate,
            "model.language_model.layers.0.mlp.experts.down_proj": fake_down,
        }

        out = adapter.from_hf(hf_state)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.language_model.layers.0.mlp.experts.down_projs"

        # DTensors should be passed through as-is (same object, no slicing)
        assert out[gate_key] is fake_gate
        assert out[down_key] is fake_down

    def test_filters_scale_inv_keys(self, adapter):
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.weight_scale_inv": torch.tensor([99.0]),
        }

        out = adapter.from_hf(hf_state)

        assert not any("scale_inv" in k for k in out)

    def test_keeps_top_level_lm_head_key_with_model_prefixed_checkpoint(self, adapter):
        lm_head = torch.randn(8, 8)
        hf_state = {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
            "lm_head.weight": lm_head,
        }

        out = adapter.from_hf(hf_state)

        assert out["lm_head.weight"] is lm_head
        assert "model.lm_head.weight" not in out


class TestConvertSingleTensorToHf:
    def test_expert_tensor_passthrough(self, adapter):
        tensor = torch.randn(4, 16, 32)
        fqn = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == "model.language_model.layers.0.mlp.experts.gate_up_proj"
        assert value is tensor  # same object, no copy

    def test_non_expert_tensor_passthrough(self, adapter):
        tensor = torch.randn(16, 16)
        fqn = "model.language_model.layers.0.self_attn.q_proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == fqn
        assert value is tensor

    def test_exclude_regex_filters_results(self, adapter):
        tensor = torch.randn(16, 16)
        fqn = "exclude.me"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

        assert result == []


# ---------------------------------------------------------------------------
# from_hf  –  ep_shard multi-node scenarios
# ---------------------------------------------------------------------------
class TestFromHFEpShard:
    """Tests for from_hf with ep_shard > 1 (multi-node expert FSDP sharding)."""

    def _setup_from_hf_mocks(self, monkeypatch, ep_range, ep_shard_size, ep_shard_rank):
        """Shared mock setup for from_hf ep_shard tests."""
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n: ep_range,
        )

        mock_ep_sub = Mock()
        mock_ep_sub.get_rank.return_value = 0

        mock_ep_shard_sub = Mock()
        mock_ep_shard_sub.size.return_value = ep_shard_size
        mock_ep_shard_sub.get_local_rank.return_value = ep_shard_rank

        def fake_get_submesh(mesh, dims):
            if dims == ("ep",):
                return mock_ep_sub
            if dims == ("ep_shard",):
                return mock_ep_shard_sub
            return Mock()

        monkeypatch.setattr("nemo_automodel.components.moe.state_dict_utils.get_submesh", fake_get_submesh)

        captured_list = []

        def fake_create_dtensor(local_tensor, mesh, rank):
            captured_list.append(local_tensor)
            return local_tensor

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.create_dtensor_from_local",
            fake_create_dtensor,
        )

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep_shard", "ep"]

        return device_mesh, captured_list

    def test_from_hf_slices_ep_shard_dim(self, adapter, monkeypatch):
        """With ep_shard_size=2, from_hf must slice dim 1 by ep_shard rank."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        inter, hidden = 8, 4
        ep_shard_size, ep_shard_rank = 2, 1
        local_experts = n_experts // 2  # 2

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, local_experts), ep_shard_size=ep_shard_size, ep_shard_rank=ep_shard_rank
        )

        gate_up = torch.arange(n_experts * inter * hidden, dtype=adapter.dtype).reshape(n_experts, inter, hidden)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(
                n_experts, hidden, inter, dtype=adapter.dtype
            ),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        # First captured tensor is gate_and_up_projs (dict is insertion-ordered)
        local_gate = captured_list[0]
        chunk = inter // ep_shard_size
        assert local_gate.shape == (local_experts, chunk, hidden)
        expected = gate_up[:local_experts, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
        torch.testing.assert_close(local_gate, expected.to(adapter.dtype))

    def test_from_hf_no_ep_shard_unchanged(self, adapter, monkeypatch):
        """With ep_shard_size=1 (single-node), from_hf must NOT slice dim 1."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        inter, hidden = 8, 4

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, n_experts), ep_shard_size=1, ep_shard_rank=0
        )

        gate_up = torch.randn(n_experts, inter, hidden, dtype=adapter.dtype)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(
                n_experts, hidden, inter, dtype=adapter.dtype
            ),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        local_gate = captured_list[0]
        assert local_gate.shape == (n_experts, inter, hidden)
        torch.testing.assert_close(local_gate, gate_up.to(adapter.dtype))

    def test_from_hf_ep_shard_roundtrip(self, adapter, monkeypatch):
        """to_hf → from_hf roundtrip: data at a specific ep_shard rank must be recoverable."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        inter, hidden = 8, 4
        ep_shard_size, ep_shard_rank = 2, 0

        original = torch.arange(n_experts * inter * hidden, dtype=adapter.dtype).reshape(n_experts, inter, hidden)

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, n_experts), ep_shard_size=ep_shard_size, ep_shard_rank=ep_shard_rank
        )

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": original.clone(),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(
                n_experts, hidden, inter, dtype=adapter.dtype
            ),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        local_gate = captured_list[0]
        chunk = inter // ep_shard_size
        expected_shard = original[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
        torch.testing.assert_close(local_gate, expected_shard)
