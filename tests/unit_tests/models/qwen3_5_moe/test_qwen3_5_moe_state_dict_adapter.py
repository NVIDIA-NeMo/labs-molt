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

import json
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file

import nemo_automodel.components.models.qwen3_5_moe.model as qwen3_5_moe_model
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import (
    Qwen3_5MoeStateDictAdapter,
    _infer_base_expert_hf_layout,
)
from nemo_automodel.components.moe.layers import MoEConfig


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 64
    cfg.shared_expert_intermediate_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg._name_or_path = "Qwen/Qwen3.6-35B-A3B"
    cfg.name_or_path = "Qwen/Qwen3.6-35B-A3B"
    return cfg


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=64,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.001,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=64,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def adapter(config, moe_config, backend_config):
    return Qwen3_5MoeStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32)


@pytest.fixture
def split_fp8_adapter(backend_config):
    moe_config = MoEConfig(
        dim=3,
        inter_dim=2,
        moe_inter_dim=2,
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
    )
    return Qwen3_5MoeStateDictAdapter(
        config=SimpleNamespace(_name_or_path="", name_or_path=""),
        moe_config=moe_config,
        backend=backend_config,
        dtype=torch.float32,
        text_only=True,
    )


def _fp8_weight(rows, cols, offset):
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return (values / 16.0 + offset).to(dtype=torch.float8_e4m3fn)


def _build_split_fp8_hf_state(adapter):
    hidden_dim = adapter.moe_config.dim
    inter_dim = adapter.moe_config.moe_inter_dim
    hf_state = {}
    expected_gate_up = []
    expected_down = []

    for expert_id in range(adapter.moe_config.n_routed_experts):
        projections = {}
        projection_specs = (
            ("gate_proj", (inter_dim, hidden_dim), 1.0),
            ("up_proj", (inter_dim, hidden_dim), 0.5),
            ("down_proj", (hidden_dim, inter_dim), 0.25),
        )
        for projection_index, (projection, shape, scale) in enumerate(projection_specs):
            key = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
            weight = _fp8_weight(*shape, offset=0.25 + expert_id + projection_index / 8.0)
            scale_inv = torch.full((1, 1), scale, dtype=torch.bfloat16)
            hf_state[key] = weight
            hf_state[f"{key}_scale_inv"] = scale_inv
            projections[projection] = weight.float() * scale_inv.float()[0, 0]

        expected_gate_up.append(
            torch.cat(
                (projections["gate_proj"].transpose(0, 1), projections["up_proj"].transpose(0, 1)),
                dim=1,
            )
        )
        expected_down.append(projections["down_proj"].transpose(0, 1))

    return hf_state, torch.stack(expected_gate_up, dim=0), torch.stack(expected_down, dim=0)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestInitialization:
    def test_sets_expected_attributes(self, config, moe_config, backend_config):
        adapter = Qwen3_5MoeStateDictAdapter(
            config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float16
        )

        assert adapter.config is config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend_config
        assert adapter.dtype == torch.float16
        assert adapter._uses_model_prefix is True

    def test_key_mappings_are_consistent(self, adapter):
        assert ".mlp.shared_expert." in adapter.hf_to_internal_map
        assert adapter.hf_to_internal_map[".mlp.shared_expert."] == ".mlp.shared_experts."
        # reverse mapping should be the inverse
        assert ".mlp.shared_experts." in adapter.internal_to_hf_map
        assert adapter.internal_to_hf_map[".mlp.shared_experts."] == ".mlp.shared_expert."

    def test_mtp_layout_explicit_override(self, config, moe_config, backend_config):
        adapter = Qwen3_5MoeStateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend_config,
            mtp_expert_hf_layout="per_expert_safetensors",
        )

        assert adapter._get_mtp_expert_hf_layout() == "split"

    def test_mtp_layout_config_override(self, config, moe_config, backend_config):
        config.mtp_expert_hf_layout = "group"
        adapter = Qwen3_5MoeStateDictAdapter(config=config, moe_config=moe_config, backend=backend_config)

        assert adapter._get_mtp_expert_hf_layout() == "grouped"

    @pytest.mark.parametrize("override_source", ["constructor", "config"])
    def test_mtp_layout_override_takes_precedence_over_local_checkpoint(
        self, tmp_path, config, moe_config, backend_config, override_source
    ):
        split_key = "mtp.layers.0.mlp.experts.0.down_proj.weight"
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {split_key: "model-00001-of-00001.safetensors"}})
        )
        config._name_or_path = str(tmp_path)
        config.name_or_path = str(tmp_path)
        explicit_layout = None
        if override_source == "constructor":
            explicit_layout = "grouped"
        else:
            config.mtp_expert_hf_layout = "grouped"
        adapter = Qwen3_5MoeStateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend_config,
            pretrained_model_name_or_path=str(tmp_path),
            mtp_expert_hf_layout=explicit_layout,
        )

        assert adapter._get_mtp_expert_hf_layout() == "grouped"

    def test_mtp_layout_rejects_unknown_override(self, config, moe_config, backend_config):
        adapter = Qwen3_5MoeStateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend_config,
            mtp_expert_hf_layout="packed",
        )

        with pytest.raises(ValueError, match="Unsupported MTP expert HF layout"):
            adapter._get_mtp_expert_hf_layout()

    def test_mtp_layout_rejects_mixed_checkpoint_keys(self, adapter):
        checkpoint_keys = {
            "mtp.layers.0.mlp.experts.down_proj",
            "mtp.layers.0.mlp.experts.0.down_proj.weight",
        }

        with pytest.raises(ValueError, match="both split and grouped MTP expert keys"):
            adapter._get_mtp_expert_hf_layout(checkpoint_keys)

    def test_base_layout_rejects_mixed_checkpoint_keys(self):
        split_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
        checkpoint_keys = {
            split_key,
            f"{split_key}_scale_inv",
            "model.layers.0.mlp.experts.gate_up_proj",
        }

        with pytest.raises(ValueError, match="both split-FP8 and grouped decoder expert keys"):
            _infer_base_expert_hf_layout(checkpoint_keys)


class TestSplitFp8BaseExperts:
    def test_from_hf_dequantizes_and_exports_split_fp8(self, split_fp8_adapter):
        hf_state, expected_gate_up, expected_down = _build_split_fp8_hf_state(split_fp8_adapter)

        assert split_fp8_adapter._get_expert_hf_layout(hf_state) == "split-fp8"

        native = split_fp8_adapter.from_hf(dict(hf_state))
        gate_key = "model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.layers.0.mlp.experts.down_projs"

        assert native[gate_key].shape == (2, 3, 4)
        assert native[down_key].shape == (2, 2, 3)
        torch.testing.assert_close(native[gate_key], expected_gate_up)
        torch.testing.assert_close(native[down_key], expected_down)

        exported = split_fp8_adapter.to_hf(native, quantization=True)

        assert set(exported) == set(hf_state)
        inter_dim = split_fp8_adapter.moe_config.moe_inter_dim
        for expert_id in range(split_fp8_adapter.moe_config.n_routed_experts):
            expected_exports = {
                "gate_proj": native[gate_key][expert_id, :, :inter_dim].transpose(0, 1),
                "up_proj": native[gate_key][expert_id, :, inter_dim:].transpose(0, 1),
                "down_proj": native[down_key][expert_id].transpose(0, 1),
            }
            for projection, expected in expected_exports.items():
                key = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
                assert exported[key].dtype == torch.float8_e4m3fn
                torch.testing.assert_close(exported[key].float(), expected.to(dtype=torch.float8_e4m3fn).float())

                scale_key = f"{key}_scale_inv"
                assert exported[scale_key].shape == (1, 1)
                assert exported[scale_key].dtype == torch.bfloat16
                torch.testing.assert_close(exported[scale_key], torch.ones_like(exported[scale_key]))


# ---------------------------------------------------------------------------
# _apply_key_mapping
# ---------------------------------------------------------------------------
class TestApplyKeyMapping:
    def test_shared_expert_to_shared_experts(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(64, 32),
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(64, 32),
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(32, 64),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }

        out = adapter._apply_key_mapping(state_dict, adapter.hf_to_internal_map)

        assert "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_experts.up_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_experts.down_proj.weight" in out
        # Original singular keys removed
        assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" not in out
        # Unrelated keys untouched
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out

    def test_reverse_mapping(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(64, 32),
        }

        out = adapter._apply_key_mapping(state_dict, adapter.internal_to_hf_map)

        assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight" not in out

    def test_multiple_layers(self, adapter):
        state_dict = {
            f"model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight": torch.randn(64, 32) for i in range(3)
        }

        out = adapter._apply_key_mapping(state_dict, adapter.hf_to_internal_map)

        for i in range(3):
            assert f"model.language_model.layers.{i}.mlp.shared_experts.gate_proj.weight" in out
            assert f"model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight" not in out


# ---------------------------------------------------------------------------
# to_hf
# ---------------------------------------------------------------------------
class TestToHF:
    def test_converts_aggregated_experts_with_transpose(self, adapter):
        # NeMo layout: [n_experts, hidden, moe_inter] (gate_and_up_projs)
        gate_up = torch.randn(4, 64, 128)
        down = torch.randn(4, 64, 64)

        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": gate_up,
            "model.language_model.layers.0.mlp.experts.down_projs": down,
        }

        out = adapter.to_hf(state_dict)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        down_key = "model.language_model.layers.0.mlp.experts.down_proj"

        assert gate_key in out
        assert down_key in out
        # Should be transposed(1,2) relative to input
        assert out[gate_key].shape == (4, 128, 64)
        assert out[down_key].shape == (4, 64, 64)

    def test_maps_shared_experts_to_shared_expert(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(64, 32),
            "model.language_model.layers.0.mlp.shared_experts.up_proj.weight": torch.randn(64, 32),
        }

        out = adapter.to_hf(state_dict)

        assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_expert.up_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight" not in out

    def test_respects_exclude_regex(self, adapter):
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 128),
            "exclude.me": torch.randn(1),
        }

        out = adapter.to_hf(state_dict, exclude_key_regex=r"^exclude")

        assert "exclude.me" not in out

    def test_passthrough_non_expert_keys(self, adapter):
        tensor = torch.randn(64, 64)
        state_dict = {
            "model.language_model.layers.0.self_attn.q_proj.weight": tensor,
        }

        out = adapter.to_hf(state_dict)

        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out
        assert out["model.language_model.layers.0.self_attn.q_proj.weight"] is tensor

    def test_transposes_expert_tensors(self, adapter):
        """to_hf should transpose expert tensors (NeMo→HF layout) without any comms."""
        gate_up = torch.randn(4, 64, 128, dtype=torch.float16)
        down = torch.randn(4, 64, 64, dtype=torch.float16)

        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": gate_up,
            "model.language_model.layers.0.mlp.experts.down_projs": down,
        }

        out = adapter.to_hf(state_dict)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        down_key = "model.language_model.layers.0.mlp.experts.down_proj"

        # Tensors should be transposed(1, 2)
        torch.testing.assert_close(out[gate_key], gate_up.transpose(1, 2))
        torch.testing.assert_close(out[down_key], down.transpose(1, 2))

    def test_round_trip_preserves_values(self, adapter):
        """HF → native → HF must produce identical tensors."""
        gate_up_hf = torch.randn(4, 128, 64)
        down_hf = torch.randn(4, 64, 64)
        attn = torch.randn(64, 64)
        shared = torch.randn(64, 32)

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up_hf,
            "model.language_model.layers.0.mlp.experts.down_proj": down_hf,
            "model.language_model.layers.0.self_attn.q_proj.weight": attn,
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": shared,
        }

        native = adapter.from_hf(dict(hf_state))
        roundtrip = adapter.to_hf(native)

        for key in hf_state:
            torch.testing.assert_close(roundtrip[key], hf_state[key])

    def test_text_only_expert_round_trip_preserves_keys_and_values(self, config, moe_config, backend_config):
        """The causal-LM adapter keeps model.layers keys free of a language_model prefix."""
        adapter = Qwen3_5MoeStateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend_config,
            dtype=torch.float32,
            text_only=True,
        )
        gate_up_hf = torch.randn(4, 128, 64)
        down_hf = torch.randn(4, 64, 64)
        hf_state = {
            "model.layers.0.mlp.experts.gate_up_proj": gate_up_hf,
            "model.layers.0.mlp.experts.down_proj": down_hf,
        }

        native = adapter.from_hf(hf_state)

        gate_up_native_key = "model.layers.0.mlp.experts.gate_and_up_projs"
        down_native_key = "model.layers.0.mlp.experts.down_projs"
        assert set(native) == {gate_up_native_key, down_native_key}
        torch.testing.assert_close(native[gate_up_native_key], gate_up_hf.transpose(1, 2))
        torch.testing.assert_close(native[down_native_key], down_hf.transpose(1, 2))

        roundtrip = adapter.to_hf(native)

        assert set(roundtrip) == set(hf_state)
        for key, value in hf_state.items():
            torch.testing.assert_close(roundtrip[key], value)

    @pytest.mark.parametrize(
        ("checkpoint_layout", "misleading_checkpoint_name", "use_index"),
        [
            ("split", "Qwen3.6-35B-A3B-consolidated", True),
            ("grouped", "Qwen3.5-35B-A3B-consolidated", False),
        ],
    )
    def test_consolidated_round_trip_infers_mtp_layout_from_checkpoint_keys(
        self,
        tmp_path,
        config,
        moe_config,
        backend_config,
        checkpoint_layout,
        misleading_checkpoint_name,
        use_index,
    ):
        gate_up = torch.arange(4 * 8 * 128, dtype=torch.float32).reshape(4, 8, 128)
        down = torch.arange(4 * 64 * 8, dtype=torch.float32).reshape(4, 64, 8)
        native_state = {
            "mtp.layers.0.mlp.experts.gate_and_up_projs": gate_up,
            "mtp.layers.0.mlp.experts.down_projs": down,
        }
        writer_adapter = Qwen3_5MoeStateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend_config,
            mtp_expert_hf_layout=checkpoint_layout,
        )
        checkpoint_state = writer_adapter.to_hf(native_state)
        checkpoint_path = tmp_path / misleading_checkpoint_name
        checkpoint_path.mkdir()
        shard_name = "model-00001-of-00001.safetensors" if use_index else "model.safetensors"
        save_file(
            {key: value.contiguous() for key, value in checkpoint_state.items()},
            checkpoint_path / shard_name,
        )
        if use_index:
            weight_map = {key: shard_name for key in checkpoint_state}
            (checkpoint_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

        target = torch.nn.Module()
        target.mtp = torch.nn.Module()
        target.mtp.layers = torch.nn.ModuleList([torch.nn.Module()])
        target.mtp.layers[0].mlp = torch.nn.Module()
        target.mtp.layers[0].mlp.experts = torch.nn.Module()
        target.mtp.layers[0].mlp.experts.gate_and_up_projs = torch.nn.Parameter(torch.zeros_like(gate_up))
        target.mtp.layers[0].mlp.experts.down_projs = torch.nn.Parameter(torch.zeros_like(down))
        checkpoint_model_path = str(checkpoint_path)
        local_config = SimpleNamespace(
            _name_or_path=checkpoint_model_path,
            name_or_path=checkpoint_model_path,
        )
        target.state_dict_adapter = Qwen3_5MoeStateDictAdapter(
            config=local_config,
            moe_config=moe_config,
            backend=backend_config,
            pretrained_model_name_or_path=checkpoint_model_path,
        )

        checkpointing_config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path / "cache"),
            model_repo_id="test/model",
            save_consolidated=False,
            is_peft=False,
        )
        with patch("torch.distributed.is_initialized", return_value=False):
            checkpointer = Checkpointer(
                checkpointing_config,
                dp_rank=0,
                tp_rank=0,
                pp_rank=0,
                moe_mesh=None,
            )
        checkpointer.load_model(target, model_path=str(checkpoint_path))

        torch.testing.assert_close(target.mtp.layers[0].mlp.experts.gate_and_up_projs, gate_up)
        torch.testing.assert_close(target.mtp.layers[0].mlp.experts.down_projs, down)
        roundtrip_state = target.state_dict_adapter.to_hf(target.state_dict())
        assert set(roundtrip_state) == set(checkpoint_state)
        for key in checkpoint_state:
            torch.testing.assert_close(roundtrip_state[key], checkpoint_state[key])

    def test_exclude_regex_filters_expert_key(self, adapter):
        """exclude_key_regex should filter expert keys after rename."""
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_projs": torch.randn(4, 64, 64),
        }

        out = adapter.to_hf(state_dict, exclude_key_regex=r".*gate_up_proj$")

        assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in out
        assert "model.language_model.layers.0.mlp.experts.down_proj" in out


# ---------------------------------------------------------------------------
# from_hf
# ---------------------------------------------------------------------------
class TestFromHF:
    def test_detects_model_prefix(self, adapter):
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
        }

        adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is True

    def test_handles_missing_prefix(self, adapter):
        hf_state = {
            "language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
        }

        out = adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is False
        assert "language_model.layers.0.mlp.experts.gate_and_up_projs" in out
        assert "language_model.layers.0.mlp.experts.down_projs" in out

    def test_combines_expert_weights_with_transpose(self, adapter):
        # HF layout: [n_experts, moe_inter, hidden]
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
        # Should be transposed(1,2) to NeMo layout
        torch.testing.assert_close(out[gate_key], gate_up.transpose(1, 2).to(adapter.dtype))
        torch.testing.assert_close(out[down_key], down.transpose(1, 2).to(adapter.dtype))

    def test_maps_shared_expert_to_shared_experts(self, adapter):
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(64, 32),
        }

        out = adapter.from_hf(hf_state)

        assert "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight" not in out

    def test_dtensor_passthrough(self, adapter, monkeypatch):
        """DCP path: DTensor values should be renamed + transposed, no slicing."""

        class FakeDTensor(torch.Tensor):
            """Minimal DTensor stand-in."""

            _is_fake_dtensor = True

            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data)

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor",
            lambda t: getattr(t, "_is_fake_dtensor", False),
        )

        gate_up_data = torch.randn(4, 32, 64)
        down_data = torch.randn(4, 64, 32)
        gate_up = FakeDTensor(gate_up_data)
        down = FakeDTensor(down_data)

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        }

        out = adapter.from_hf(hf_state)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.language_model.layers.0.mlp.experts.down_projs"

        # Should be transposed(1,2), no EP slicing
        assert out[gate_key].shape == (4, 64, 32)
        assert out[down_key].shape == (4, 32, 64)
        # Verify values are correct transpose
        torch.testing.assert_close(out[gate_key], gate_up_data.transpose(1, 2))
        torch.testing.assert_close(out[down_key], down_data.transpose(1, 2))

    def test_dtensor_skips_ep_slicing(self, adapter, monkeypatch):
        """DCP path with device_mesh: DTensors must NOT be sliced, only renamed + transposed."""

        class FakeDTensor(torch.Tensor):
            _is_fake_dtensor = True

            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data)

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor",
            lambda t: getattr(t, "_is_fake_dtensor", False),
        )
        # These should NOT be called for DTensor path
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n: (0, 2),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_submesh",
            lambda mesh, dims: Mock(get_rank=lambda: 0),
        )
        create_dtensor_called = []
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.create_dtensor_from_local",
            lambda t, m, r: create_dtensor_called.append(1) or t,
        )

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]

        gate_up = FakeDTensor(torch.randn(4, 32, 64))
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": FakeDTensor(torch.randn(4, 64, 32)),
        }

        out = adapter.from_hf(hf_state, device_mesh=device_mesh)

        # DTensor path: no create_dtensor_from_local calls, full shape preserved (not sliced to 2)
        assert len(create_dtensor_called) == 0
        assert out["model.language_model.layers.0.mlp.experts.gate_and_up_projs"].shape[0] == 4

    def test_non_prefixed_keys_get_model_prefix(self, adapter):
        """Non-expert keys without model. prefix should get it added."""
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "language_model.layers.0.input_layernorm.weight": torch.randn(64),
        }

        out = adapter.from_hf(hf_state)

        # The non-prefixed key should get model. prefix since other keys have it
        assert "model.language_model.layers.0.input_layernorm.weight" in out

    def test_device_mesh_rank_fallback_no_ep_dim(self, adapter, monkeypatch):
        """When device_mesh has no 'ep' dim, from_hf should use mesh.get_rank()."""
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n: (0, 4),
        )

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["dp"]  # no "ep"
        device_mesh.get_rank.return_value = 0

        def fake_create_dtensor(local_tensor, mesh, rank):
            return local_tensor

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.create_dtensor_from_local",
            fake_create_dtensor,
        )

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
        }

        out = adapter.from_hf(hf_state, device_mesh=device_mesh)

        device_mesh.get_rank.assert_called_once()
        assert "model.language_model.layers.0.mlp.experts.gate_and_up_projs" in out

    def test_skips_scale_inv_keys(self, adapter):
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "model.language_model.layers.0.mlp.experts.gate_up_proj_scale_inv": torch.randn(4),
        }

        out = adapter.from_hf(hf_state)

        assert not any(k.endswith("_scale_inv") for k in out.keys())

    def test_passthrough_non_expert_keys(self, adapter):
        tensor = torch.randn(64, 64)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "model.language_model.layers.0.self_attn.q_proj.weight": tensor,
        }

        out = adapter.from_hf(hf_state)

        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out

    def test_maps_vlm_lm_head_to_outer_model(self, adapter):
        lm_head = torch.randn(128, 64)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "model.lm_head.weight": lm_head,
        }

        out = adapter.from_hf(hf_state)

        assert "lm_head.weight" in out
        assert "model.lm_head.weight" not in out
        assert out["lm_head.weight"] is lm_head

    def test_keeps_native_lm_head_outer_when_model_prefix_is_detected(self, adapter):
        native_lm_head = torch.zeros(128, 64)
        checkpoint_lm_head = torch.ones(128, 64)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "lm_head.weight": native_lm_head,
            "model.lm_head.weight": checkpoint_lm_head,
        }

        out = adapter.from_hf(hf_state)

        assert "lm_head.weight" in out
        assert "model.lm_head.weight" not in out
        assert out["lm_head.weight"] is checkpoint_lm_head

    def test_maps_mtp_fusion_keys_without_model_prefix(self, adapter):
        tensor = torch.randn(64, 128)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64),
            "mtp.fc.weight": tensor,
        }

        out = adapter.from_hf(hf_state)

        assert "mtp.layers.0.eh_proj.weight" in out
        assert "model.mtp.layers.0.eh_proj.weight" not in out
        assert out["mtp.layers.0.eh_proj.weight"] is tensor

    def test_converts_mtp_aggregated_experts(self, adapter):
        gate_up = torch.randn(4, 32, 64)
        down = torch.randn(4, 64, 32)

        out = adapter.from_hf(
            {
                "mtp.layers.0.mlp.experts.gate_up_proj": gate_up,
                "mtp.layers.0.mlp.experts.down_proj": down,
            }
        )

        torch.testing.assert_close(
            out["mtp.layers.0.mlp.experts.gate_and_up_projs"], gate_up.transpose(1, 2).to(adapter.dtype)
        )
        torch.testing.assert_close(out["mtp.layers.0.mlp.experts.down_projs"], down.transpose(1, 2).to(adapter.dtype))

    def test_converts_mtp_split_experts(self, adapter):
        hf_state = {}
        expected_gate_up = []
        expected_down = []
        for expert_id in range(adapter.moe_config.n_routed_experts):
            gate = torch.randn(64, 64)
            up = torch.randn(64, 64)
            down = torch.randn(64, 64)
            hf_state[f"mtp.layers.0.mlp.experts.{expert_id}.gate_proj.weight"] = gate
            hf_state[f"mtp.layers.0.mlp.experts.{expert_id}.up_proj.weight"] = up
            hf_state[f"mtp.layers.0.mlp.experts.{expert_id}.down_proj.weight"] = down
            expected_gate_up.append(torch.cat((gate.transpose(0, 1), up.transpose(0, 1)), dim=1))
            expected_down.append(down.transpose(0, 1))

        out = adapter.from_hf(hf_state)

        torch.testing.assert_close(
            out["mtp.layers.0.mlp.experts.gate_and_up_projs"],
            torch.stack(expected_gate_up, dim=0).to(adapter.dtype),
        )
        torch.testing.assert_close(
            out["mtp.layers.0.mlp.experts.down_projs"],
            torch.stack(expected_down, dim=0).to(adapter.dtype),
        )
        roundtrip = adapter.to_hf(out)
        assert set(roundtrip) == set(hf_state)
        for key in hf_state:
            torch.testing.assert_close(roundtrip[key], hf_state[key])

    def test_converts_mtp_split_dtensors_from_dcp(self, adapter, monkeypatch):
        """DCP split experts must unwrap their residual DTensor before rebuilding the EP DTensor."""

        class FakeDTensor(torch.Tensor):
            _is_fake_dtensor = True

            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data)

            def to_local(self):
                return self.as_subclass(torch.Tensor)

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.is_dtensor",
            lambda tensor: getattr(tensor, "_is_fake_dtensor", False),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n_experts: (0, n_experts),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_submesh",
            lambda mesh, dims: Mock(get_rank=lambda: 0),
        )

        rebuilt_locals = []

        def fake_create_dtensor(local_tensor, mesh, rank):
            assert not getattr(local_tensor, "_is_fake_dtensor", False)
            rebuilt_locals.append(local_tensor)
            return local_tensor

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.create_dtensor_from_local",
            fake_create_dtensor,
        )

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]
        hf_state = {}
        expected_gate_up = []
        expected_down = []
        for expert_id in range(adapter.moe_config.n_routed_experts):
            gate = torch.randn(32, 64)
            up = torch.randn(32, 64)
            down = torch.randn(64, 32)
            hf_state[f"mtp.layers.0.mlp.experts.{expert_id}.gate_proj.weight"] = FakeDTensor(gate)
            hf_state[f"mtp.layers.0.mlp.experts.{expert_id}.up_proj.weight"] = FakeDTensor(up)
            hf_state[f"mtp.layers.0.mlp.experts.{expert_id}.down_proj.weight"] = FakeDTensor(down)
            expected_gate_up.append(torch.cat((gate.transpose(0, 1), up.transpose(0, 1)), dim=1))
            expected_down.append(down.transpose(0, 1))

        out = adapter.from_hf(hf_state, device_mesh=device_mesh)

        assert len(rebuilt_locals) == 2
        torch.testing.assert_close(
            out["mtp.layers.0.mlp.experts.gate_and_up_projs"],
            torch.stack(expected_gate_up, dim=0).to(adapter.dtype),
        )
        torch.testing.assert_close(
            out["mtp.layers.0.mlp.experts.down_projs"],
            torch.stack(expected_down, dim=0).to(adapter.dtype),
        )

    def test_expert_parallel_sharding(self, adapter, monkeypatch):
        """When device_mesh is provided, from_hf should slice experts by rank."""
        gate_up = torch.randn(4, 32, 64)
        down = torch.randn(4, 64, 32)

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["ep"]

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_expert_range_for_rank_from_mesh",
            lambda mesh, n_experts: (1, 3),
        )
        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.get_submesh",
            lambda mesh, dims: Mock(get_rank=lambda: 0),
        )

        def fake_create_dtensor(local_tensor, mesh, rank):
            return local_tensor

        monkeypatch.setattr(
            "nemo_automodel.components.moe.state_dict_utils.create_dtensor_from_local",
            fake_create_dtensor,
        )

        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        }

        out = adapter.from_hf(hf_state, device_mesh=device_mesh)

        gate_key = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"
        down_key = "model.language_model.layers.0.mlp.experts.down_projs"

        # Only experts 1 and 2 should be sliced
        assert out[gate_key].shape[0] == 2
        assert out[down_key].shape[0] == 2


# ---------------------------------------------------------------------------
# convert_single_tensor_to_hf
# ---------------------------------------------------------------------------
class TestConvertSingleTensorToHf:
    def test_gate_and_up_projs_conversion(self, adapter):
        tensor = torch.randn(4, 64, 128)
        fqn = "model.language_model.layers.0.mlp.experts.gate_and_up_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == "model.language_model.layers.0.mlp.experts.gate_up_proj"
        # Should be transposed(1,2)
        assert value.shape == (4, 128, 64)

    def test_down_projs_conversion(self, adapter):
        tensor = torch.randn(4, 64, 32)
        fqn = "model.language_model.layers.0.mlp.experts.down_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == "model.language_model.layers.0.mlp.experts.down_proj"
        assert value.shape == (4, 32, 64)

    def test_shared_experts_key_mapping(self, adapter):
        tensor = torch.randn(64, 32)
        fqn = "model.language_model.layers.0.mlp.shared_experts.gate_proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
        assert torch.equal(result[0][1], tensor)

    def test_strips_fp32_holder_segment_on_save(self, adapter):
        # The fp32 SSM-gating holder is stripped back to the bare HF key on save.
        tensor = torch.randn(8)
        fqn = "model.language_model.layers.0.linear_attn._fp32_params.A_log"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "model.language_model.layers.0.linear_attn.A_log"
        assert torch.equal(result[0][1], tensor)

    def test_non_expert_tensor_passthrough(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "model.language_model.layers.0.self_attn.q_proj.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == fqn
        assert result[0][1] is tensor

    def test_exclude_regex_filters_results(self, adapter):
        tensor = torch.randn(64, 64)
        fqn = "exclude.me"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

        assert result == []

    def test_expert_key_with_no_model_prefix(self, adapter):
        adapter._uses_model_prefix = False
        tensor = torch.randn(4, 64, 128)
        fqn = "language_model.layers.0.mlp.experts.gate_and_up_projs"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        key, _ = result[0]
        assert key == "language_model.layers.0.mlp.experts.gate_up_proj"

    def test_mtp_fusion_key_conversion(self, adapter):
        tensor = torch.randn(64, 128)
        result = adapter.convert_single_tensor_to_hf("mtp.layers.0.eh_proj.weight", tensor)

        assert result == [("mtp.fc.weight", tensor)]

    def test_mtp_expert_key_conversion_grouped_layout(self, adapter):
        # Qwen3.6 MTP experts use the grouped HF layout, identical to the main
        # decoder layers, not per-expert keys (AM-442).
        tensor = torch.randn(4, 64, 128)
        result = adapter.convert_single_tensor_to_hf("mtp.layers.0.mlp.experts.gate_and_up_projs", tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == "mtp.layers.0.mlp.experts.gate_up_proj"
        torch.testing.assert_close(value, tensor.transpose(1, 2))
        # No per-expert keys are emitted.
        assert not any(".experts.0." in k for k, _ in result)

    def test_mtp_down_expert_key_conversion_grouped_layout(self, adapter):
        tensor = torch.randn(4, 64, 32)
        result = adapter.convert_single_tensor_to_hf("mtp.layers.0.mlp.experts.down_projs", tensor)

        assert len(result) == 1
        key, value = result[0]
        assert key == "mtp.layers.0.mlp.experts.down_proj"
        torch.testing.assert_close(value, tensor.transpose(1, 2))

    def test_mtp_experts_emit_no_per_expert_keys_for_grouped_layout(self, adapter):
        """AM-442 regression: to_hf must not fabricate per-expert MTP keys that are
        absent from the grouped checkpoint (e.g. ``...experts.224.down_proj.weight``)."""
        for native in ("mtp.layers.0.mlp.experts.gate_and_up_projs", "mtp.layers.0.mlp.experts.down_projs"):
            result = adapter.convert_single_tensor_to_hf(native, torch.randn(4, 64, 32))
            assert len(result) == 1
            key = result[0][0]
            assert key in ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj")
            assert not re.search(r"\.experts\.\d+\.", key)

    def test_mtp_expert_key_conversion_split_layout_for_qwen35(self, adapter):
        adapter.pretrained_model_name_or_path = "Qwen/Qwen3.5-35B-A3B"
        tensor = torch.randn(4, 64, 128)

        result = adapter.convert_single_tensor_to_hf("mtp.layers.0.mlp.experts.gate_and_up_projs", tensor)

        assert len(result) == 8
        result_by_key = dict(result)
        expected_keys = set()
        for expert_id in range(4):
            expected_keys.add(f"mtp.layers.0.mlp.experts.{expert_id}.gate_proj.weight")
            expected_keys.add(f"mtp.layers.0.mlp.experts.{expert_id}.up_proj.weight")
        assert set(result_by_key) == expected_keys
        for expert_id in range(4):
            torch.testing.assert_close(
                result_by_key[f"mtp.layers.0.mlp.experts.{expert_id}.gate_proj.weight"],
                tensor[expert_id, :, :64].transpose(0, 1),
            )
            torch.testing.assert_close(
                result_by_key[f"mtp.layers.0.mlp.experts.{expert_id}.up_proj.weight"],
                tensor[expert_id, :, 64:].transpose(0, 1),
            )

    def test_mtp_down_expert_key_conversion_split_layout_for_qwen35(self, adapter):
        adapter.pretrained_model_name_or_path = "Qwen/Qwen3.5-35B-A3B"
        tensor = torch.randn(4, 64, 32)

        result = adapter.convert_single_tensor_to_hf("mtp.layers.0.mlp.experts.down_projs", tensor)

        assert len(result) == 4
        result_by_key = dict(result)
        expected_keys = {f"mtp.layers.0.mlp.experts.{expert_id}.down_proj.weight" for expert_id in range(4)}
        assert set(result_by_key) == expected_keys
        for expert_id in range(4):
            torch.testing.assert_close(
                result_by_key[f"mtp.layers.0.mlp.experts.{expert_id}.down_proj.weight"],
                tensor[expert_id].transpose(0, 1),
            )


# ---------------------------------------------------------------------------
# Qwen3_5MoeForConditionalGeneration state dict adapter wiring
# ---------------------------------------------------------------------------
class TestConditionalGenerationStateDictAdapterWiring:
    def test_passes_top_level_model_name_to_state_dict_adapter(self):
        class DummyRotary:
            inv_freq = torch.ones(4)

        class DummyVisual:
            def __init__(self):
                self.rotary_pos_emb = DummyRotary()

        class DummyParentModel:
            def __init__(self):
                self.visual = DummyVisual()
                self.language_model = SimpleNamespace()

        class DummyTextBackend:
            def __init__(self, *args, **kwargs):
                self.moe_config = Mock(name="moe_config")

        class DummyMTPConfig:
            enabled = False

        class DummyFp32SafeRotary:
            def __init__(self, dim):
                self.dim = dim

            def register_buffer(self, *args, **kwargs):
                pass

            def to(self, *args, **kwargs):
                return self

        adapter_calls = []

        class DummyAdapter:
            def __init__(self, *args, **kwargs):
                adapter_calls.append((args, kwargs))

        def fake_hf_init(self, config):
            self.model = DummyParentModel()
            self.lm_head = None

        text_config = SimpleNamespace(torch_dtype=None, hidden_size=8, vocab_size=16, pad_token_id=None)
        config = SimpleNamespace(
            text_config=text_config,
            torch_dtype=None,
            _name_or_path="model-under-test",
            name_or_path=None,
        )
        backend = BackendConfig(enable_hf_state_dict_adapter=True)

        with (
            patch.object(qwen3_5_moe_model.HFQwen3_5MoeForConditionalGeneration, "__init__", fake_hf_init),
            patch.object(qwen3_5_moe_model, "Qwen3_5MoeModel", DummyParentModel),
            patch.object(qwen3_5_moe_model, "Qwen3_5MoeTextModelBackend", DummyTextBackend),
            patch.object(qwen3_5_moe_model, "initialize_linear_module", Mock(return_value=Mock())),
            patch.object(qwen3_5_moe_model, "build_mtp_config_from_hf", Mock(return_value=DummyMTPConfig())),
            patch.object(qwen3_5_moe_model, "Qwen3_5MoeStateDictAdapter", DummyAdapter),
            patch.object(
                qwen3_5_moe_model,
                "Fp32SafeQwen3_5MoeVisionRotaryEmbedding",  # pragma: allowlist secret
                DummyFp32SafeRotary,
            ),
        ):
            qwen3_5_moe_model.Qwen3_5MoeForConditionalGeneration(config, backend=backend)

        assert len(adapter_calls) == 1
        args, kwargs = adapter_calls[0]
        assert args[0] is text_config
        assert kwargs["pretrained_model_name_or_path"] == "model-under-test"


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
        """With ep_shard_size=2, from_hf must slice dim 1 of the transposed tensor."""
        n_experts = adapter.moe_config.n_routed_experts  # 4
        # HF: [n_experts, inter, hidden]; native (after transpose): [n_experts, hidden, inter]
        inter, hidden = 8, 4
        ep_shard_size, ep_shard_rank = 2, 1

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, n_experts), ep_shard_size=ep_shard_size, ep_shard_rank=ep_shard_rank
        )

        gate_up_hf = torch.arange(n_experts * inter * hidden, dtype=adapter.dtype).reshape(n_experts, inter, hidden)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up_hf,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(
                n_experts, hidden, inter, dtype=adapter.dtype
            ),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        # First captured tensor is gate_and_up_projs
        local_gate = captured_list[0]
        # After transpose(1,2): [n_experts, hidden, inter]; ep_shard slices dim 1 (hidden)
        chunk = hidden // ep_shard_size
        native_full = gate_up_hf.transpose(1, 2).to(adapter.dtype)
        expected = native_full[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
        assert local_gate.shape == (n_experts, chunk, inter)
        torch.testing.assert_close(local_gate, expected)

    def test_from_hf_no_ep_shard_unchanged(self, adapter, monkeypatch):
        """With ep_shard_size=1 (single-node), from_hf must NOT slice dim 1."""
        n_experts = adapter.moe_config.n_routed_experts
        inter, hidden = 8, 4

        device_mesh, captured_list = self._setup_from_hf_mocks(
            monkeypatch, ep_range=(0, n_experts), ep_shard_size=1, ep_shard_rank=0
        )

        gate_up_hf = torch.randn(n_experts, inter, hidden, dtype=adapter.dtype)
        hf_state = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up_hf,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(
                n_experts, hidden, inter, dtype=adapter.dtype
            ),
        }

        adapter.from_hf(hf_state, device_mesh=device_mesh)

        local_gate = captured_list[0]
        # No ep_shard slicing — full transposed tensor
        assert local_gate.shape == (n_experts, hidden, inter)
        torch.testing.assert_close(local_gate, gate_up_hf.transpose(1, 2).to(adapter.dtype))


class TestFp32ParamRouting:
    """Routing/stripping of SSM-gating params into/out of the fp32 holder."""

    def test_strip_fp32_params_removes_holder_segment(self):
        from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import _strip_fp32_params

        assert (
            _strip_fp32_params("model.language_model.layers.0.linear_attn._fp32_params.A_log")
            == "model.language_model.layers.0.linear_attn.A_log"
        )
        assert (
            _strip_fp32_params("model.language_model.layers.0.linear_attn._fp32_params.dt_bias")
            == "model.language_model.layers.0.linear_attn.dt_bias"
        )

    def test_strip_fp32_params_passthrough(self):
        from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import _strip_fp32_params

        for key in (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.linear_attn.norm.weight",
        ):
            assert _strip_fp32_params(key) == key

    def test_route_fp32_params_routes_gating_keys(self):
        from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import _route_fp32_params

        assert (
            _route_fp32_params("model.language_model.layers.0.linear_attn.A_log")
            == "model.language_model.layers.0.linear_attn._fp32_params.A_log"
        )
        assert (
            _route_fp32_params("model.language_model.layers.0.linear_attn.dt_bias")
            == "model.language_model.layers.0.linear_attn._fp32_params.dt_bias"
        )

    def test_route_fp32_params_passthrough(self):
        from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import _route_fp32_params

        # Already routed, non-gating param, and a non-linear_attn A_log all pass through.
        for key in (
            "model.language_model.layers.0.linear_attn._fp32_params.A_log",
            "model.language_model.layers.0.linear_attn.norm.weight",
            "model.some.other.path.A_log",
        ):
            assert _route_fp32_params(key) == key

    def test_route_strip_round_trip(self):
        from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import (
            _route_fp32_params,
            _strip_fp32_params,
        )

        bare = "model.language_model.layers.3.linear_attn.A_log"
        assert _strip_fp32_params(_route_fp32_params(bare)) == bare

    def test_from_hf_routes_gating_keys_into_holder(self, adapter):
        # On load, bare HF SSM-gating keys are routed into the fp32 holder.
        hf_state = {
            "model.language_model.layers.0.linear_attn.A_log": torch.randn(8),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.randn(8),
        }

        out = adapter.from_hf(hf_state)

        assert "model.language_model.layers.0.linear_attn._fp32_params.A_log" in out
        assert "model.language_model.layers.0.linear_attn._fp32_params.dt_bias" in out
        assert "model.language_model.layers.0.linear_attn.A_log" not in out

    def test_to_hf_strips_a_log_holder(self, adapter):
        sd = {"model.language_model.layers.0.linear_attn._fp32_params.A_log": torch.zeros(4)}
        out = adapter.to_hf(sd)
        assert "model.language_model.layers.0.linear_attn.A_log" in out
        assert all("_fp32_params" not in k for k in out)

    def test_to_hf_strips_dt_bias_holder(self, adapter):
        sd = {"model.language_model.layers.2.linear_attn._fp32_params.dt_bias": torch.ones(4)}
        out = adapter.to_hf(sd)
        assert "model.language_model.layers.2.linear_attn.dt_bias" in out
        assert all("_fp32_params" not in k for k in out)

    def test_to_hf_upcasts_gdn_fp32_params_saved_as_bf16(self, adapter):
        sd = {
            "model.language_model.layers.0.linear_attn._fp32_params.A_log": torch.zeros(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.linear_attn._fp32_params.dt_bias": torch.ones(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        }

        out = adapter.to_hf(sd)

        assert out["model.language_model.layers.0.linear_attn.A_log"].dtype == torch.float32
        assert out["model.language_model.layers.0.linear_attn.dt_bias"].dtype == torch.float32
        q_proj_key = "model.language_model.layers.0.self_attn.q_proj.weight"
        assert out[q_proj_key] is sd[q_proj_key]
        assert out[q_proj_key].dtype == torch.bfloat16

    def test_forced_hf_dtype_mapping_marks_gdn_fp32_params(self, adapter):
        state_dict = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4, dtype=torch.float32),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(4, dtype=torch.float32),
            "model.language_model.layers.0.linear_attn.conv1d.weight": torch.zeros(4, dtype=torch.float32),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2, dtype=torch.float32),
        }

        assert adapter.forced_hf_dtype_mapping(state_dict) == {
            "model.language_model.layers.0.linear_attn.A_log": "F32",
            "model.language_model.layers.0.linear_attn.dt_bias": "F32",
        }

    def test_convert_single_tensor_strips_holder(self, adapter):
        result = adapter.convert_single_tensor_to_hf(
            "model.language_model.layers.1.linear_attn._fp32_params.A_log", torch.zeros(4)
        )
        assert [k for k, _ in result] == ["model.language_model.layers.1.linear_attn.A_log"]

    def test_convert_single_tensor_upcasts_gdn_fp32_params(self, adapter):
        result = adapter.convert_single_tensor_to_hf(
            "model.language_model.layers.1.linear_attn._fp32_params.dt_bias",
            torch.zeros(4, dtype=torch.bfloat16),
        )
        assert result[0][0] == "model.language_model.layers.1.linear_attn.dt_bias"
        assert result[0][1].dtype == torch.float32

    def test_bare_key_unchanged(self, adapter):
        result = adapter.convert_single_tensor_to_hf("model.language_model.layers.0.linear_attn.A_log", torch.zeros(4))
        assert [k for k, _ in result] == ["model.language_model.layers.0.linear_attn.A_log"]

    def test_from_hf_routes_and_upcasts_gdn_fp32_params_loaded_as_bf16(self, adapter):
        hf_state = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        }

        out = adapter.from_hf(hf_state)

        a_log_key = "model.language_model.layers.0.linear_attn._fp32_params.A_log"
        dt_bias_key = "model.language_model.layers.0.linear_attn._fp32_params.dt_bias"
        q_proj_key = "model.language_model.layers.0.self_attn.q_proj.weight"
        assert out[a_log_key].dtype == torch.float32
        assert out[dt_bias_key].dtype == torch.float32
        assert out[q_proj_key] is hf_state[q_proj_key]
        assert out[q_proj_key].dtype == torch.bfloat16
