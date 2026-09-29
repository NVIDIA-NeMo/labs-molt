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

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import (
    NemotronV3StateDictAdapter,
)
from nemo_automodel.components.moe.config import MoEConfig

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")


class MockNemotronV3Config:
    """Mock configuration for NemotronV3 model."""

    def __init__(self, **overrides):
        self.num_hidden_layers = 2
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.intermediate_size = 512

        for key, value in overrides.items():
            setattr(self, key, value)


class TestNemotronV3StateDictAdapter:
    """Test NemotronV3StateDictAdapter."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_bias=False,
            expert_activation="relu2",  # NemotronV3 uses relu2
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            dispatcher="torch",
        )

    def test_adapter_init(self, config, moe_config, backend):
        """Test adapter initialization."""
        adapter = NemotronV3StateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend,
            dtype=torch.bfloat16,
        )

        assert adapter.config == config
        assert adapter.moe_config == moe_config
        assert adapter.backend == backend
        assert adapter.dtype == torch.bfloat16
        assert adapter._uses_model_prefix is False

    def test_hf_prefix_property(self, config, moe_config, backend):
        """The output prefix tracks the namespace detected from the source checkpoint."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._hf_prefix == "backbone."
        adapter._uses_model_prefix = True
        assert adapter._hf_prefix == "model."

    def test_peft_target_module_mapping_tracks_export_format(self, config, moe_config, backend):
        """Default PEFT targets the built-in model; v4 preserves the source namespace."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)
        native_name = "model.layers.0.mixer.in_proj"

        assert adapter.map_peft_target_module_to_hf(native_name) == native_name
        assert (
            adapter.map_peft_target_module_to_hf(native_name, v4_compatible=True) == "backbone.layers.0.mixer.in_proj"
        )
        adapter._uses_model_prefix = True
        assert adapter.map_peft_target_module_to_hf(native_name) == native_name
        assert adapter.map_peft_target_module_to_hf(native_name, v4_compatible=True) == native_name

    def test_expert_path_segment_property(self, config, moe_config, backend):
        """Test _expert_path_segment property returns 'mixer.experts'."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._expert_path_segment == "mixer.experts"

    def test_low_memory_dcp_capability_requires_model_backed_experts(self, config, moe_config, backend):
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)
        assert adapter.supports_low_memory_dcp_load is True

        adapter.backend.experts = "te"
        adapter.backend.dispatcher = "deepep"
        with patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=8):
            assert adapter.supports_low_memory_dcp_load is False

        with patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=1):
            assert adapter.supports_low_memory_dcp_load is True

        adapter.backend.experts = "gmm"
        adapter.backend.dispatcher = "mok"
        assert adapter.supports_low_memory_dcp_load is False

    def test_te_single_device_fallback_loads_through_grouped_parameter_views(self, config, moe_config, backend):
        backend.experts = "te"
        backend.dispatcher = "deepep"
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)
        gate_and_up = torch.zeros(
            moe_config.n_routed_experts,
            moe_config.expert_dim,
            moe_config.moe_inter_dim,
        )
        down = torch.zeros(
            moe_config.n_routed_experts,
            moe_config.moe_inter_dim,
            moe_config.expert_dim,
        )

        with patch("nemo_automodel.components.moe.state_dict_mixin.get_world_size_safe", return_value=1):
            destinations = adapter.to_hf(
                {
                    "model.layers.0.mixer.experts.gate_and_up_projs": gate_and_up,
                    "model.layers.0.mixer.experts.down_projs": down,
                },
                for_checkpoint_load=True,
            )
            assert adapter.supports_low_memory_dcp_load is True

        assert set(destinations) == {
            f"backbone.layers.0.mixer.experts.{expert_id}.{projection}.weight"
            for expert_id in range(moe_config.n_routed_experts)
            for projection in ("up_proj", "down_proj")
        }
        for expert_id in range(moe_config.n_routed_experts):
            up_destination = destinations[f"backbone.layers.0.mixer.experts.{expert_id}.up_proj.weight"]
            down_destination = destinations[f"backbone.layers.0.mixer.experts.{expert_id}.down_proj.weight"]
            assert up_destination.untyped_storage().data_ptr() == gate_and_up.untyped_storage().data_ptr()
            assert down_destination.untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
            assert not up_destination.is_contiguous()
            assert not down_destination.is_contiguous()

    def test_from_hf_map_structure(self, config, moe_config, backend):
        """Test from_hf_map structure."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        # Check that mapping uses 'mixer.experts' path
        assert "model.layers.{}.mixer.experts.{}.up_proj.weight" in adapter.from_hf_map
        assert "model.layers.{}.mixer.experts.{}.down_proj.weight" in adapter.from_hf_map


class TestNemotronV3AdapterDense:
    """Dense Nemotron-H adapter (moe_config=None) — issue #2004.

    Dense checkpoints carry no '.mixer.experts.' keys, so from_hf/to_hf reduce to
    pure renames (backbone<->model, norm_f<->norm, embeddings<->embed_tokens) and
    must not dereference moe_config.
    """

    @pytest.fixture
    def backend(self):
        return BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", enable_deepep=False)

    @pytest.fixture
    def adapter(self, backend):
        return NemotronV3StateDictAdapter(config=MockNemotronV3Config(), moe_config=None, backend=backend)

    def test_init_accepts_none_moe_config(self, adapter):
        assert adapter.moe_config is None
        assert adapter.supports_low_memory_dcp_load is True

    def test_from_hf_renames_without_experts(self, adapter):
        hf_sd = {
            "backbone.embeddings.weight": torch.randn(100, 256),
            "backbone.layers.0.mixer.A_log": torch.randn(4),
            "backbone.layers.1.mixer.up_proj.weight": torch.randn(512, 256),
            "backbone.norm_f.weight": torch.randn(256),
            "lm_head.weight": torch.randn(100, 256),
        }
        native = adapter.from_hf(dict(hf_sd))

        assert "model.embed_tokens.weight" in native
        assert "model.norm.weight" in native
        assert "model.layers.0.mixer._fp32_params.A_log" in native
        assert "model.layers.1.mixer.up_proj.weight" in native
        assert "lm_head.weight" in native
        assert not any(k.startswith("backbone.") for k in native)
        assert not any(k.endswith("norm_f.weight") for k in native)
        assert adapter._uses_model_prefix is False

    @pytest.mark.parametrize("embed_key", ["backbone.embedding.weight", "backbone.embeddings.weight", "model.embedding.weight"])
    def test_from_hf_accepts_singular_or_plural_embedding(self, adapter, embed_key):
        """RL#4211: transformers save_pretrained may emit singular ``embedding``."""
        tensor = torch.randn(100, 256)
        native = adapter.from_hf({embed_key: tensor})

        assert set(native) == {"model.embed_tokens.weight"}
        torch.testing.assert_close(native["model.embed_tokens.weight"], tensor, rtol=0, atol=0)

    def test_singular_embedding_exports_as_plural(self, adapter):
        """Canonical HF export stays plural to match released nvidia checkpoints."""
        tensor = torch.randn(100, 256)
        native = adapter.from_hf({"backbone.embedding.weight": tensor})
        exported = adapter.to_hf(dict(native))

        assert set(exported) == {"backbone.embeddings.weight"}
        torch.testing.assert_close(exported["backbone.embeddings.weight"], tensor, rtol=0, atol=0)
        assert "backbone.embedding.weight" not in exported

    def test_round_trip_dense(self, adapter):
        hf_sd = {
            "backbone.embeddings.weight": torch.randn(100, 256),
            "backbone.layers.0.mixer.up_proj.weight": torch.randn(512, 256),
            "backbone.norm_f.weight": torch.randn(256),
        }
        native = adapter.from_hf(dict(hf_sd))
        back = adapter.to_hf(dict(native))

        assert set(back.keys()) == set(hf_sd.keys())

    def test_dense_mtp_keys_do_not_enter_moe_conversion(self, adapter):
        hf_tensor = torch.randn(4, dtype=torch.bfloat16)
        native = adapter.from_hf({"mtp.layers.0.mixer.A_log": hf_tensor})

        assert set(native) == {"mtp.layers.0.mixer._fp32_params.A_log"}
        assert native["mtp.layers.0.mixer._fp32_params.A_log"].dtype == torch.float32

        exported = adapter.convert_single_tensor_to_hf(
            "mtp.layers.0.mixer._fp32_params.A_log", native["mtp.layers.0.mixer._fp32_params.A_log"]
        )
        assert exported[0][0] == "mtp.layers.0.mixer.A_log"

    @pytest.mark.parametrize("v4_compatible", [False, True])
    def test_peft_outer_prefix_round_trip(self, adapter, v4_compatible):
        prefix = "backbone" if v4_compatible else "model"
        hf_key = f"base_model.model.{prefix}.layers.0.mixer.in_proj.lora_A.weight"
        native_key = "base_model.model.model.layers.0.mixer.in_proj.lora_A.weight"
        tensor = torch.randn(8, 256)

        exported = adapter.to_hf({native_key: tensor}, v4_compatible=v4_compatible)
        assert adapter._uses_model_prefix is False, "PEFT export must not change the full-checkpoint layout"
        restored = adapter.from_hf(dict(exported))

        assert list(exported) == [hf_key]
        assert list(restored) == [native_key]
        torch.testing.assert_close(restored[native_key], tensor)

    def test_legacy_peft_adapter_loads_and_reexports_for_v5(self, adapter):
        """A saved backbone adapter remains loadable and can be exported to built-in HF."""
        old_key = "base_model.model.backbone.layers.0.mixer.in_proj.lora_A.weight"
        tensor = torch.randn(8, 256)

        native = adapter.from_hf({old_key: tensor})
        exported = adapter.to_hf(native)

        key = "base_model.model.model.layers.0.mixer.in_proj.lora_A.weight"
        assert set(exported) == {key}
        torch.testing.assert_close(exported[key], tensor, rtol=0, atol=0)

    @pytest.mark.parametrize("base_prefix", ["backbone", "model"])
    @pytest.mark.parametrize("adapter_prefix", ["backbone", "model"])
    def test_adapter_restore_preserves_base_checkpoint_namespace(self, adapter, base_prefix, adapter_prefix):
        """Legacy LoRA names must not rename frozen embeddings in the merged export."""
        base_state = {
            f"{base_prefix}.embeddings.weight": torch.randn(16, 8),
            f"{base_prefix}.norm_f.weight": torch.randn(8),
        }
        native_base = adapter.from_hf(dict(base_state))
        lora_tensor = torch.randn(2, 8)
        native_lora = adapter.from_hf(
            {f"base_model.model.{adapter_prefix}.layers.0.mixer.in_proj.lora_A.weight": lora_tensor}
        )
        native_key = "base_model.model.model.layers.0.mixer.in_proj.lora_A.weight"
        assert set(native_lora) == {native_key}
        torch.testing.assert_close(native_lora[native_key], lora_tensor, rtol=0, atol=0)

        exported = adapter.to_hf(dict(native_base))
        assert set(exported) == set(base_state)
        for key in base_state:
            torch.testing.assert_close(exported[key], base_state[key], rtol=0, atol=0)


class TestNemotronV3AdapterMTP:
    """MTP checkpoint namespace regressions."""

    def test_view_loaded_mtp_experts_keep_native_namespace(self):
        config = MockNemotronV3Config()
        moe_config = MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        adapter = NemotronV3StateDictAdapter(config, moe_config, BackendConfig(), dtype=torch.bfloat16)

        # DCP writes these split tensors through views into the native grouped
        # parameters. The merge therefore returns no tensor for them and uses
        # view_loaded_native_keys to tell the loader they were loaded.
        adapter._inplace_loaded_native_keys = {
            "layers.1.mixer.experts.gate_and_up_projs",
            "layers.1.mixer.experts.down_projs",
        }
        hf_state = {}
        for expert_id in range(2):
            hf_state[f"mtp.layers.1.mixer.experts.{expert_id}.up_proj.weight"] = torch.randn(128, 256)
            hf_state[f"mtp.layers.1.mixer.experts.{expert_id}.down_proj.weight"] = torch.randn(256, 128)

        native = adapter.from_hf(hf_state)

        assert not native
        assert adapter.view_loaded_native_keys == {
            "mtp.layers.1.mixer.experts.gate_and_up_projs",
            "mtp.layers.1.mixer.experts.down_projs",
        }

    def test_mtp_expert_conversion_forwards_checkpoint_kwargs(self):
        moe_config = MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        adapter = NemotronV3StateDictAdapter(MockNemotronV3Config(), moe_config, BackendConfig())
        tensor = torch.randn(moe_config.n_routed_experts, 256, 128)

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None) as convert:
            adapter.convert_single_tensor_to_hf(
                "mtp.layers.1.mixer.experts.gate_and_up_projs",
                tensor,
                for_checkpoint_load=True,
                v4_compatible=True,
            )

        convert.assert_called_once_with(
            "mtp.layers.1.mixer.experts.gate_and_up_projs",
            tensor,
            prefix_override="mtp.",
            for_checkpoint_load=True,
            v4_compatible=True,
        )


class TestNemotronV3AdapterToHf:
    """Test to_hf conversion."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_to_hf_converts_keys(self, config, moe_config, backend):
        """Test to_hf converts internal keys to HF format."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, 256),
            "model.norm.weight": torch.randn(256),
            "lm_head.weight": torch.randn(100, 256),
        }

        hf_state_dict = adapter.to_hf(state_dict)

        # model → backbone
        assert "backbone.embeddings.weight" in hf_state_dict
        # norm → norm_f
        assert "backbone.norm_f.weight" in hf_state_dict
        # lm_head stays the same
        assert "lm_head.weight" in hf_state_dict

    def test_to_hf_exclude_key_regex(self, config, moe_config, backend):
        """Test to_hf with exclude_key_regex."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, 256),
            "exclude_me.weight": torch.randn(10, 10),
        }

        hf_state_dict = adapter.to_hf(state_dict, exclude_key_regex=r"exclude.*")

        assert "backbone.embeddings.weight" in hf_state_dict
        assert "exclude_me.weight" not in hf_state_dict

    def test_to_hf_hides_mamba_fp32_holder(self, config, moe_config, backend):
        """Test to_hf maps internal Mamba fp32 holder keys back to public HF keys."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.layers.0.mixer._fp32_params.A_log": torch.randn(4, dtype=torch.bfloat16),
            "model.layers.0.mixer._fp32_params.dt_bias": torch.randn(4, dtype=torch.bfloat16),
            "model.layers.0.mixer._fp32_params.D": torch.randn(4, dtype=torch.bfloat16),
        }

        hf_state_dict = adapter.to_hf(state_dict)

        assert "backbone.layers.0.mixer.A_log" in hf_state_dict
        assert "backbone.layers.0.mixer.dt_bias" in hf_state_dict
        assert "backbone.layers.0.mixer.D" in hf_state_dict
        assert "backbone.layers.0.mixer._fp32_params.A_log" not in hf_state_dict
        assert "backbone.layers.0.mixer._fp32_params.dt_bias" not in hf_state_dict
        assert "backbone.layers.0.mixer._fp32_params.D" not in hf_state_dict
        assert hf_state_dict["backbone.layers.0.mixer.A_log"].dtype == torch.float32
        assert hf_state_dict["backbone.layers.0.mixer.dt_bias"].dtype == torch.float32
        assert hf_state_dict["backbone.layers.0.mixer.D"].dtype == torch.float32

    def test_forced_hf_dtype_mapping_marks_fp32_protected_keys(self, config, moe_config, backend):
        """Test Nemotron fp32-protected HF export keys keep fp32 consolidated metadata."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "backbone.layers.0.mixer.A_log": torch.randn(4, dtype=torch.float32),
            "backbone.layers.0.mixer.dt_bias": torch.randn(4, dtype=torch.float32),
            "backbone.layers.0.mixer.D": torch.randn(4, dtype=torch.float32),
            "backbone.layers.0.mixer.router.e_score_correction_bias": torch.randn(4, dtype=torch.float32),
            "backbone.layers.0.mixer.in_proj.weight": torch.randn(4, 4, dtype=torch.float32),
            "lm_head.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        }

        forced = adapter.forced_hf_dtype_mapping(state_dict)

        assert forced == {
            "backbone.layers.0.mixer.A_log": "F32",
            "backbone.layers.0.mixer.dt_bias": "F32",
            "backbone.layers.0.mixer.D": "F32",
            "backbone.layers.0.mixer.router.e_score_correction_bias": "F32",
        }


class TestNemotronV3AdapterFromHf:
    """Test from_hf conversion."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_from_hf_renames_backbone_to_model(self, config, moe_config, backend):
        """Test from_hf renames backbone → model."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "backbone.embeddings.weight": torch.randn(100, 256),
            "backbone.norm_f.weight": torch.randn(256),
            "lm_head.weight": torch.randn(100, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {
                "model.embed_tokens.weight": torch.randn(100, 256),
                "model.norm.weight": torch.randn(256),
                "lm_head.weight": torch.randn(100, 256),
            }

            adapter.from_hf(hf_state_dict)

            # Check that _from_hf_w_merged_experts was called with renamed state dict
            call_args = mock_merge.call_args[0][0]
            assert "model.embed_tokens.weight" in call_args
            assert "model.norm.weight" in call_args
            assert "backbone.embeddings.weight" not in call_args
            assert "backbone.norm_f.weight" not in call_args

    def test_from_hf_detects_backbone_prefix(self, config, moe_config, backend):
        """Test from_hf detects backbone prefix from HF checkpoint."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": torch.randn(128, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {}
            adapter.from_hf(hf_state_dict)

            # _uses_model_prefix should be False when starting with 'backbone.'
            assert adapter._uses_model_prefix is False

    def test_from_hf_detects_model_prefix(self, config, moe_config, backend):
        """Test from_hf detects model prefix from HF checkpoint."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "model.layers.0.mixer.experts.0.up_proj.weight": torch.randn(128, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {}
            adapter.from_hf(hf_state_dict)

            assert adapter._uses_model_prefix is True

    def test_from_hf_routes_mamba_fp32_params_to_holder(self, config, moe_config, backend):
        """Test from_hf maps public Mamba fp32 keys into the internal holder."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "backbone.layers.0.mixer.A_log": torch.randn(4, dtype=torch.bfloat16),
            "backbone.layers.0.mixer.dt_bias": torch.randn(4, dtype=torch.bfloat16),
            "backbone.layers.0.mixer.D": torch.randn(4, dtype=torch.bfloat16),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {}
            adapter.from_hf(hf_state_dict)

            call_args = mock_merge.call_args[0][0]
            assert "model.layers.0.mixer._fp32_params.A_log" in call_args
            assert "model.layers.0.mixer._fp32_params.dt_bias" in call_args
            assert "model.layers.0.mixer._fp32_params.D" in call_args
            assert "model.layers.0.mixer.A_log" not in call_args
            assert "model.layers.0.mixer.dt_bias" not in call_args
            assert "model.layers.0.mixer.D" not in call_args
            assert call_args["model.layers.0.mixer._fp32_params.A_log"].dtype == torch.float32
            assert call_args["model.layers.0.mixer._fp32_params.dt_bias"].dtype == torch.float32
            assert call_args["model.layers.0.mixer._fp32_params.D"].dtype == torch.float32


class TestNemotronV3AdapterConvertSingleTensor:
    """Test convert_single_tensor_to_hf method."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_convert_embed_tokens(self, config, moe_config, backend):
        """Test converting embed_tokens weight."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(100, 256)
        fqn = "model.embed_tokens.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "backbone.embeddings.weight"
        assert torch.equal(result[0][1], tensor)

    def test_convert_norm_weight(self, config, moe_config, backend):
        """Test converting norm weight."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256)
        fqn = "model.norm.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "backbone.norm_f.weight"
        assert torch.equal(result[0][1], tensor)

    def test_convert_layer_weight(self, config, moe_config, backend):
        """Test converting layer weight (model → backbone)."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256, 256)
        fqn = "model.layers.0.mixer.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "backbone.layers.0.mixer.weight"
        assert torch.equal(result[0][1], tensor)

    def test_convert_mamba_fp32_holder_weight(self, config, moe_config, backend):
        """Test single-tensor conversion hides Mamba fp32 holder keys."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(4, dtype=torch.bfloat16)
        fqn = "model.layers.0.mixer._fp32_params.D"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "backbone.layers.0.mixer.D"
        assert result[0][1].dtype == torch.float32

    def test_convert_expert_tensor(self, config, moe_config, backend):
        """Test converting merged expert tensor to split experts."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        # For relu2 (non-gated), gate_and_up_projs has shape [n_experts, dim, inter_dim]
        # instead of [n_experts, dim, 2*inter_dim]
        tensor = torch.randn(2, 256, 128)
        fqn = "model.layers.0.mixer.experts.gate_and_up_projs"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts") as mock_convert:
            mock_convert.return_value = [
                ("backbone.layers.0.mixer.experts.0.up_proj.weight", torch.randn(128, 256)),
                ("backbone.layers.0.mixer.experts.1.up_proj.weight", torch.randn(128, 256)),
            ]

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            mock_convert.assert_called_once_with(fqn, tensor)
            assert len(result) == 2

    def test_convert_with_exclude_regex(self, config, moe_config, backend):
        """Test converting with exclude regex."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256)
        fqn = "exclude.this.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

        assert len(result) == 0


class TestNemotronV3AdapterNonGatedExperts:
    """Test adapter handling of non-gated (relu2) experts."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",  # Non-gated activation
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_is_gated_moe_property(self, config, moe_config, backend):
        """Test _is_gated_moe property returns False for relu2."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._is_gated_moe is False

    def test_gated_activation_returns_true(self, config, backend):
        """Test _is_gated_moe returns True for gated activations."""
        moe_config = MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="swiglu",  # Gated activation
            dtype=torch.bfloat16,
        )

        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._is_gated_moe is True


class TestNemotronV3AdapterMixerExperts:
    """Test adapter handling of mixer.experts path (vs mlp.experts)."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_expert_path_segment(self, config, moe_config, backend):
        """Test that expert path segment is 'mixer.experts'."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._expert_path_segment == "mixer.experts"

    def test_default_lora_export_uses_v5_for_non_gated_experts(self, config, moe_config, backend):
        """Nemotron V3 emits ParamWrapper weights for its fused up projection.

        The default export uses the corrected peft >= 0.19.1 layout, where the
        native A tensor feeds ``lora_A`` (huggingface/peft#3165).
        """
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)
        tensor = torch.randn(moe_config.n_routed_experts, moe_config.dim, 8)

        result = adapter.convert_single_tensor_to_hf("model.layers.0.mixer.experts.lora_gate_and_up_A", tensor)

        keys = {key for key, _ in result}
        assert keys == {"model.layers.0.mixer.experts.base_layer.lora_A.weight"}
        assert adapter._v5_peft_target_parameters == ("mixer.experts.up_proj", "mixer.experts.down_proj")

    def test_legacy_layout_option_restores_the_pre_flip_suffix(self, config, moe_config, backend):
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)
        tensor = torch.randn(moe_config.n_routed_experts, moe_config.dim, 8)

        result = adapter.convert_single_tensor_to_hf(
            "model.layers.0.mixer.experts.lora_gate_and_up_A", tensor, legacy_paramwrapper_layout=True
        )

        keys = {key for key, _ in result}
        assert keys == {"model.layers.0.mixer.experts.base_layer.lora_B.weight"}

    def test_v4_lora_export_stays_per_expert_for_non_gated_experts(self, config, moe_config, backend):
        """Explicit v4 compatibility retains the per-expert up projection."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)
        tensor = torch.randn(moe_config.n_routed_experts, moe_config.dim, 8)

        result = adapter.convert_single_tensor_to_hf(
            "model.layers.0.mixer.experts.lora_gate_and_up_A", tensor, v4_compatible=True
        )

        keys = {key for key, _ in result}
        assert "backbone.layers.0.mixer.experts.0.up_proj.lora_A.weight" in keys
        assert not any("base_layer" in key for key in keys)

    def test_from_hf_uses_mixer_experts_path(self, config, moe_config, backend):
        """Test that from_hf correctly parses mixer.experts paths."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        # Simulate HF checkpoint with mixer.experts path
        hf_state_dict = {
            "model.layers.0.mixer.experts.0.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mixer.experts.1.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mixer.experts.0.down_proj.weight": torch.randn(256, 128),
            "model.layers.0.mixer.experts.1.down_proj.weight": torch.randn(256, 128),
        }

        with patch.object(adapter, "_validate_expert_availability"):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch(
                    "nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local",
                    side_effect=lambda x, *args: x,
                ):
                    result = adapter._from_hf_w_merged_experts(hf_state_dict)

        # Should have created merged expert tensors with mixer.experts path
        assert "model.layers.0.mixer.experts.gate_and_up_projs" in result
        assert "model.layers.0.mixer.experts.down_projs" in result

    def test_convert_merged_expert_to_hf_split_uses_mixer_path(self, config, moe_config, backend):
        """Test that convert_single_merged_expert uses mixer.experts path."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        # For relu2, tensor has shape [n_experts, dim, inter_dim]
        tensor = torch.randn(2, 256, 128)
        fqn = "model.layers.0.mixer.experts.gate_and_up_projs"

        result = adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)

        assert result is not None
        # Should produce up_proj weights with mixer.experts path (non-gated, so no gate_proj)
        keys = [k for k, _ in result]
        assert all("mixer.experts" in k for k in keys)
        # For relu2, only up_proj is created (no gate_proj)
        assert all("up_proj" in k for k in keys)
