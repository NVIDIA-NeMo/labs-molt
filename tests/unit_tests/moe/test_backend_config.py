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

import logging

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig, CudaGraphConfig
from nemo_automodel.components.moe.config import MoEConfig


class TestBackendConfigSparseAttention:
    """Test the independent sparse-attention backend selector."""

    def test_sparse_attention_backend_contract(self):
        assert BackendConfig().sparse_attn == "generic"
        assert BackendConfig(sparse_attn="msa").sparse_attn == "msa"
        with pytest.raises(ValueError, match="expected 'generic' or 'msa'"):
            BackendConfig(sparse_attn="unknown")


class TestBackendConfigGatePrecision:
    """Test BackendConfig gate_precision field."""

    def test_gate_precision_string_input_fp32(self):
        """Test that BackendConfig gate_precision accepts string input and converts to torch.dtype."""
        backend_config = BackendConfig(gate_precision="torch.float32")
        assert backend_config.gate_precision == torch.float32

    def test_gate_precision_string_input_fp64(self):
        """Test that BackendConfig gate_precision accepts fp64 string input."""
        backend_config = BackendConfig(gate_precision="torch.float64")
        assert backend_config.gate_precision == torch.float64

    def test_gate_precision_string_input_short_form(self):
        """Test that BackendConfig gate_precision accepts short form string input."""
        backend_config = BackendConfig(gate_precision="float32")
        assert backend_config.gate_precision == torch.float32

    def test_gate_precision_none_default(self):
        """Test that BackendConfig gate_precision defaults to None."""
        backend_config = BackendConfig()
        assert backend_config.gate_precision is None

    def test_gate_precision_torch_dtype_input(self):
        """Test that BackendConfig gate_precision accepts torch.dtype directly."""
        backend_config = BackendConfig(gate_precision=torch.float32)
        assert backend_config.gate_precision == torch.float32


class TestBackendConfigExpertsDispatcherValidation:
    """Test BackendConfig validation for experts and dispatcher fields."""

    def test_te_experts_falls_back_to_torch(self):
        """Test that BackendConfig falls back TE experts to native torch_mm with the torch dispatcher."""
        config = BackendConfig(experts="te", dispatcher="torch")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "torch"

    def test_gmm_experts_alias_is_deprecated(self):
        """Test that the old gmm name warns while remaining compatible."""
        with pytest.warns(FutureWarning, match="experts='gmm' is deprecated"):
            config = BackendConfig(experts="gmm", dispatcher="torch")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "torch"

    def test_te_experts_with_deepep_valid(self):
        """Test that te experts with deepep dispatcher is valid."""
        config = BackendConfig(experts="te", dispatcher="deepep")
        assert config.experts == "te"
        assert config.dispatcher == "deepep"

    def test_torch_experts_with_torch_dispatcher_valid(self):
        """Test that torch experts with torch dispatcher is valid."""
        config = BackendConfig(experts="torch", dispatcher="torch")
        assert config.experts == "torch"
        assert config.dispatcher == "torch"

    def test_torch_experts_with_deepep_dispatcher_valid(self):
        """Test that torch experts with deepep dispatcher is valid."""
        config = BackendConfig(experts="torch", dispatcher="deepep")
        assert config.experts == "torch"
        assert config.dispatcher == "deepep"

    def test_torch_mm_experts_with_torch_dispatcher_valid(self):
        """Test that torch_mm experts with torch dispatcher is valid."""
        config = BackendConfig(experts="torch_mm", dispatcher="torch")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "torch"

    def test_torch_mm_experts_with_deepep_dispatcher_valid(self):
        """Test that torch_mm experts with deepep dispatcher is valid."""
        config = BackendConfig(experts="torch_mm", dispatcher="deepep")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "deepep"


class TestBackendConfigFakeGateNoise:
    """Test BackendConfig fake_gate_noise field."""

    def test_fake_gate_noise_default(self):
        """Test that fake_gate_noise defaults to 0.0."""
        config = BackendConfig()
        assert config.fake_gate_noise == 0.0

    def test_fake_gate_noise_custom_value(self):
        """Test that fake_gate_noise accepts a custom float value."""
        config = BackendConfig(fake_gate_noise=0.5)
        assert config.fake_gate_noise == 0.5

    def test_fake_gate_noise_with_fake_balanced_gate(self):
        """Test that fake_gate_noise can be set alongside fake_balanced_gate."""
        config = BackendConfig(fake_balanced_gate=True, fake_gate_noise=0.3)
        assert config.fake_balanced_gate is True
        assert config.fake_gate_noise == 0.3


class TestBackendConfigEnableDeepepRemoved:
    """enable_deepep was removed: it is ignored (with a warning) and never alters dispatcher/experts."""

    def test_enable_deepep_true_is_ignored_and_warns(self, caplog):
        """enable_deepep=True is ignored; dispatcher/experts keep their explicit values and a warning is logged."""
        with caplog.at_level(logging.WARNING):
            config = BackendConfig(dispatcher="hybridep", experts="torch_mm", enable_deepep=True)
        assert config.dispatcher == "hybridep"  # not overridden to "deepep"
        assert config.experts == "torch_mm"
        assert config.enable_deepep is None  # cleared after the warning
        assert "enable_deepep is no longer supported" in caplog.text

    def test_enable_deepep_false_is_ignored_and_warns(self, caplog):
        """enable_deepep=False is ignored; the dispatcher is NOT forced to torch and a warning is logged."""
        with caplog.at_level(logging.WARNING):
            config = BackendConfig(dispatcher="deepep", experts="torch_mm", enable_deepep=False)
        assert config.dispatcher == "deepep"  # not forced to "torch"
        assert config.experts == "torch_mm"
        assert config.enable_deepep is None
        assert "enable_deepep is no longer supported" in caplog.text

    def test_enable_deepep_none_no_warning(self, caplog):
        """enable_deepep=None (default) leaves the field as None and logs no warning."""
        with caplog.at_level(logging.WARNING):
            config = BackendConfig()
        assert config.enable_deepep is None
        assert "enable_deepep" not in caplog.text

    def test_enable_deepep_does_not_override_explicit_dispatcher(self, caplog):
        """A stale enable_deepep no longer wins over an explicit dispatcher/experts."""
        with caplog.at_level(logging.WARNING):
            config = BackendConfig(dispatcher="torch", experts="torch", enable_deepep=True)
        assert config.dispatcher == "torch"
        assert config.experts == "torch"

    def test_dispatcher_without_enable_deepep(self):
        """dispatcher works correctly without enable_deepep (field stays None)."""
        config = BackendConfig(dispatcher="deepep")
        assert config.dispatcher == "deepep"
        assert config.enable_deepep is None

        config = BackendConfig(dispatcher="torch")
        assert config.dispatcher == "torch"
        assert config.enable_deepep is None


class TestBackendConfigHybridEP:
    """Test BackendConfig HybridEP dispatcher support."""

    def test_hybridep_dispatcher_valid(self):
        """Test that BackendConfig accepts hybridep dispatcher."""
        config = BackendConfig(dispatcher="hybridep")
        assert config.dispatcher == "hybridep"

    def test_hybridep_dispatcher_num_sms_default(self):
        """Test that dispatcher_num_sms defaults to 20."""
        config = BackendConfig(dispatcher="hybridep")
        assert config.dispatcher_num_sms == 20

    def test_hybridep_dispatcher_num_sms_custom(self):
        """Test that dispatcher_num_sms accepts a custom value."""
        config = BackendConfig(dispatcher="hybridep", dispatcher_num_sms=24)
        assert config.dispatcher_num_sms == 24

    def test_dispatcher_share_token_dispatcher_default(self):
        """Test that dispatcher_share_token_dispatcher defaults to enabled."""
        config = BackendConfig(dispatcher="deepep")
        assert config.dispatcher_share_token_dispatcher is True

    def test_dispatcher_share_token_dispatcher_custom(self):
        """Test that dispatcher_share_token_dispatcher accepts an explicit value."""
        config = BackendConfig(dispatcher="deepep", dispatcher_share_token_dispatcher=False)
        assert config.dispatcher_share_token_dispatcher is False

    def test_dispatcher_async_dispatch_default(self):
        """Test that dispatcher_async_dispatch defaults to disabled."""
        config = BackendConfig(dispatcher="deepep")
        assert config.dispatcher_async_dispatch is False

    def test_dispatcher_async_dispatch_custom(self):
        """Test that dispatcher_async_dispatch accepts an explicit value."""
        config = BackendConfig(dispatcher="deepep", dispatcher_async_dispatch=True)
        assert config.dispatcher_async_dispatch is True

    def test_te_experts_falls_back_with_hybridep(self):
        """Test that te experts with hybridep dispatcher is valid (no fallback)."""
        config = BackendConfig(experts="te", dispatcher="hybridep")
        assert config.experts == "te"
        assert config.dispatcher == "hybridep"

    def test_torch_mm_experts_with_hybridep(self):
        """Test that torch_mm experts with hybridep dispatcher is valid."""
        config = BackendConfig(experts="torch_mm", dispatcher="hybridep")
        assert config.experts == "torch_mm"
        assert config.dispatcher == "hybridep"


class TestMoEConfig:
    """Test MoEConfig dataclass."""

    @pytest.fixture
    def base_moe_config_kwargs(self):
        """Base kwargs for creating a MoEConfig."""
        return {
            "n_routed_experts": 8,
            "n_shared_experts": 0,
            "n_activated_experts": 2,
            "n_expert_groups": 1,
            "n_limited_groups": 1,
            "train_gate": False,
            "gate_bias_update_factor": 0.0,
            "aux_loss_coeff": 0.0,
            "score_func": "softmax",
            "route_scale": 1.0,
            "dim": 128,
            "inter_dim": 256,
            "moe_inter_dim": 256,
            "norm_topk_prob": False,
        }

    def test_dtype_string_input_torch_prefix(self, base_moe_config_kwargs):
        """Test that MoEConfig dtype accepts string input with torch prefix."""
        config = MoEConfig(**base_moe_config_kwargs, dtype="torch.float16")
        assert config.dtype == torch.float16

    def test_dtype_string_input_short_form(self, base_moe_config_kwargs):
        """Test that MoEConfig dtype accepts short form string input."""
        config = MoEConfig(**base_moe_config_kwargs, dtype="bfloat16")
        assert config.dtype == torch.bfloat16

    def test_dtype_torch_dtype_input(self, base_moe_config_kwargs):
        """Test that MoEConfig dtype accepts torch.dtype directly."""
        config = MoEConfig(**base_moe_config_kwargs, dtype=torch.float32)
        assert config.dtype == torch.float32

    def test_dtype_default_bfloat16(self, base_moe_config_kwargs):
        """Test that MoEConfig dtype defaults to bfloat16."""
        config = MoEConfig(**base_moe_config_kwargs)
        assert config.dtype == torch.bfloat16

    def test_expert_activation_default(self, base_moe_config_kwargs):
        """Test that expert_activation defaults to swiglu."""
        config = MoEConfig(**base_moe_config_kwargs)
        assert config.expert_activation == "swiglu"

    def test_expert_activation_quick_geglu(self, base_moe_config_kwargs):
        """Test that expert_activation can be set to quick_geglu."""
        config = MoEConfig(**base_moe_config_kwargs, expert_activation="quick_geglu")
        assert config.expert_activation == "quick_geglu"

    def test_optional_fields_defaults(self, base_moe_config_kwargs):
        """Test that optional fields have correct defaults."""
        config = MoEConfig(**base_moe_config_kwargs)
        assert config.router_bias is False
        assert config.expert_bias is False
        assert config.softmax_before_topk is False
        assert config.shared_expert_gate is False
        assert config.shared_expert_inter_dim is None

    def test_moeconfig_importable_from_layers(self, base_moe_config_kwargs):
        """Test that MoEConfig is still importable from layers for backwards compatibility."""
        from nemo_automodel.components.moe.layers import MoEConfig as MoEConfigFromLayers

        config = MoEConfigFromLayers(**base_moe_config_kwargs)
        assert config.n_routed_experts == 8

    def test_swiglu_limit_default_zero(self, base_moe_config_kwargs):
        """``swiglu_limit`` defaults to 0.0 (preserves the legacy fused swiglu path)."""
        config = MoEConfig(**base_moe_config_kwargs)
        assert config.swiglu_limit == 0.0

    @pytest.mark.parametrize("limit", [1.0, 7.0, 100.5])
    def test_swiglu_limit_custom_positive(self, base_moe_config_kwargs, limit):
        """``swiglu_limit`` accepts positive floats for the DSV4 clamped variant."""
        config = MoEConfig(**base_moe_config_kwargs, swiglu_limit=limit)
        assert config.swiglu_limit == limit


class TestBackendConfigMXFP8:
    """MXFP8 backend wiring: the torch_mm_mxfp8 experts option + use_mxfp8 derivation."""

    def test_experts_accepts_torch_mm_mxfp8(self):
        """BackendConfig.experts accepts the torch_mm_mxfp8 value (dispatcher torch -> kept)."""
        config = BackendConfig(experts="torch_mm_mxfp8", dispatcher="torch")
        assert config.experts == "torch_mm_mxfp8"

    def test_use_mxfp8_true_only_for_torch_mm_mxfp8(self):
        """The use_mxfp8 predicate (as derived in experts.py) is True only for torch_mm_mxfp8."""

        def _use_mxfp8(experts):
            return experts == "torch_mm_mxfp8"

        assert _use_mxfp8("torch_mm_mxfp8") is True
        for other in ("torch", "torch_mm", "gmm", "te"):
            assert _use_mxfp8(other) is False

    def test_use_torch_mm_includes_deprecated_gmm_alias(self):
        """The grouped-MM selector covers torch_mm, its deprecated alias, and MXFP8."""

        def _use_torch_mm(experts):
            return experts in ("gmm", "torch_mm", "torch_mm_mxfp8")

        assert _use_torch_mm("gmm") is True
        assert _use_torch_mm("torch_mm") is True
        assert _use_torch_mm("torch_mm_mxfp8") is True


class TestTEFp8ConfigRecipe:
    """TEFp8Config.build_recipe recipe-string mapping (the 'mxfp8' shorthand)."""

    def test_recipe_field_accepts_mxfp8(self):
        from nemo_automodel.components.models.common.utils import TEFp8Config

        cfg = TEFp8Config(recipe="mxfp8")
        assert cfg.recipe == "mxfp8"

    def test_build_recipe_mxfp8_maps_to_mxfp8blockscaling(self):
        """recipe='mxfp8' -> a TE MXFP8BlockScaling instance (when TE is importable)."""
        from nemo_automodel.components.models.common.utils import HAVE_TE, TEFp8Config

        if not HAVE_TE:
            pytest.skip("transformer_engine not importable")
        from transformer_engine.common.recipe import MXFP8BlockScaling

        recipe = TEFp8Config(recipe="mxfp8").build_recipe()
        assert isinstance(recipe, MXFP8BlockScaling)

    def test_build_recipe_prebuilt_object_passthrough(self):
        """A pre-built recipe object is returned unchanged (when TE is importable)."""
        from nemo_automodel.components.models.common.utils import HAVE_TE, TEFp8Config

        if not HAVE_TE:
            pytest.skip("transformer_engine not importable")
        sentinel = object()
        assert TEFp8Config(recipe=sentinel).build_recipe() is sentinel


class TestBackendConfigCompileAttn:
    """BackendConfig.compile_attn fullgraph-compile flag (drives both MLA and GQA attention)."""

    def test_compile_attn_default_false(self):
        assert BackendConfig().compile_attn is False

    def test_compile_attn_explicit_true(self):
        assert BackendConfig(compile_attn=True).compile_attn is True

    def test_compile_mla_removed(self):
        # compile_mla was consolidated into the generic compile_attn flag.
        with pytest.raises(TypeError):
            BackendConfig(compile_mla=True)


class TestBackendConfigRopeFusionDisabled:
    """TE fused RoPE is temporarily force-disabled globally in __post_init__ (see #3027)."""

    def test_rope_fusion_default_forced_false(self):
        """The default resolves to False regardless of TE/CUDA availability."""
        assert BackendConfig().rope_fusion is False

    def test_rope_fusion_explicit_true_is_overridden(self, caplog):
        """An explicit rope_fusion=True is forced back to False and warns once."""
        with caplog.at_level(logging.WARNING):
            config = BackendConfig(rope_fusion=True)
        assert config.rope_fusion is False
        assert "rope_fusion is temporarily force-disabled globally" in caplog.text

    def test_rope_fusion_explicit_false_no_warning(self, caplog):
        """An explicit rope_fusion=False stays False and logs no override warning."""
        with caplog.at_level(logging.WARNING):
            config = BackendConfig(rope_fusion=False)
        assert config.rope_fusion is False
        assert "force-disabled" not in caplog.text


class TestBackendConfigPartialCudaGraphs:
    def test_empty_module_list_disables_cuda_graphs(self):
        assert BackendConfig().cuda_graph.modules == []

    def test_mapping_is_normalized_to_typed_config(self):
        config = BackendConfig(attn="te", cuda_graph={"modules": ["te_dpa"]})

        assert config.cuda_graph == CudaGraphConfig(modules=["te_dpa"])

    def test_accepts_megatron_moe_module_names(self):
        modules = ["attn", "moe_router", "moe_preprocess"]
        config = BackendConfig(
            attn="te",
            linear="torch",
            rms_norm="torch",
            rope_fusion=False,
            dispatcher="hybridep",
            cuda_graph=CudaGraphConfig(modules=modules),
        )

        assert config.cuda_graph.modules == modules

    @pytest.mark.parametrize("modules", ["te_dpa", ("te_dpa",)])
    def test_modules_must_be_a_list(self, modules):
        with pytest.raises(TypeError, match="must be a list"):
            CudaGraphConfig(modules=modules)

    def test_module_names_must_be_strings(self):
        with pytest.raises(TypeError, match="entries must be strings"):
            CudaGraphConfig(modules=[1])

    def test_rejects_unknown_or_duplicate_modules(self):
        with pytest.raises(ValueError, match=r"Unsupported cuda_graph\.modules"):
            CudaGraphConfig(modules=["mlp"])
        with pytest.raises(ValueError, match="must not contain duplicates"):
            CudaGraphConfig(modules=["te_dpa", "te_dpa"])
        with pytest.raises(ValueError, match="cannot contain both"):
            CudaGraphConfig(modules=["attn", "te_dpa"])

    @pytest.mark.parametrize("module", ["attn", "te_dpa"])
    def test_attention_requires_te_backend(self, module):
        with pytest.raises(ValueError, match="requires attn='te'"):
            BackendConfig(
                attn="sdpa",
                cuda_graph=CudaGraphConfig(modules=[module]),
            )

    def test_whole_attention_requires_mixed_pytorch_te_backend(self):
        with pytest.raises(ValueError, match="requires linear='torch'"):
            BackendConfig(
                attn="te",
                linear="te",
                rms_norm="torch",
                rope_fusion=False,
                cuda_graph=CudaGraphConfig(modules=["attn"]),
            )
        with pytest.raises(ValueError, match="requires rms_norm='torch'"):
            BackendConfig(
                attn="te",
                linear="torch",
                rms_norm="te",
                rope_fusion=False,
                cuda_graph=CudaGraphConfig(modules=["attn"]),
            )

    def test_whole_attention_uses_resolved_rope_fusion_value(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = BackendConfig(
                attn="te",
                linear="torch",
                rms_norm="torch",
                rope_fusion=True,
                cuda_graph=CudaGraphConfig(modules=["attn"]),
            )

        assert config.rope_fusion is False
        assert config.cuda_graph.modules == ["attn"]

    def test_preprocess_requires_router_and_hybridep(self):
        with pytest.raises(ValueError, match="requires 'moe_router'"):
            CudaGraphConfig(modules=["moe_preprocess"])
        with pytest.raises(ValueError, match="requires dispatcher='hybridep'"):
            BackendConfig(
                dispatcher="deepep",
                cuda_graph=CudaGraphConfig(modules=["moe_router", "moe_preprocess"]),
            )
