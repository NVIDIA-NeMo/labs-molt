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

"""Tests for newly added FLOPs calculator functions:

- minimax_m2_flops
- qwen3_5_flops (hybrid GDN/full attention, MoE and Dense variants)
- step3_5_flash_flops (hybrid full/SWA + MoE)
- mla_moe_flops (shared MLA + MoE helper for Kimi K2, GLM-5, Mistral Small 4)
- deepseekv3_flops DSA (sparse attention) extension
- VL composite config text_config fallback in qwen3_flops and qwen3_5_flops
- _mamba_layer_flops refactored formula
- _hybrid_model_flops conditional accumulation
- kimi_k3_flops registration (hybrid KDA / gated-MLA + latent MoE)
"""

from types import SimpleNamespace

import pytest

from nemo_automodel.components.utils import flops_utils

# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def _qwen3_5_moe_cfg() -> SimpleNamespace:
    """Qwen3.5-35B-A3B MoE config (simplified)."""
    return SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=151936,
        head_dim=128,
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        num_experts=64,
        num_experts_per_tok=4,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=2048,
        max_position_embeddings=4096,
        attn_output_gate=True,
    )


def _qwen3_5_dense_cfg() -> SimpleNamespace:
    """Qwen3.5 Dense variant (no MoE)."""
    return SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=151936,
        head_dim=128,
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        intermediate_size=5632,
        max_position_embeddings=4096,
        attn_output_gate=True,
    )


def _mla_moe_cfg() -> SimpleNamespace:
    """Kimi K2 / GLM-5 style MLA + MoE config (simplified)."""
    return SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=16,
        num_attention_heads=32,
        vocab_size=131072,
        q_lora_rank=512,
        kv_lora_rank=256,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        intermediate_size=8192,
        moe_intermediate_size=1024,
        num_experts_per_tok=4,
        n_shared_experts=1,
        first_k_dense_replace=2,
        max_position_embeddings=4096,
    )


# ---------------------------------------------------------------------------
# Tests: minimax_m2_flops
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: qwen3_5_flops
# ---------------------------------------------------------------------------


class TestQwen35Flops:
    def test_moe_basic(self):
        cfg = _qwen3_5_moe_cfg()
        result = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)
        assert result > 0

    def test_dense_basic(self):
        cfg = _qwen3_5_dense_cfg()
        result = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)
        assert result > 0

    def test_gbs_scaling_moe(self):
        cfg = _qwen3_5_moe_cfg()
        r1 = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)
        r2 = flops_utils.qwen3_5_flops(cfg, gbs=4, seq_len=1024)
        assert r2 == pytest.approx(4 * r1, rel=1e-6)

    def test_moe_counts_activated_experts_not_total_experts(self):
        cfg = _qwen3_5_moe_cfg()
        base = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)

        cfg.num_experts *= 2
        more_total_experts = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)
        cfg.num_experts_per_tok *= 2
        more_activated_experts = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)

        assert more_total_experts == base
        assert more_activated_experts > more_total_experts

    def test_moe_counts_shared_expert_width(self):
        cfg = _qwen3_5_moe_cfg()
        base = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)
        added_width = 512

        cfg.shared_expert_intermediate_size += added_width
        with_wider_shared_expert = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)

        expected_delta = 6 * 1 * cfg.num_hidden_layers * 1024 * cfg.hidden_size * 3 * added_width
        assert with_wider_shared_expert - base == expected_delta

    def test_vl_text_config_fallback(self):
        """VL composite config should extract text_config when num_hidden_layers is missing."""
        text_cfg = _qwen3_5_moe_cfg()
        vl_cfg = SimpleNamespace(text_config=text_cfg)
        direct = flops_utils.qwen3_5_flops(text_cfg, gbs=1, seq_len=1024)
        via_vl = flops_utils.qwen3_5_flops(vl_cfg, gbs=1, seq_len=1024)
        assert via_vl == direct

    def test_layer_types_override(self):
        """Explicit layer_types should override full_attention_interval."""
        cfg = _qwen3_5_moe_cfg()
        # 16 layers, 4 full + 12 GDN (interval=4)
        result_interval = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)

        # Now override with all full attention
        cfg.layer_types = ["full_attention"] * 16
        result_all_full = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)

        # All full attention should differ from hybrid
        assert result_all_full != result_interval

    def test_mtp_increases_flops(self):
        cfg = _qwen3_5_moe_cfg()
        base = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)
        cfg.mtp_num_hidden_layers = 2
        with_mtp = flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024)
        assert with_mtp > base

    def test_precomputed_moe(self):
        cfg = _qwen3_5_moe_cfg()
        actual = int(flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024))
        assert actual == 6200003395584

    def test_precomputed_dense(self):
        cfg = _qwen3_5_dense_cfg()
        actual = int(flops_utils.qwen3_5_flops(cfg, gbs=1, seq_len=1024))
        assert actual == 5890765750272


# ---------------------------------------------------------------------------
# Tests: qwen3_flops VL text_config fallback
# ---------------------------------------------------------------------------


class TestQwen3VLFallback:
    def test_vl_text_config_fallback(self):
        """qwen3_flops should extract text_config for VL composite configs."""
        text_cfg = SimpleNamespace(
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            vocab_size=32000,
            head_dim=128,
            max_position_embeddings=2048,
        )
        vl_cfg = SimpleNamespace(text_config=text_cfg)
        direct = flops_utils.qwen3_flops(text_cfg, gbs=1, seq_len=1024)
        via_vl = flops_utils.qwen3_flops(vl_cfg, gbs=1, seq_len=1024)
        assert via_vl == direct


# ---------------------------------------------------------------------------
# Tests: mla_moe_flops
# ---------------------------------------------------------------------------


class TestMlaMoeFlops:
    def test_basic(self):
        cfg = _mla_moe_cfg()
        result = flops_utils.mla_moe_flops(cfg, gbs=1, seq_len=1024)
        assert result > 0

    def test_gbs_scaling(self):
        cfg = _mla_moe_cfg()
        r1 = flops_utils.mla_moe_flops(cfg, gbs=1, seq_len=1024)
        r2 = flops_utils.mla_moe_flops(cfg, gbs=3, seq_len=1024)
        assert r2 == pytest.approx(3 * r1, rel=1e-6)

    def test_vl_text_config_fallback(self):
        """Should transparently unwrap VL configs with text_config."""
        text_cfg = _mla_moe_cfg()
        vl_cfg = SimpleNamespace(text_config=text_cfg)
        direct = flops_utils.mla_moe_flops(text_cfg, gbs=1, seq_len=1024)
        via_vl = flops_utils.mla_moe_flops(vl_cfg, gbs=1, seq_len=1024)
        assert via_vl == direct

    def test_precomputed_value(self):
        cfg = _mla_moe_cfg()
        actual = int(flops_utils.mla_moe_flops(cfg, gbs=1, seq_len=1024))
        assert actual == 12962211299328


# ---------------------------------------------------------------------------
# Tests: step3_5_flash_flops
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: deepseekv3_flops with DSA (sparse attention)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: _mamba_layer_flops (refactored formula)
# ---------------------------------------------------------------------------


class TestMambaLayerFlops:
    def test_basic(self):
        cfg = SimpleNamespace(
            hidden_size=2688,
            mamba_num_heads=64,
            mamba_head_dim=64,
            mamba_state_dim=128,
            n_groups=8,
        )
        result = flops_utils._mamba_layer_flops(cfg, gbs=1, seq_len=1024)
        assert result > 0

    def test_scaling_with_gbs(self):
        cfg = SimpleNamespace(
            hidden_size=2688,
            mamba_num_heads=64,
            mamba_head_dim=64,
            mamba_state_dim=128,
            n_groups=8,
        )
        r1 = flops_utils._mamba_layer_flops(cfg, gbs=1, seq_len=1024)
        r2 = flops_utils._mamba_layer_flops(cfg, gbs=2, seq_len=1024)
        assert r2 == pytest.approx(2 * r1, rel=1e-6)

    def test_precomputed_value(self):
        cfg = SimpleNamespace(
            hidden_size=2688,
            mamba_num_heads=64,
            mamba_head_dim=64,
            mamba_state_dim=128,
            n_groups=8,
        )
        actual = int(flops_utils._mamba_layer_flops(cfg, gbs=1, seq_len=1024))
        assert actual == 249091325952


# ---------------------------------------------------------------------------
# Tests: _hybrid_model_flops (conditional accumulation)
# ---------------------------------------------------------------------------


class TestHybridModelFlops:
    def test_basic(self):
        cfg = SimpleNamespace(
            hidden_size=2688,
            num_hidden_layers=4,
            num_attention_heads=32,
            num_key_value_heads=2,
            intermediate_size=1856,
            vocab_size=131072,
            mamba_num_heads=64,
            mamba_head_dim=64,
            ssm_state_size=128,
            n_groups=8,
            num_experts_per_tok=6,
            moe_intermediate_size=1856,
            moe_shared_expert_intermediate_size=3712,
            n_routed_experts=128,
            hybrid_override_pattern="MEME",
        )
        result = flops_utils._hybrid_model_flops(cfg, gbs=1, seq_len=1024)
        assert result > 0

    def test_matches_nemotronh(self):
        """_hybrid_model_flops should match nemotronh_flops for a nano config."""
        cfg = SimpleNamespace(
            hidden_size=2688,
            num_hidden_layers=4,
            num_attention_heads=32,
            num_key_value_heads=2,
            intermediate_size=1856,
            vocab_size=131072,
            mamba_num_heads=64,
            mamba_head_dim=64,
            ssm_state_size=128,
            n_groups=8,
            num_experts_per_tok=6,
            moe_intermediate_size=1856,
            moe_shared_expert_intermediate_size=3712,
            n_routed_experts=128,
            hybrid_override_pattern="MEME",
            max_position_embeddings=2048,
        )
        hybrid = flops_utils._hybrid_model_flops(cfg, gbs=1, seq_len=1024)
        nemo = flops_utils.nemotronh_flops(cfg, gbs=1, seq_len=1024)
        assert hybrid == nemo


# ---------------------------------------------------------------------------
# Tests: get_flops_formula_for_hf_config dispatch
# ---------------------------------------------------------------------------


class TestGetFlopsFormula:
    def _make_config(self, class_name):
        """Create an object whose class name matches a given HF config class."""
        cls = type(class_name, (), {})
        return cls()


    def test_qwen3_5_moe(self):
        cfg = self._make_config("Qwen3_5MoeConfig")
        assert flops_utils.get_flops_formula_for_hf_config(cfg) == flops_utils.qwen3_5_flops

    def test_qwen3_5_dense(self):
        cfg = self._make_config("Qwen3_5Config")
        assert flops_utils.get_flops_formula_for_hf_config(cfg) == flops_utils.qwen3_5_flops


    def test_glm_moe_dsa(self):
        cfg = self._make_config("GlmMoeDsaConfig")
        assert flops_utils.get_flops_formula_for_hf_config(cfg) == flops_utils.mla_moe_flops


    def test_unknown_falls_back_to_transformer(self):
        cfg = self._make_config("UnknownModelConfig")
        assert flops_utils.get_flops_formula_for_hf_config(cfg) == flops_utils.transformer_flops
