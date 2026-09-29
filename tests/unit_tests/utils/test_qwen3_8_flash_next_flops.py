# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the Qwen3.8-Flash-Next FLOPs formula (GDN + compressed-block QSA + HyperConnections + Engram PLE + MoE)."""

import pytest

from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig,
    Qwen3_8_FlashNextLegacyConfig,
    Qwen3_8_FlashNextLegacyTextConfig,
    Qwen3_8_FlashNextTextConfig,
)
from nemo_automodel.components.utils import flops_utils

# Qwen/Qwen3.8-Flash-Next model card and docs/model-coverage page: 125B backbone parameters (the 51.2B
# Engram table on top) and 6B activated per token. The formula counts GEMM weights only (no norms, no
# embedding gather, no Engram table rows); the LM head is excluded from the "activated" comparison.
CARD_BACKBONE_PARAMS = 125e9
CARD_ACTIVE_PARAMS = 6e9

# Released checkpoint values (config.json text_config) that the config defaults do not fix.
CHECKPOINT_OVERRIDES = dict(ple_layer_ids=[2])


def _text_config(**overrides) -> Qwen3_8_FlashNextTextConfig:
    return Qwen3_8_FlashNextTextConfig(**{**CHECKPOINT_OVERRIDES, **overrides})


def _gemm_params_per_token(cfg) -> float:
    """FLOPs / 6 at seq_len=1: active GEMM parameters plus the negligible one-key attention/indexer terms."""
    return flops_utils.qwen3_8_flash_next_flops(cfg, gbs=1, seq_len=1) / 6


def _backbone_params(cfg) -> float:
    """Backbone parameters implied by the formula: all routed experts instead of top-k, plus the input embedding."""
    per_expert = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    return (
        _gemm_params_per_token(cfg)
        - cfg.num_hidden_layers * cfg.num_experts_per_tok * per_expert
        + cfg.num_hidden_layers * cfg.num_experts * per_expert
        + cfg.vocab_size * cfg.hidden_size
    )


def test_registered_for_text_multimodal_and_legacy_configs():
    for config_cls in (
        Qwen3_8_FlashNextTextConfig,
        Qwen3_8_FlashNextConfig,
        Qwen3_8_FlashNextLegacyTextConfig,
        Qwen3_8_FlashNextLegacyConfig,
    ):
        assert flops_utils.get_flops_formula_for_hf_config(config_cls()) is flops_utils.qwen3_8_flash_next_flops


def test_multimodal_wrapper_uses_text_config():
    text = _text_config()
    wrapper = Qwen3_8_FlashNextConfig(text_config=text)
    assert flops_utils.qwen3_8_flash_next_flops(wrapper, gbs=2, seq_len=512) == flops_utils.qwen3_8_flash_next_flops(
        text, gbs=2, seq_len=512
    )


def test_active_and_backbone_params_match_the_model_card():
    cfg = _text_config()
    active_without_lm_head = _gemm_params_per_token(cfg) - cfg.hidden_size * cfg.vocab_size
    assert active_without_lm_head == pytest.approx(CARD_ACTIVE_PARAMS, rel=0.03)  # 6.14B vs "6B"
    assert _backbone_params(cfg) == pytest.approx(CARD_BACKBONE_PARAMS, rel=0.01)  # 125.8B vs "125B"


def test_sum_mod_matches_brute_force():
    for seq_len in (0, 1, 3, 4, 5, 127, 128, 4096):
        for ratio in (2, 4, 7):
            assert flops_utils._sum_mod(seq_len, ratio) == sum(j % ratio for j in range(1, seq_len + 1))


def test_sparse_attention_saturates_with_sequence_length():
    """Past the 2048-token budget the per-token cost barely grows: 32K costs < 3% more per token than 4K."""
    cfg = _text_config()
    per_token_4k = flops_utils.qwen3_8_flash_next_flops(cfg, gbs=1, seq_len=4096) / 4096
    per_token_32k = flops_utils.qwen3_8_flash_next_flops(cfg, gbs=1, seq_len=32768) / 32768
    assert 1.0 < per_token_32k / per_token_4k < 1.03


def test_linear_in_global_batch_size():
    cfg = _text_config()
    one = flops_utils.qwen3_8_flash_next_flops(cfg, gbs=1, seq_len=2048)
    assert flops_utils.qwen3_8_flash_next_flops(cfg, gbs=64, seq_len=2048) == pytest.approx(64 * one)


def test_qsa_attention_and_indexer_follow_the_layer_pattern():
    """Every extra QSA layer adds its linears (6x), its routed-key BMMs (6x) and its forward-only indexer (2x)."""
    seq_len = 4096
    base = _text_config(num_hidden_layers=8, full_attention_interval=4)  # 2 QSA layers
    more = _text_config(num_hidden_layers=8, full_attention_interval=2)  # 4 QSA layers
    delta = flops_utils.qwen3_8_flash_next_flops(more, gbs=1, seq_len=seq_len) - flops_utils.qwen3_8_flash_next_flops(
        base, gbs=1, seq_len=seq_len
    )

    hs, heads, kv_heads, head_dim = base.hidden_size, base.num_attention_heads, base.num_key_value_heads, base.head_dim
    ratio, budget = base.indexer_compress_ratio, base.indexer_budget
    attn_linear = 6 * seq_len * (hs * heads * head_dim * 2 + 2 * hs * kv_heads * head_dim + heads * head_dim * hs)
    routed_keys = sum(ratio * min(budget // ratio, j // ratio) + (j % ratio) for j in range(1, seq_len + 1))
    attn_bmm = 6 * 2 * heads * head_dim * routed_keys
    indexer = 2 * (
        seq_len * hs * (base.indexer_n_heads + base.indexer_kv_heads) * base.indexer_head_dim
        + base.indexer_n_heads * base.indexer_head_dim * sum(j // ratio for j in range(1, seq_len + 1))
    )
    gdn = flops_utils._gdn_attention_per_layer_flops(
        1,
        seq_len,
        hs,
        base.linear_key_head_dim,
        base.linear_value_head_dim,
        base.linear_num_key_heads,
        base.linear_num_value_heads,
        base.linear_conv_kernel_dim,
    )
    # Two layers switch from GDN to QSA; MoE / HyperConnection / PLE / LM-head terms are unchanged.
    assert delta == pytest.approx(2 * (attn_linear + attn_bmm + indexer - gdn), rel=1e-12)


def test_engram_ple_projection_is_counted_once_per_ple_layer():
    seq_len = 1024
    with_ple = _text_config(ple_layer_ids=[2])
    without_ple = _text_config(ple_layer_ids=[])
    hs, hc = with_ple.hidden_size, with_ple.hc_count
    expected = (
        6
        * seq_len
        * (with_ple.ple_embed_dim * hc * hs + with_ple.ple_embed_dim * hs + hc * hs * with_ple.ple_conv_kernel_size)
    )
    assert flops_utils.qwen3_8_flash_next_flops(
        with_ple, gbs=1, seq_len=seq_len
    ) - flops_utils.qwen3_8_flash_next_flops(without_ple, gbs=1, seq_len=seq_len) == pytest.approx(expected)
