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

"""Tests for the DeepSeek-V4.1 FLOPs formula (CSA2 sparse attention + single-pass mHC + Engram + MoE)."""

import pytest

from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
from nemo_automodel.components.utils import flops_utils

# deepseek-ai/DeepSeek-V4.1-Flash model card: 552B backbone parameters (Engram tables 196B on top),
# 8B activated per token during prefill / 16B during decode. Training runs every layer, so the
# decode figure is the active count; the formula counts GEMM weights only (no norms, sinks, biases).
CARD_BACKBONE_PARAMS = 552e9
CARD_ACTIVE_PARAMS = 16e9


def _text_config(**overrides) -> DeepseekV41TextConfig:
    return DeepseekV41TextConfig(**overrides)


def _active_params(cfg) -> float:
    """Per-token FLOPs / 6 at seq_len=1: active GEMM parameters plus the negligible one-key attention/indexer terms."""
    return flops_utils.deepseek_v41_flops(cfg, gbs=1, seq_len=1) / 6


def _backbone_params(cfg) -> float:
    """Total backbone parameters implied by the formula: all routed experts instead of top-k, plus the embedding."""
    per_expert = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    active = _active_params(cfg)
    return (
        active
        - cfg.num_hidden_layers * cfg.num_experts_per_tok * per_expert
        + cfg.num_hidden_layers * cfg.n_routed_experts * per_expert
        + cfg.vocab_size * cfg.hidden_size
    )


def test_registered_for_text_and_multimodal_configs():
    assert flops_utils.get_flops_formula_for_hf_config(DeepseekV41TextConfig()) is flops_utils.deepseek_v41_flops
    assert flops_utils.get_flops_formula_for_hf_config(DeepseekV41Config()) is flops_utils.deepseek_v41_flops


def test_multimodal_wrapper_uses_text_config():
    text = _text_config()
    wrapper = DeepseekV41Config(text_config=text)
    assert flops_utils.deepseek_v41_flops(wrapper, gbs=2, seq_len=512) == flops_utils.deepseek_v41_flops(
        text, gbs=2, seq_len=512
    )


def test_active_and_backbone_params_match_the_model_card():
    cfg = _text_config()
    assert _active_params(cfg) == pytest.approx(CARD_ACTIVE_PARAMS, rel=0.01)  # 16.09B vs "16B"
    assert _backbone_params(cfg) == pytest.approx(CARD_BACKBONE_PARAMS, rel=0.005)  # 551.8B vs "552B"


def test_closed_form_sums_match_brute_force():
    for seq_len in (1, 5, 127, 128, 129, 1000, 1024, 4096):
        assert flops_utils._sum_min_window(seq_len, 128) == sum(min(i + 1, 128) for i in range(seq_len))
        for ratio in (1, 2, 4):
            for cap in (None, 3, 512):
                expected = sum((j // ratio) if cap is None else min(cap, j // ratio) for j in range(1, seq_len + 1))
                assert flops_utils._sum_min_floor_div(seq_len, ratio, cap) == expected, (seq_len, ratio, cap)


def test_sparse_attention_saturates_with_sequence_length():
    """Per-token cost barely grows past window + top-k: 32K costs < 2% more per token than 4K."""
    cfg = _text_config()
    per_token_4k = flops_utils.deepseek_v41_flops(cfg, gbs=1, seq_len=4096) / 4096
    per_token_32k = flops_utils.deepseek_v41_flops(cfg, gbs=1, seq_len=32768) / 32768
    assert 1.0 < per_token_32k / per_token_4k < 1.02
    # and the model FLOPs sit just above the dense-FFN identity 6 x active params
    assert 1.05 < per_token_4k / (6 * _active_params(cfg)) < 1.15


def test_linear_in_global_batch_size():
    cfg = _text_config()
    one = flops_utils.deepseek_v41_flops(cfg, gbs=1, seq_len=2048)
    assert flops_utils.deepseek_v41_flops(cfg, gbs=64, seq_len=2048) == pytest.approx(64 * one)


def test_indexer_is_forward_only_and_frozen_layers_are_read_from_the_config():
    """Removing the indexer's layers drops exactly its forward-only cost; disabling Engram drops its projection."""
    seq_len = 4096
    base = _text_config()
    no_index = _text_config(index_source_layer_ids=[2, 8, 14, 20], kv_source_layer_ids=[2, 8, 14, 20])
    f_base = flops_utils.deepseek_v41_flops(base, gbs=1, seq_len=seq_len)
    f_less = flops_utils.deepseek_v41_flops(no_index, gbs=1, seq_len=seq_len)
    removed_layers = [24, 28, 32, 36]
    expected_drop = 0
    for lid in removed_layers:
        ratio = base.compress_ratios[lid]
        expected_drop += (
            2
            * seq_len
            * (base.q_lora_rank * base.index_n_heads * base.index_head_dim + base.hidden_size * base.index_n_heads)
        )
        expected_drop += 2 * base.index_n_heads * base.index_head_dim * sum(j // ratio for j in range(1, seq_len + 1))
    assert f_base - f_less == pytest.approx(expected_drop)

    no_engram = _text_config(engram_layer_ids=[], engram_num_embeddings=[])
    hash_heads = (base.engram_max_ngram_size - 1) * base.engram_n_heads
    engram_proj = 2 * hash_heads * base.engram_head_dim * base.hidden_size * (base.hc_mult + 1)
    assert f_base - flops_utils.deepseek_v41_flops(no_engram, gbs=1, seq_len=seq_len) == pytest.approx(
        6 * seq_len * engram_proj
    )


# Pinned by evaluating the formula on the released Flash configuration (see the formula docstring):
# 96.63 GFLOPs for one token (6 x 16.09B active GEMM parameters plus the one-key attention and indexer terms) and
# 432.71 TFLOPs for a 4096-token sequence (105.6 GFLOPs/token).
PRECOMPUTED_SEQ1 = 96629071872
PRECOMPUTED_SEQ4096 = 432713811099648


def test_precomputed_values():
    cfg = _text_config()
    assert int(flops_utils.deepseek_v41_flops(cfg, gbs=1, seq_len=1)) == PRECOMPUTED_SEQ1
    assert int(flops_utils.deepseek_v41_flops(cfg, gbs=1, seq_len=4096)) == PRECOMPUTED_SEQ4096
