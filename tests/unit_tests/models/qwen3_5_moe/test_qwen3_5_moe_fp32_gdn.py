# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Qwen3.5-MoE fp32-aware GatedDeltaNet construction."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.common.gated_delta_net_fp32 import HOLDER_NAME
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet


def _text_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        layer_types=["linear_attention"],
    )


def test_constructor_forces_tracked_params_fp32_under_bf16_default_dtype():
    cfg = _text_config()
    cfg.torch_dtype = torch.bfloat16

    old_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        gdn = CPAwareGatedDeltaNet(cfg, layer_idx=0)
    finally:
        torch.set_default_dtype(old_default_dtype)

    assert gdn.A_log.dtype == torch.float32
    assert gdn.dt_bias.dtype == torch.float32

    assert HOLDER_NAME in gdn._modules
    assert "A_log" not in gdn._parameters
    assert "dt_bias" not in gdn._parameters
    assert gdn._fp32_params.A_log.dtype == torch.float32
    assert gdn._fp32_params.dt_bias.dtype == torch.float32
