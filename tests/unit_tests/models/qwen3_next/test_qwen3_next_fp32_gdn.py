# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Qwen3-Next fp32-aware GatedDeltaNet gate routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from nemo_automodel.components.models.common.gated_delta_net_fp32 import HOLDER_NAME
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextFp32GatedDeltaNet


def _bare_gdn(num_v: int = 4, dtype: torch.dtype = torch.float32) -> Qwen3NextFp32GatedDeltaNet:
    """Build a Qwen3NextFp32GatedDeltaNet shell with only A_log/dt_bias (no full init)."""
    gdn = Qwen3NextFp32GatedDeltaNet.__new__(Qwen3NextFp32GatedDeltaNet)
    nn.Module.__init__(gdn)
    gdn.A_log = nn.Parameter(torch.ones(num_v, dtype=dtype))
    gdn.dt_bias = nn.Parameter(torch.zeros(num_v, dtype=dtype))
    return gdn


def _expected_gate(a: torch.Tensor) -> torch.Tensor:
    return -torch.ones(a.shape[-1]).exp() * F.softplus(torch.zeros(a.shape[-1]))


def test_compute_gate_fallback_without_holder():
    gdn = _bare_gdn()
    a = torch.zeros_like(gdn.A_log)
    g = gdn._compute_gate(a)
    assert g.dtype == torch.float32
    assert torch.allclose(g, _expected_gate(a), atol=1e-5)


def test_compute_gate_routes_through_holder():
    gdn = _constructed_gdn()
    gdn._fp32_params.A_log.data.fill_(1.0)
    gdn._fp32_params.dt_bias.data.zero_()
    assert HOLDER_NAME in gdn._modules
    assert "A_log" not in gdn._parameters
    assert "dt_bias" not in gdn._parameters
    assert gdn.A_log is gdn._fp32_params._parameters["A_log"]
    assert gdn.dt_bias is gdn._fp32_params._parameters["dt_bias"]

    a = torch.zeros_like(gdn.A_log)
    g = gdn._compute_gate(a)
    assert g.dtype == torch.float32
    assert torch.allclose(g, _expected_gate(a), atol=1e-5)


def test_compute_gate_fp32_with_bf16_input():
    gdn = _constructed_gdn()
    g = gdn._compute_gate(torch.zeros_like(gdn.A_log, dtype=torch.bfloat16))
    assert g.dtype == torch.float32


def test_constructor_forces_tracked_params_fp32_under_bf16_default_dtype():
    cfg = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    cfg.torch_dtype = torch.bfloat16

    old_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        gdn = Qwen3NextFp32GatedDeltaNet(cfg, layer_idx=0)
    finally:
        torch.set_default_dtype(old_default_dtype)

    assert gdn.A_log.dtype == torch.float32
    assert gdn.dt_bias.dtype == torch.float32
    assert HOLDER_NAME in gdn._modules
    assert "A_log" not in gdn._parameters
    assert "dt_bias" not in gdn._parameters
    assert gdn._fp32_params.A_log.dtype == torch.float32
    assert gdn._fp32_params.dt_bias.dtype == torch.float32


def _constructed_gdn() -> Qwen3NextFp32GatedDeltaNet:
    cfg = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    return Qwen3NextFp32GatedDeltaNet(cfg, layer_idx=0)
