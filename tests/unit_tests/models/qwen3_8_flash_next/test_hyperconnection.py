# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU tests for the exact Qwen3.8-Flash-Next HyperConnection equations."""

import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.layers import (
    Qwen3_8_FlashNextGroupedRMSNorm,
    Qwen3_8_FlashNextHyperConnection,
)


def _reference_group_norm(x: torch.Tensor, weight: torch.Tensor, groups: int, eps: float) -> torch.Tensor:
    """Evaluate the branch-local Gemma RMSNorm reference equation.

    Args:
        x: Tensor of shape ``[..., groups * group_size]``.
        weight: Tensor of shape ``[groups * group_size]`` containing additive
            Gemma scale parameters.
        groups: Number of independently normalized branches.
        eps: Variance epsilon.

    Returns:
        Tensor of shape ``[..., groups * group_size]`` in the input dtype.
    """
    grouped = x.float().unflatten(-1, (groups, x.shape[-1] // groups))
    grouped = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    return (grouped.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


def test_grouped_rms_norm_is_branch_local() -> None:
    norm = Qwen3_8_FlashNextGroupedRMSNorm(hidden_size=12, group_size=4, eps=1e-6)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0]])
    norm.weight.detach().copy_(torch.linspace(-0.2, 0.2, 12))

    actual = norm(x)
    expected = _reference_group_norm(x, norm.weight, groups=3, eps=norm.eps)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hyperconnection_matches_reference_read_and_write() -> None:
    torch.manual_seed(7)
    layer = Qwen3_8_FlashNextHyperConnection(
        hidden_size=4,
        hc_count=3,
        lowrank_size=5,
        rms_norm_eps=1e-6,
        backend=BackendConfig(linear="torch"),
        dtype=torch.float32,
    )
    for parameter in layer.parameters():
        parameter.detach().uniform_(-0.25, 0.25)
    x = torch.randn(2, 3, 12)
    block_output = torch.randn(2, 3, 4)

    actual_mixed, residual = layer.mix(x)
    actual_combined = layer.combine(block_output, residual)

    normalized = _reference_group_norm(x, layer.hc_norm.weight, groups=3, eps=1e-6)
    read_gate = F.silu(F.linear(normalized, layer.input_mix_weight_down.weight) / 3)
    read_gate = torch.sigmoid(F.linear(read_gate, layer.input_mix_weight_up.weight)).unflatten(-1, (3, 4))
    expected_mixed = (read_gate * normalized.unflatten(-1, (3, 4))).mean(dim=-2)
    write_gate = 2.0 * torch.sigmoid(F.linear(normalized, layer.block_inject_weight.weight) / 3)
    expected_combined = (x.unflatten(-1, (3, 4)) + block_output.unsqueeze(-2) * write_gate.unsqueeze(-1)).flatten(-2)

    torch.testing.assert_close(actual_mixed, expected_mixed, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_combined, expected_combined, rtol=1e-6, atol=1e-6)


def test_final_hyperconnection_has_read_only_state_dict() -> None:
    layer = Qwen3_8_FlashNextHyperConnection(
        hidden_size=4,
        hc_count=2,
        lowrank_size=3,
        rms_norm_eps=1e-6,
        backend=BackendConfig(linear="torch"),
        use_combine=False,
        dtype=torch.float32,
    )

    assert "block_inject_weight.weight" not in layer.state_dict()
    mixed, _ = layer.mix(torch.randn(2, 8))
    assert mixed.shape == (2, 4)


def test_hyperconnection_preserves_fp32_residual_with_bf16_projections() -> None:
    torch.manual_seed(11)
    layer = Qwen3_8_FlashNextHyperConnection(
        hidden_size=4,
        hc_count=2,
        lowrank_size=3,
        rms_norm_eps=1e-6,
        backend=BackendConfig(linear="torch"),
        dtype=torch.bfloat16,
    )
    x = torch.randn(2, 3, 8, requires_grad=True)
    block_output = torch.randn(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)

    mixed, residual = layer.mix(x)
    combined = layer.combine(block_output, residual)

    normalized = _reference_group_norm(x, layer.hc_norm.weight, groups=2, eps=1e-6)
    projection_input = normalized.to(torch.bfloat16)
    read_gate = F.silu(F.linear(projection_input, layer.input_mix_weight_down.weight) / 2)
    read_gate = torch.sigmoid(F.linear(read_gate, layer.input_mix_weight_up.weight)).unflatten(-1, (2, 4))
    expected_mixed = (read_gate * normalized.unflatten(-1, (2, 4))).mean(dim=-2)
    write_gate = 2.0 * torch.sigmoid(F.linear(projection_input, layer.block_inject_weight.weight) / 2)
    expected_combined = (x.unflatten(-1, (2, 4)) + block_output.unsqueeze(-2) * write_gate.unsqueeze(-1)).flatten(-2)

    assert mixed.dtype == torch.float32
    assert combined.dtype == torch.float32
    torch.testing.assert_close(mixed, expected_mixed, rtol=0, atol=0)
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)

    (mixed.square().mean() + combined.square().mean()).backward()
    assert x.grad is not None and x.grad.dtype == torch.float32 and torch.isfinite(x.grad).all()
    assert block_output.grad is not None and block_output.grad.dtype == torch.bfloat16
    for parameter in layer.parameters():
        assert parameter.grad is not None and parameter.grad.dtype == parameter.dtype
        assert torch.isfinite(parameter.grad).all()
