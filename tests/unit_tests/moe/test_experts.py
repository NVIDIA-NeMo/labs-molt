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

import importlib.util
from unittest.mock import Mock, patch

import pytest
import torch

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_CUDA = torch.cuda.is_available()
SKIP_TE_TESTS = not (HAVE_TE and HAVE_CUDA)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _apply_bias,
    _DeterministicBiasRepeatInterleave,
    _permute_tokens_for_grouped_mm,
    _stabilize_empty_routing_probs_dtype,
    _torch_mm_experts_fwd,
    get_expert_activation_for_deepep,
    is_gated_activation,
    swiglu_clamped_deepep,
)

# Over the default 5s budget on purpose: this module launches a fresh interpreter, which re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.1,
        aux_loss_coeff=0.01,
        score_func="softmax",
        route_scale=1.0,
        dim=128,
        inter_dim=256,
        moe_inter_dim=256,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.bfloat16,
    )


class TestActivationFunctions:
    """Test activation functions used in MoE layers."""

    def test_get_expert_activation_for_deepep_swiglu(self, moe_config):
        """Test getting swiglu activation for DeepEP."""
        moe_config.expert_activation = "swiglu"

        with patch("nemo_automodel.components.moe.experts.weighted_bias_swiglu_impl") as mock_swiglu:
            activation_fn = get_expert_activation_for_deepep(moe_config)
            assert activation_fn == mock_swiglu

    def test_get_expert_activation_for_deepep_swiglu_default_uses_fused(self, moe_config):
        """``swiglu_limit == 0`` (default) keeps the fast fused ``weighted_bias_swiglu_impl`` path."""
        moe_config.expert_activation = "swiglu"
        moe_config.swiglu_limit = 0.0

        with patch("nemo_automodel.components.moe.experts.weighted_bias_swiglu_impl") as mock_swiglu:
            assert get_expert_activation_for_deepep(moe_config) is mock_swiglu

    def test_get_expert_activation_for_deepep_swiglu_with_limit_uses_clamped(self, moe_config):
        """``swiglu_limit > 0`` dispatches to the clamped FP32 variant for DSV4."""
        from functools import partial

        moe_config.expert_activation = "swiglu"
        moe_config.swiglu_limit = 7.0

        activation_fn = get_expert_activation_for_deepep(moe_config)

        # Should be a functools.partial wrapping swiglu_clamped_deepep with limit=7.0
        assert isinstance(activation_fn, partial)
        assert activation_fn.func is swiglu_clamped_deepep
        assert activation_fn.keywords == {"limit": 7.0}

    def test_get_expert_activation_for_deepep_swiglu_negative_limit_uses_fused(self, moe_config):
        """A non-positive ``swiglu_limit`` (e.g. 0.0 or negative) falls back to the fused path."""
        moe_config.expert_activation = "swiglu"
        moe_config.swiglu_limit = -1.0

        with patch("nemo_automodel.components.moe.experts.weighted_bias_swiglu_impl") as mock_swiglu:
            assert get_expert_activation_for_deepep(moe_config) is mock_swiglu


class TestSwigluClampedDeepEP:
    """Tests for the DSV4-style clamped FP32 SwiGLU activation."""

    def _eager_reference(self, x, permuted_probs, limit):
        """Reference implementation matching the DSV4 official Expert.forward."""
        gate, up = torch.chunk(x, 2, dim=-1)
        gate = gate.float().clamp(max=limit)
        up = up.float().clamp(min=-limit, max=limit)
        inter = torch.nn.functional.silu(gate) * up
        return (inter * permuted_probs).to(x.dtype)

    def test_output_shape_and_dtype(self):
        """Output should have shape ``[..., inter_dim]`` and the input's dtype."""
        torch.manual_seed(0)
        n_tokens = 4
        inter_dim = 8
        # x: [n_tokens, 2*inter_dim]
        x = torch.randn(n_tokens, 2 * inter_dim, dtype=torch.float32)
        probs = torch.rand(n_tokens, 1, dtype=torch.float32)

        out = swiglu_clamped_deepep(x, probs, limit=2.0)

        assert out.shape == (n_tokens, inter_dim)
        assert out.dtype == x.dtype

    @pytest.mark.parametrize("limit", [0.5, 2.0, 7.0])
    def test_matches_eager_reference_fp32(self, limit):
        """Compiled output must match an eager FP32 reference within tight tolerance."""
        torch.manual_seed(42)
        n_tokens = 8
        inter_dim = 16
        x = torch.randn(n_tokens, 2 * inter_dim, dtype=torch.float32) * 5.0  # exercise clamping
        probs = torch.rand(n_tokens, 1, dtype=torch.float32)

        try:
            out = swiglu_clamped_deepep(x, probs, limit=limit)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile path unavailable on this host: {exc}")

        ref = self._eager_reference(x, probs, limit)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    def test_clamping_caps_gate_above_limit(self):
        """When gate >> limit, silu(gate) saturates near gate≈limit (gate.clamp(max=limit))."""
        n_tokens = 2
        inter_dim = 4
        limit = 1.0
        # gate: very large positive => clamped at limit; up: modest (within range).
        gate = torch.full((n_tokens, inter_dim), 50.0, dtype=torch.float32)
        up = torch.full((n_tokens, inter_dim), 0.5, dtype=torch.float32)
        x = torch.cat([gate, up], dim=-1)
        probs = torch.ones(n_tokens, 1, dtype=torch.float32)

        try:
            out = swiglu_clamped_deepep(x, probs, limit=limit)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile path unavailable on this host: {exc}")

        ref = self._eager_reference(x, probs, limit)
        # silu(1.0) * 0.5 ≈ 0.7311 * 0.5 ≈ 0.3655
        expected = torch.full_like(out, torch.nn.functional.silu(torch.tensor(limit)).item() * 0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_clamping_caps_up_outside_range(self):
        """``up`` is clamped symmetrically to [-limit, limit]."""
        n_tokens = 2
        inter_dim = 4
        limit = 1.0
        gate = torch.zeros((n_tokens, inter_dim), dtype=torch.float32)  # silu(0) = 0
        up = torch.full((n_tokens, inter_dim), -100.0, dtype=torch.float32)  # clamps to -1.0
        x = torch.cat([gate, up], dim=-1)
        probs = torch.ones(n_tokens, 1, dtype=torch.float32)

        try:
            out = swiglu_clamped_deepep(x, probs, limit=limit)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile path unavailable on this host: {exc}")

        # silu(0) * (-1.0) = 0
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)

    def test_dtype_roundtrip_bf16(self):
        """Output dtype must equal input dtype (bf16 in, bf16 out)."""
        torch.manual_seed(0)
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        probs = torch.rand(2, 1, dtype=torch.bfloat16)

        try:
            out = swiglu_clamped_deepep(x, probs, limit=4.0)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile path unavailable on this host: {exc}")

        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 4)


class TestGroupedExpertsRouteWeightAfterDown:
    """Tests for Kimi-style expert output weighting."""

    def _tiny_config(self, *, apply_router_weight_after_down: bool = True, expert_activation: str = "swiglu"):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=2,
            inter_dim=4,
            moe_inter_dim=2,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=True,
            expert_activation=expert_activation,
            apply_router_weight_after_down=apply_router_weight_after_down,
            dtype=torch.float32,
        )

    def test_loop_matches_reference_weight_after_down_projection(self):
        config = self._tiny_config()
        experts = GroupedExperts(config)
        with torch.no_grad():
            experts.gate_and_up_projs.copy_(
                torch.tensor(
                    [
                        [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]],
                        [[0.25, -0.5, 1.0, 0.25], [0.75, 0.5, -0.25, 1.0]],
                    ]
                )
            )
            experts.down_projs.copy_(
                torch.tensor(
                    [
                        [[1.0, 0.5], [-0.25, 0.75]],
                        [[0.5, -1.0], [1.25, 0.25]],
                    ]
                )
            )
            experts.gate_up_proj_bias.copy_(torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.2, 0.1, -0.1, 0.4]]))
            experts.down_proj_bias.copy_(torch.tensor([[0.05, -0.1], [-0.2, 0.15]]))

        x = torch.tensor([[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25]])
        weights = torch.tensor([[0.2, 0.7], [0.4, 0.3], [0.9, 0.1]])
        indices = torch.tensor([[0, 1], [1, 0], [0, 1]])
        token_mask = torch.ones(x.shape[0], dtype=torch.bool)

        output = experts(x, token_mask, weights, indices)

        expected = torch.zeros_like(x)
        for token_idx in range(x.shape[0]):
            for top_idx in range(indices.shape[1]):
                expert_idx = indices[token_idx, top_idx]
                gate_up = x[token_idx : token_idx + 1] @ experts.gate_and_up_projs[expert_idx]
                gate_up = gate_up + experts.gate_up_proj_bias[expert_idx]
                gate, up = torch.chunk(gate_up, 2, dim=-1)
                expert_out = torch.nn.functional.silu(gate) * up
                expert_out = expert_out @ experts.down_projs[expert_idx]
                expert_out = expert_out + experts.down_proj_bias[expert_idx]
                expected[token_idx] += expert_out.squeeze(0) * weights[token_idx, top_idx]

        torch.testing.assert_close(output, expected)

    def test_default_down_bias_stays_weighted_by_route(self):
        config = self._tiny_config(apply_router_weight_after_down=False)
        experts = GroupedExperts(config)
        with torch.no_grad():
            experts.gate_and_up_projs.zero_()
            experts.down_projs.zero_()
            experts.gate_up_proj_bias.zero_()
            experts.down_proj_bias.copy_(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))

        x = torch.tensor([[0.5, -1.0], [1.5, 2.0]])
        weights = torch.tensor([[0.2, 0.7], [0.4, 0.3]])
        indices = torch.tensor([[0, 1], [1, 0]])
        token_mask = torch.ones(x.shape[0], dtype=torch.bool)

        output = experts(x, token_mask, weights, indices)

        expected = torch.stack(
            [
                experts.down_proj_bias[0] * weights[0, 0] + experts.down_proj_bias[1] * weights[0, 1],
                experts.down_proj_bias[1] * weights[1, 0] + experts.down_proj_bias[0] * weights[1, 1],
            ]
        )
        torch.testing.assert_close(output, expected)


class TestGroupedExpertsZeroActiveExperts:
    """Test GroupedExperts handling of zero active local experts.

    When using expert parallelism, it's possible for no tokens to be routed
    to the local experts on a particular rank. This test class verifies that
    the GroupedExperts module correctly handles this edge case by:
    1. Returning correct output shape (all zeros for the local contribution)
    2. Maintaining gradient flow through expert parameters
    """

    @pytest.fixture
    def initialized_experts(self, moe_config, device):
        """Create GroupedExperts with properly initialized weights."""
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights to avoid NaN issues
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
        return experts

    @pytest.fixture
    def initialized_experts_with_bias(self, moe_config, device):
        """Create GroupedExperts with bias and properly initialized weights."""
        moe_config.expert_bias = True
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights to avoid NaN issues
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
            experts.gate_up_proj_bias.zero_()
            experts.down_proj_bias.zero_()
        return experts

    def test_zero_active_experts_forward_shape(self, initialized_experts, moe_config, device):
        """Test forward pass returns correct shape when no tokens select any expert."""
        experts = initialized_experts

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)

        # Set indices to an expert ID that doesn't exist (out of range)
        # This simulates the case where all tokens select experts on other ranks
        # In EP scenario, experts_start_idx to experts_end_idx defines local experts
        # Setting indices outside this range means no local experts are selected
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,  # Non-existent expert
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert output.device == device
        # Check that output doesn't contain NaN
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_zero_active_experts_backward_no_error(self, moe_config, device):
        """Test backward pass completes without error when no tokens select any expert.

        When combined with other model outputs (like residual connections), the backward
        pass should complete without errors even when no local experts are active.
        """
        # Use float32 dtype for gradient computation
        moe_config.dtype = torch.float32
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

        # Set indices to non-existent expert (simulates all tokens routed elsewhere)
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # Verify forward pass produces correct output
        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

        # Simulate real training: MoE output combined with other model components
        # (e.g., residual connection). This ensures backward can run without error.
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        combined = output + residual
        loss = combined.sum()
        loss.backward()

        # Input should have gradients from the residual path
        assert x.grad is not None, "Input should have gradients from residual path"

    def test_zero_active_experts_with_bias_backward_no_error(self, moe_config, device):
        """Test backward pass completes without error with bias when no tokens select any expert.

        When combined with other model outputs (like residual connections), the backward
        pass should complete without errors even when no local experts are active.
        """
        # Use float32 dtype for gradient computation
        moe_config.dtype = torch.float32
        moe_config.expert_bias = True
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights and biases
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
            experts.gate_up_proj_bias.zero_()
            experts.down_proj_bias.zero_()

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

        # Set indices to non-existent expert
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # Verify forward pass produces correct output
        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

        # Simulate real training: MoE output combined with other model components
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        combined = output + residual
        loss = combined.sum()
        loss.backward()

        # Input should have gradients from the residual path
        assert x.grad is not None, "Input should have gradients from residual path"

    def test_zero_active_experts_partial_token_mask(self, initialized_experts, moe_config, device):
        """Test zero active experts case with partial token mask (some masked tokens)."""
        experts = initialized_experts

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        # Mask half the tokens
        token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        token_mask[: num_tokens // 2] = True
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)

        # Non-existent expert indices
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        # Check that output doesn't contain NaN
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_zero_active_experts_quick_geglu_activation(self, moe_config, device):
        """Test zero active experts case with quick_geglu activation function."""
        # Use float32 dtype for gradient computation
        moe_config.dtype = torch.float32
        moe_config.expert_activation = "quick_geglu"
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # Verify forward pass produces correct output
        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

        # Simulate real training: MoE output combined with other model components
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        combined = output + residual
        loss = combined.sum()
        loss.backward()

        # Input should have gradients from the residual path
        assert x.grad is not None, "Input should have gradients from residual path"

    def test_mixed_active_and_inactive_experts(self, initialized_experts, moe_config, device):
        """Test when some tokens select local experts and others don't."""
        experts = initialized_experts

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)

        # Half tokens go to valid experts, half to non-existent
        indices = torch.zeros((num_tokens, moe_config.n_activated_experts), dtype=torch.long, device=device)
        indices[: num_tokens // 2] = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens // 2, moe_config.n_activated_experts), device=device
        )
        indices[num_tokens // 2 :] = moe_config.n_routed_experts + 100  # Non-existent

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        # Check that output doesn't contain NaN
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_zero_active_experts_output_is_minimal(self, initialized_experts, moe_config, device):
        """Test that output contribution from zero-active-experts path is minimal.

        When no tokens select any expert, the dummy computation should contribute
        minimally to the output (the contribution is multiplied by weights which
        could be small, and uses zeros as input).
        """
        experts = initialized_experts

        num_tokens = 8
        # Use bfloat16 to match the initialized_experts dtype
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        # Use small weights to ensure minimal contribution
        weights = torch.full((num_tokens, moe_config.n_activated_experts), 0.01, dtype=torch.bfloat16, device=device)

        # Non-existent expert indices
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # The output should be very small since we're using zeros as input
        # and multiplying by small weights
        assert output.abs().max() < 1.0, "Output magnitude should be small for zero active experts"

    def test_zero_active_experts_grad_norm_no_hang(self, moe_config, device):
        """Test that computing gradient norm doesn't hang when no tokens select any expert.

        This test verifies that torch.nn.utils.clip_grad_norm_ completes without hanging,
        which is important for distributed training where all ranks must participate in
        gradient synchronization.
        """
        # Use float32 dtype for gradient computation
        moe_config.dtype = torch.float32
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

        # Set indices to non-existent expert (simulates all tokens routed elsewhere)
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # Simulate real training: MoE output combined with residual connection
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        combined = output + residual
        loss = combined.sum()
        loss.backward()

        # This is the critical test: clip_grad_norm_ should complete without hanging
        # In distributed training, if gradients don't exist, this could cause a hang
        grad_norm = torch.nn.utils.clip_grad_norm_(experts.parameters(), max_norm=1.0)

        # Verify grad_norm is a valid finite number (not NaN or Inf)
        assert torch.isfinite(grad_norm), f"Gradient norm should be finite, got {grad_norm}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_zero_active_experts_has_expert_gradients(self, moe_config, device):
        """Test that expert parameters have gradients when no tokens select any expert.

        Note: This test runs in a subprocess to avoid caching issues
        when run alongside other tests. The test code is in run_zero_active_experts_gradient_test.py.
        """
        import subprocess
        import sys

        # Run test as a module to avoid path resolution issues with torch.compile caching
        result = subprocess.run(
            [sys.executable, "-m", "tests.unit_tests.moe.run_zero_active_experts_gradient_test", str(device)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Subprocess test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert "SUCCESS" in result.stdout, (
            f"Test did not complete successfully:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestGroupedExperts:
    """Test GroupedExperts module."""

    def test_grouped_experts_init(self, moe_config):
        """Test GroupedExperts initialization."""
        experts = GroupedExperts(moe_config)

        assert experts.n_routed_experts == moe_config.n_routed_experts
        assert experts.expert_bias == moe_config.expert_bias
        expected_shape = (moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim * 2)
        assert experts.gate_and_up_projs.shape == expected_shape

        down_shape = (moe_config.n_routed_experts, moe_config.moe_inter_dim, moe_config.dim)
        assert experts.down_projs.shape == down_shape

    def test_grouped_experts_init_with_bias(self, moe_config):
        """Test GroupedExperts initialization with bias."""
        moe_config.expert_bias = True
        experts = GroupedExperts(moe_config)

        assert experts.gate_up_proj_bias is not None
        assert experts.down_proj_bias is not None
        assert experts.gate_up_proj_bias.shape == (moe_config.n_routed_experts, moe_config.moe_inter_dim * 2)
        assert experts.down_proj_bias.shape == (moe_config.n_routed_experts, moe_config.dim)

    def test_grouped_experts_forward_shape(self, moe_config, device):
        """Test GroupedExperts forward pass shape preservation."""
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert output.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_grouped_experts_gpu_execution(self, moe_config):
        """Test GroupedExperts execution on GPU."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        try:
            output = experts(x, token_mask, weights, indices)
            assert output.shape == x.shape
            assert output.device == device
            # Test passes if no exception is raised
        except Exception as e:
            pytest.fail(f"GPU execution failed: {e}")


class TestGroupedExpertsForwardLoopDTensorBias:
    """Test that _forward_loop correctly handles DTensor biases.

    When expert parallelism shards bias parameters as DTensors, the
    _forward_loop path must convert them to local tensors before arithmetic
    with the plain-tensor matmul outputs.  A missing conversion causes:
        RuntimeError: aten.add.Tensor got mixed torch.Tensor and DTensor
    """

    @staticmethod
    def _init_experts(moe_config, device):
        moe_config.expert_bias = True
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        with torch.no_grad():
            for p in experts.parameters():
                p.normal_(0, 0.02)
        return experts

    def test_forward_loop_with_bias_produces_correct_shape(self, moe_config, device):
        """Forward pass with expert_bias=True through _forward_loop should work."""
        experts = self._init_experts(moe_config, device)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)
        assert output.shape == x.shape

    def test_forward_loop_bias_affects_output(self, moe_config, device):
        """Verify that biases actually influence the output (not silently ignored)."""
        experts = self._init_experts(moe_config, device)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        # Output with zero biases
        with torch.no_grad():
            experts.gate_up_proj_bias.zero_()
            experts.down_proj_bias.zero_()
        output_zero_bias = experts(x, token_mask, weights, indices)

        # Output with non-zero biases
        with torch.no_grad():
            experts.gate_up_proj_bias.fill_(1.0)
            experts.down_proj_bias.fill_(1.0)
        output_nonzero_bias = experts(x, token_mask, weights, indices)

        assert not torch.allclose(output_zero_bias, output_nonzero_bias), "Bias should change the output"

    def test_forward_loop_dtensor_bias_converted_to_local(self, moe_config, device, monkeypatch):
        """Verify that isinstance(bias, DTensor) triggers .to_local() in forward.

        We monkeypatch the isinstance check in experts.py so that the plain
        bias tensors are treated as DTensors.  A .to_local() method is attached
        to confirm the conversion path is exercised.
        """
        import builtins

        from torch.distributed.tensor import DTensor

        experts = self._init_experts(moe_config, device)

        to_local_calls = []
        original_isinstance = builtins.isinstance

        def patched_isinstance(obj, classinfo):
            """Make bias parameters appear as DTensor instances."""
            if original_isinstance(classinfo, type) and classinfo is DTensor:
                if hasattr(obj, "_fake_dtensor"):
                    return True
            if original_isinstance(classinfo, tuple) and DTensor in classinfo:
                if hasattr(obj, "_fake_dtensor"):
                    return True
            return original_isinstance(obj, classinfo)

        def fake_to_local(self_tensor):
            to_local_calls.append(self_tensor)
            return self_tensor.data

        # Mark biases as fake DTensors and add .to_local()
        experts.gate_up_proj_bias._fake_dtensor = True
        experts.gate_up_proj_bias.to_local = lambda: fake_to_local(experts.gate_up_proj_bias)
        experts.down_proj_bias._fake_dtensor = True
        experts.down_proj_bias.to_local = lambda: fake_to_local(experts.down_proj_bias)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        monkeypatch.setattr(builtins, "isinstance", patched_isinstance)
        try:
            output = experts(x, token_mask, weights, indices)
        finally:
            monkeypatch.undo()

        assert output.shape == x.shape
        assert len(to_local_calls) >= 2, f"Expected .to_local() called for both biases, got {len(to_local_calls)} calls"


class TestGroupedExpertsDeepEP:
    """Test GroupedExpertsDeepEP module."""

    def test_grouped_experts_deepep_init(self, moe_config):
        """Test GroupedExpertsDeepEP initialization."""
        experts = GroupedExpertsDeepEP(moe_config)

        assert experts.config == moe_config
        assert experts.expert_bias == moe_config.expert_bias
        expected_shape = (moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim * 2)
        assert experts.gate_and_up_projs.shape == expected_shape

    def test_empty_routing_probs_match_activation_dtype(self):
        """Empty DeepEP metadata is stable across checkpoint recomputation."""
        empty_probs = torch.empty(0, dtype=torch.float32)
        nonempty_probs = torch.ones(1, dtype=torch.float32)

        stabilized = _stabilize_empty_routing_probs_dtype(empty_probs, torch.bfloat16)

        assert stabilized.shape == empty_probs.shape
        assert stabilized.dtype == torch.bfloat16
        assert _stabilize_empty_routing_probs_dtype(nonempty_probs, torch.bfloat16) is nonempty_probs

    def test_grouped_experts_deepep_token_dispatcher_init(self, moe_config):
        """Test token dispatcher initialization."""
        experts = GroupedExpertsDeepEP(moe_config)

        # Mock device mesh with proper integer returns
        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()

        # Patch the MoEFlexTokenDispatcher to avoid the TPxEP assertion and the
        # DeepEP buffer allocation, which requires the optional runtime.
        with (
            patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher,
            patch.object(experts, "_init_deepep_buffer") as mock_init_buffer,
        ):
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            assert hasattr(experts, "token_dispatcher")
            assert experts.ep_size == 2
            assert experts.ep_rank == 0
            mock_init_buffer.assert_called_once_with(mock_mesh.get_group.return_value)

    def test_grouped_experts_deepep_apply_bias_no_bias(self, moe_config):
        """Test _apply_bias method with no bias."""
        _ = GroupedExpertsDeepEP(moe_config)

        value = torch.randn(4, 8)
        tokens_per_expert = torch.tensor([2, 2])

        result = _apply_bias(value, bias=None, tokens_per_expert=tokens_per_expert)

        torch.testing.assert_close(result, value)

    def test_grouped_experts_deepep_apply_bias_with_bias(self, moe_config):
        """Test _apply_bias method with bias."""
        _ = GroupedExpertsDeepEP(moe_config)

        value = torch.randn(4, 8, requires_grad=True)
        bias = [torch.randn(8, requires_grad=True), torch.randn(8, requires_grad=True)]
        tokens_per_expert = torch.tensor([2, 2])

        result = _apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert)
        expected = torch.cat([value[:2] + bias[0], value[2:] + bias[1]])

        torch.testing.assert_close(result, expected)

        result.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))
        for expert_bias in bias:
            torch.testing.assert_close(expert_bias.grad, torch.full_like(expert_bias, 2))

    def test_grouped_experts_deepep_apply_bias_with_probs(self, moe_config):
        """Test _apply_bias method with permuted probabilities."""
        _ = GroupedExpertsDeepEP(moe_config)

        value = torch.randn(4, 8, requires_grad=True)
        bias = torch.randn(2, 8, requires_grad=True)
        tokens_per_expert = torch.tensor([2, 2])
        permuted_probs = torch.randn(4, 1, requires_grad=True)

        result = _apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert, permuted_probs=permuted_probs)
        expected_bias = torch.cat([bias[0].expand(2, -1), bias[1].expand(2, -1)])
        expected = value + expected_bias * permuted_probs

        torch.testing.assert_close(result, expected)

        result.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))
        expected_bias_grad = torch.stack(
            [
                permuted_probs[:2].detach().sum().expand_as(bias[0]),
                permuted_probs[2:].detach().sum().expand_as(bias[1]),
            ]
        )
        expected_probs_grad = expected_bias.detach().sum(dim=-1, keepdim=True)
        torch.testing.assert_close(bias.grad, expected_bias_grad)
        torch.testing.assert_close(permuted_probs.grad, expected_probs_grad)

    @pytest.mark.parametrize(
        ("bias_requires_grad", "use_probs", "expected_bias_grad_dtype", "expected_probs_grad_dtype"),
        [
            pytest.param(True, False, torch.bfloat16, None, id="trainable-unweighted"),
            pytest.param(True, True, torch.bfloat16, torch.float32, id="trainable-fp32-weighted"),
            pytest.param(False, True, None, torch.float32, id="frozen-fp32-weighted"),
        ],
    )
    def test_grouped_experts_deepep_apply_bias_dtypes(
        self,
        moe_config,
        bias_requires_grad,
        use_probs,
        expected_bias_grad_dtype,
        expected_probs_grad_dtype,
    ):
        """Trainable, promoted, and frozen bias paths preserve their public dtypes."""
        _ = GroupedExpertsDeepEP(moe_config)
        value = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=bias_requires_grad)
        tokens_per_expert = torch.tensor([1, 3])
        permuted_probs = torch.rand(4, 1, dtype=torch.float32, requires_grad=True) if use_probs else None

        result = _apply_bias(value, bias, tokens_per_expert, permuted_probs)
        result.backward(torch.randn_like(result))

        assert result.dtype == torch.bfloat16
        assert value.grad is not None
        assert value.grad.dtype == torch.bfloat16
        if expected_bias_grad_dtype is None:
            assert bias.grad is None
        else:
            assert bias.grad is not None
            assert bias.grad.dtype == expected_bias_grad_dtype
        if expected_probs_grad_dtype is not None:
            assert permuted_probs is not None
            assert permuted_probs.grad is not None
            assert permuted_probs.grad.dtype == expected_probs_grad_dtype

    def test_grouped_experts_deepep_apply_bias_with_empty_experts(self, moe_config):
        """Zero-token experts and higher-rank outputs preserve grouped order."""
        _ = GroupedExpertsDeepEP(moe_config)

        value = torch.randn(2, 2, 8)
        bias = torch.randn(4, 8)
        tokens_per_expert = torch.tensor([2, 0, 2, 0])

        result = _apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert)
        flat_value = value.view(-1, value.shape[-1])
        expected = torch.cat([flat_value[:2] + bias[0], flat_value[2:] + bias[2]]).view_as(value)

        torch.testing.assert_close(result, expected)

    def test_grouped_experts_deepep_apply_bias_cpu_fallback_matches_fp64_cancellation(self, moe_config):
        """The CPU fallback retains small residuals after large cancellation."""
        _ = GroupedExpertsDeepEP(moe_config)
        n_tokens = 4096
        hidden = 8
        value = torch.zeros(n_tokens, hidden, dtype=torch.float32)
        bias = torch.zeros(2, hidden, dtype=torch.float32, requires_grad=True)
        tokens_per_expert = torch.tensor([n_tokens, 0])
        permuted_probs = torch.ones(n_tokens, 1, dtype=torch.float32)
        permuted_probs[n_tokens // 2 :] = 0.9999
        upstream_grad = torch.ones(n_tokens, hidden, dtype=torch.float32)
        upstream_grad[n_tokens // 2 :] = -1

        _apply_bias(value, bias, tokens_per_expert, permuted_probs).backward(upstream_grad)

        expected_grad = torch.zeros_like(bias)
        expected_grad[0] = (upstream_grad * permuted_probs).double().sum(dim=0).float()
        assert bias.grad is not None
        torch.testing.assert_close(bias.grad, expected_grad, rtol=0, atol=0)

    def test_deterministic_bias_repeat_interleave_supports_higher_order_gradients(self):
        """The differentiable fallback preserves the custom function's double backward."""
        bias = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        token_counts = torch.tensor([0, 2, 1], dtype=torch.long)

        def expand(candidate_bias: torch.Tensor) -> torch.Tensor:
            """Expand an expert bias for numerical differentiation.

            Args:
                candidate_bias: Tensor of shape [experts, hidden].

            Returns:
                Tensor of shape [tokens, hidden].
            """
            return _DeterministicBiasRepeatInterleave.apply(candidate_bias, token_counts, 3)

        assert torch.autograd.gradcheck(expand, (bias,))
        assert torch.autograd.gradgradcheck(expand, (bias,))

    @pytest.mark.parametrize("use_probs", [False, True], ids=["unweighted", "fp32-weighted"])
    def test_grouped_experts_deepep_apply_bias_backward_is_fullgraph_compatible(self, moe_config, use_probs):
        """The deterministic bias backward remains traceable by AOTAutograd."""
        _ = GroupedExpertsDeepEP(moe_config)
        value = torch.randn(4, 8, requires_grad=True)
        bias = torch.randn(3, 8, requires_grad=True)
        tokens_per_expert = torch.tensor([0, 1, 3])
        permuted_probs = torch.rand(4, 1, dtype=torch.float32, requires_grad=True) if use_probs else None
        compiled_apply_bias = torch.compile(_apply_bias, backend="aot_eager", fullgraph=True)

        compiled_apply_bias(value, bias, tokens_per_expert, permuted_probs).square().sum().backward()

        assert value.grad is not None
        assert bias.grad is not None
        if permuted_probs is not None:
            assert permuted_probs.grad is not None

    def test_grouped_experts_deepep_init_with_hybridep_backend(self, moe_config):
        """Test GroupedExpertsDeepEP initialization with hybridep backend."""
        experts = GroupedExpertsDeepEP(
            moe_config,
            dispatcher_backend="hybridep",
            dispatcher_num_sms=24,
            dispatcher_share_token_dispatcher=False,
            dispatcher_async_dispatch=True,
        )

        assert experts.dispatcher_backend == "hybridep"
        assert experts.dispatcher_num_sms == 24
        assert experts.dispatcher_share_token_dispatcher is False
        assert experts.dispatcher_async_dispatch is True
        assert experts.config == moe_config

    def test_grouped_experts_deepep_token_dispatcher_init_hybridep(self, moe_config):
        """Test init_token_dispatcher passes hybridep config to TokenDispatcherConfig."""
        experts = GroupedExpertsDeepEP(
            moe_config,
            dispatcher_backend="hybridep",
            dispatcher_num_sms=24,
            dispatcher_share_token_dispatcher=False,
            dispatcher_async_dispatch=True,
        )

        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()

        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            # Verify the TokenDispatcherConfig was created with hybridep settings
            call_kwargs = mock_dispatcher.call_args
            config_arg = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            if config_arg is None:
                config_arg = call_kwargs[0][2]  # positional arg
            assert config_arg.moe_flex_dispatcher_backend == "hybridep"
            assert config_arg.moe_hybridep_num_sms == 24
            assert config_arg.moe_deepep_num_sms == 24
            assert config_arg.moe_share_token_dispatcher is False
            assert config_arg.moe_deepep_async_dispatch is True


class TestNonGatedActivations:
    """Test non-gated activation support (ReLU²) for memory-efficient MoE.

    Non-gated activations like ReLU² only need up_projs with shape [n_experts, dim, inter_dim]
    instead of gate_and_up_projs with shape [n_experts, dim, 2*inter_dim], saving 50% memory.
    """

    @pytest.fixture
    def relu2_config(self):
        """Create MoEConfig with ReLU² activation (non-gated)."""
        return MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def swiglu_config(self):
        """Create MoEConfig with SwiGLU activation (gated)."""
        return MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            dtype=torch.bfloat16,
        )

    def test_is_gated_activation_swiglu(self):
        """Test is_gated_activation returns True for swiglu."""
        assert is_gated_activation("swiglu") is True

    def test_is_gated_activation_quick_geglu(self):
        """Test is_gated_activation returns True for quick_geglu."""
        assert is_gated_activation("quick_geglu") is True

    def test_is_gated_activation_relu2(self):
        """Test is_gated_activation returns False for relu2."""
        assert is_gated_activation("relu2") is False

    def test_grouped_experts_relu2_uses_smaller_projections(self, relu2_config):
        """Test that GroupedExperts with ReLU² uses smaller gate_and_up_projs (inter_dim, not 2*inter_dim)."""
        experts = GroupedExperts(relu2_config)

        # Should have gate_and_up_projs with shape [n_experts, dim, inter_dim] (not 2*inter_dim)
        assert experts.gate_and_up_projs is not None
        assert experts.gate_and_up_projs.shape == (
            relu2_config.n_routed_experts,
            relu2_config.dim,
            relu2_config.moe_inter_dim,  # inter_dim, not 2*inter_dim
        )

        # Should have down_projs (same for both gated and non-gated)
        assert experts.down_projs is not None
        assert experts.down_projs.shape == (
            relu2_config.n_routed_experts,
            relu2_config.moe_inter_dim,
            relu2_config.dim,
        )

    def test_grouped_experts_swiglu_uses_gate_and_up_projs(self, swiglu_config):
        """Test that GroupedExperts with SwiGLU creates gate_and_up_projs with 2*inter_dim."""
        experts = GroupedExperts(swiglu_config)

        # Should have gate_and_up_projs with shape [n_experts, dim, 2*inter_dim]
        assert experts.gate_and_up_projs is not None
        assert experts.gate_and_up_projs.shape == (
            swiglu_config.n_routed_experts,
            swiglu_config.dim,
            swiglu_config.moe_inter_dim * 2,
        )

    def test_grouped_experts_relu2_with_bias(self, relu2_config):
        """Test GroupedExperts with ReLU² and bias uses smaller gate_up_proj_bias (inter_dim)."""
        relu2_config.expert_bias = True
        experts = GroupedExperts(relu2_config)

        # Should have gate_up_proj_bias with shape [n_experts, inter_dim] (not 2*inter_dim)
        assert experts.gate_up_proj_bias is not None
        assert experts.gate_up_proj_bias.shape == (
            relu2_config.n_routed_experts,
            relu2_config.moe_inter_dim,  # inter_dim, not 2*inter_dim
        )

        # Should have down_proj_bias
        assert experts.down_proj_bias is not None

    def test_grouped_experts_swiglu_with_bias(self, swiglu_config):
        """Test GroupedExperts with SwiGLU and bias uses gate_up_proj_bias with 2*inter_dim."""
        swiglu_config.expert_bias = True
        experts = GroupedExperts(swiglu_config)

        # Should have gate_up_proj_bias with shape [n_experts, 2*inter_dim]
        assert experts.gate_up_proj_bias is not None
        assert experts.gate_up_proj_bias.shape == (
            swiglu_config.n_routed_experts,
            swiglu_config.moe_inter_dim * 2,
        )

    def test_grouped_experts_relu2_forward(self, relu2_config, device):
        """Test GroupedExperts with ReLU² forward pass works correctly."""
        experts = GroupedExperts(relu2_config)
        experts = experts.to(device)

        # Initialize weights
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, relu2_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, relu2_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, relu2_config.n_routed_experts, (num_tokens, relu2_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert output.device == device
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_relu2_memory_efficiency(self, relu2_config, swiglu_config):
        """Test that ReLU² uses ~50% less memory for up projection weights than SwiGLU."""
        relu2_experts = GroupedExperts(relu2_config)
        swiglu_experts = GroupedExperts(swiglu_config)

        # Calculate parameter sizes
        relu2_up_params = relu2_experts.gate_and_up_projs.numel()
        swiglu_up_params = swiglu_experts.gate_and_up_projs.numel()

        # ReLU² should have exactly half the up projection parameters
        assert relu2_up_params * 2 == swiglu_up_params

    def test_grouped_experts_deepep_relu2_uses_smaller_projections(self, relu2_config):
        """Test that GroupedExpertsDeepEP with ReLU² uses smaller gate_and_up_projs."""
        experts = GroupedExpertsDeepEP(relu2_config)

        # Should have gate_and_up_projs with shape [n_experts, dim, inter_dim] (not 2*inter_dim)
        assert experts.gate_and_up_projs is not None
        assert experts.gate_and_up_projs.shape == (
            relu2_config.n_routed_experts,
            relu2_config.dim,
            relu2_config.moe_inter_dim,  # inter_dim, not 2*inter_dim
        )

    def test_grouped_experts_deepep_swiglu_uses_gate_and_up_projs(self, swiglu_config):
        """Test that GroupedExpertsDeepEP with SwiGLU creates gate_and_up_projs with 2*inter_dim."""
        experts = GroupedExpertsDeepEP(swiglu_config)

        # Should have gate_and_up_projs with shape [n_experts, dim, 2*inter_dim]
        assert experts.gate_and_up_projs is not None
        assert experts.gate_and_up_projs.shape == (
            swiglu_config.n_routed_experts,
            swiglu_config.dim,
            swiglu_config.moe_inter_dim * 2,
        )


def test_grouped_experts_te_rejects_router_weight_after_down_without_te_import(moe_config):
    """TE must fail loudly before importing optional TE when route weights would be applied incorrectly."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTE

    moe_config.apply_router_weight_after_down = True

    with pytest.raises(ValueError, match="GroupedExpertsTE"):
        GroupedExpertsTE(moe_config)


@pytest.mark.skipif(SKIP_TE_TESTS, reason="TransformerEngine and CUDA required")
class TestGroupedExpertsTE:
    """Test GroupedExpertsTE module using Transformer Engine's GroupedLinear."""

    @pytest.fixture
    def te_moe_config(self):
        """Create MoE config for TE tests."""
        return MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            activation_alpha=1.702,
            activation_limit=7.0,
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def te_moe_config_with_bias(self):
        """Create MoE config with bias for TE tests."""
        return MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=True,
            expert_activation="swiglu",
            activation_alpha=1.702,
            activation_limit=7.0,
            dtype=torch.bfloat16,
        )

    def _materialize_weights(self, experts, device):
        """Materialize meta device weights to actual device."""
        from transformer_engine.pytorch import GroupedLinear

        config = experts.config
        gate_up_out_features = config.moe_inter_dim * 2 if experts.is_gated else config.moe_inter_dim
        # Re-create on actual device
        experts.gate_up_linear = GroupedLinear(
            num_gemms=experts.num_local_experts,
            in_features=config.dim,
            out_features=gate_up_out_features,
            bias=experts.expert_bias,
            params_dtype=config.dtype,
            device=device,
        )
        experts.down_linear = GroupedLinear(
            num_gemms=experts.num_local_experts,
            in_features=config.moe_inter_dim,
            out_features=config.dim,
            bias=experts.expert_bias,
            params_dtype=config.dtype,
            device=device,
        )

    def test_grouped_experts_te_init(self, te_moe_config):
        """Test GroupedExpertsTE initialization."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        experts = GroupedExpertsTE(te_moe_config)

        assert experts.config == te_moe_config
        assert experts.expert_bias == te_moe_config.expert_bias
        assert experts.num_local_experts == te_moe_config.n_routed_experts
        assert experts.dim == te_moe_config.dim
        assert experts.moe_inter_dim == te_moe_config.moe_inter_dim
        assert experts.token_dispatcher is None
        assert experts.ep_mesh is None
        assert experts.ep_rank == 0

    def test_grouped_experts_te_init_with_bias(self, te_moe_config_with_bias):
        """Test GroupedExpertsTE initialization with bias."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        experts = GroupedExpertsTE(te_moe_config_with_bias)

        assert experts.expert_bias is True
        assert experts.gate_up_linear.use_bias is True
        assert experts.down_linear.use_bias is True

    def test_grouped_experts_te_weight_properties(self, te_moe_config):
        """Test weight property getters return correct shapes."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Test gate_and_up_projs shape: [n_experts, dim, moe_inter_dim * 2]
        gate_up = experts.gate_and_up_projs
        expected_shape = (te_moe_config.n_routed_experts, te_moe_config.dim, te_moe_config.moe_inter_dim * 2)
        assert gate_up.shape == expected_shape

        # Test down_projs shape: [n_experts, moe_inter_dim, dim]
        down = experts.down_projs
        expected_shape = (te_moe_config.n_routed_experts, te_moe_config.moe_inter_dim, te_moe_config.dim)
        assert down.shape == expected_shape

    def test_grouped_experts_te_weight_setters(self, te_moe_config):
        """Test weight property setters."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Create new weights
        new_gate_up = torch.randn(
            te_moe_config.n_routed_experts,
            te_moe_config.dim,
            te_moe_config.moe_inter_dim * 2,
            dtype=te_moe_config.dtype,
            device=device,
        )
        new_down = torch.randn(
            te_moe_config.n_routed_experts,
            te_moe_config.moe_inter_dim,
            te_moe_config.dim,
            dtype=te_moe_config.dtype,
            device=device,
        )

        # Set weights
        experts.gate_and_up_projs = new_gate_up
        experts.down_projs = new_down

        # Verify weights were set (check internal flag)
        assert hasattr(experts, "_weights_loaded_from_checkpoint")
        assert experts._weights_loaded_from_checkpoint is True

        # Verify shapes match
        assert experts.gate_and_up_projs.shape == new_gate_up.shape
        assert experts.down_projs.shape == new_down.shape

    def test_grouped_experts_te_bias_properties(self, te_moe_config_with_bias):
        """Test bias property getters and setters."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config_with_bias)
        self._materialize_weights(experts, device)

        # Test gate_up_proj_bias shape: [n_experts, moe_inter_dim * 2]
        gate_up_bias = experts.gate_up_proj_bias
        expected_shape = (te_moe_config_with_bias.n_routed_experts, te_moe_config_with_bias.moe_inter_dim * 2)
        assert gate_up_bias.shape == expected_shape

        # Test down_proj_bias shape: [n_experts, dim]
        down_bias = experts.down_proj_bias
        expected_shape = (te_moe_config_with_bias.n_routed_experts, te_moe_config_with_bias.dim)
        assert down_bias.shape == expected_shape

    def test_grouped_experts_te_bias_setter(self, te_moe_config_with_bias):
        """Test bias property setters."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config_with_bias)
        self._materialize_weights(experts, device)

        # Create new biases
        new_gate_up_bias = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.moe_inter_dim * 2,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )
        new_down_bias = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.dim,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )

        # Set biases
        experts.gate_up_proj_bias = new_gate_up_bias
        experts.down_proj_bias = new_down_bias

        # Verify shapes match
        assert experts.gate_up_proj_bias.shape == new_gate_up_bias.shape
        assert experts.down_proj_bias.shape == new_down_bias.shape

    def test_grouped_experts_te_no_bias_returns_none(self, te_moe_config):
        """Test bias properties return None when expert_bias is False."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        assert experts.gate_up_proj_bias is None
        assert experts.down_proj_bias is None

    def test_grouped_experts_te_state_dict(self, te_moe_config):
        """Test state_dict returns correct keys and shapes."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        state = experts.state_dict()

        # Check keys
        assert "gate_and_up_projs" in state
        assert "down_projs" in state

        # Check shapes
        expected_gate_up_shape = (te_moe_config.n_routed_experts, te_moe_config.dim, te_moe_config.moe_inter_dim * 2)
        expected_down_shape = (te_moe_config.n_routed_experts, te_moe_config.moe_inter_dim, te_moe_config.dim)

        assert state["gate_and_up_projs"].shape == expected_gate_up_shape
        assert state["down_projs"].shape == expected_down_shape

        # No bias keys since expert_bias is False
        assert "gate_up_proj_bias" not in state
        assert "down_proj_bias" not in state

    def test_grouped_experts_te_state_dict_with_bias(self, te_moe_config_with_bias):
        """Test state_dict includes bias when enabled."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config_with_bias)
        self._materialize_weights(experts, device)

        state = experts.state_dict()

        # Check bias keys exist
        assert "gate_up_proj_bias" in state
        assert "down_proj_bias" in state

        # Check bias shapes
        expected_gate_up_bias_shape = (
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.moe_inter_dim * 2,
        )
        expected_down_bias_shape = (te_moe_config_with_bias.n_routed_experts, te_moe_config_with_bias.dim)

        assert state["gate_up_proj_bias"].shape == expected_gate_up_bias_shape
        assert state["down_proj_bias"].shape == expected_down_bias_shape

    def test_grouped_experts_te_state_dict_with_prefix(self, te_moe_config):
        """Test state_dict with prefix."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        prefix = "layer.experts."
        state = experts.state_dict(prefix=prefix)

        assert f"{prefix}gate_and_up_projs" in state
        assert f"{prefix}down_projs" in state

    def test_grouped_experts_te_load_state_dict(self, te_moe_config):
        """Test _load_from_state_dict loads weights correctly."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Create a state dict with known values
        gate_up_weights = torch.randn(
            te_moe_config.n_routed_experts,
            te_moe_config.dim,
            te_moe_config.moe_inter_dim * 2,
            dtype=te_moe_config.dtype,
            device=device,
        )
        down_weights = torch.randn(
            te_moe_config.n_routed_experts,
            te_moe_config.moe_inter_dim,
            te_moe_config.dim,
            dtype=te_moe_config.dtype,
            device=device,
        )

        state_dict = {
            "gate_and_up_projs": gate_up_weights.clone(),
            "down_projs": down_weights.clone(),
        }

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        experts._load_from_state_dict(state_dict, "", None, True, missing_keys, unexpected_keys, error_msgs)

        assert len(missing_keys) == 0
        assert len(error_msgs) == 0

        # Verify weights were loaded
        loaded_gate_up = experts.gate_and_up_projs
        loaded_down = experts.down_projs

        torch.testing.assert_close(loaded_gate_up, gate_up_weights, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(loaded_down, down_weights, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_load_state_dict_with_bias(self, te_moe_config_with_bias):
        """Test _load_from_state_dict loads biases correctly."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config_with_bias)
        self._materialize_weights(experts, device)

        # Create state dict with known values
        gate_up_weights = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.dim,
            te_moe_config_with_bias.moe_inter_dim * 2,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )
        down_weights = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.moe_inter_dim,
            te_moe_config_with_bias.dim,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )
        gate_up_bias = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.moe_inter_dim * 2,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )
        down_bias = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.dim,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )

        state_dict = {
            "gate_and_up_projs": gate_up_weights.clone(),
            "down_projs": down_weights.clone(),
            "gate_up_proj_bias": gate_up_bias.clone(),
            "down_proj_bias": down_bias.clone(),
        }

        missing_keys = []
        experts._load_from_state_dict(state_dict, "", None, True, missing_keys, [], [])

        assert len(missing_keys) == 0

        # Verify biases were loaded
        torch.testing.assert_close(experts.gate_up_proj_bias, gate_up_bias, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(experts.down_proj_bias, down_bias, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_load_state_dict_missing_keys(self, te_moe_config):
        """Test _load_from_state_dict reports missing keys."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Empty state dict
        state_dict = {}
        missing_keys = []

        experts._load_from_state_dict(state_dict, "", None, True, missing_keys, [], [])

        assert "gate_and_up_projs" in missing_keys
        assert "down_projs" in missing_keys

    def test_grouped_experts_te_init_token_dispatcher(self, te_moe_config):
        """Test init_token_dispatcher initializes correctly."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        experts = GroupedExpertsTE(te_moe_config)

        # Mock device mesh
        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()
        mock_mesh.mesh_dim_names = ("ep",)

        # Patch MoEFlexTokenDispatcher
        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            assert experts.ep_mesh == mock_mesh
            assert experts.ep_size == 2
            assert experts.ep_rank == 0
            assert experts.num_local_experts == te_moe_config.n_routed_experts // 2
            assert experts.token_dispatcher is not None

    def test_grouped_experts_te_init_token_dispatcher_updates_linear_layers(self, te_moe_config):
        """Test init_token_dispatcher recreates linear layers with correct num_gemms."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        experts = GroupedExpertsTE(te_moe_config)

        # Initial num_gemms should be full expert count
        initial_gate_up_num_gemms = experts.gate_up_linear.num_gemms
        assert initial_gate_up_num_gemms == te_moe_config.n_routed_experts

        # Mock device mesh with ep_size=2
        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()
        mock_mesh.mesh_dim_names = ("ep",)

        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            # After init, num_gemms should be n_routed_experts / ep_size
            expected_local_experts = te_moe_config.n_routed_experts // 2
            assert experts.gate_up_linear.num_gemms == expected_local_experts
            assert experts.down_linear.num_gemms == expected_local_experts

    def test_grouped_experts_te_state_dict_roundtrip(self, te_moe_config):
        """Test state_dict -> load_state_dict roundtrip preserves weights."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # Create first instance and set specific weights
        experts1 = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts1, device)

        # Initialize weights with specific values
        with torch.no_grad():
            for i in range(experts1.gate_up_linear.num_gemms):
                getattr(experts1.gate_up_linear, f"weight{i}").normal_(0, 0.02)
                getattr(experts1.down_linear, f"weight{i}").normal_(0, 0.02)

        # Get state dict
        state = experts1.state_dict()

        # Create second instance and load state
        experts2 = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts2, device)

        missing_keys = []
        experts2._load_from_state_dict(state, "", None, True, missing_keys, [], [])

        assert len(missing_keys) == 0

        # Compare weights
        torch.testing.assert_close(experts1.gate_and_up_projs, experts2.gate_and_up_projs, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(experts1.down_projs, experts2.down_projs, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_weight_setter_with_none(self, te_moe_config):
        """Test weight setters handle None gracefully."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Get original weights
        original_gate_up = experts.gate_and_up_projs.clone()

        # Setting None should be a no-op
        experts.gate_and_up_projs = None
        experts.down_projs = None

        # Weights should be unchanged
        torch.testing.assert_close(experts.gate_and_up_projs, original_gate_up, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_init_weights(self, te_moe_config):
        """Test init_weights method."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Get weights before init
        old_gate_up = experts.gate_and_up_projs.clone()

        # Initialize weights
        experts.init_weights(device, init_std=0.02)

        # Weights should have changed
        new_gate_up = experts.gate_and_up_projs
        assert not torch.equal(old_gate_up, new_gate_up)

    def test_grouped_experts_te_relu2_init(self):
        """Test GroupedExpertsTE initialization with ReLU² (non-gated)."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)

        assert experts.is_gated is False
        # gate_up_linear out_features should be moe_inter_dim (not 2*moe_inter_dim)
        assert experts.gate_up_linear.out_features == config.moe_inter_dim

    def test_grouped_experts_te_relu2_weight_shapes(self):
        """Test GroupedExpertsTE weight shapes with ReLU² (non-gated)."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)
        self._materialize_weights(experts, device)

        # gate_and_up_projs: [n_experts, dim, moe_inter_dim] (not 2*moe_inter_dim)
        gate_up = experts.gate_and_up_projs
        assert gate_up.shape == (config.n_routed_experts, config.dim, config.moe_inter_dim)

        # down_projs: [n_experts, moe_inter_dim, dim] (same for gated and non-gated)
        down = experts.down_projs
        assert down.shape == (config.n_routed_experts, config.moe_inter_dim, config.dim)

    def test_grouped_experts_te_relu2_with_bias(self):
        """Test GroupedExpertsTE with ReLU² and bias uses smaller gate_up_proj_bias."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=True,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)
        self._materialize_weights(experts, device)

        # gate_up_proj_bias: [n_experts, moe_inter_dim] (not 2*moe_inter_dim)
        gate_up_bias = experts.gate_up_proj_bias
        assert gate_up_bias.shape == (config.n_routed_experts, config.moe_inter_dim)

        # down_proj_bias: [n_experts, dim]
        down_bias = experts.down_proj_bias
        assert down_bias.shape == (config.n_routed_experts, config.dim)

    def test_grouped_experts_te_relu2_state_dict(self):
        """Test state_dict with ReLU² returns correct shapes."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)
        self._materialize_weights(experts, device)

        state = experts.state_dict()

        assert "gate_and_up_projs" in state
        assert "down_projs" in state

        # ReLU² uses moe_inter_dim, not 2*moe_inter_dim
        assert state["gate_and_up_projs"].shape == (config.n_routed_experts, config.dim, config.moe_inter_dim)
        assert state["down_projs"].shape == (config.n_routed_experts, config.moe_inter_dim, config.dim)

    def test_grouped_experts_te_relu2_load_state_dict_roundtrip(self):
        """Test state_dict -> load_state_dict roundtrip with ReLU²."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

        # Create first instance with known weights
        experts1 = GroupedExpertsTE(config)
        self._materialize_weights(experts1, device)
        with torch.no_grad():
            for i in range(experts1.gate_up_linear.num_gemms):
                getattr(experts1.gate_up_linear, f"weight{i}").normal_(0, 0.02)
                getattr(experts1.down_linear, f"weight{i}").normal_(0, 0.02)

        state = experts1.state_dict()

        # Load into second instance
        experts2 = GroupedExpertsTE(config)
        self._materialize_weights(experts2, device)
        missing_keys = []
        experts2._load_from_state_dict(state, "", None, True, missing_keys, [], [])

        assert len(missing_keys) == 0
        torch.testing.assert_close(experts1.gate_and_up_projs, experts2.gate_and_up_projs, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(experts1.down_projs, experts2.down_projs, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_relu2_memory_efficiency(self):
        """Test that TE ReLU² uses ~50% less memory for up projection than SwiGLU."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        base_kwargs = dict(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            dtype=torch.bfloat16,
        )

        relu2_experts = GroupedExpertsTE(MoEConfig(**base_kwargs, expert_activation="relu2"))
        swiglu_experts = GroupedExpertsTE(MoEConfig(**base_kwargs, expert_activation="swiglu"))

        # ReLU² gate_up_linear out_features should be half of SwiGLU's
        assert relu2_experts.gate_up_linear.out_features * 2 == swiglu_experts.gate_up_linear.out_features

    def test_grouped_experts_te_relu2_init_token_dispatcher(self):
        """Test init_token_dispatcher with ReLU² creates correctly sized linear layers."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)

        # Before init_token_dispatcher, out_features should be moe_inter_dim
        assert experts.gate_up_linear.out_features == config.moe_inter_dim

        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()
        mock_mesh.mesh_dim_names = ("ep",)

        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()
            experts.init_token_dispatcher(mock_mesh)

            # After init_token_dispatcher, out_features should still be moe_inter_dim (not 2*)
            assert experts.gate_up_linear.out_features == config.moe_inter_dim
            assert experts.gate_up_linear.num_gemms == config.n_routed_experts // 2

    def test_grouped_experts_te_zero_routed_tokens_grads_all_local_experts(self, te_moe_config):
        """All local expert params must get (zero) grads when no tokens are routed locally.

        Grouped GEMM produces explicit zero gradients for every local expert.
        The zero-token fallback must do the same: a parameter left with
        ``grad=None`` skips momentum decay, decoupled weight decay, and its
        optimizer step, so its optimizer state silently diverges from ranks
        that did receive tokens.
        """
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        num_tokens = 4
        x = torch.randn(num_tokens, te_moe_config.dim, dtype=te_moe_config.dtype, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, te_moe_config.n_activated_experts, device=device)
        indices = torch.zeros(num_tokens, te_moe_config.n_activated_experts, dtype=torch.long, device=device)

        # Simulate an EP step where the dispatcher hands this rank zero tokens.
        empty_hidden = torch.zeros(0, te_moe_config.dim, dtype=te_moe_config.dtype, device=device)
        empty_probs = torch.zeros(0, dtype=te_moe_config.dtype, device=device)
        zero_splits = [0] * te_moe_config.n_routed_experts
        dispatcher = Mock()
        dispatcher.token_permutation2 = Mock(return_value=(empty_hidden, zero_splits, empty_probs))
        dispatcher.token_unpermutation = Mock(side_effect=lambda output: output)
        experts.token_dispatcher = dispatcher
        experts.ep_size = 1

        y = experts(x, token_mask, weights, indices)
        y.sum().backward()

        params = list(experts.gate_up_linear.parameters()) + list(experts.down_linear.parameters())
        assert len(params) == 2 * te_moe_config.n_routed_experts
        for param in params:
            assert param.grad is not None, "expert parameter received no gradient in the zero-token fallback"
            assert torch.count_nonzero(param.grad) == 0


class TestPermuteTokensForGroupedMM:
    """Test _permute_tokens_for_grouped_mm helper function."""

    def test_basic_permutation(self, device):
        """Test tokens are sorted by expert and outputs have correct shapes."""
        n_local_experts = 4
        num_tokens = 8
        topk = 2
        indices = torch.tensor(
            [[0, 1], [2, 3], [0, 2], [1, 3], [0, 1], [2, 3], [0, 2], [1, 3]],
            device=device,
        )
        weights = torch.rand(num_tokens, topk, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts, experts_start_idx=0
        )

        # All 16 slots should be assigned (8 tokens * topk=2, all local)
        assert sorted_ids.shape[0] == num_tokens * topk
        assert sorted_weights.shape == sorted_ids.shape
        assert tpe.shape == (n_local_experts,)
        assert offs.shape == (n_local_experts,)
        assert offs.dtype == torch.int32
        assert tpe.sum().item() == num_tokens * topk
        # offs is cumulative sum
        torch.testing.assert_close(offs, tpe.cumsum(0).to(torch.int32))

    def test_expert_offset(self, device):
        """Test that experts_start_idx correctly filters to local experts."""
        # 8 experts total, local experts are 4..7
        indices = torch.tensor([[0, 5], [3, 7], [4, 6]], device=device)
        weights = torch.ones(3, 2, device=device)
        token_mask = torch.ones(3, dtype=torch.bool, device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts=4, experts_start_idx=4
        )

        # Only experts 4,5,6,7 are local. Assignments: token0->5, token1->7, token2->4, token2->6
        assert tpe.sum().item() == 4
        assert tpe[0].item() == 1  # expert 4: token2
        assert tpe[1].item() == 1  # expert 5: token0
        assert tpe[2].item() == 1  # expert 6: token2
        assert tpe[3].item() == 1  # expert 7: token1

    def test_masked_tokens_excluded(self, device):
        """Test that masked tokens are excluded from permutation."""
        indices = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], device=device)
        weights = torch.ones(4, 2, device=device)
        token_mask = torch.tensor([True, True, False, False], device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts=2, experts_start_idx=0
        )

        # Only first 2 tokens are valid -> 4 assignments (2 tokens * topk=2)
        assert tpe.sum().item() == 4

    def test_no_local_tokens(self, device):
        """Test when no tokens route to local experts."""
        # Local experts 0..1, all indices go to experts 2,3
        indices = torch.tensor([[2, 3], [2, 3]], device=device)
        weights = torch.ones(2, 2, device=device)
        token_mask = torch.ones(2, dtype=torch.bool, device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts=2, experts_start_idx=0
        )

        assert tpe.sum().item() == 0
        assert sorted_ids.shape[0] == 0

    def test_weights_preserved(self, device):
        """Test that sorted weights correspond to the correct tokens."""
        indices = torch.tensor([[1, 0]], device=device)  # token 0 -> expert 1 (slot0), expert 0 (slot1)
        weights = torch.tensor([[0.7, 0.3]], device=device)
        token_mask = torch.ones(1, dtype=torch.bool, device=device)

        sorted_ids, sorted_slots, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices,
            weights,
            token_mask,
            n_local_experts=2,
            experts_start_idx=0,
            return_slot_ids=True,
        )

        # Sorted by expert: expert 0 first (weight 0.3), expert 1 second (weight 0.7)
        assert tpe[0].item() == 1  # expert 0
        assert tpe[1].item() == 1  # expert 1
        torch.testing.assert_close(sorted_slots, torch.tensor([1, 0], device=device))
        torch.testing.assert_close(sorted_weights[0], torch.tensor(0.3, device=device))
        torch.testing.assert_close(sorted_weights[1], torch.tensor(0.7, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
class TestGroupedMM:
    """Test GroupedExperts with torch._grouped_mm backend (use_torch_mm=True)."""

    @pytest.fixture
    def torch_mm_backend(self):
        return BackendConfig(experts="torch_mm", dispatcher="torch")

    @pytest.fixture
    def torch_mm_config(self):
        return MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            activation_alpha=1.702,
            activation_limit=7.0,
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def torch_mm_config_with_bias(self, torch_mm_config):
        torch_mm_config.expert_bias = True
        return torch_mm_config

    @staticmethod
    def _unwrap_compiled(fn):
        """Unwrap a torch.compile decorated function to its eager version."""
        from functools import partial

        if isinstance(fn, partial):
            inner = TestGroupedMM._unwrap_compiled(fn.func)
            if inner is not fn.func:
                return partial(inner, *fn.args, **fn.keywords)
            return fn
        if hasattr(fn, "_torchdynamo_orig_callable"):
            return fn._torchdynamo_orig_callable
        return fn

    def _init_experts(self, config, backend, device):
        """Create and initialize GroupedExperts on device."""
        experts = GroupedExperts(config, backend=backend).to(device)
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
            if experts.expert_bias:
                experts.gate_up_proj_bias.zero_()
                experts.down_proj_bias.zero_()
        # Use eager (non-compiled) activation functions to avoid recompilation issues in tests
        experts.expert_activation_grouped = self._unwrap_compiled(experts.expert_activation_grouped)
        return experts

    def test_init_sets_use_torch_mm(self, torch_mm_config, torch_mm_backend):
        """Test that torch_mm selects the native grouped operation."""
        experts = GroupedExperts(torch_mm_config, backend=torch_mm_backend)
        assert experts.use_torch_mm is True
        assert hasattr(experts, "expert_activation_grouped")

    def test_gmm_alias_selects_grouped_mm(self, torch_mm_config):
        """Test that the deprecated backend name remains a compatibility alias."""
        with pytest.warns(FutureWarning, match="experts='gmm' is deprecated"):
            backend = BackendConfig(experts="gmm", dispatcher="torch")
        assert GroupedExperts(torch_mm_config, backend=backend).use_torch_mm is True

    def test_init_without_backend_disables_grouped_mm(self, torch_mm_config):
        """Test that grouped MM is disabled without a backend."""
        experts = GroupedExperts(torch_mm_config)
        assert experts.use_torch_mm is False
        # expert_activation_grouped is always initialized (used by both loop and grouped_mm paths)
        assert hasattr(experts, "expert_activation_grouped")

    def test_forward_shape(self, torch_mm_config, torch_mm_backend, device):
        """Test grouped_mm forward produces correct output shape."""
        experts = self._init_experts(torch_mm_config, torch_mm_backend, device)

        num_tokens = 16
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, torch_mm_config.n_routed_experts, (num_tokens, torch_mm_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert output.device == device
        assert not torch.isnan(output).any()

    def test_forward_with_bias(self, torch_mm_config_with_bias, torch_mm_backend, device):
        """Test grouped_mm forward with expert bias."""
        experts = self._init_experts(torch_mm_config_with_bias, torch_mm_backend, device)

        num_tokens = 16
        x = torch.randn(num_tokens, torch_mm_config_with_bias.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(
            num_tokens, torch_mm_config_with_bias.n_activated_experts, dtype=torch.bfloat16, device=device
        )
        indices = torch.randint(
            0,
            torch_mm_config_with_bias.n_routed_experts,
            (num_tokens, torch_mm_config_with_bias.n_activated_experts),
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_forward_matches_loop_path(self, torch_mm_config, torch_mm_backend, device):
        """Test that torch_mm and loop paths produce similar outputs."""
        torch_mm_config.dtype = torch.float32

        experts_mm = self._init_experts(torch_mm_config, torch_mm_backend, device)
        experts_loop = GroupedExperts(torch_mm_config).to(device)
        experts_loop.expert_activation_grouped = self._unwrap_compiled(experts_loop.expert_activation_grouped)
        # Copy weights
        with torch.no_grad():
            experts_loop.gate_and_up_projs.copy_(experts_mm.gate_and_up_projs)
            experts_loop.down_projs.copy_(experts_mm.down_projs)

        num_tokens = 16
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(
            0, torch_mm_config.n_routed_experts, (num_tokens, torch_mm_config.n_activated_experts), device=device
        )

        out_mm = experts_mm(x, token_mask, weights, indices)
        out_loop = experts_loop(x, token_mask, weights, indices)

        torch.testing.assert_close(out_mm, out_loop, rtol=1e-3, atol=1e-3)

    def test_backward(self, torch_mm_config, torch_mm_backend, device):
        """Test backward pass completes and produces gradients."""
        torch_mm_config.dtype = torch.float32
        experts = self._init_experts(torch_mm_config, torch_mm_backend, device)

        num_tokens = 8
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(
            0, torch_mm_config.n_routed_experts, (num_tokens, torch_mm_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert experts.gate_and_up_projs.grad is not None
        assert experts.down_projs.grad is not None

    def test_zero_active_experts(self, torch_mm_config, torch_mm_backend, device):
        """Test grouped_mm path when no tokens route to any expert."""
        torch_mm_config.dtype = torch.float32
        experts = self._init_experts(torch_mm_config, torch_mm_backend, device)

        num_tokens = 8
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.float32, device=device)
        # Route all tokens to non-existent experts
        indices = torch.full(
            (num_tokens, torch_mm_config.n_activated_experts),
            fill_value=torch_mm_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        # Backward should still work (dummy computation for gradient flow)
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        (output + residual).sum().backward()
        assert x.grad is not None

    def test_partial_token_mask(self, torch_mm_config, torch_mm_backend, device):
        """Test grouped_mm with partially masked tokens."""
        experts = self._init_experts(torch_mm_config, torch_mm_backend, device)

        num_tokens = 16
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        token_mask[: num_tokens // 2] = True
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, torch_mm_config.n_routed_experts, (num_tokens, torch_mm_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_relu2_activation(self, torch_mm_backend, device):
        """Test grouped_mm with ReLU² (non-gated) activation."""
        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = self._init_experts(config, torch_mm_backend, device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_deepep_init_with_torch_mm(self, torch_mm_config, torch_mm_backend):
        """Test GroupedExpertsDeepEP initializes with torch_mm backend."""
        experts = GroupedExpertsDeepEP(torch_mm_config, backend=torch_mm_backend)
        assert experts.use_mxfp8 is False

    def test_deepep_init_without_backend_uses_native_gmm(self, torch_mm_config):
        """Test GroupedExpertsDeepEP defaults to native grouped MM."""
        experts = GroupedExpertsDeepEP(torch_mm_config)
        assert experts.use_mxfp8 is False


class TestTorchMMExpertsFwd:
    """Test _torch_mm_experts_fwd helper function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_basic_forward(self, device):
        """Test _torch_mm_experts_fwd produces correct shape output."""
        n_experts = 2
        dim = 32
        inter_dim = 64
        total_tokens = 6

        hidden = torch.randn(total_tokens, dim, dtype=torch.bfloat16, device=device)
        gate_up = torch.randn(n_experts, dim, inter_dim * 2, dtype=torch.bfloat16, device=device) * 0.02
        down = torch.randn(n_experts, inter_dim, dim, dtype=torch.bfloat16, device=device) * 0.02
        tpe = torch.tensor([3, 3], device=device)
        probs = torch.rand(total_tokens, 1, dtype=torch.float32, device=device)

        from nemo_automodel.components.moe.megatron.moe_utils import weighted_bias_swiglu_impl

        output = _torch_mm_experts_fwd(hidden, gate_up, down, tpe, probs, weighted_bias_swiglu_impl)

        assert output.shape == (total_tokens, dim)
        assert not torch.isnan(output).any()


class TestGroupedExpertsConvergenceFixes:
    """Test fixes for GroupedExperts convergence with expert parallelism.

    These tests verify:
    1. expert_activation_grouped is always initialized (needed for restructured loop path)
    2. Loop path uses WeightedSwiGLUFunction (matching DeepEP compute pattern)
    3. Float32 scatter_add accumulation with correct output dtype
    4. Backward gradient flow through both loop and grouped_mm paths
    """

    @pytest.fixture
    def config(self):
        return MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            activation_alpha=1.702,
            activation_limit=7.0,
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def torch_mm_backend(self):
        return BackendConfig(experts="torch_mm", dispatcher="torch")

    @staticmethod
    def _unwrap_compiled(fn):
        from functools import partial

        if isinstance(fn, partial):
            inner = TestGroupedExpertsConvergenceFixes._unwrap_compiled(fn.func)
            if inner is not fn.func:
                return partial(inner, *fn.args, **fn.keywords)
            return fn
        if hasattr(fn, "_torchdynamo_orig_callable"):
            return fn._torchdynamo_orig_callable
        return fn

    def _init_experts(self, config, backend=None, device=None):
        experts = GroupedExperts(config, backend=backend)
        if device:
            experts = experts.to(device)
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
            if experts.expert_bias:
                experts.gate_up_proj_bias.zero_()
                experts.down_proj_bias.zero_()
        experts.expert_activation_grouped = self._unwrap_compiled(experts.expert_activation_grouped)
        return experts

    # --- Test 1: expert_activation_grouped always initialized ---

    def test_expert_activation_grouped_always_present(self, config):
        """expert_activation_grouped must be available for both loop and grouped_mm paths."""
        experts_no_backend = GroupedExperts(config)
        assert hasattr(experts_no_backend, "expert_activation_grouped")
        assert callable(experts_no_backend.expert_activation_grouped)

        experts_with_backend = GroupedExperts(config, backend=BackendConfig(experts="torch_mm", dispatcher="torch"))
        assert hasattr(experts_with_backend, "expert_activation_grouped")
        assert callable(experts_with_backend.expert_activation_grouped)

    # --- Test 2: Output dtype matches input dtype (float32 accumulation cast back) ---

    def test_output_dtype_matches_input_bf16(self, config, device):
        """Output should be bf16 when input is bf16 (float32 accumulation cast back)."""
        experts = self._init_experts(config, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        assert output.dtype == torch.bfloat16

    def test_output_dtype_matches_input_fp32(self, config, device):
        """Output should be float32 when input is float32."""
        config.dtype = torch.float32
        experts = self._init_experts(config, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        assert output.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_output_dtype_grouped_mm_bf16(self, config, torch_mm_backend, device):
        """grouped_mm path output should be bf16 when input is bf16."""
        experts = self._init_experts(config, backend=torch_mm_backend, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        assert output.dtype == torch.bfloat16

    # --- Test 3: Loop path matches grouped_mm (restructured to use WeightedSwiGLUFunction) ---

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_loop_path_matches_grouped_mm_path(self, config, torch_mm_backend, device):
        """Loop path (restructured) should produce similar output to grouped_mm path."""
        config.dtype = torch.float32

        experts_mm = self._init_experts(config, backend=torch_mm_backend, device=device)
        experts_loop = self._init_experts(config, device=device)
        with torch.no_grad():
            experts_loop.gate_and_up_projs.copy_(experts_mm.gate_and_up_projs)
            experts_loop.down_projs.copy_(experts_mm.down_projs)

        num_tokens = 16
        x = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        out_mm = experts_mm(x, token_mask, weights, indices)
        out_loop = experts_loop(x, token_mask, weights, indices)

        torch.testing.assert_close(out_mm, out_loop, rtol=1e-3, atol=1e-3)

    # --- Test 4: Backward produces correct gradients ---

    def test_loop_path_backward_all_params_have_grad(self, config, device):
        """Loop path backward should produce gradients for input and all expert params."""
        config.dtype = torch.float32
        experts = self._init_experts(config, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        output.sum().backward()

        assert x.grad is not None, "Input x should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients should not be NaN"
        assert experts.gate_and_up_projs.grad is not None, "gate_and_up_projs should have gradients"
        assert experts.down_projs.grad is not None, "down_projs should have gradients"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_loop_and_grouped_mm_backward_gradients_match(self, config, torch_mm_backend, device):
        """Loop and grouped_mm paths should produce similar gradients."""
        config.dtype = torch.float32

        experts_mm = self._init_experts(config, backend=torch_mm_backend, device=device)
        experts_loop = self._init_experts(config, device=device)
        with torch.no_grad():
            experts_loop.gate_and_up_projs.copy_(experts_mm.gate_and_up_projs)
            experts_loop.down_projs.copy_(experts_mm.down_projs)

        num_tokens = 16
        x_mm = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device, requires_grad=True)
        x_loop = x_mm.detach().clone().requires_grad_(True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        experts_mm(x_mm, token_mask, weights, indices).sum().backward()
        experts_loop(x_loop, token_mask, weights, indices).sum().backward()

        torch.testing.assert_close(x_mm.grad, x_loop.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            experts_mm.gate_and_up_projs.grad, experts_loop.gate_and_up_projs.grad, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(experts_mm.down_projs.grad, experts_loop.down_projs.grad, rtol=1e-3, atol=1e-3)

    # --- Test 5: Loop path with bias ---

    def test_loop_path_with_bias_forward_and_backward(self, config, device):
        """Loop path should work correctly with expert bias."""
        config.dtype = torch.float32
        config.expert_bias = True
        experts = self._init_experts(config, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        output.sum().backward()
        assert x.grad is not None
        assert experts.gate_up_proj_bias.grad is not None, "gate_up_proj_bias should have gradients"
        assert experts.down_projs.grad is not None
