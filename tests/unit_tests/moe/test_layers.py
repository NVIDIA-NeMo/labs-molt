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
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    GroupedExpertsTE,
)
from nemo_automodel.components.moe.layers import (
    MLP,
    FakeBalancedGate,
    Gate,
    MoE,
)
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_CUDA = torch.cuda.is_available()
SKIP_TE_TESTS = not (HAVE_TE and HAVE_CUDA)


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


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class TestMLP:
    """Test MLP layer."""

    def test_mlp_init(self, device):
        """Test MLP initialization."""
        dim, inter_dim = 64, 128
        mlp = MLP(dim, inter_dim, backend="torch")

        assert mlp.gate_proj.in_features == dim
        assert mlp.gate_proj.out_features == inter_dim
        assert mlp.down_proj.in_features == inter_dim
        assert mlp.down_proj.out_features == dim
        assert mlp.up_proj.in_features == dim
        assert mlp.up_proj.out_features == inter_dim

    def test_mlp_forward_shape(self, device):
        """Test MLP forward pass shape preservation."""
        dim, inter_dim = 64, 128
        mlp = MLP(dim, inter_dim, backend="torch")
        mlp = mlp.to(device)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device=device)

        output = mlp(x)

        assert output.shape == (batch_size, seq_len, dim)
        assert output.device == device

    def test_mlp_forward_computation(self, device):
        """Test MLP forward computation correctness."""
        dim, inter_dim = 4, 8
        mlp = MLP(dim, inter_dim, backend="torch")
        mlp = mlp.to(device)

        x = torch.randn(1, 1, dim, dtype=torch.bfloat16, device=device)

        # Manual computation for verification
        gate_out = mlp.gate_proj(x)
        up_out = mlp.up_proj(x)
        expected = mlp.down_proj(F.silu(gate_out) * up_out)

        output = mlp(x)

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_mlp_init_weights(self, device):
        """Test MLP weight initialization."""
        mlp = MLP(64, 128, backend="torch")

        original_gate_weight = mlp.gate_proj.weight.clone().detach()

        with torch.no_grad():
            mlp.init_weights(device, init_std=0.02)

        # Weights should have changed
        assert not torch.equal(mlp.gate_proj.weight.detach(), original_gate_weight)


class TestFakeBalancedGate:
    """Test FakeBalancedGate for uniform expert routing."""

    def test_fake_balanced_gate_init(self, moe_config):
        """Test FakeBalancedGate initialization."""
        gate = FakeBalancedGate(moe_config)

        assert gate.n_routed_experts == moe_config.n_routed_experts
        assert gate.n_activated_experts == moe_config.n_activated_experts

    def test_fake_balanced_gate_forward_shape(self, moe_config, device):
        """Test FakeBalancedGate forward output shapes."""
        gate = FakeBalancedGate(moe_config)
        gate = gate.to(device)

        batch_size, seq_len = 4, 8
        x = torch.randn(batch_size * seq_len, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(batch_size * seq_len, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        expected_shape = (batch_size * seq_len, moe_config.n_activated_experts)
        assert weights.shape == expected_shape
        assert indices.shape == expected_shape
        assert aux_loss is None

    def test_fake_balanced_gate_uniform_weights(self, moe_config, device):
        """Test that FakeBalancedGate produces uniform weights."""
        gate = FakeBalancedGate(moe_config)
        gate = gate.to(device)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # All weights should be 1/n_activated_experts
        expected_weight = 1.0 / moe_config.n_activated_experts
        torch.testing.assert_close(weights, torch.full_like(weights, expected_weight))

    def test_fake_balanced_gate_cycling_indices(self, moe_config, device):
        """Test that FakeBalancedGate cycles through experts."""
        gate = FakeBalancedGate(moe_config)
        gate = gate.to(device)

        num_tokens = moe_config.n_routed_experts * 2  # Two full cycles
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Check that we cycle through experts
        flat_indices = indices.flatten()
        for i in range(moe_config.n_routed_experts):
            assert i in flat_indices

    def test_routing_with_skip_first_expert(self, moe_config, device):
        """Test routing when skipping the first expert."""
        skip_n = 1
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
        gate = gate.to(device)

        batch_size = 16
        x = torch.randn(batch_size, moe_config.dim, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Check indices skip the first expert (expert 0)
        assert indices.min() >= skip_n
        assert indices.max() < moe_config.n_routed_experts

        # Check that expert 0 is never selected
        assert (indices == 0).sum() == 0

    def test_routing_with_skip_multiple_experts(self, moe_config, device):
        """Test routing when skipping multiple experts."""
        skip_n = 3
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
        gate = gate.to(device)

        batch_size = 32
        x = torch.randn(batch_size, moe_config.dim, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Check indices skip the first 3 experts (experts 0, 1, 2)
        assert indices.min() >= skip_n
        assert indices.max() < moe_config.n_routed_experts

        # Check that experts 0, 1, 2 are never selected
        for i in range(skip_n):
            assert (indices == i).sum() == 0

    def test_load_balancing_with_skip(self, moe_config, device):
        """Test that load is balanced across available experts when skipping."""
        skip_n = 2
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
        gate = gate.to(device)

        # Use a large batch to ensure good distribution
        batch_size = 1000
        x = torch.randn(batch_size, moe_config.dim, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Count how many times each expert is selected
        available_experts = moe_config.n_routed_experts - skip_n
        expert_counts = torch.zeros(moe_config.n_routed_experts, dtype=torch.int64, device=device)
        for i in range(moe_config.n_routed_experts):
            expert_counts[i] = (indices == i).sum()

        # First skip_n experts should have 0 assignments
        assert expert_counts[:skip_n].sum() == 0

        # Remaining experts should have roughly equal assignments
        remaining_counts = expert_counts[skip_n:]
        expected_count = (batch_size * moe_config.n_activated_experts) // available_experts

        # Allow some tolerance for distribution
        assert torch.all(remaining_counts > 0), "All available experts should be used"
        assert torch.all(torch.abs(remaining_counts - expected_count) < expected_count * 0.2), (
            "Load should be roughly balanced"
        )

    def test_weights_are_uniform_with_skip(self, moe_config, device):
        """Test that weights are always uniform regardless of skip parameter."""
        for skip_n in [0, 1, 3]:
            gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
            gate = gate.to(device)

            batch_size = 8
            x = torch.randn(batch_size, moe_config.dim, device=device)
            token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

            weights, _, _ = gate(x, token_mask, cp_mesh=None)

            # All weights should be 1 / n_activated_experts
            expected_weight = 1.0 / moe_config.n_activated_experts
            assert torch.allclose(weights, torch.ones_like(weights) * expected_weight)

    def test_skip_almost_all_experts(self, moe_config, device):
        """Test edge case where we skip all but one expert."""
        skip_n = moe_config.n_routed_experts - 1
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
        gate = gate.to(device)

        batch_size = 8
        x = torch.randn(batch_size, moe_config.dim, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # All tokens should route to the last expert
        assert torch.all(indices == moe_config.n_routed_experts - 1)

    def test_output_dtype_matches_input(self, moe_config, device):
        """Test that output weights match input dtype."""
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=0)
        gate = gate.to(device)

        batch_size = 8

        # Test with float32
        x_fp32 = torch.randn(batch_size, moe_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        weights_fp32, _, _ = gate(x_fp32, token_mask, cp_mesh=None)
        assert weights_fp32.dtype == torch.float32

        # Test with float16
        x_fp16 = torch.randn(batch_size, moe_config.dim, dtype=torch.float16, device=device)
        weights_fp16, _, _ = gate(x_fp16, token_mask, cp_mesh=None)
        assert weights_fp16.dtype == torch.float16


class TestFakeBalancedGateNoise:
    """Test FakeBalancedGate with configurable noise."""

    def test_noise_zero_matches_balanced(self, moe_config, device):
        """Test that noise=0 gives identical results to the default balanced gate."""
        gate_default = FakeBalancedGate(moe_config)
        gate_zero = FakeBalancedGate(moe_config, noise=0.0)
        gate_default.to(device)
        gate_zero.to(device)

        x = torch.randn(16, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(16, dtype=torch.bool, device=device)

        w1, i1, _ = gate_default(x, mask, cp_mesh=None)
        w2, i2, _ = gate_zero(x, mask, cp_mesh=None)

        torch.testing.assert_close(w1, w2)
        assert torch.equal(i1, i2)

    def test_noise_one_random_indices(self, moe_config, device):
        """Test that noise=1.0 produces fully random indices (not perfectly balanced)."""
        gate = FakeBalancedGate(moe_config, noise=1.0)
        gate.to(device)

        # Large batch to ensure statistical detection of randomness
        n_tokens = 1000
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        weights, indices, _ = gate(x, mask, cp_mesh=None)

        # With noise=1.0, weights should NOT be uniform
        expected_uniform = 1.0 / moe_config.n_activated_experts
        assert not torch.allclose(weights, torch.full_like(weights, expected_uniform))

        # Weights should still sum to ~1 per token
        weight_sums = weights.sum(dim=-1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-3, rtol=1e-3)

    def test_noise_creates_load_imbalance(self, moe_config, device):
        """Test that noise creates measurable load imbalance across experts."""
        gate = FakeBalancedGate(moe_config, noise=0.5)
        gate.to(device)

        n_tokens = 2000
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        _, indices, _ = gate(x, mask, cp_mesh=None)

        # Count tokens per expert
        counts = torch.zeros(moe_config.n_routed_experts, device=device)
        for i in range(moe_config.n_routed_experts):
            counts[i] = (indices == i).sum()

        # With noise, std dev should be nonzero (not perfectly balanced)
        assert counts.float().std() > 0

    def test_noise_output_shapes(self, moe_config, device):
        """Test that noisy gate outputs have correct shapes."""
        gate = FakeBalancedGate(moe_config, noise=0.3)
        gate.to(device)

        n_tokens = 32
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, mask, cp_mesh=None)

        assert weights.shape == (n_tokens, moe_config.n_activated_experts)
        assert indices.shape == (n_tokens, moe_config.n_activated_experts)
        assert aux_loss is None

    def test_noise_indices_in_valid_range(self, moe_config, device):
        """Test that noisy indices stay within valid expert range."""
        gate = FakeBalancedGate(moe_config, noise=1.0)
        gate.to(device)

        n_tokens = 200
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        _, indices, _ = gate(x, mask, cp_mesh=None)

        assert indices.min() >= 0
        assert indices.max() < moe_config.n_routed_experts

    def test_noise_with_skip_experts(self, moe_config, device):
        """Test that noise respects skip_first_n_experts."""
        skip_n = 2
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n, noise=0.8)
        gate.to(device)

        n_tokens = 200
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        _, indices, _ = gate(x, mask, cp_mesh=None)

        # Skipped experts should never appear
        assert indices.min() >= skip_n
        assert indices.max() < moe_config.n_routed_experts

    def test_noise_indices_unique_per_token(self, moe_config, device):
        """Test that each token's expert indices are unique (no duplicates).

        Duplicate expert indices per token cause undefined behavior in the
        scatter-back step (y[token_ids] += ...) during the MoE forward pass.
        """
        gate = FakeBalancedGate(moe_config, noise=1.0)
        gate.to(device)

        n_tokens = 500
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        _, indices, _ = gate(x, mask, cp_mesh=None)

        # Each row should have unique expert indices
        for i in range(n_tokens):
            row = indices[i]
            assert row.unique().numel() == row.numel(), f"Token {i} has duplicate experts: {row}"

    def test_noise_weights_positive(self, moe_config, device):
        """Test that noisy weights are all positive."""
        gate = FakeBalancedGate(moe_config, noise=0.9)
        gate.to(device)

        n_tokens = 64
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        weights, _, _ = gate(x, mask, cp_mesh=None)

        assert (weights > 0).all()

    def test_noise_deterministic_for_same_input(self, moe_config, device):
        """Test that same input produces same routing (activation checkpointing safe).

        The seed is derived from the input content, so forward and recompute
        (which receive the same x) produce identical routing.
        """
        gate = FakeBalancedGate(moe_config, noise=0.7)
        gate.to(device)

        n_tokens = 64
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        w1, i1, _ = gate(x, mask, cp_mesh=None)
        w2, i2, _ = gate(x, mask, cp_mesh=None)

        torch.testing.assert_close(w1, w2)
        assert torch.equal(i1, i2)

    def test_noise_dynamic_for_different_input(self, moe_config, device):
        """Test that different inputs produce different routing.

        This mimics real training where each step has different hidden states,
        reproducing the dynamic tokens_per_expert pattern of real Gate.
        """
        gate = FakeBalancedGate(moe_config, noise=0.7)
        gate.to(device)

        n_tokens = 64
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        x1 = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        x2 = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)

        _, i1, _ = gate(x1, mask, cp_mesh=None)
        _, i2, _ = gate(x2, mask, cp_mesh=None)

        assert not torch.equal(i1, i2)

    def test_noise_stored_as_attribute(self, moe_config):
        """Test that noise is stored as an attribute."""
        gate = FakeBalancedGate(moe_config, noise=0.42)
        assert gate.noise == 0.42

        gate_default = FakeBalancedGate(moe_config)
        assert gate_default.noise == 0.0


class TestGate:
    """Test Gate (router) module."""

    def test_parameterless_routing_core_matches_eager_path(self, moe_config, device):
        gate = Gate(moe_config).to(device)
        gate.eval()
        torch.nn.init.normal_(gate.weight)
        gate.bias_update_factor = 0.0
        gate.aux_loss_coeff = 0.0
        inputs = torch.randn(4, moe_config.dim, device=device)
        token_mask = torch.ones(4, dtype=torch.bool, device=device)

        eager = gate(inputs, token_mask, None)
        gate.use_routing_core = True
        scoped = gate(inputs, token_mask, None)

        assert not tuple(gate.routing_core.parameters())
        for eager_value, scoped_value in zip(eager, scoped):
            if eager_value is None:
                assert scoped_value is None
            else:
                torch.testing.assert_close(eager_value, scoped_value)

    def test_gate_init_basic(self, moe_config):
        """Test Gate initialization with basic config."""
        gate = Gate(moe_config)

        assert gate.dim == moe_config.dim
        assert gate.n_experts == moe_config.n_routed_experts
        assert gate.topk == moe_config.n_activated_experts
        assert gate.weight.shape == (moe_config.n_routed_experts, moe_config.dim)
        assert gate.bias is None  # router_bias is False in fixture

    def test_gate_init_with_bias(self, moe_config):
        """Test Gate initialization with bias enabled."""
        moe_config.router_bias = True
        gate = Gate(moe_config)

        assert gate.bias is not None
        assert gate.bias.shape == (moe_config.n_routed_experts,)

    def test_gate_init_with_correction_bias(self, moe_config):
        """Test Gate initialization with bias update factor."""
        moe_config.gate_bias_update_factor = 0.1
        gate = Gate(moe_config)

        assert gate.e_score_correction_bias is not None
        assert gate.e_score_correction_bias.shape == (moe_config.n_routed_experts,)

    def test_gate_forward_softmax_mode(self, moe_config, device):
        """Test Gate forward pass in softmax mode."""
        moe_config.score_func = "softmax"
        gate = Gate(moe_config)
        gate = gate.to(device)

        # Initialize weights to avoid NaN issues
        with torch.no_grad():
            gate.weight.normal_(0, 0.02)
            if gate.bias is not None:
                gate.bias.zero_()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        # In softmax mode, weights should sum to 1 along last dim
        # Use detach() to avoid gradient warnings
        weights_detached = weights.detach()
        expected = torch.ones(num_tokens, dtype=torch.bfloat16, device=device)
        torch.testing.assert_close(weights_detached.sum(dim=-1), expected, rtol=1e-4, atol=1e-4)

    def test_gate_forward_sigmoid_mode(self, moe_config, device):
        """Test Gate forward pass in sigmoid mode."""
        moe_config.score_func = "sigmoid"
        gate = Gate(moe_config)
        gate = gate.to(device)

        # Initialize weights to avoid NaN issues
        with torch.no_grad():
            gate.weight.normal_(0, 0.02)
            if gate.bias is not None:
                gate.bias.zero_()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        # In sigmoid mode, all weights should be between 0 and 1
        weights_detached = weights.detach()
        assert (weights_detached >= 0).all() and (weights_detached <= 1).all()

    def test_gate_forward_softmax_with_bias_mode(self, moe_config, device):
        """Test Gate forward pass in softmax_with_bias mode (no groups)."""
        moe_config.score_func = "softmax_with_bias"
        moe_config.force_e_score_correction_bias = True
        gate = Gate(moe_config)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)
            if gate.bias is not None:
                gate.bias.zero_()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        # Weights should be gathered from unbiased softmax scores, so all >= 0
        weights_detached = weights.detach()
        assert (weights_detached >= 0).all()

    def test_gate_forward_softmax_with_bias_groups(self, moe_config, device):
        """Test Gate forward pass in softmax_with_bias mode with group routing."""
        moe_config.score_func = "softmax_with_bias"
        moe_config.n_routed_experts = 16
        moe_config.n_expert_groups = 4
        moe_config.n_limited_groups = 2
        moe_config.n_activated_experts = 4
        moe_config.force_e_score_correction_bias = True
        gate = Gate(moe_config)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        # All selected expert indices should be valid
        assert (indices >= 0).all() and (indices < moe_config.n_routed_experts).all()

    def test_gate_forward_softmax_with_bias_no_correction_bias(self, moe_config, device):
        """Test softmax_with_bias without e_score_correction_bias falls back to unbiased selection."""
        moe_config.score_func = "softmax_with_bias"
        moe_config.gate_bias_update_factor = 0
        moe_config.force_e_score_correction_bias = False
        gate = Gate(moe_config)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)

    def test_gate_forward_sigmoid_with_bias_groups_matches_reference(self, moe_config, device):
        """MiMo routing uses bias for selection, but final weights come from unbiased sigmoid scores."""
        moe_config.score_func = "sigmoid_with_bias"
        moe_config.n_routed_experts = 16
        moe_config.n_expert_groups = 4
        moe_config.n_limited_groups = 2
        moe_config.n_activated_experts = 4
        moe_config.norm_topk_prob = True
        moe_config.route_scale = 1.0
        moe_config.gate_bias_update_factor = 0
        moe_config.aux_loss_coeff = 0
        moe_config.force_e_score_correction_bias = True
        moe_config.dtype = torch.float32
        gate = Gate(moe_config, gate_precision=torch.float32).to(device)

        torch.manual_seed(1234)
        with torch.no_grad():
            gate.weight.normal_(0, 0.02)
            gate.e_score_correction_bias.copy_(torch.linspace(-0.75, 0.75, moe_config.n_routed_experts, device=device))

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, _ = gate(x, token_mask, cp_mesh=None)

        with torch.no_grad():
            scores = F.linear(x, gate.weight).sigmoid()
            scores_for_choice = scores + gate.e_score_correction_bias
            scores_for_choice = scores_for_choice.view(num_tokens, moe_config.n_expert_groups, -1)
            group_scores = scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=moe_config.n_limited_groups, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand_as(scores_for_choice).reshape(num_tokens, -1)
            scores_for_choice = scores_for_choice.reshape(num_tokens, -1).masked_fill(~score_mask.bool(), float("-inf"))
            expected_indices = torch.topk(scores_for_choice, k=moe_config.n_activated_experts, dim=-1, sorted=False)[1]
            expected_weights = scores.gather(1, expected_indices)
            expected_weights = expected_weights / (expected_weights.sum(dim=-1, keepdim=True) + 1e-20)

        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights.detach(), expected_weights, rtol=1e-5, atol=1e-5)

    def test_gate_forward_sqrtsoftplus_basic(self, moe_config, device):
        """``score_func='sqrtsoftplus'`` should compute weights as sqrt(softplus(logits))."""
        moe_config.score_func = "sqrtsoftplus"
        moe_config.norm_topk_prob = False
        moe_config.route_scale = 1.0
        # No correction bias path on this test
        moe_config.gate_bias_update_factor = 0
        moe_config.force_e_score_correction_bias = False
        gate = Gate(moe_config)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Shape checks
        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        # sqrt(softplus(...)) is non-negative
        weights_detached = weights.detach()
        assert (weights_detached >= 0).all()
        # Indices must be valid expert ids
        assert (indices >= 0).all() and (indices < moe_config.n_routed_experts).all()

    def test_gate_forward_sqrtsoftplus_matches_reference(self, moe_config, device):
        """sqrtsoftplus weights should match a manual reference: weights = sqrt(softplus(logits))[gather indices]."""
        moe_config.score_func = "sqrtsoftplus"
        moe_config.norm_topk_prob = False
        moe_config.route_scale = 1.0
        moe_config.gate_bias_update_factor = 0
        moe_config.force_e_score_correction_bias = False
        # Use float32 throughout for clean comparison
        moe_config.dtype = torch.float32
        gate = Gate(moe_config)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        torch.manual_seed(0)
        num_tokens = 12
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, _ = gate(x, token_mask, cp_mesh=None)

        # Reference: same transform on logits
        with torch.no_grad():
            logits = F.linear(x, gate.weight)
            ref_scores = torch.sqrt(F.softplus(logits.float())).to(logits.dtype)
            ref_weights = ref_scores.gather(1, indices)

        torch.testing.assert_close(weights.detach(), ref_weights, rtol=1e-5, atol=1e-5)

    def test_gate_forward_sqrtsoftplus_correction_bias_only_for_selection(self, moe_config, device):
        """``e_score_correction_bias`` shifts SELECTION but final weights come from UNBIASED scores."""
        moe_config.score_func = "sqrtsoftplus"
        moe_config.norm_topk_prob = False
        moe_config.route_scale = 1.0
        moe_config.gate_bias_update_factor = 0
        moe_config.force_e_score_correction_bias = True
        moe_config.dtype = torch.float32
        gate = Gate(moe_config)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)
            # Strong, non-uniform bias to ensure selection actually shifts
            gate.e_score_correction_bias.copy_(torch.linspace(-2.0, 2.0, moe_config.n_routed_experts, device=device))

        torch.manual_seed(7)
        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, _ = gate(x, token_mask, cp_mesh=None)

        # Reference: unbiased sqrt(softplus(logits)) gathered at the selected indices.
        with torch.no_grad():
            logits = F.linear(x, gate.weight)
            unbiased_scores = torch.sqrt(F.softplus(logits.float())).to(logits.dtype)
            ref_weights = unbiased_scores.gather(1, indices)

        torch.testing.assert_close(weights.detach(), ref_weights, rtol=1e-5, atol=1e-5)

        # Sanity: indices should match those produced by topk on (unbiased + bias)
        with torch.no_grad():
            biased_scores = unbiased_scores + gate.e_score_correction_bias
            ref_indices = torch.topk(biased_scores, moe_config.n_activated_experts, dim=-1)[1]
        # topk picks may differ in tie-breaking; compare the set of selected experts per row
        for r in range(num_tokens):
            assert set(indices[r].tolist()) == set(ref_indices[r].tolist())

    def test_gate_forward_sqrtsoftplus_with_norm_topk_prob(self, moe_config, device):
        """When ``norm_topk_prob=True`` and topk > 1, weights are renormalised after gather."""
        moe_config.score_func = "sqrtsoftplus"
        moe_config.norm_topk_prob = True
        moe_config.route_scale = 1.0
        moe_config.gate_bias_update_factor = 0
        moe_config.force_e_score_correction_bias = False
        moe_config.n_activated_experts = 2  # ensure topk > 1
        moe_config.dtype = torch.float32
        gate = Gate(moe_config)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        torch.manual_seed(0)
        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, _, _ = gate(x, token_mask, cp_mesh=None)

        # After norm_topk_prob, each row sums to (approximately) 1
        weights_detached = weights.detach()
        torch.testing.assert_close(
            weights_detached.sum(dim=-1),
            torch.ones(num_tokens, dtype=weights_detached.dtype, device=device),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_gate_forward_sqrtsoftplus_route_scale(self, moe_config, device):
        """``route_scale`` multiplies the final weights for the sqrtsoftplus branch too."""
        moe_config.score_func = "sqrtsoftplus"
        moe_config.norm_topk_prob = False
        moe_config.route_scale = 3.5
        moe_config.gate_bias_update_factor = 0
        moe_config.force_e_score_correction_bias = False
        moe_config.dtype = torch.float32
        gate = Gate(moe_config)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        torch.manual_seed(0)
        num_tokens = 6
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, _ = gate(x, token_mask, cp_mesh=None)

        with torch.no_grad():
            logits = F.linear(x, gate.weight)
            ref_scores = torch.sqrt(F.softplus(logits.float())).to(logits.dtype)
            ref_weights = ref_scores.gather(1, indices) * 3.5

        torch.testing.assert_close(weights.detach(), ref_weights, rtol=1e-5, atol=1e-5)

    def test_gate_forward_with_aux_loss(self, moe_config, device):
        """Test Gate forward pass with auxiliary loss computation."""
        moe_config.aux_loss_coeff = 0.01
        gate = Gate(moe_config)
        gate = gate.to(device)
        gate.train()  # Enable training mode for aux loss

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert aux_loss is not None
        assert aux_loss.numel() == 1  # Scalar loss
        assert aux_loss.requires_grad

    @pytest.mark.parametrize("gate_precision", [None, torch.float32])
    def test_gate_aux_loss_under_activation_checkpointing(self, moe_config, device, gate_precision):
        """Aux-loss path must be activation-checkpoint safe: saved tensors that AC
        observes during forward must match dtype on backward recompute. Regression
        test for the case where Gate cast `original_scores` back to bf16 right
        before `_compute_aux_loss`, producing bf16 saved tensors that recomputed as
        fp32 inside `torch.utils.checkpoint.NO_REENTRANT` and tripped
        `check_recomputed_tensors_match`.

        Parametrized over gate_precision so both Nemotron-3-Nano's forced fp32
        path and the default (None) path are covered.
        """
        moe_config.aux_loss_coeff = 0.01
        gate = Gate(moe_config, gate_precision=gate_precision)
        gate = gate.to(device)
        gate.train()

        num_tokens = 16
        x = torch.randn(
            num_tokens,
            moe_config.dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        def gate_fwd(x_):
            weights, _, aux_loss = gate(x_, token_mask, cp_mesh=None)
            # Inject the aux_loss gradient the same way the real model does, so
            # MoEAuxLossAutoScaler's saved tensor is exercised on backward.
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=x_.device)
            return weights.sum()

        # Wrap in bf16 autocast so any op that promotes to fp32 (e.g. softmax) on
        # the original forward but not on AC's recompute would surface a dtype
        # mismatch. Mirrors the FSDP2 MixedPrecisionPolicy(cast_forward_inputs=True)
        # pattern that the production crash was hitting on cluster.
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            loss = torch.utils.checkpoint.checkpoint(gate_fwd, x, use_reentrant=False)
        # Pre-fix this raised CheckpointError("Recomputed values for the
        # following tensors have different metadata") inside the unpack hook.
        loss.backward()
        assert x.grad is not None

    @pytest.mark.skipif(not HAVE_CUDA, reason="bf16-true reproduction needs CUDA for parity with cluster")
    @pytest.mark.parametrize("gate_precision", [None, torch.float32])
    def test_aux_loss_under_bf16_true_default_and_ac(self, moe_config, gate_precision):
        """Faithful reproduction of the production crash. Lightning's
        `bf16-true` precision toggles `torch.set_default_dtype(bf16)` for
        the duration of the forward only, so intermediate tensors created
        inside the gate (broadcasted scalars, accumulator zeros, etc.)
        default to bf16 on the original forward but fp32 on AC's recompute
        (which fires during backward, after default_dtype was reset).

        Pre-fix this raises `CheckpointError` on a `[n_experts]` tensor
        (per-expert expert_scores from `_compute_aux_loss`) and a scalar
        (the aux_loss saved by `MoEAuxLossAutoScaler.forward`) — the same
        structural pattern as the cluster failure on 128 experts. With
        Hunks 2 (`_compute_aux_loss` fp32 pin) and 3
        (`MoEAuxLossAutoScaler` save fp32) applied, both saved tensors are
        fp32 on both passes and the check passes.
        """
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        moe_config.aux_loss_coeff = 0.01
        # bf16-true: master params are bf16, no fp32 master copy.
        gate = Gate(moe_config, gate_precision=gate_precision).to(device).to(torch.bfloat16)
        gate.train()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        def gate_call(x_):
            weights, _, aux_loss = gate(x_, token_mask, cp_mesh=None)
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=x_.device)
            return weights.sum()

        # Mimic Lightning bf16-true: default_dtype=bf16 ONLY around the outer
        # forward call. AC's recompute will run during loss.backward() below,
        # at which point default_dtype is back to fp32.
        saved_default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            loss = torch.utils.checkpoint.checkpoint(gate_call, x, use_reentrant=False)
        finally:
            torch.set_default_dtype(saved_default)

        loss.backward()
        assert x.grad is not None

    def test_compute_aux_loss_returns_fp32(self, moe_config, device):
        """`_compute_aux_loss` must return fp32 regardless of input dtypes.
        This is the dtype contract that makes the path AC-safe: saved tensors
        inside the function are all fp32, so forward and recompute cannot
        diverge by dtype no matter what `original_scores.dtype` happens to be.
        """
        moe_config.aux_loss_coeff = 0.01
        gate = Gate(moe_config, gate_precision=None).to(device)
        gate.train()

        num_tokens = 16
        n_experts = moe_config.n_routed_experts
        original_scores_bf16 = torch.randn(
            num_tokens, n_experts, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        expert_load_int = torch.randint(0, 8, (n_experts,), dtype=torch.int64, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        loss = gate._compute_aux_loss(original_scores_bf16, expert_load_int, token_mask, cp_mesh=None)
        assert loss.dtype == torch.float32, (
            f"_compute_aux_loss must return fp32 to satisfy the AC saved/recomputed dtype contract; got {loss.dtype}."
        )

    def test_gate_update_bias(self, moe_config, device):
        """Test gate bias update mechanism."""
        moe_config.gate_bias_update_factor = 0.1
        gate = Gate(moe_config)
        gate = gate.to(device)
        gate.train()

        # Simulate some expert load
        expert_load = torch.rand(moe_config.n_routed_experts, dtype=torch.bfloat16, device=device) * 10
        gate._cumulative_expert_load = expert_load

        original_bias = gate.e_score_correction_bias.clone()

        gate.update_bias()

        # Bias should have been updated
        assert not torch.equal(gate.e_score_correction_bias, original_bias)
        # Cumulative load should be reset
        assert gate._cumulative_expert_load is None

    def test_gate_init_weights(self, moe_config, device):
        """Test Gate weight initialization."""
        gate = Gate(moe_config)

        original_weight = gate.weight.clone().detach()

        with torch.no_grad():
            gate.init_weights(device, init_std=0.02)

        # Weight should have changed
        assert not torch.equal(gate.weight.detach(), original_weight)

    def test_gate_init_with_precision(self, moe_config):
        """Test Gate initialization with gate_precision set."""
        gate = Gate(moe_config, gate_precision=torch.float32)
        assert gate.gate_precision == torch.float32

        gate = Gate(moe_config, gate_precision=torch.float64)
        assert gate.gate_precision == torch.float64

    def test_gate_init_default_precision(self, moe_config):
        """Test Gate initialization with default precision (None)."""
        gate = Gate(moe_config, gate_precision=None)
        assert gate.gate_precision is None

        gate = Gate(moe_config)
        assert gate.gate_precision is None

    def test_gate_forward_with_fp32_precision(self, moe_config, device):
        """Test Gate forward pass with fp32 precision."""
        moe_config.score_func = "softmax"
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_forward_with_fp64_precision(self, moe_config, device):
        """Test Gate forward pass with fp64 precision."""
        moe_config.score_func = "softmax"
        gate = Gate(moe_config, gate_precision=torch.float64)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_precision_output_dtype_matches_input(self, moe_config, device):
        """Test that output dtype matches input dtype regardless of gate_precision."""
        moe_config.score_func = "softmax"

        for input_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for gate_precision in [None, torch.float32, torch.float64]:
                gate = Gate(moe_config, gate_precision=gate_precision)
                gate = gate.to(device)

                with torch.no_grad():
                    gate.weight.normal_(0, 0.02)

                num_tokens = 8
                x = torch.randn(num_tokens, moe_config.dim, dtype=input_dtype, device=device)
                token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

                weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

                assert weights.dtype == input_dtype, (
                    f"Expected output dtype {input_dtype} but got {weights.dtype} with gate_precision={gate_precision}"
                )

    def test_gate_router_weights_fp32_keeps_precision_output(self, moe_config, device):
        """Test that router_weights_fp32 keeps Kimi-style routing weights in fp32."""
        moe_config.score_func = "softmax"
        moe_config.router_weights_fp32 = True
        gate = Gate(moe_config, gate_precision=torch.float32).to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        x = torch.randn(8, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)

        weights, _, _ = gate(x, token_mask, cp_mesh=None)

        assert weights.dtype == torch.float32

    def test_gate_router_weights_can_use_score_correction_bias(self, device):
        """Test Kimi's HF router behavior where biased scores are also gathered as weights."""
        config = MoEConfig(
            n_routed_experts=2,
            n_shared_experts=0,
            n_activated_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=2,
            inter_dim=4,
            moe_inter_dim=4,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=False,
            force_e_score_correction_bias=True,
            router_weights_fp32=True,
            router_weight_uses_score_correction_bias=True,
            dtype=torch.float32,
        )
        gate = Gate(config, gate_precision=torch.float32).to(device)
        with torch.no_grad():
            gate.weight.zero_()
            gate.e_score_correction_bias.copy_(torch.tensor([10.0, 0.0], device=device))

        x = torch.zeros(3, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)

        weights, indices, _ = gate(x, token_mask, cp_mesh=None)

        torch.testing.assert_close(indices, torch.zeros_like(indices))
        torch.testing.assert_close(weights, torch.full_like(weights, 10.5))

    def test_gate_precision_with_sigmoid(self, moe_config, device):
        """Test Gate precision with sigmoid score function."""
        moe_config.score_func = "sigmoid"
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16
        weights_detached = weights.detach()
        assert (weights_detached >= 0).all() and (weights_detached <= 1).all()

    def test_gate_precision_with_correction_bias(self, moe_config, device):
        """Test Gate precision with correction bias enabled."""
        moe_config.score_func = "sigmoid"
        moe_config.gate_bias_update_factor = 0.1
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_precision_with_norm_topk_prob(self, moe_config, device):
        """Test Gate precision with norm_topk_prob enabled."""
        moe_config.score_func = "softmax"
        moe_config.norm_topk_prob = True
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_precision_with_softmax_before_topk(self, moe_config, device):
        """Test Gate precision with softmax_before_topk enabled."""
        moe_config.score_func = "softmax"
        moe_config.softmax_before_topk = True
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_precision_consistency_across_calls(self, moe_config, device):
        """Test that Gate with precision produces consistent results across calls."""
        moe_config.score_func = "softmax"
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)
        gate.eval()

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        with torch.no_grad():
            weights1, indices1, _ = gate(x, token_mask, cp_mesh=None)
            weights2, indices2, _ = gate(x, token_mask, cp_mesh=None)

        torch.testing.assert_close(weights1, weights2)
        torch.testing.assert_close(indices1, indices2)

    def test_dtype_string_input(self):
        """Test that dtype field accepts string input and converts to torch.dtype."""
        config = MoEConfig(
            n_routed_experts=8,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=128,
            inter_dim=256,
            moe_inter_dim=256,
            norm_topk_prob=False,
            dtype="torch.float16",
        )

        assert config.dtype == torch.float16

    def test_gate_forward_with_string_precision_via_backend(self, device):
        """Test Gate forward pass with string precision input via BackendConfig."""
        config = MoEConfig(
            n_routed_experts=8,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=128,
            inter_dim=256,
            moe_inter_dim=256,
            norm_topk_prob=False,
            dtype="bfloat16",
        )

        backend_config = BackendConfig(gate_precision="float32")
        assert backend_config.gate_precision == torch.float32

        gate = Gate(config, gate_precision=backend_config.gate_precision)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, config.n_activated_experts)
        assert weights.dtype == torch.bfloat16
        assert gate.gate_precision == torch.float32


class TestMoE:
    """Test MoE (Mixture of Experts) module."""

    def test_moe_init_with_fake_balanced_gate(self, moe_config, backend_config):
        """Test MoE initialization with fake balanced gate."""
        backend_config.fake_balanced_gate = True
        moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, FakeBalancedGate)
        assert isinstance(moe.experts, GroupedExperts)

    def test_moe_init_fake_gate_noise_passed_through(self, moe_config, backend_config):
        """Test that fake_gate_noise from BackendConfig is passed to FakeBalancedGate."""
        backend_config.fake_balanced_gate = True
        backend_config.fake_gate_noise = 0.5
        moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, FakeBalancedGate)
        assert moe.gate.noise == 0.5

    def test_moe_init_with_deepep_single_device(self, moe_config, backend_config):
        """DeepEP dispatcher enabled but world size == 1 should fall back to GroupedExperts."""
        backend_config.experts = "te"
        backend_config.dispatcher = "deepep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=1):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExperts)

    @pytest.mark.skipif(SKIP_TE_TESTS, reason="TransformerEngine and CUDA required")
    def test_moe_init_with_deepep_multi_device(self, moe_config, backend_config):
        """DeepEP dispatcher enabled and world size > 1 should use GroupedExpertsTE."""
        backend_config.experts = "te"
        backend_config.dispatcher = "deepep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExpertsTE)

    def test_moe_init_with_gmm_experts_with_deepep(self, moe_config, backend_config):
        """GMM experts with deepep dispatcher should use GroupedExpertsDeepEP."""
        backend_config.experts = "gmm"
        backend_config.dispatcher = "deepep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExpertsDeepEP)

    def test_moe_init_with_hybridep_single_device(self, moe_config, backend_config):
        """HybridEP dispatcher enabled but world size == 1 should fall back to GroupedExperts."""
        backend_config.experts = "torch_mm"
        backend_config.dispatcher = "hybridep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=1):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExperts)

    def test_moe_init_with_hybridep_multi_device(self, moe_config, backend_config):
        """HybridEP dispatcher enabled and world size > 1 should use GroupedExpertsDeepEP."""
        backend_config.experts = "torch_mm"
        backend_config.dispatcher = "hybridep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExpertsDeepEP)
        assert moe.experts.dispatcher_backend == "hybridep"
        assert moe.experts.dispatcher_num_sms == backend_config.dispatcher_num_sms

    def test_moe_forwards_dispatcher_config_to_experts(self, moe_config, backend_config):
        """MoE should pass BackendConfig dispatcher knobs to flex dispatcher experts."""
        backend_config.experts = "torch_mm"
        backend_config.dispatcher = "deepep"
        backend_config.dispatcher_num_sms = 12
        backend_config.dispatcher_share_token_dispatcher = False
        backend_config.dispatcher_async_dispatch = True
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.experts, GroupedExpertsDeepEP)
        assert moe.experts.dispatcher_backend == "deepep"
        assert moe.experts.dispatcher_num_sms == 12
        assert moe.experts.dispatcher_share_token_dispatcher is False
        assert moe.experts.dispatcher_async_dispatch is True

    def test_moe_init_with_shared_experts(self, moe_config, backend_config):
        """Test MoE initialization with shared experts."""
        moe_config.n_shared_experts = 2
        moe = MoE(moe_config, backend_config)

        assert moe.shared_experts is not None
        assert isinstance(moe.shared_experts, MLP)

    def test_moe_init_without_shared_experts(self, moe_config, backend_config):
        """Test MoE initialization without shared experts."""
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)

        assert moe.shared_experts is None

    def test_moe_forward_without_shared_experts(self, moe_config, backend_config, device):
        """Test MoE forward pass without shared experts."""
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)
        moe = moe.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, device=device)

        with patch.object(moe.gate, "forward") as mock_gate, patch.object(moe.experts, "forward") as mock_experts:
            # Mock gate outputs
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )

            # Mock expert outputs
            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            output = moe(x)

            assert output.shape == x.shape
            assert output.device == device

    def test_moe_forward_with_shared_experts(self, moe_config, backend_config, device):
        """Test MoE forward pass with shared experts."""
        moe_config.n_shared_experts = 2
        moe = MoE(moe_config, backend_config)
        moe = moe.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, device=device)

        with (
            patch.object(moe.gate, "forward") as mock_gate,
            patch.object(moe.experts, "forward") as mock_experts,
            patch.object(moe.shared_experts, "forward") as mock_shared,
        ):
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )

            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)
            mock_shared.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            # Shared experts run inline on the main stream (no side-stream overlap).
            output = moe(x)

            assert output.shape == x.shape
            assert output.device == device

    def test_moe_forward_with_padding_mask(self, moe_config, backend_config, device):
        """Test MoE forward pass with padding mask."""
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)
        moe = moe.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, device=device)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        padding_mask[:, -2:] = True  # Mask last 2 tokens

        with patch.object(moe.gate, "forward") as mock_gate, patch.object(moe.experts, "forward") as mock_experts:
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )

            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            output = moe(x, padding_mask=padding_mask)

            assert output.shape == x.shape
            # Verify gate was called with correct token mask
            mock_gate.assert_called_once()
            gate_args = mock_gate.call_args[0]
            token_mask = gate_args[1]
            expected_mask = (~padding_mask).flatten()
            torch.testing.assert_close(token_mask.float(), expected_mask.float())

    def test_moe_forward_return_tuple_with_aux_loss(self, moe_config, backend_config, device):
        """Test MoE forward returns tuple when there's auxiliary loss."""
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)
        moe = moe.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, device=device)

        with patch.object(moe.gate, "forward") as mock_gate, patch.object(moe.experts, "forward") as mock_experts:
            aux_loss = torch.tensor(0.01, device=device)
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                aux_loss,
            )

            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            result = moe(x)

            # Should return the reshaped output since aux_loss handling is done in gate
            assert result.shape == x.shape


class TestMoEAuxLossAutoScaler:
    """Tests for MoEAuxLossAutoScaler gradient flow and scaling."""

    def setup_method(self):
        MoEAuxLossAutoScaler.main_loss_backward_scale = None

    def teardown_method(self):
        MoEAuxLossAutoScaler.main_loss_backward_scale = None

    def test_apply_returns_output_unchanged(self):
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.5, requires_grad=True)
        result = MoEAuxLossAutoScaler.apply(output, aux_loss)
        assert torch.equal(result.data, output.data)

    def test_apply_return_has_grad_fn(self):
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.5, requires_grad=True)
        result = MoEAuxLossAutoScaler.apply(output, aux_loss)
        assert result.grad_fn is not None

    def test_backward_scales_aux_loss_grad(self):
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.5, requires_grad=True)
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(10.0)

        result = MoEAuxLossAutoScaler.apply(output, aux_loss)
        result.sum().backward()

        assert aux_loss.grad is not None
        assert aux_loss.grad.item() == pytest.approx(10.0)

    def test_backward_default_scale_is_one(self):
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.5, requires_grad=True)

        result = MoEAuxLossAutoScaler.apply(output, aux_loss)
        result.sum().backward()

        assert aux_loss.grad.item() == pytest.approx(1.0)

    def test_backward_passes_grad_output_through(self):
        output = torch.randn(4, 8, requires_grad=True)
        aux_loss = torch.tensor(0.5, requires_grad=True)
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(5.0)

        result = MoEAuxLossAutoScaler.apply(output, aux_loss)
        loss = (result * 2).sum()
        loss.backward()

        assert output.grad is not None
        assert torch.allclose(output.grad, torch.full_like(output, 2.0))

    def test_apply_inside_activation_checkpoint_with_bf16_aux_loss(self):
        """The autoscaler must be activation-checkpoint safe even when callers
        pass a bf16 `aux_loss`: forward saves a tensor whose dtype must equal
        the recomputed dtype, and the only durable way to guarantee that is to
        pin the saved tensor to fp32. Regression test for Hunk 3."""
        output = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
        aux_loss_bf16 = torch.tensor(0.5, dtype=torch.bfloat16, requires_grad=True)

        def fwd(out_, aux_):
            return MoEAuxLossAutoScaler.apply(out_, aux_).sum()

        loss = torch.utils.checkpoint.checkpoint(fwd, output, aux_loss_bf16, use_reentrant=False)
        # Pre-fix this raised CheckpointError if any inner op produced an fp32
        # tensor on recompute that was saved as bf16 on forward.
        loss.backward()
        assert output.grad is not None
        assert aux_loss_bf16.grad is not None


class TestGateAuxLossGradientFlow:
    """Tests that Gate.forward() correctly wires aux loss into the autograd graph."""

    def setup_method(self):
        MoEAuxLossAutoScaler.main_loss_backward_scale = None

    def teardown_method(self):
        MoEAuxLossAutoScaler.main_loss_backward_scale = None

    def test_gate_weights_carry_aux_loss_grad_fn(self, moe_config, device):
        moe_config.aux_loss_coeff = 0.01
        moe_config.gate_bias_update_factor = 0.0
        gate = Gate(moe_config).to(device)
        gate.train()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.grad_fn is not None, "weights must have a grad_fn from MoEAuxLossAutoScaler.apply()"

    def test_aux_loss_receives_gradient_through_weights(self, moe_config, device):
        moe_config.aux_loss_coeff = 0.01
        moe_config.gate_bias_update_factor = 0.0
        gate = Gate(moe_config).to(device)
        gate.train()
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Backward through the weights (simulating main loss path)
        weights.sum().backward()

        # Gate router weight should have gradients from aux loss
        assert gate.weight.grad is not None

    def test_aux_loss_coeff_scales_aux_loss_input(self, moe_config, device):
        moe_config.aux_loss_coeff = 0.05
        moe_config.gate_bias_update_factor = 0.0
        gate = Gate(moe_config).to(device)
        with torch.no_grad():
            gate.init_weights(device, init_std=0.02)
        gate.train()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        captured = {}
        original_apply = MoEAuxLossAutoScaler.apply

        def spy_apply(output, aux_loss):
            captured["scaled_aux_loss"] = aux_loss.detach().clone()
            return original_apply(output, aux_loss)

        with patch.object(MoEAuxLossAutoScaler, "apply", side_effect=spy_apply):
            weights, indices, raw_aux_loss = gate(x, token_mask, cp_mesh=None)

        expected = moe_config.aux_loss_coeff * raw_aux_loss.detach()
        assert captured["scaled_aux_loss"].item() == pytest.approx(expected.item(), rel=1e-2)

    def test_no_aux_loss_when_coeff_zero(self, moe_config, device):
        moe_config.aux_loss_coeff = 0.0
        moe_config.gate_bias_update_factor = 0.0
        gate = Gate(moe_config).to(device)
        gate.train()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert aux_loss is None


class TestAuxLossSoftmaxFix:
    """Test that aux_loss uses proper probabilities for softmax routing without softmax_before_topk.

    GPT-OSS uses softmax scoring with topk-first (softmax_before_topk=False).
    Previously, original_scores passed raw logits to _compute_aux_loss, causing
    P_i to be negative and aux_loss to diverge negative during training.
    The fix applies softmax to original_scores so P_i represents proper probabilities.
    """

    def _make_gate(self, device, score_func="softmax", softmax_before_topk=False, aux_loss_coeff=0.01):
        config = MoEConfig(
            n_routed_experts=8,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0,
            aux_loss_coeff=aux_loss_coeff,
            score_func=score_func,
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=False,
            router_bias=True,
            expert_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=softmax_before_topk,
            dtype=torch.float32,
        )
        gate = Gate(config).to(device)
        gate.train()
        gate.gate_precision = torch.float32
        # Deterministic init: use known weights to avoid CUDA-state-dependent NaN.
        # Small uniform weights ensure softmax/sigmoid produce well-conditioned scores.
        with torch.no_grad():
            gate.weight.uniform_(-0.1, 0.1)
            if gate.bias is not None:
                gate.bias.zero_()
        return gate

    def test_softmax_topk_first_aux_loss_is_positive(self, device):
        """aux_loss must be non-negative when using softmax with topk-first (GPT-OSS style)."""
        gate = self._make_gate(device, score_func="softmax", softmax_before_topk=False)
        x = torch.randn(256, 64, device=device)
        token_mask = torch.ones(256, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert aux_loss is not None
        assert not torch.isnan(aux_loss), "aux_loss is NaN"
        assert aux_loss.item() >= 0, f"aux_loss should be non-negative, got {aux_loss.item()}"

    def test_softmax_before_topk_aux_loss_is_positive(self, device):
        """aux_loss must be non-negative when using softmax_before_topk (Qwen3-MoE style)."""
        gate = self._make_gate(device, score_func="softmax", softmax_before_topk=True)
        x = torch.randn(256, 64, device=device)
        token_mask = torch.ones(256, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert aux_loss is not None
        assert not torch.isnan(aux_loss), "aux_loss is NaN"
        assert aux_loss.item() >= 0, f"aux_loss should be non-negative, got {aux_loss.item()}"

    def test_sigmoid_aux_loss_is_positive(self, device):
        """aux_loss must be non-negative when using sigmoid scoring (Moonlight/DeepSeek style).

        Sigmoid scores are always in [0, 1], so P_i is always non-negative.
        This test verifies the sigmoid path was not broken by the softmax fix.
        """
        gate = self._make_gate(device, score_func="sigmoid", softmax_before_topk=False)
        # Use 512 tokens with topk=2 and 8 experts to ensure good expert coverage
        x = torch.randn(512, 64, device=device)
        token_mask = torch.ones(512, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert aux_loss is not None
        assert torch.isfinite(aux_loss), f"aux_loss is not finite: {aux_loss.item()}"
        assert aux_loss.item() >= 0, f"aux_loss should be non-negative, got {aux_loss.item()}"

    def test_softmax_topk_first_aux_loss_stays_positive_after_gradient_steps(self, device):
        """aux_loss should remain non-negative even after multiple gradient updates.

        This is the regression test for the GPT-OSS divergence bug where raw logits
        caused aux_loss to go increasingly negative during training.
        """
        gate = self._make_gate(device, score_func="softmax", softmax_before_topk=False, aux_loss_coeff=0.1)
        optimizer = torch.optim.SGD(gate.parameters(), lr=0.01)

        for step in range(20):
            x = torch.randn(64, 64, device=device)
            token_mask = torch.ones(64, dtype=torch.bool, device=device)

            weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

            assert aux_loss is not None
            assert not torch.isnan(aux_loss), f"aux_loss went to NaN at step {step}"
            assert aux_loss.item() >= 0, f"aux_loss went negative at step {step}: {aux_loss.item()}"

            # Simulate training: backward through aux_loss via the MoEAuxLossAutoScaler
            loss = weights.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class TestApplyBiasNotCompiled:
    """Test that _apply_bias works correctly without torch.compile."""

    def test_apply_bias_basic(self, device):
        """_apply_bias should add per-expert bias to grouped GEMM output."""
        from nemo_automodel.components.moe.experts import _apply_bias

        n_experts = 4
        hidden_dim = 8
        tokens_per_expert = torch.tensor([3, 2, 4, 1], device=device)
        total_tokens = tokens_per_expert.sum().item()

        value = torch.randn(total_tokens, hidden_dim, device=device)
        bias = torch.ones(n_experts, hidden_dim, device=device)

        result = _apply_bias(value, bias, tokens_per_expert)

        # Each token should have bias[expert_idx] added
        assert result.shape == value.shape
        offset = 0
        for expert_idx, count in enumerate(tokens_per_expert):
            expected = value[offset : offset + count] + bias[expert_idx]
            torch.testing.assert_close(result[offset : offset + count], expected)
            offset += count

    def test_apply_bias_with_probs(self, device):
        """_apply_bias with permuted_probs should weight bias by routing probabilities."""
        from nemo_automodel.components.moe.experts import _apply_bias

        n_experts = 2
        hidden_dim = 4
        tokens_per_expert = torch.tensor([2, 3], device=device)
        total_tokens = 5

        value = torch.zeros(total_tokens, hidden_dim, device=device)
        bias = torch.ones(n_experts, hidden_dim, device=device) * 2.0
        probs = torch.tensor([0.5, 0.5, 1.0, 1.0, 1.0], device=device).unsqueeze(-1)

        result = _apply_bias(value, bias, tokens_per_expert, permuted_probs=probs)

        # Expert 0 tokens: bias * prob = 2.0 * 0.5 = 1.0
        torch.testing.assert_close(result[0], torch.full((hidden_dim,), 1.0, device=device))
        torch.testing.assert_close(result[1], torch.full((hidden_dim,), 1.0, device=device))
        # Expert 1 tokens: bias * prob = 2.0 * 1.0 = 2.0
        torch.testing.assert_close(result[2], torch.full((hidden_dim,), 2.0, device=device))

    def test_apply_bias_is_not_compiled(self):
        """_apply_bias should not be wrapped with torch.compile."""
        from nemo_automodel.components.moe.experts import _apply_bias

        # torch.compile wraps functions in OptimizedModule or similar
        assert not hasattr(_apply_bias, "_torchdynamo_orig_callable"), "_apply_bias should not be torch.compiled"


class TestSigmoidGateScoringPrecision:
    """Sigmoid routing must score in fp32 by default, like the softmax path.

    HF sigmoid-router references compute ``sigmoid(logits.float())``; scoring in
    bf16 quantizes scores at ~2e-3, which flips knife-edge selections against
    fine-grained ``e_score_correction_bias`` lattices (AMINT-286).
    """

    def _sigmoid_config(self):
        return MoEConfig(
            n_routed_experts=16,
            n_shared_experts=0,
            n_activated_experts=4,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=64,
            inter_dim=128,
            moe_inter_dim=128,
            norm_topk_prob=True,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            force_e_score_correction_bias=True,
            dtype=torch.bfloat16,
        )

    def test_sigmoid_scoring_matches_fp32_reference_on_bf16_inputs(self):
        torch.manual_seed(0)
        config = self._sigmoid_config()
        gate = Gate(config)
        with torch.no_grad():
            gate.weight.copy_(torch.randn_like(gate.weight) * 0.05)
            # Knife-edge lattice bias like the MiniMax-M2.7 / GLM-4.7 checkpoints:
            # large magnitude, 1e-3 spacing.
            gate.e_score_correction_bias.copy_(8.0 + torch.arange(config.n_routed_experts, dtype=torch.float32) * 1e-3)
        gate.eval()

        # x: Tensor of shape [tokens, hidden] in bf16, like real routed inputs.
        x = torch.randn(512, config.dim, dtype=torch.bfloat16)
        token_mask = torch.ones(x.shape[0], dtype=torch.bool)
        weights, indices, _aux = gate(x, token_mask, None)

        # fp32 reference: the bf16 gate matmul followed by fp32 sigmoid,
        # bias-augmented selection, and fp32 top-k normalization.
        scores_bf16 = F.linear(x, gate.weight.to(x.dtype))
        ref_scores = torch.sigmoid(scores_bf16.float())
        choice = ref_scores + gate.e_score_correction_bias
        ref_indices = torch.topk(choice, config.n_activated_experts, dim=-1)[1]

        assert torch.equal(indices.sort(dim=-1).values, ref_indices.sort(dim=-1).values)

        ref_weights = ref_scores.gather(1, indices)
        ref_weights = ref_weights / (ref_weights.sum(dim=-1, keepdim=True) + 1e-20)
        # The gate casts final weights back to the input dtype, mirroring the
        # HF reference's top_k_weights.to(router_logits.dtype).
        assert weights.dtype == x.dtype
        assert torch.equal(weights, ref_weights.to(x.dtype))
