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

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

try:
    import transformer_engine  # noqa: F401

    HAS_TE = True
except ImportError:
    HAS_TE = False

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules, patch_moe_module
from nemo_automodel.components._peft.lora_experts import (
    GroupedExpertsDeepEPLoRA,
    GroupedExpertsLoRA,
    _pad_lora_rank_for_grouped_mm,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import GroupedExperts, GroupedExpertsDeepEP, GroupedExpertsTE


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_lora_init(moe_config, device):
    """Test initialization of GroupedExpertsLoRA, verifying shapes and frozen weights."""
    orig_experts = GroupedExperts(moe_config).to(device)
    # Initialize weights to avoid NaNs
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8)

    assert isinstance(lora_experts, GroupedExpertsLoRA)
    assert lora_experts.lora_dim == 4
    assert lora_experts.scale == 2.0

    # Check shapes
    # lora_gate_and_up_A: [n_experts, in_dim, lora_dim] -> [4, 16, 4]
    assert lora_experts.lora_gate_and_up_A.shape == (4, 16, 4)
    # lora_gate_and_up_B: [n_experts, lora_dim, out_dim] -> [4, 4, 64]
    assert lora_experts.lora_gate_and_up_B.shape == (4, 4, 64)  # 32 * 2
    # lora_down_A: [n_experts, inter_dim, lora_dim] -> [4, 32, 4]
    assert lora_experts.lora_down_A.shape == (4, 32, 4)
    # lora_down_B: [n_experts, lora_dim, out_dim] -> [4, 4, 16]
    assert lora_experts.lora_down_B.shape == (4, 4, 16)

    # Check requires_grad
    assert not lora_experts.gate_and_up_projs.requires_grad
    assert not lora_experts.down_projs.requires_grad
    assert lora_experts.lora_gate_and_up_A.requires_grad
    assert lora_experts.lora_gate_and_up_B.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_lora_equivalence(moe_config, device):
    """Test that applying LoRA to a model maintains output equivalence upon initialization."""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.linear = nn.Linear(16, 16)

        def forward(self, x, token_mask, weights, indices):
            return self.experts(x, token_mask, weights, indices) + self.linear(x)

    model = MockModel().to(device)
    # Initialize weights
    with torch.no_grad():
        model.experts.init_weights(buffer_device=device)
        nn.init.normal_(model.linear.weight)
        nn.init.zeros_(model.linear.bias)

    # Mock input
    bs = 2
    seq_len = 5
    dim = 16
    model = model.to(device)
    x = torch.randn(bs * seq_len, dim, device=device)
    token_mask = torch.ones(bs * seq_len, dtype=torch.bool, device=device)
    weights = torch.rand(bs * seq_len, 2, device=device)
    indices = torch.randint(0, 4, (bs * seq_len, 2), device=device)

    # Baseline output
    with torch.no_grad():
        out_orig = model(x, token_mask, weights, indices)

    # Apply LoRA
    peft_config = PeftConfig(target_modules=["*experts*"], dim=4)
    apply_lora_to_linear_modules(model, peft_config)
    model = model.to(device)
    # LoRA output
    with torch.no_grad():
        out_lora = model(x, token_mask, weights, indices)

    assert torch.allclose(out_orig, out_lora, atol=1e-6)
    assert isinstance(model.experts, GroupedExpertsLoRA)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_deepep_lora_init(moe_config, device):
    """Test initialization of GroupedExpertsDeepEPLoRA, verifying shapes."""
    orig_experts = GroupedExpertsDeepEP(moe_config).to(device)
    # Initialize weights
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4, alpha=8)

    assert isinstance(lora_experts, GroupedExpertsDeepEPLoRA)
    assert lora_experts.lora_dim == 4

    # Check shapes
    assert lora_experts.lora_gate_and_up_A.shape == (4, 16, 4)
    assert lora_experts.lora_gate_and_up_B.shape == (4, 4, 64)
    assert lora_experts.lora_down_A.shape == (4, 32, 4)
    assert lora_experts.lora_down_B.shape == (4, 4, 16)

    # Check requires_grad
    assert not lora_experts.gate_and_up_projs.requires_grad
    assert not lora_experts.down_projs.requires_grad
    assert lora_experts.lora_gate_and_up_A.requires_grad
    assert lora_experts.lora_gate_and_up_B.requires_grad


def test_grouped_experts_deepep_lora_preserves_dispatcher_settings(moe_config):
    """Test that LoRA wrapping preserves the source expert dispatcher backend."""
    orig_experts = GroupedExpertsDeepEP(
        moe_config,
        dispatcher_backend="hybridep",
        dispatcher_num_sms=24,
        dispatcher_share_token_dispatcher=False,
        dispatcher_async_dispatch=True,
    )
    orig_experts.use_mxfp8 = True

    lora_experts = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4, alpha=8)

    assert lora_experts.dispatcher_backend == "hybridep"
    assert lora_experts.dispatcher_num_sms == 24
    assert lora_experts.dispatcher_share_token_dispatcher is False
    assert lora_experts.dispatcher_async_dispatch is True
    assert lora_experts.use_mxfp8 is True


def test_pad_lora_rank_for_grouped_mm_aligns_bf16_rank():
    """Test rank padding for torch._grouped_mm stride alignment."""
    lora_A = torch.randn(2, 16, 4, dtype=torch.bfloat16)
    lora_B = torch.randn(2, 4, 32, dtype=torch.bfloat16)

    padded_A, padded_B = _pad_lora_rank_for_grouped_mm(lora_A, lora_B)

    assert padded_A.shape == (2, 16, 8)
    assert padded_B.shape == (2, 8, 32)
    assert torch.equal(padded_A[..., :4], lora_A)
    assert torch.equal(padded_B[:, :4], lora_B)
    assert torch.count_nonzero(padded_A[..., 4:]) == 0
    assert torch.count_nonzero(padded_B[:, 4:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_patch_moe_module(moe_config, device):
    """Test that patch_moe_module correctly wraps the original experts with the appropriate LoRA class."""
    orig_experts = GroupedExperts(moe_config).to(device)
    patched = patch_moe_module(orig_experts, dim=4)
    assert isinstance(patched, GroupedExpertsLoRA)

    orig_experts_deep = GroupedExpertsDeepEP(moe_config).to(device)
    patched_deep = patch_moe_module(orig_experts_deep, dim=4)
    assert isinstance(patched_deep, GroupedExpertsDeepEPLoRA)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_lora_patching_logic(moe_config, device):
    """
    Test the patching logic of apply_lora_to_linear_modules.
    Verifies that:
    1. Exact name matching works for MoE modules.
    2. Wildcard matching works for MoE modules.
    3. Non-target modules (e.g., standard Linear layers not in target list) are NOT patched.
    """

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.linear = nn.Linear(16, 16)

    model = MockModel().to(device)
    peft_config = PeftConfig(target_modules=["experts"], dim=4)

    count = apply_lora_to_linear_modules(model, peft_config)
    assert count == 1
    assert isinstance(model.experts, GroupedExpertsLoRA)
    assert isinstance(model.linear, nn.Linear)  # Should not be patched

    # Test wildcard matching
    model = MockModel().to(device)
    peft_config = PeftConfig(target_modules=["*experts*"], dim=4)
    count = apply_lora_to_linear_modules(model, peft_config)
    assert count == 1
    assert isinstance(model.experts, GroupedExpertsLoRA)
    assert isinstance(model.linear, nn.Linear)  # Should not be patched


class MockDeepEPDispatcher:
    """Mock dispatcher that simulates DeepEP's token permutation locally."""

    def token_permutation2(self, hidden_states, num_local_tokens, token_probs, token_indices):
        # Simply return the hidden states as if it was a single expert local dispatch
        # Return one token-count entry per local expert.
        tokens_per_expert = torch.zeros(4, dtype=torch.long, device=hidden_states.device)
        return hidden_states, tokens_per_expert, token_probs

    def token_unpermutation(self, hidden_states):
        return hidden_states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.timeout(60)
def test_grouped_experts_deepep_lora_forward_mocked(moe_config, device):
    """
    Test Forward pass of GroupedExpertsDeepEPLoRA using a Mock Dispatcher.

    This test verifies the LoRA-wrapped native grouped-MM logic
    independently of the DeepEP communication backend. This allows verification on
    non-Hopper (non-sm_90) hardware where DeepEP is physically unavailable.
    """
    moe_config.n_routed_experts = 4
    moe_config.dim = 16
    moe_config.moe_inter_dim = 32
    moe_config.dtype = torch.bfloat16

    orig_experts = GroupedExpertsDeepEP(moe_config).to(device).to(torch.bfloat16)
    # Initialize expert weights BEFORE creating LoRA module so they match after copy
    with torch.no_grad():
        orig_experts.init_weights(device)

    # Manually inject mock state since DeepEP init fails on non-Hopper hardware
    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1

    lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4).to(device).to(torch.bfloat16)
    mock_dispatcher = MockDeepEPDispatcher()

    # Mock tokens_per_expert; counts must sum to num_tokens.
    num_tokens = 8
    # One expert gets all tokens for simplicity
    tokens_per_expert = torch.tensor([num_tokens, 0, 0, 0], dtype=torch.long, device="cpu")

    # Capture deterministic data to return from the mock dispatcher
    dtype = torch.bfloat16
    permuted_x = torch.randn(num_tokens, 16, device=device).to(dtype)
    # permuted_probs should be 1D [num_tokens] because it's unsqueezed in forward
    permuted_probs = torch.ones(num_tokens, device=device).to(dtype)

    # Set the same mock on both modules to ensure they see the same "dispatched" data
    mock_dispatcher.token_permutation2 = MagicMock(return_value=(permuted_x, tokens_per_expert, permuted_probs))
    lora_module.token_dispatcher = mock_dispatcher
    orig_experts.token_dispatcher = mock_dispatcher

    x = torch.randn(num_tokens, 16, device=device).to(dtype)
    # weights should also be [num_tokens, TopK] where TopK=1
    weights = torch.ones(num_tokens, 1, device=device).to(dtype)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    # This reaches the native grouped-MM calls.
    out = lora_module(x, token_mask, weights, indices)

    # Verify equivalence with zero LoRA weights (DeepEP LoRA B is zero-init by default)
    # GroupedExpertsDeepEP.forward hardcodes a call to .to_local() which regular
    # Parameters don't have. We surgicaly patch it only for this unit test.
    with torch.no_grad(), patch.object(torch.Tensor, "to_local", new=lambda self: self, create=True):
        out_orig = orig_experts(x, token_mask, weights, indices)

    assert out.shape == (num_tokens, 16)
    assert torch.allclose(out, out_orig, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_lora_forward(moe_config, device):
    """Test forward pass of GroupedExpertsLoRA."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Create input
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    # Forward pass
    out = lora_experts(x, token_mask, weights, indices)

    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_weight_initialization_xavier(moe_config, device):
    """Test Xavier initialization for LoRA weights."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8, lora_A_init_method="xavier").to(device)

    # Check that B matrices are zero-initialized
    assert torch.allclose(lora_experts.lora_gate_and_up_B, torch.zeros_like(lora_experts.lora_gate_and_up_B))
    assert torch.allclose(lora_experts.lora_down_B, torch.zeros_like(lora_experts.lora_down_B))

    # Check that A matrices are not zero (xavier initialized)
    assert not torch.allclose(lora_experts.lora_gate_and_up_A, torch.zeros_like(lora_experts.lora_gate_and_up_A))
    assert not torch.allclose(lora_experts.lora_down_A, torch.zeros_like(lora_experts.lora_down_A))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_weight_initialization_kaiming(moe_config, device):
    """Test Kaiming initialization for LoRA weights."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8, lora_A_init_method="kaiming").to(device)

    # Check that B matrices are zero-initialized
    assert torch.allclose(lora_experts.lora_gate_and_up_B, torch.zeros_like(lora_experts.lora_gate_and_up_B))
    assert torch.allclose(lora_experts.lora_down_B, torch.zeros_like(lora_experts.lora_down_B))

    # Check that A matrices are not zero (kaiming initialized)
    assert not torch.allclose(lora_experts.lora_gate_and_up_A, torch.zeros_like(lora_experts.lora_gate_and_up_A))
    assert not torch.allclose(lora_experts.lora_down_A, torch.zeros_like(lora_experts.lora_down_A))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_scale_parameter(moe_config, device):
    """Test that LoRA scale parameter is correctly computed and applied."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    # Test different alpha/dim combinations
    lora_dim = 4
    alpha = 16
    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=lora_dim, alpha=alpha).to(device)

    expected_scale = alpha / lora_dim
    assert lora_experts.scale == expected_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_gradient_flow(moe_config, device):
    """Test that gradients flow only through LoRA parameters, not base weights."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Create input
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device, requires_grad=True)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    # Forward pass
    out = lora_experts(x, token_mask, weights, indices)
    loss = out.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    assert lora_experts.gate_and_up_projs.grad is None
    assert lora_experts.down_projs.grad is None
    assert lora_experts.lora_gate_and_up_A.grad is not None
    assert lora_experts.lora_gate_and_up_B.grad is not None
    assert lora_experts.lora_down_A.grad is not None
    assert lora_experts.lora_down_B.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_zero_active_experts(moe_config, device):
    """Test the edge case where no tokens are routed to any local experts."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Create input where no tokens match any expert
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device, requires_grad=True)
    token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)  # All masked out
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    # Forward pass - should handle gracefully
    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)

    # Test backward pass for gradient flow
    loss = out.sum()
    loss.backward()

    # Check that gradients exist for LoRA parameters
    assert lora_experts.lora_gate_and_up_A.grad is not None
    assert lora_experts.lora_gate_and_up_B.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_with_expert_bias(device):
    """Test LoRA with expert bias enabled."""
    moe_config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.float32,
        expert_bias=True,  # Enable bias
    )

    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Check that bias parameters exist and are frozen
    assert hasattr(lora_experts, "gate_up_proj_bias")
    assert hasattr(lora_experts, "down_proj_bias")
    assert not lora_experts.gate_up_proj_bias.requires_grad
    assert not lora_experts.down_proj_bias.requires_grad

    # Test forward pass
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quick_geglu_activation_with_lora(device):
    """Test GroupedExpertsLoRA with QuickGEGLU activation."""
    moe_config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="quick_geglu",  # Use QuickGEGLU
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.float32,
    )

    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Test forward pass
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_dtype_conversion(moe_config, device):
    """Test LoRA with explicit dtype specification."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    # Create LoRA with explicit bfloat16 dtype
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8, lora_dtype="bfloat16").to(device)

        # Check that LoRA weights have the correct dtype
        assert lora_experts.lora_gate_and_up_A.dtype == torch.bfloat16
        assert lora_experts.lora_gate_and_up_B.dtype == torch.bfloat16
        assert lora_experts.lora_down_A.dtype == torch.bfloat16
        assert lora_experts.lora_down_B.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_backward_pass_values(moe_config, device):
    """Test that LoRA backward pass produces non-zero gradients."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Manually set LoRA B weights to non-zero to ensure LoRA contributes to output
    with torch.no_grad():
        lora_experts.lora_gate_and_up_B.normal_(0, 0.01)
        lora_experts.lora_down_B.normal_(0, 0.01)

    # Create input
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device, requires_grad=True)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    # Forward and backward
    out = lora_experts(x, token_mask, weights, indices)
    loss = out.sum()
    loss.backward()

    # Check that gradients are non-zero
    assert not torch.allclose(
        lora_experts.lora_gate_and_up_A.grad, torch.zeros_like(lora_experts.lora_gate_and_up_A.grad)
    )
    assert not torch.allclose(
        lora_experts.lora_gate_and_up_B.grad, torch.zeros_like(lora_experts.lora_gate_and_up_B.grad)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepep_lora_zero_tokens(moe_config, device):
    """Test DeepEP LoRA forward pass with zero tokens routed to experts."""
    moe_config.n_routed_experts = 4
    moe_config.dim = 16
    moe_config.moe_inter_dim = 32
    moe_config.dtype = torch.bfloat16

    orig_experts = GroupedExpertsDeepEP(moe_config).to(device).to(torch.bfloat16)
    with torch.no_grad():
        orig_experts.init_weights(device)

    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1

    lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4).to(device).to(torch.bfloat16)
    mock_dispatcher = MockDeepEPDispatcher()

    num_tokens = 8
    # All experts get zero tokens
    tokens_per_expert = torch.tensor([0, 0, 0, 0], dtype=torch.long, device="cpu")

    dtype = torch.bfloat16
    permuted_x = torch.randn(num_tokens, 16, device=device).to(dtype)
    permuted_probs = torch.ones(num_tokens, device=device).to(dtype)

    mock_dispatcher.token_permutation2 = MagicMock(return_value=(permuted_x, tokens_per_expert, permuted_probs))
    lora_module.token_dispatcher = mock_dispatcher

    x = torch.randn(num_tokens, 16, device=device).to(dtype)
    weights = torch.ones(num_tokens, 1, device=device).to(dtype)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    # Should handle zero tokens gracefully
    out = lora_module(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TE, reason="Transformer Engine required")
def test_patch_moe_module_rejects_te_experts(moe_config, device):
    """Test that patch_moe_module raises NotImplementedError for GroupedExpertsTE."""
    orig_experts = GroupedExpertsTE(moe_config)
    orig_experts.init_weights(buffer_device=device)
    orig_experts = orig_experts.to(device)
    with pytest.raises(NotImplementedError, match="LoRA is not supported for Transformer Engine"):
        patch_moe_module(orig_experts, dim=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TE, reason="Transformer Engine required")
def test_apply_lora_rejects_te_experts(moe_config, device):
    """Test that apply_lora_to_linear_modules raises NotImplementedError for GroupedExpertsTE."""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExpertsTE(moe_config)

    model = MockModel()
    model.experts.init_weights(buffer_device=device)
    model = model.to(device)
    peft_config = PeftConfig(target_modules=["experts"], dim=4)

    with pytest.raises(NotImplementedError, match="LoRA is not supported for Transformer Engine"):
        apply_lora_to_linear_modules(model, peft_config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_relu2_non_gated_shapes(device):
    """Test LoRA weight shapes for non-gated ReLU² activation (half the gate+up dim)."""
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="relu2",
        dtype=torch.float32,
    )

    orig_experts = GroupedExperts(config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Non-gated: lora_gate_and_up_B should be [n_experts, lora_dim, moe_inter_dim] (not 2x)
    assert lora_experts.lora_gate_and_up_B.shape == (4, 4, 32)
    # Down A still uses moe_inter_dim
    assert lora_experts.lora_down_A.shape == (4, 32, 4)

    # Forward pass should work
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_copies_use_torch_mm_flag(moe_config, device):
    """Test that use_torch_mm flag is copied from the original module."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    # Default (no backend) -> use_torch_mm is False
    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8)
    assert lora_experts.use_torch_mm is False

    # Simulate a module that had use_torch_mm=True
    orig_experts.use_torch_mm = True
    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8)
    assert lora_experts.use_torch_mm is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepep_lora_relu2_non_gated_shapes(device):
    """Test DeepEP LoRA weight shapes for non-gated ReLU² activation."""
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="relu2",
        dtype=torch.float32,
    )

    orig_experts = GroupedExpertsDeepEP(config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(device)

    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1

    lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4, alpha=8)

    # Non-gated: lora_gate_and_up_B should be [n_experts, lora_dim, moe_inter_dim] (not 2x)
    assert lora_module.lora_gate_and_up_B.shape == (4, 4, 32)
    assert lora_module.lora_down_A.shape == (4, 32, 4)


# ---------- torch_mm forward path tests ----------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_lora_forward_torch_mm(moe_config, device):
    """Test forward pass of GroupedExpertsLoRA using _forward_grouped_mm (use_torch_mm=True)."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    # Force use_torch_mm so forward dispatches to _forward_grouped_mm
    orig_experts.use_torch_mm = True
    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)
    assert lora_experts.use_torch_mm is True

    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_lora_forward_torch_mm_equivalence(moe_config, device):
    """With zero-init B, torch_mm path should match the loop path output."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    # Create two LoRA wrappers: one with loop, one with torch_mm
    lora_loop = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)
    lora_loop.use_torch_mm = False

    lora_mm = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)
    lora_mm.use_torch_mm = True
    # Copy LoRA weights so both are identical
    with torch.no_grad():
        lora_mm.lora_gate_and_up_A.copy_(lora_loop.lora_gate_and_up_A)
        lora_mm.lora_gate_and_up_B.copy_(lora_loop.lora_gate_and_up_B)
        lora_mm.lora_down_A.copy_(lora_loop.lora_down_A)
        lora_mm.lora_down_B.copy_(lora_loop.lora_down_B)

    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    with torch.no_grad():
        out_loop = lora_loop(x, token_mask, weights, indices)
        out_mm = lora_mm(x, token_mask, weights, indices)

    assert torch.allclose(out_loop, out_mm, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_lora_forward_torch_mm_zero_tokens(moe_config, device):
    """Test _forward_grouped_mm dummy computation path when no tokens are routed."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    orig_experts.use_torch_mm = True
    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)  # All masked
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_lora_forward_torch_mm_with_bias(device):
    """Test _forward_grouped_mm path with expert_bias=True."""
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.float32,
        expert_bias=True,
    )

    orig_experts = GroupedExperts(config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    orig_experts.use_torch_mm = True
    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepep_lora_forward_torch_mm(moe_config, device):
    """Test DeepEP LoRA forward with the native grouped-MM backend via a mock dispatcher."""
    moe_config.n_routed_experts = 4
    moe_config.dim = 16
    moe_config.moe_inter_dim = 32
    moe_config.dtype = torch.bfloat16

    orig_experts = GroupedExpertsDeepEP(moe_config).to(device).to(torch.bfloat16)
    with torch.no_grad():
        orig_experts.init_weights(device)

    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1
    # lora_dim must be >= 8 for bf16 to satisfy torch._grouped_mm 16-byte stride alignment
    lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=8).to(device).to(torch.bfloat16)

    mock_dispatcher = MockDeepEPDispatcher()

    num_tokens = 8
    tokens_per_expert = torch.tensor([num_tokens, 0, 0, 0], dtype=torch.long, device="cpu")

    dtype = torch.bfloat16
    permuted_x = torch.randn(num_tokens, 16, device=device).to(dtype)
    permuted_probs = torch.ones(num_tokens, device=device).to(dtype)

    mock_dispatcher.token_permutation2 = MagicMock(return_value=(permuted_x, tokens_per_expert, permuted_probs))
    lora_module.token_dispatcher = mock_dispatcher

    x = torch.randn(num_tokens, 16, device=device).to(dtype)
    weights = torch.ones(num_tokens, 1, device=device).to(dtype)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    out = lora_module(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_dtype_string(moe_config, device):
    """Test that lora_dtype can be passed as a string."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8, lora_dtype="bfloat16").to(device)
        assert lora_experts.lora_gate_and_up_A.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepep_lora_dtype_string(moe_config, device):
    """Test that DeepEP LoRA lora_dtype can be passed as a string."""
    orig_experts = GroupedExpertsDeepEP(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(device)
    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4, lora_dtype="bfloat16").to(device)
        assert lora_module.lora_gate_and_up_A.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepep_lora_kaiming_init(moe_config, device):
    """Test Kaiming initialization for DeepEP LoRA weights."""
    orig_experts = GroupedExpertsDeepEP(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(device)
    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1

    lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4, alpha=8, lora_A_init_method="kaiming")

    # B matrices zero-initialized
    assert torch.allclose(lora_module.lora_gate_and_up_B, torch.zeros_like(lora_module.lora_gate_and_up_B))
    assert torch.allclose(lora_module.lora_down_B, torch.zeros_like(lora_module.lora_down_B))

    # A matrices not zero (kaiming initialized)
    assert not torch.allclose(lora_module.lora_gate_and_up_A, torch.zeros_like(lora_module.lora_gate_and_up_A))
    assert not torch.allclose(lora_module.lora_down_A, torch.zeros_like(lora_module.lora_down_A))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deepep_lora_forward_torch_mm_with_bias(device):
    """Test DeepEP LoRA forward with native grouped MM and expert_bias=True."""
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
        expert_bias=True,
    )

    orig_experts = GroupedExpertsDeepEP(config).to(device).to(torch.bfloat16)
    with torch.no_grad():
        orig_experts.init_weights(device)
    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1
    # lora_dim must be >= 8 for bf16 to satisfy torch._grouped_mm 16-byte stride alignment
    lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=8).to(device).to(torch.bfloat16)

    mock_dispatcher = MockDeepEPDispatcher()
    num_tokens = 8
    tokens_per_expert = torch.tensor([num_tokens, 0, 0, 0], dtype=torch.long, device="cpu")
    dtype = torch.bfloat16
    permuted_x = torch.randn(num_tokens, 16, device=device).to(dtype)
    permuted_probs = torch.ones(num_tokens, device=device).to(dtype)

    mock_dispatcher.token_permutation2 = MagicMock(return_value=(permuted_x, tokens_per_expert, permuted_probs))
    lora_module.token_dispatcher = mock_dispatcher

    x = torch.randn(num_tokens, 16, device=device).to(dtype)
    weights = torch.ones(num_tokens, 1, device=device).to(dtype)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    out = lora_module(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()


# ---------- moe_rank_scaling tests ----------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_rank_scaling_basic(moe_config, device):
    """MoE experts get dim // n_activated_experts; Linear keeps full dim."""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.linear = nn.Linear(16, 16)

    model = MockModel().to(device)
    with torch.no_grad():
        model.experts.init_weights(buffer_device=device)

    peft_config = PeftConfig(
        target_modules=["experts", "linear"],
        dim=16,
        moe_rank_scaling=True,
    )
    count = apply_lora_to_linear_modules(model, peft_config)
    assert count == 2

    # MoE module should have rank = 16 // 2 = 8
    assert isinstance(model.experts, GroupedExpertsLoRA)
    assert model.experts.lora_dim == 8

    # Linear module should keep full rank = 16
    assert model.linear.dim == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_rank_scaling_default_off(moe_config, device):
    """With moe_rank_scaling=False (default), both get the full dim."""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.linear = nn.Linear(16, 16)

    model = MockModel().to(device)
    with torch.no_grad():
        model.experts.init_weights(buffer_device=device)

    peft_config = PeftConfig(
        target_modules=["experts", "linear"],
        dim=16,
    )
    count = apply_lora_to_linear_modules(model, peft_config)
    assert count == 2

    # Both should keep full rank
    assert model.experts.lora_dim == 16
    assert model.linear.dim == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_rank_scaling_floor_division_warning(device):
    """When dim is not evenly divisible by n_activated_experts, log a warning and use floor division."""
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=3,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(config)

    model = MockModel().to(device)
    with torch.no_grad():
        model.experts.init_weights(buffer_device=device)

    peft_config = PeftConfig(
        target_modules=["experts"],
        dim=16,
        moe_rank_scaling=True,
    )

    with patch("nemo_automodel.components._peft.lora.logger") as mock_logger:
        count = apply_lora_to_linear_modules(model, peft_config)
        mock_logger.warning.assert_called_once()
        assert "not evenly divisible" in mock_logger.warning.call_args[0][0]

    assert count == 1
    # 16 // 3 = 5
    assert model.experts.lora_dim == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_rank_scaling_dim_too_small(moe_config, device):
    """When dim < n_activated_experts, raise ValueError."""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)

    model = MockModel().to(device)
    with torch.no_grad():
        model.experts.init_weights(buffer_device=device)

    # moe_config has n_activated_experts=2, dim=1 -> 1 // 2 = 0 -> error
    peft_config = PeftConfig(
        target_modules=["experts"],
        dim=1,
        moe_rank_scaling=True,
    )
    with pytest.raises(ValueError, match="Increase dim to at least n_activated_experts"):
        apply_lora_to_linear_modules(model, peft_config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_rank_scaling_output_equivalence(moe_config, device):
    """With zero-init B, scaled-rank MoE LoRA should produce the same output as the original model."""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.linear = nn.Linear(16, 16)

        def forward(self, x, token_mask, weights, indices):
            return self.experts(x, token_mask, weights, indices) + self.linear(x)

    model = MockModel().to(device)
    with torch.no_grad():
        model.experts.init_weights(buffer_device=device)
        nn.init.normal_(model.linear.weight)
        nn.init.zeros_(model.linear.bias)

    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    with torch.no_grad():
        out_orig = model(x, token_mask, weights, indices)

    peft_config = PeftConfig(
        target_modules=["experts", "linear"],
        dim=16,
        moe_rank_scaling=True,
    )
    apply_lora_to_linear_modules(model, peft_config)
    model = model.to(device)

    with torch.no_grad():
        out_lora = model(x, token_mask, weights, indices)

    assert torch.allclose(out_orig, out_lora, atol=1e-6)
