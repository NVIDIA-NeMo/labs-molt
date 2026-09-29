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

"""Tests for MoE latent projection layers (fc1_latent_proj, fc2_latent_proj)."""

from functools import partial
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE, _init_weights


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=8,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
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


class TestMoELatentProjectionInit:
    """Test MoE initialization with latent projection layers."""

    def test_latent_proj_created_when_moe_latent_size_set(self, moe_config, backend_config):
        """Test that latent projection layers are created when moe_latent_size is set."""
        moe_config.moe_latent_size = 64
        moe = MoE(moe_config, backend_config)

        assert moe.fc1_latent_proj is not None
        assert moe.fc2_latent_proj is not None

    def test_latent_proj_none_when_moe_latent_size_unset(self, moe_config, backend_config):
        """Test that latent projection layers are None when moe_latent_size is None."""
        assert moe_config.moe_latent_size is None
        moe = MoE(moe_config, backend_config)

        assert moe.fc1_latent_proj is None
        assert moe.fc2_latent_proj is None

    def test_fc1_latent_proj_dimensions(self, moe_config, backend_config):
        """Test fc1_latent_proj projects from dim to moe_latent_size."""
        moe_config.moe_latent_size = 64
        moe = MoE(moe_config, backend_config)

        assert moe.fc1_latent_proj.in_features == moe_config.dim
        assert moe.fc1_latent_proj.out_features == moe_config.moe_latent_size

    def test_fc2_latent_proj_dimensions(self, moe_config, backend_config):
        """Test fc2_latent_proj projects from moe_latent_size back to dim."""
        moe_config.moe_latent_size = 64
        moe = MoE(moe_config, backend_config)

        assert moe.fc2_latent_proj.in_features == moe_config.moe_latent_size
        assert moe.fc2_latent_proj.out_features == moe_config.dim

    def test_latent_proj_no_bias_by_default(self, moe_config, backend_config):
        """Test latent projections have no bias when expert_bias is False."""
        moe_config.moe_latent_size = 64
        assert moe_config.expert_bias is False
        moe = MoE(moe_config, backend_config)

        assert moe.fc1_latent_proj.bias is None
        assert moe.fc2_latent_proj.bias is None

    def test_latent_proj_with_bias(self, moe_config, backend_config):
        """Test latent projections have bias when expert_bias is True."""
        moe_config.moe_latent_size = 64
        moe_config.expert_bias = True
        moe = MoE(moe_config, backend_config)

        assert moe.fc1_latent_proj.bias is not None
        assert moe.fc2_latent_proj.bias is not None
        assert moe.fc1_latent_proj.bias.shape == (moe_config.moe_latent_size,)
        assert moe.fc2_latent_proj.bias.shape == (moe_config.dim,)


class TestMoELatentProjectionForward:
    """Test MoE forward pass with latent projection layers."""

    def test_forward_without_shared_experts_latent_enabled(self, moe_config, backend_config, device):
        """Test that latent projections are applied around experts when no shared experts."""
        moe_config.moe_latent_size = 64
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config).to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, dtype=torch.bfloat16, device=device)

        with (
            patch.object(moe.gate, "forward") as mock_gate,
            patch.object(moe.experts, "forward") as mock_experts,
        ):
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )
            # Experts receive latent-dim input and return latent-dim output
            mock_experts.return_value = torch.randn(
                batch_size * seq_len, moe_config.moe_latent_size, dtype=torch.bfloat16, device=device
            )

            output = moe(x)

            assert output.shape == x.shape

            # Verify experts received latent-dim input (64), not original dim (128)
            experts_input = mock_experts.call_args[0][0]
            assert experts_input.shape[-1] == moe_config.moe_latent_size

    def test_fc2_latent_proj_accepts_float32_expert_output(self, moe_config, backend_config):
        """Test back-projection re-enters bf16 after an FSDP-style fp32 expert output."""
        moe_config.moe_latent_size = 64
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)

        batch_size, seq_len = 2, 3
        x = torch.randn(batch_size, seq_len, moe_config.dim, dtype=torch.bfloat16)
        expert_output = torch.randn(
            batch_size * seq_len,
            moe_config.moe_latent_size,
            dtype=torch.float32,
            requires_grad=True,
        )

        with (
            patch.object(moe.gate, "forward") as mock_gate,
            patch.object(moe.experts, "forward") as mock_experts,
        ):
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, dtype=torch.bfloat16),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                ),
                None,
            )
            mock_experts.return_value = expert_output

            output = moe(x)

        assert output.dtype == torch.bfloat16
        output.float().sum().backward()
        assert expert_output.grad is not None
        assert expert_output.grad.dtype == torch.float32
        assert torch.isfinite(expert_output.grad).all()

    def test_forward_gate_receives_original_input(self, moe_config, backend_config, device):
        """Test that the gate receives the original (non-projected) input."""
        moe_config.moe_latent_size = 64
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config).to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, dtype=torch.bfloat16, device=device)

        with (
            patch.object(moe.gate, "forward") as mock_gate,
            patch.object(moe.experts, "forward") as mock_experts,
        ):
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )
            mock_experts.return_value = torch.randn(
                batch_size * seq_len, moe_config.moe_latent_size, dtype=torch.bfloat16, device=device
            )

            moe(x)

            # Gate should receive original dim, not latent dim
            gate_input = mock_gate.call_args[0][0]
            assert gate_input.shape[-1] == moe_config.dim

    def test_forward_with_shared_experts_latent_enabled(self, moe_config, backend_config, device):
        """Test latent projections with shared experts enabled."""
        moe_config.moe_latent_size = 64
        moe_config.n_shared_experts = 2
        moe = MoE(moe_config, backend_config).to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, dtype=torch.bfloat16, device=device)

        with (
            patch.object(moe.gate, "forward") as mock_gate,
            patch.object(moe.experts, "forward") as mock_experts,
            patch.object(moe.shared_experts, "forward") as mock_shared,
        ):
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )
            mock_experts.return_value = torch.randn(
                batch_size * seq_len, moe_config.moe_latent_size, dtype=torch.bfloat16, device=device
            )
            mock_shared.return_value = torch.randn(
                batch_size * seq_len, moe_config.dim, dtype=torch.bfloat16, device=device
            )

            # Shared experts run inline on the main stream (no side-stream overlap).
            output = moe(x)

            assert output.shape == x.shape

            # Routed experts should get latent-dim input
            experts_input = mock_experts.call_args[0][0]
            assert experts_input.shape[-1] == moe_config.moe_latent_size

            # Shared experts should get original-dim input
            shared_input = mock_shared.call_args[0][0]
            assert shared_input.shape[-1] == moe_config.dim

    def test_forward_no_latent_experts_receive_original_dim(self, moe_config, backend_config, device):
        """Test that without latent projections, experts receive original dim input."""
        assert moe_config.moe_latent_size is None
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config).to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, dtype=torch.bfloat16, device=device)

        with (
            patch.object(moe.gate, "forward") as mock_gate,
            patch.object(moe.experts, "forward") as mock_experts,
        ):
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )
            mock_experts.return_value = torch.randn(
                batch_size * seq_len, moe_config.dim, dtype=torch.bfloat16, device=device
            )

            moe(x)

            # Without latent projection, experts get the original dim
            experts_input = mock_experts.call_args[0][0]
            assert experts_input.shape[-1] == moe_config.dim

    def test_forward_with_padding_mask_and_latent(self, moe_config, backend_config, device):
        """Test latent projection works correctly with padding masks."""
        moe_config.moe_latent_size = 64
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config).to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, dtype=torch.bfloat16, device=device)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        padding_mask[:, -2:] = True

        with (
            patch.object(moe.gate, "forward") as mock_gate,
            patch.object(moe.experts, "forward") as mock_experts,
        ):
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )
            mock_experts.return_value = torch.randn(
                batch_size * seq_len, moe_config.moe_latent_size, dtype=torch.bfloat16, device=device
            )

            output = moe(x, padding_mask=padding_mask)

            assert output.shape == x.shape

            # Verify token mask was correctly derived
            mock_gate.assert_called_once()
            token_mask = mock_gate.call_args[0][1]
            expected_mask = (~padding_mask).flatten()
            torch.testing.assert_close(token_mask.float(), expected_mask.float())


class TestMoELatentProjectionInitWeights:
    """Test weight initialization for latent projection layers."""

    def test_init_weights_latent_proj(self, moe_config, backend_config, device):
        """Test that _init_weights initializes latent projection weights."""
        moe_config.moe_latent_size = 64
        moe = MoE(moe_config, backend_config)

        original_fc1_weight = moe.fc1_latent_proj.weight.clone().detach()
        original_fc2_weight = moe.fc2_latent_proj.weight.clone().detach()

        init_fn = partial(_init_weights, buffer_device=device, init_std=0.02)
        with torch.no_grad():
            init_fn(moe)

        assert not torch.equal(moe.fc1_latent_proj.weight.detach(), original_fc1_weight)
        assert not torch.equal(moe.fc2_latent_proj.weight.detach(), original_fc2_weight)

    def test_init_weights_latent_proj_bias_zeroed(self, moe_config, backend_config, device):
        """Test that _init_weights zeros latent projection biases."""
        moe_config.moe_latent_size = 64
        moe_config.expert_bias = True
        moe = MoE(moe_config, backend_config)

        # Set biases to non-zero
        with torch.no_grad():
            moe.fc1_latent_proj.bias.fill_(1.0)
            moe.fc2_latent_proj.bias.fill_(1.0)

        init_fn = partial(_init_weights, buffer_device=device, init_std=0.02)
        with torch.no_grad():
            init_fn(moe)

        torch.testing.assert_close(moe.fc1_latent_proj.bias, torch.zeros_like(moe.fc1_latent_proj.bias))
        torch.testing.assert_close(moe.fc2_latent_proj.bias, torch.zeros_like(moe.fc2_latent_proj.bias))

    def test_init_weights_no_latent_proj_noop(self, moe_config, backend_config, device):
        """Test that _init_weights is a no-op for MoE without latent projections."""
        assert moe_config.moe_latent_size is None
        moe = MoE(moe_config, backend_config)

        # Should not raise
        init_fn = partial(_init_weights, buffer_device=device, init_std=0.02)
        with torch.no_grad():
            init_fn(moe)

        assert moe.fc1_latent_proj is None
        assert moe.fc2_latent_proj is None

    def test_moe_init_weights_method(self, moe_config, backend_config, device):
        """Test the full MoE.init_weights method works with latent projections."""
        moe_config.moe_latent_size = 64
        moe = MoE(moe_config, backend_config)

        original_fc1_weight = moe.fc1_latent_proj.weight.clone().detach()

        with torch.no_grad():
            moe.init_weights(device, init_std=0.02)

        assert not torch.equal(moe.fc1_latent_proj.weight.detach(), original_fc1_weight)


class TestMoEConfigExpertDim:
    """Test the MoEConfig.expert_dim property."""

    def test_expert_dim_returns_latent_size_when_set(self):
        """Test expert_dim returns moe_latent_size when it's set."""
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
            moe_latent_size=64,
        )
        assert config.expert_dim == 64

    def test_expert_dim_returns_dim_when_latent_unset(self):
        """Test expert_dim returns dim when moe_latent_size is None."""
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
        )
        assert config.expert_dim == 128
