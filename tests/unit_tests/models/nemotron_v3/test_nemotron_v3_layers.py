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


import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.layers import (
    NemotronV3Attention,
    NemotronV3Block,
)
from nemo_automodel.components.moe.config import MoEConfig

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")

try:
    import mamba_ssm  # noqa: F401

    _has_mamba_ssm = True
except ImportError:
    _has_mamba_ssm = False

skip_if_no_mamba = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_mamba_ssm, reason="CUDA and mamba_ssm required for Mamba Triton kernels"
)


class MockNemotronV3Config:
    """Mock configuration for NemotronV3 model."""

    def __init__(self, **overrides):
        # Attention configuration
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 64
        self.hidden_size = 512
        self.attention_bias = False
        self.attention_dropout = 0.0

        # MLP/MoE configuration
        self.intermediate_size = 1024
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"

        # Mamba configuration
        self.mamba_num_heads = 8
        self.mamba_head_dim = 64
        self.ssm_state_size = 16
        self.n_groups = 1
        self.chunk_size = 256
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False

        # General configuration
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 4
        self.vocab_size = 1000
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False

        # Layer types for hybrid architecture
        self.layers_block_type = ["attention", "mlp", "moe", "mamba"]

        # MoE configuration
        self.n_routed_experts = 8
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.moe_intermediate_size = 256
        self.norm_topk_prob = False
        self.moe_shared_expert_intermediate_size = 256

        # Apply overrides
        for key, value in overrides.items():
            setattr(self, key, value)


class TestNemotronV3Attention:
    """Test NemotronV3Attention module."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    def test_attention_init(self, config):
        """Test attention initialization."""
        attn = NemotronV3Attention(config)

        assert attn.num_attention_heads == config.num_attention_heads
        assert attn.num_key_value_heads == config.num_key_value_heads
        assert attn.head_dim == config.head_dim
        assert attn.hidden_size == config.hidden_size

        # Check projections exist
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "o_proj")

    def test_attention_init_with_bias(self, config):
        """Test attention initialization with bias enabled."""
        config.attention_bias = True
        attn = NemotronV3Attention(config)

        assert attn.q_proj.bias is not None
        assert attn.k_proj.bias is not None
        assert attn.v_proj.bias is not None
        assert attn.o_proj.bias is not None

    @skip_if_no_gpu
    def test_attention_forward_shape(self, config):
        """Test attention forward pass produces correct shapes."""
        attn = NemotronV3Attention(config).cuda()

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_attention_forward_with_mask(self, config):
        """Test attention forward pass with attention mask."""
        attn = NemotronV3Attention(config).cuda()

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        # 2D padding mask (1=valid, 0=pad) — TE handles causality internally
        attention_mask = torch.ones(batch_size, seq_len, device="cuda", dtype=torch.long)

        output = attn(hidden_states, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_attention_gqa_with_different_kv_heads(self):
        """Test GQA with different number of key-value heads."""
        config = MockNemotronV3Config(
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=32,
            hidden_size=512,
        )
        attn = NemotronV3Attention(config).cuda()

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_attention_init_weights(self, config):
        """Test attention weight initialization."""
        config.attention_bias = True
        attn = NemotronV3Attention(config).cuda()

        device = torch.device("cuda")
        attn.init_weights(
            num_hidden_layers=config.num_hidden_layers,
            rescale_prenorm_residual=True,
            buffer_device=device,
        )

        # Check biases are zeroed
        assert torch.allclose(attn.q_proj.bias, torch.zeros_like(attn.q_proj.bias))
        assert torch.allclose(attn.k_proj.bias, torch.zeros_like(attn.k_proj.bias))

    @skip_if_no_gpu
    def test_attention_forward_single_token(self, config):
        """Test attention with single token (seqlen=1)."""
        attn = NemotronV3Attention(config).cuda()

        batch_size, seq_len = 2, 1
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestNemotronV3Block:
    """Test NemotronV3Block module."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

    @pytest.fixture
    def moe_config(self, config):
        return MoEConfig(
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=1,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=config.routed_scaling_factor,
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=config.mlp_bias,
            expert_activation="relu2",
            dtype=torch.bfloat16,
            shared_expert_gate=False,
            shared_expert_inter_dim=config.moe_shared_expert_intermediate_size,
            shared_expert_activation="relu2",
        )

    def test_block_init_attention(self, config, backend):
        """Test block initialization with attention layer type."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        assert block.block_type == "attention"
        assert isinstance(block.mixer, NemotronV3Attention)

    def test_block_init_mlp(self, config, backend):
        """Test block initialization with MLP layer type."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        assert block.block_type == "mlp"
        # MLP should use relu2 activation by default for NemotronV3
        assert hasattr(block.mixer, "up_proj")
        assert hasattr(block.mixer, "down_proj")

    def test_block_init_moe(self, config, backend, moe_config):
        """Test block initialization with MoE layer type."""
        config.layers_block_type = ["moe"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=moe_config, backend=backend)

        assert block.block_type == "moe"
        assert hasattr(block.mixer, "gate")
        assert hasattr(block.mixer, "experts")

    def test_block_init_invalid_type(self, config, backend):
        """Test block initialization with invalid layer type raises error."""
        config.layers_block_type = ["invalid"]

        with pytest.raises(ValueError, match="Invalid block_type"):
            NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

    @skip_if_no_gpu
    def test_block_forward_attention(self, config, backend):
        """Test block forward pass with attention layer."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend).cuda()

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        output = block(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_block_forward_mlp(self, config, backend):
        """Test block forward pass with MLP layer."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)
        block = block.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = block(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_block_forward_moe(self, config, backend, moe_config):
        """Test block forward pass with MoE layer."""
        config.layers_block_type = ["moe"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=moe_config, backend=backend)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = block(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_block_residual_connection(self, config, backend):
        """Test that block applies residual connection."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)
        block = block.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
        hidden_states_clone = hidden_states.clone()

        output = block(hidden_states)

        # Output should differ from input due to MLP transformation
        assert not torch.allclose(output, hidden_states_clone)

    def test_block_residual_fp32(self, config, backend):
        """Test block with fp32 residual option."""
        config.layers_block_type = ["mlp"]
        config.residual_in_fp32 = True
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        assert block.residual_in_fp32 is True

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = block(hidden_states)

        # Should still produce correct shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_block_mlp_property(self, config, backend, moe_config):
        """Test mlp property returns mixer for MoE blocks."""
        config.layers_block_type = ["moe"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=moe_config, backend=backend)

        assert block.mlp is not None
        assert block.mlp is block.mixer

    def test_block_mlp_property_non_moe(self, config, backend):
        """Test mlp property returns None for non-MoE blocks."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        assert block.mlp is None

    def test_block_init_weights_attention(self, config, backend):
        """Test weight initialization for attention block."""
        config.layers_block_type = ["attention"]
        config.attention_bias = True
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        device = torch.device("cpu")
        block.init_weights(buffer_device=device)

        # Verify biases are zeroed
        assert torch.allclose(block.mixer.q_proj.bias, torch.zeros_like(block.mixer.q_proj.bias))

    def test_block_init_weights_mlp(self, config, backend):
        """Test weight initialization for MLP block."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        device = torch.device("cpu")
        block.init_weights(buffer_device=device)

        # Weights should be initialized (not all zeros)
        assert not torch.allclose(block.mixer.up_proj.weight, torch.zeros_like(block.mixer.up_proj.weight))

    def test_block_uses_relu2_for_mlp(self, config, backend):
        """Test that MLP block uses relu2 activation by default."""
        config.layers_block_type = ["mlp"]
        config.mlp_hidden_act = "relu2"
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        # MLP should not have gate_proj (non-gated activation)
        assert block.mixer.gate_proj is None
        assert block.mixer.up_proj is not None
        assert block.mixer.down_proj is not None


class TestNemotronV3MambaRMSNormGated:
    """Test NemotronV3MambaRMSNormGated module."""

    def test_gated_rmsnorm_init(self):
        """Test gated RMSNorm initialization."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3MambaRMSNormGated

        hidden_size = 512
        group_size = 128
        norm = NemotronV3MambaRMSNormGated(hidden_size, group_size, eps=1e-5)

        assert norm.weight.shape == (hidden_size,)
        assert norm.variance_epsilon == 1e-5
        assert norm.group_size == group_size

    @skip_if_no_mamba
    def test_gated_rmsnorm_forward_requires_cuda(self):
        """Test that forward requires CUDA for Triton kernels."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3MambaRMSNormGated

        hidden_size = 512
        group_size = 128
        norm = NemotronV3MambaRMSNormGated(hidden_size, group_size).cuda()

        # Forward requires CUDA tensors for Triton kernels
        hidden_states = torch.randn(2, 8, hidden_size, device="cuda")
        output = norm(hidden_states)
        assert output.shape == hidden_states.shape


class TestNemotronV3Mamba2Mixer:
    """Test NemotronV3Mamba2Mixer module initialization."""

    def test_full_tensor_if_dtensor_clones_gathered_value(self, monkeypatch):
        """Gathered fp32 holder outputs must not alias storage FSDP may reshard."""
        from nemo_automodel.components.models.nemotron_v3 import layers as layers_mod

        full = torch.arange(4, dtype=torch.float32, requires_grad=True)

        class FakeDTensor:
            def full_tensor(self):
                return full

        monkeypatch.setattr(layers_mod, "DTensor", FakeDTensor)

        gathered = layers_mod._full_tensor_if_dtensor(FakeDTensor())

        assert torch.equal(gathered, full)
        assert gathered.data_ptr() != full.data_ptr()
        gathered.sum().backward()
        assert torch.equal(full.grad, torch.ones_like(full))

    def test_full_tensor_if_dtensor_clones_regular_tensor(self):
        """FSDP can reshard a regular-looking tensor returned from an fp32 holder."""
        from nemo_automodel.components.models.nemotron_v3 import layers as layers_mod

        tensor = torch.arange(4, dtype=torch.float32, requires_grad=True)

        gathered = layers_mod._full_tensor_if_dtensor(tensor)

        assert torch.equal(gathered, tensor)
        assert gathered.data_ptr() != tensor.data_ptr()
        gathered.sum().backward()
        assert torch.equal(tensor.grad, torch.ones_like(tensor))

    def test_get_fp32_ssm_params_passes_reference(self, config):
        """FSDP2 root pre-forward expects a non-empty args tuple for the holder."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)
        reference = torch.randn(1, 2, mixer.in_proj.out_features)
        seen = {}

        def fake_forward(ref=None):
            seen["reference"] = ref
            return mixer.A_log, mixer.dt_bias, mixer.D

        mixer._fp32_params.forward = fake_forward

        mixer._get_fp32_ssm_params(reference)

        assert seen["reference"] is reference

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    def test_mamba2_mixer_init(self, config):
        """Test Mamba2Mixer initialization."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        assert mixer.layer_idx == 0
        assert mixer.hidden_size == config.hidden_size
        assert mixer.num_heads == config.mamba_num_heads
        assert mixer.head_dim == config.mamba_head_dim
        assert mixer.ssm_state_size == config.ssm_state_size
        assert mixer.n_groups == config.n_groups
        assert mixer.chunk_size == config.chunk_size
        assert mixer.conv_kernel_size == config.conv_kernel

    def test_mamba2_mixer_derived_dimensions(self, config):
        """Test Mamba2Mixer derived dimensions."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        expected_intermediate = config.mamba_num_heads * config.mamba_head_dim
        expected_conv_dim = expected_intermediate + 2 * config.n_groups * config.ssm_state_size

        assert mixer.intermediate_size == expected_intermediate
        assert mixer.conv_dim == expected_conv_dim

    def test_mamba2_mixer_projection_sizes(self, config):
        """Test Mamba2Mixer projection dimensions."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        expected_proj_size = mixer.intermediate_size + mixer.conv_dim + mixer.num_heads
        assert mixer.in_proj.in_features == config.hidden_size
        assert mixer.in_proj.out_features == expected_proj_size

        assert mixer.out_proj.in_features == mixer.intermediate_size
        assert mixer.out_proj.out_features == config.hidden_size

    def test_mamba2_mixer_ssm_parameters(self, config):
        """Test Mamba2Mixer SSM parameters."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        assert mixer.dt_bias.shape == (config.mamba_num_heads,)
        assert mixer.A_log.shape == (config.mamba_num_heads,)
        assert mixer.D.shape == (config.mamba_num_heads,)

        params = dict(mixer.named_parameters())
        assert "_fp32_params.A_log" in params
        assert "_fp32_params.dt_bias" in params
        assert "_fp32_params.D" in params
        assert "A_log" not in mixer._parameters
        assert "dt_bias" not in mixer._parameters
        assert "D" not in mixer._parameters

        # Check no_weight_decay attributes
        assert getattr(mixer.A_log, "_no_weight_decay", False)
        assert getattr(mixer.D, "_no_weight_decay", False)

    def test_mamba2_mixer_conv1d(self, config):
        """Test Mamba2Mixer conv1d layer."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        assert mixer.conv1d.in_channels == mixer.conv_dim
        assert mixer.conv1d.out_channels == mixer.conv_dim
        assert mixer.conv1d.kernel_size[0] == config.conv_kernel
        assert mixer.conv1d.groups == mixer.conv_dim


class TestBackwardCompatibility:
    """Verify that attention and block layers still work without cache args."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

    @skip_if_no_gpu
    def test_attention_no_cache_args(self, config):
        """Verify attn(hidden) still works without cache args."""
        attn = NemotronV3Attention(config).cuda()
        hidden = torch.randn(2, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out = attn(hidden)
        assert out.shape == (2, 8, config.hidden_size)

    @skip_if_no_gpu
    def test_attention_mask_only(self, config):
        """Verify attn(hidden, attention_mask=...) still works without cache args."""
        attn = NemotronV3Attention(config).cuda()
        batch_size, seq_len = 2, 8
        hidden = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        # 2D padding mask (1=valid, 0=pad) — TE handles causality internally
        mask = torch.ones(batch_size, seq_len, device="cuda", dtype=torch.long)
        out = attn(hidden, attention_mask=mask)
        assert out.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_block_attention_no_cache_args(self, config, backend):
        """Verify block(hidden) still works for attention block without cache args."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend).cuda()
        hidden = torch.randn(2, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out = block(hidden)
        assert out.shape == (2, 8, config.hidden_size)

    def test_block_mlp_no_cache_args(self, config, backend):
        """Verify block(hidden) still works for mlp block without cache args."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)
        block = block.to(torch.bfloat16)
        hidden = torch.randn(2, 8, config.hidden_size, dtype=torch.bfloat16)
        out = block(hidden)
        assert out.shape == (2, 8, config.hidden_size)


class TestMambaMixerSeqIdxConstruction:
    """Regression tests for the seq_idx construction in NemotronV3Mamba2Mixer.

    The mamba kernel asserts ``seq_idx.shape == (batch_size, seqlen)``. When
    the model receives a (B, S, H) BSHD batch with a globally-cumulated
    ``cu_seqlens`` (derived in ``model.py`` from a 2D attention_mask), the
    pre-fix mixer produced ``seq_idx`` of shape ``(1, S)`` and crashed. After
    the fix the mixer must produce a per-row ``seq_idx`` of shape ``(B, S)``.
    """

    @pytest.fixture
    def config(self):
        # Small dimensions so CPU init is fast.
        return MockNemotronV3Config(
            hidden_size=64,
            mamba_num_heads=4,
            mamba_head_dim=16,
            n_groups=1,
            ssm_state_size=8,
            chunk_size=8,
            conv_kernel=4,
        )

    @staticmethod
    def _patch_mamba_kernel(monkeypatch, capture):
        """Install a fake ``mamba_split_conv1d_scan_combined`` that records
        the ``seq_idx`` kwarg and returns a correctly-shaped output tensor."""
        import sys
        import types

        def _fake_kernel(projected_states, *args, **kwargs):
            capture["seq_idx"] = kwargs.get("seq_idx", None)
            # Out shape matches the projected gate path: (B, S, intermediate_size)
            # after the outproj_weight matmul → (B, S, hidden_size).
            outproj_weight = kwargs["outproj_weight"]
            B, S, _ = projected_states.shape
            return projected_states.new_zeros((B, S, outproj_weight.shape[0]))

        # Stub the import chain so the deferred ``from mamba_ssm.ops.triton.
        # ssd_combined import mamba_split_conv1d_scan_combined`` resolves to
        # our fake without requiring the real package.
        for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        fake_ssd = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
        fake_ssd.mamba_split_conv1d_scan_combined = _fake_kernel
        monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton.ssd_combined", fake_ssd)

    def test_bshd_b_gt_1_with_cu_seqlens_yields_per_row_seq_idx(self, monkeypatch, config):
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        B, S = 2, 16
        hidden_states = torch.randn(B, S, config.hidden_size)
        # cu_seqlens as model.py derives it from a 2D attention_mask: a global
        # cumsum across the batch (one boundary per row).
        cu_seqlens = torch.tensor([0, S, 2 * S], dtype=torch.int32)

        capture: dict = {}
        self._patch_mamba_kernel(monkeypatch, capture)
        mixer.forward(hidden_states, cu_seqlens=cu_seqlens)

        seq_idx = capture["seq_idx"]
        assert seq_idx is not None, "mamba kernel did not receive a seq_idx"
        assert seq_idx.shape == (B, S), f"expected (B, S)=({B}, {S}), got {tuple(seq_idx.shape)}"
        assert seq_idx.dtype == torch.int32

    def test_bshd_b_eq_1_with_cu_seqlens_keeps_flat_construction(self, monkeypatch, config):
        """When B == 1 (THD flat layout), the legacy searchsorted construction
        still applies and should yield shape (1, S)."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        B, S = 1, 16
        hidden_states = torch.randn(B, S, config.hidden_size)
        # Two intra-row sub-sequences: boundaries at positions 8 and 16.
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

        capture: dict = {}
        self._patch_mamba_kernel(monkeypatch, capture)
        mixer.forward(hidden_states, cu_seqlens=cu_seqlens)

        seq_idx = capture["seq_idx"]
        assert seq_idx is not None
        assert seq_idx.shape == (1, S)
        # right=True so boundary positions map to the new sub-seq.
        assert seq_idx[0, :8].tolist() == [0] * 8
        assert seq_idx[0, 8:].tolist() == [1] * 8

    def test_cp_bshd_uses_sequence_transport_while_preserving_packed_seq_idx(self, monkeypatch, config):
        """BSHD packing metadata must not select the CP helper's 2D THD path."""
        import sys
        import types

        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)
        calls: dict[str, object] = {}

        class FakeCP:
            cp_size = 2
            n_groups_local = mixer.n_groups

            def pre_conv_ssm(self, tensor, *, cu_seqlens=None):
                calls["pre_cu_seqlens"] = cu_seqlens
                calls["pre_shape"] = tensor.shape
                return tensor

            def post_conv_ssm(self, tensor, *, cu_seqlens=None):
                calls["post_cu_seqlens"] = cu_seqlens
                return tensor

            @staticmethod
            def get_A_log(tensor):
                return tensor

            @staticmethod
            def get_dt_bias(tensor):
                return tensor

            @staticmethod
            def get_D(tensor):
                return tensor

            def get_conv1d_weight(self):
                return mixer.conv1d.weight.squeeze(1)

            def get_conv1d_bias(self):
                return mixer.conv1d.bias

        def fake_kernel(projected_states, *args, **kwargs):
            del args
            calls["seq_idx"] = kwargs["seq_idx"]
            return projected_states[..., : mixer.intermediate_size]

        for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        fake_ssd = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
        fake_ssd.mamba_split_conv1d_scan_combined = fake_kernel
        monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton.ssd_combined", fake_ssd)

        class GateIdentity(torch.nn.Module):
            def forward(self, tensor, gate=None):
                del gate
                return tensor

        mixer.norm = GateIdentity()
        mixer.cp = FakeCP()
        hidden_states = torch.randn(1, 8, config.hidden_size)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

        output = mixer(hidden_states, cu_seqlens=cu_seqlens)

        assert output.shape == hidden_states.shape
        assert calls["pre_shape"] == mixer.in_proj(hidden_states).shape
        assert calls["pre_cu_seqlens"] is None
        assert calls["post_cu_seqlens"] is None
        # The local BSHD shard has 8 positions, while CP=2 transport presents
        # 16 positions to the kernel. cu_seqlens describes 8 real packed tokens;
        # searchsorted assigns the trailing 8 CP-padding positions the sentinel
        # sequence ID immediately after the two real documents.
        assert calls["seq_idx"].tolist() == [[0] * 4 + [1] * 4 + [2] * 8]

    def test_bshd_b_gt_1_passes_through_upstream_seq_idx(self, monkeypatch, config):
        """If seq_idx is supplied upstream (e.g. via _packed_seq_ids), the
        construction path must NOT override it — neat-packing semantics."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        B, S = 2, 16
        hidden_states = torch.randn(B, S, config.hidden_size)
        # Pretend the wrapper has already set seq_idx from _packed_seq_ids.
        upstream = torch.tensor(
            [[0] * 8 + [1] * 8, [0] * 6 + [1] * 5 + [2] * 5],
            dtype=torch.int32,
        )

        capture: dict = {}
        self._patch_mamba_kernel(monkeypatch, capture)
        mixer.forward(hidden_states, cu_seqlens=torch.tensor([0, 16, 32], dtype=torch.int32), seq_idx=upstream)

        seq_idx = capture["seq_idx"]
        assert seq_idx is upstream, "upstream seq_idx must not be replaced by the construction path"
