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
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.moe.config import MoEConfig

# Over the default 5s budget on purpose: CUDA Mamba kernels compile on their first invocation.
# Reduce cold compiler startup before lowering this further.
pytestmark = pytest.mark.timeout(60)

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")


class MockNemotronV3Config:
    """Mock configuration for NemotronV3 model."""

    def __init__(self, **overrides):
        # Attention configuration
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 64
        self.hidden_size = 256
        self.attention_bias = False
        self.attention_dropout = 0.0

        # MLP/MoE configuration
        self.intermediate_size = 512
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"

        # Mamba configuration
        self.mamba_num_heads = 4
        self.mamba_head_dim = 32
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
        self.num_hidden_layers = 2
        self.vocab_size = 100
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False

        # Layer types for hybrid architecture - using only attention and mlp for tests
        self.layers_block_type = ["attention", "mlp"]

        # MoE configuration
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.moe_intermediate_size = 128
        self.norm_topk_prob = False
        self.moe_shared_expert_intermediate_size = 128

        # Apply overrides
        for key, value in overrides.items():
            setattr(self, key, value)

    def to_dict(self):
        return vars(self)


class TestNemotronV3Model:
    """Test NemotronV3Model base model."""

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

    def test_model_init(self, config, backend):
        """Test NemotronV3Model initialization."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert model.config == config
        assert model.backend == backend
        assert hasattr(model, "embed_tokens")
        assert hasattr(model, "layers")
        assert hasattr(model, "norm")

    def test_model_layers_count(self, config, backend):
        """Test that model has correct number of layers."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert len(model.layers) == config.num_hidden_layers

    def test_model_layer_types(self, config, backend):
        """Test that model layers have correct types."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert model.layers["0"].block_type == "attention"
        assert model.layers["1"].block_type == "mlp"

    def test_dense_model_builds_with_no_moe_config(self, backend):
        """Dense Nemotron-H (no 'moe' layers, expert fields absent) builds with
        moe_config=None and runs forward. Regression guard for issue #2004."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        config = MockNemotronV3Config(layers_block_type=["attention", "mlp"])
        # Real dense configs (e.g. NVIDIA-Nemotron-3-Nano-4B-BF16) omit the MoE fields.
        for attr in (
            "n_routed_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
            "moe_intermediate_size",
            "norm_topk_prob",
            "moe_shared_expert_intermediate_size",
        ):
            delattr(config, attr)

        model = NemotronV3Model(config, backend=backend).to(torch.bfloat16)
        assert model.moe_config is None
        assert len(model.layers) == config.num_hidden_layers

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = model(input_ids)
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_get_capabilities_moe_supports_ep(self):
        """MoE config (has experts) reports supports_ep=True."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        caps = NemotronHForCausalLM.get_capabilities(MockNemotronV3Config())
        assert caps.supports_ep is True

    def test_get_capabilities_dense_no_ep(self):
        """Dense config (no experts) reports supports_ep=False (PR #2670 review / #2004)."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        config = MockNemotronV3Config(layers_block_type=["attention", "mlp"])
        for attr in (
            "n_routed_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
            "moe_intermediate_size",
            "norm_topk_prob",
            "moe_shared_expert_intermediate_size",
        ):
            delattr(config, attr)
        caps = NemotronHForCausalLM.get_capabilities(config)
        assert caps.supports_ep is False
        assert caps.supports_cp is True
        assert caps.supports_pp is True
        assert caps.supports_mtp_cp is True
        assert caps.supports_mtp_cp_pp is False
        assert caps.supports_tp is False

    def test_model_embedding_dimensions(self, config, backend):
        """Test that embeddings have correct dimensions."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert model.embed_tokens.num_embeddings == config.vocab_size
        assert model.embed_tokens.embedding_dim == config.hidden_size

    def test_model_forward_shape(self, config, backend):
        """Test model forward pass produces correct shape."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_with_mask(self, config, backend):
        """Test model forward pass with attention mask."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = model(input_ids, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_with_causal_mask_mapping(self, config, backend):
        """Test model forward pass with causal mask mapping."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Create causal mask mapping
        causal_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), float("-inf")
        )
        causal_mask_mapping = {"full_attention": causal_mask}

        output = model(input_ids, causal_mask_mapping=causal_mask_mapping)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_with_inputs_embeds(self, config, backend):
        """Test model forward pass with inputs_embeds instead of input_ids."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = model(inputs_embeds=inputs_embeds)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_inputs_embeds_bypasses_embedding(self, config, backend):
        """Test that inputs_embeds bypasses the embedding layer."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        # Should work even with input_ids=None (the default)
        output = model(input_ids=None, inputs_embeds=inputs_embeds)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_inputs_embeds_takes_precedence(self, config, backend):
        """Test that inputs_embeds takes precedence over input_ids when both provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        # When both are provided, inputs_embeds should be used (input_ids ignored)
        output = model(input_ids, inputs_embeds=inputs_embeds)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_forward_no_input_ids_no_inputs_embeds_raises(self, config, backend):
        """Test that ValueError is raised when neither input_ids nor inputs_embeds is provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        with pytest.raises(ValueError, match="input_ids must be provided if inputs_embeds is not provided"):
            model(input_ids=None)

    def test_model_forward_inputs_embeds_with_mask(self, config, backend):
        """Test model forward pass with inputs_embeds and attention mask."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
        attention_mask = torch.ones(batch_size, seq_len)

        output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_model_moe_config_creation(self, config, backend):
        """Test that model creates MoE config correctly."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        # Add MoE layer
        config.layers_block_type = ["moe", "attention"]

        model = NemotronV3Model(config, backend=backend)

        assert model.moe_config is not None
        assert model.moe_config.n_routed_experts == config.n_routed_experts
        assert model.moe_config.n_activated_experts == config.num_experts_per_tok
        assert model.moe_config.expert_activation == "relu2"
        assert model.moe_config.shared_expert_activation == "relu2"

    def test_model_custom_moe_config(self, config, backend):
        """Test that model uses custom MoE config when provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        custom_moe_config = MoEConfig(
            n_routed_experts=16,
            n_shared_experts=2,
            n_activated_experts=4,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

        config.layers_block_type = ["moe", "attention"]

        model = NemotronV3Model(config, backend=backend, moe_config=custom_moe_config)

        assert model.moe_config == custom_moe_config
        assert model.moe_config.n_routed_experts == 16

    def test_model_dtype(self, config, backend):
        """Test that model uses correct dtype."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        # Embeddings should be bfloat16
        assert model.embed_tokens.weight.dtype == torch.bfloat16


class TestNemotronHForCausalLM:
    """Test NemotronHForCausalLM (causal LM wrapper)."""

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

    def test_causal_lm_init(self, config, backend):
        """Test NemotronHForCausalLM initialization."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)

        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")
        assert model.config == config

    def test_causal_lm_lm_head_shape(self, config, backend):
        """Test that lm_head has correct shape."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)

        assert model.lm_head.in_features == config.hidden_size
        assert model.lm_head.out_features == config.vocab_size

    def test_causal_lm_forward_shape(self, config, backend):
        """Test causal LM forward pass produces logits with correct shape."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_causal_lm_forward_logits_dtype(self, config, backend):
        """Test that logits preserve model dtype (no float32 upcast)."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        assert output.logits.dtype == torch.bfloat16

    def test_causal_lm_forward_with_inputs_embeds(self, config, backend):
        """Test causal LM forward pass with inputs_embeds."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = model(inputs_embeds=inputs_embeds)

        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)
        assert output.logits.dtype == torch.bfloat16

    def test_causal_lm_forward_no_input_ids_no_inputs_embeds_raises(self, config, backend):
        """Test that ValueError is raised when neither input_ids nor inputs_embeds is provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        with pytest.raises(ValueError, match="input_ids must be provided if inputs_embeds is not provided"):
            model()

    def _build_stub_inner_model(self, hidden):
        """Tiny ``nn.Module`` whose forward returns a fixed tensor.  Lets us
        replace ``NemotronHForCausalLM.model`` (an ``nn.Module``) without
        tripping ``nn.Module.__setattr__``'s child-module type check."""

        class _StubInner(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return hidden

        return _StubInner()

    def test_causal_lm_thd_inputs_embeds_does_not_double_unsqueeze(self, config, backend):
        """Regression test: in THD mode with ``inputs_embeds``-only inputs, the
        outer ``NemotronHForCausalLM.forward`` used to double-unsqueeze the
        logits to ``[1, 1, T, V]``. The inner ``NemotronHModel.forward`` already
        restores the batch dim (``squeezed_for_thd`` branch), so the outer must
        only re-add it when the inner returned 2D logits.

        We bypass the attention stack (which needs TE/GPU for THD shapes) by
        replacing ``model.model`` with a stub that returns a fixed 3D tensor —
        the same shape the real inner forward returns when it took the
        ``inputs_embeds`` → squeeze → unsqueeze round-trip.
        """
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        seq_len = 8
        # Stand in for the inner forward that unsqueezed back to [1, T, H].
        stub_hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
        model.model = self._build_stub_inner_model(stub_hidden)

        inputs_embeds = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
        max_seqlen = torch.tensor(seq_len, dtype=torch.int32)

        output = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

        assert output.logits.shape == (1, seq_len, config.vocab_size)
        assert output.logits.dim() == 3

    def test_causal_lm_thd_input_ids_unsqueezes_2d_logits(self, config, backend):
        """The original THD path (``input_ids`` only, no ``inputs_embeds``) is
        the one most callers use; the unsqueeze fix must not regress it. The
        inner forward returns ``[T, H]`` (2D, never went through the
        ``squeezed_for_thd`` round-trip because ``embed_tokens(input_ids[T])``
        is already 2D), so the outer still has to add the batch dim."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        seq_len = 8
        # Stand in for the inner forward that returned 2D hidden_states.
        stub_hidden = torch.randn(seq_len, config.hidden_size, dtype=torch.bfloat16)
        model.model = self._build_stub_inner_model(stub_hidden)

        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
        max_seqlen = torch.tensor(seq_len, dtype=torch.int32)

        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

        assert output.logits.shape == (1, seq_len, config.vocab_size)
        assert output.logits.dim() == 3

    def test_causal_lm_from_config(self, config, backend):
        """Test from_config classmethod."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM.from_config(config, backend=backend)

        assert isinstance(model, NemotronHForCausalLM)
        assert model.config == config

    def test_causal_lm_state_dict_adapter_disabled(self, config, backend):
        """Test that state_dict_adapter is not created when disabled."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        backend.enable_hf_state_dict_adapter = False
        model = NemotronHForCausalLM(config, backend=backend)

        assert not hasattr(model, "state_dict_adapter")

    def test_causal_lm_state_dict_adapter_enabled(self, config, backend):
        """Test that state_dict_adapter is created when enabled."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        backend.enable_hf_state_dict_adapter = True
        model = NemotronHForCausalLM(config, backend=backend)

        assert hasattr(model, "state_dict_adapter")

    def test_mamba_decay_params_stay_fp32_after_bf16_cast(self, config, backend):
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        config.layers_block_type = ["mamba", "attention"]
        model = NemotronHForCausalLM(config, backend=backend)
        expected = {}
        for name, param in model.named_parameters():
            if name.endswith(("A_log", "dt_bias", "D")):
                values = torch.linspace(0.00123, 0.00456, param.numel(), dtype=torch.float32).reshape_as(param)
                param.data.copy_(values)
                expected[name] = values

        assert expected, "Nemotron V3 Mamba layers should create decay parameters"
        assert all("._fp32_params." in name for name in expected)

        cast_model_to_dtype(model, torch.bfloat16)

        params = dict(model.named_parameters())
        for name, values in expected.items():
            assert params[name].dtype == torch.float32
            torch.testing.assert_close(params[name], values)
        assert model.lm_head.weight.dtype == torch.bfloat16

    def test_model_class_export(self):
        """Test that ModelClass is exported correctly."""
        from nemo_automodel.components.models.nemotron_v3.model import (
            ModelClass,
            NemotronHForCausalLM,
        )

        assert ModelClass is NemotronHForCausalLM

    def test_causal_lm_generation_mixin_in_mro(self, config, backend):
        """Test that GenerationMixin is in the MRO."""
        from transformers.generation import GenerationMixin

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        assert GenerationMixin in NemotronHForCausalLM.__mro__

    def test_causal_lm_is_stateful(self, config, backend):
        """Test that _is_stateful is True to prevent DynamicCache creation."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        assert NemotronHForCausalLM._is_stateful is True

    def test_causal_lm_has_generation_config(self, config, backend):
        """Test that model has generation_config set after __init__."""
        from transformers.generation import GenerationConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)

        assert hasattr(model, "generation_config")
        assert isinstance(model.generation_config, GenerationConfig)

    def test_causal_lm_generation_config_takes_stop_tokens_from_config(self, config, backend):
        """Seeded from the model config like PreTrainedModel, not a bare GenerationConfig()."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, bos_token_id=3, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend)

        assert model.generation_config.bos_token_id == 3
        assert model.generation_config.eos_token_id == 1
        assert model.generation_config.pad_token_id == 0

    @staticmethod
    def _tiny_nemotron_h_config():
        """A real NemotronHConfig small enough to build on CPU (one Mamba and one MLP layer)."""
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

        return NemotronHConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            hybrid_override_pattern="M-",
            num_attention_heads=2,
            num_key_value_heads=1,
            mamba_num_heads=2,
            mamba_head_dim=8,
            ssm_state_size=8,
            conv_kernel=4,
            n_groups=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )

    def test_causal_lm_from_pretrained_restores_and_exports_checkpoint_generation_config(self, backend, tmp_path):
        """The checkpoint's generation_config.json survives the load and the consolidated export.

        Hub Nemotron-3 checkpoints list their extra stop token and sampling defaults
        only there. Exporting a blank generation_config.json instead used to leave
        the exported model with no stop token, since that file beats config.json
        on reload.
        """
        from transformers import GenerationConfig

        from nemo_automodel.components.checkpoint.addons import _save_generated_hf_assets
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        checkpoint_dir = tmp_path / "checkpoint"
        self._tiny_nemotron_h_config().save_pretrained(checkpoint_dir)
        GenerationConfig(
            bos_token_id=1, eos_token_id=[2, 11], pad_token_id=0, do_sample=True, temperature=0.7
        ).save_pretrained(checkpoint_dir)

        model = NemotronHForCausalLM.from_pretrained(str(checkpoint_dir), backend=backend)

        assert model.generation_config.eos_token_id == [2, 11]
        assert model.generation_config.do_sample is True
        assert model.generation_config.temperature == 0.7

        export_dir = tmp_path / "export"
        export_dir.mkdir()
        _save_generated_hf_assets(model, None, str(export_dir), tokenizer=None, v4_compatible=False)
        exported = GenerationConfig.from_pretrained(export_dir)

        assert exported.eos_token_id == [2, 11]
        assert exported.do_sample is True

    def test_causal_lm_from_pretrained_without_generation_file_keeps_config_stop_tokens(self, backend, tmp_path):
        """With no generation_config.json in the checkpoint, the export still carries config.json's eos."""
        from transformers import GenerationConfig

        from nemo_automodel.components.checkpoint.addons import _save_generated_hf_assets
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        checkpoint_dir = tmp_path / "checkpoint"
        self._tiny_nemotron_h_config().save_pretrained(checkpoint_dir)

        model = NemotronHForCausalLM.from_pretrained(str(checkpoint_dir), backend=backend)

        assert model.generation_config.eos_token_id == 2

        export_dir = tmp_path / "export"
        export_dir.mkdir()
        _save_generated_hf_assets(model, None, str(export_dir), tokenizer=None, v4_compatible=False)

        assert GenerationConfig.from_pretrained(export_dir).eos_token_id == 2

    def test_causal_lm_forward_returns_causal_lm_output(self, config, backend):
        """Test that forward returns CausalLMOutputWithPast."""
        from transformers.modeling_outputs import CausalLMOutputWithPast

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        output = model(input_ids)

        assert isinstance(output, CausalLMOutputWithPast)
        assert output.past_key_values is None
        assert output.loss is None

    def test_causal_lm_forward_with_labels(self, config, backend):
        """Test that forward computes loss when labels are provided."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids, labels=labels)

        assert output.loss is not None
        assert output.loss.ndim == 0  # scalar loss

    def test_causal_lm_output_hidden_states(self, config, backend):
        """Test output_hidden_states parameter controls hidden state return."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Default: hidden_states should be None
        output = model(input_ids)
        assert output.hidden_states is None

        # Enabled: returns a tuple with one tensor of correct shape/dtype
        output = model(input_ids, output_hidden_states=True)
        assert isinstance(output.hidden_states, tuple)
        assert len(output.hidden_states) == 1
        assert output.hidden_states[0].shape == (batch_size, seq_len, config.hidden_size)
        assert output.hidden_states[0].dtype == torch.bfloat16

        # return_dict accepted without error (API compatibility)
        output = model(input_ids, output_hidden_states=True, return_dict=True)
        assert output.hidden_states is not None

    def test_causal_lm_hidden_states_config_and_override(self, config, backend):
        """Test config.output_hidden_states and explicit parameter override."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        config.output_hidden_states = True
        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        # Config enables hidden states when parameter is not passed
        output = model(input_ids)
        assert output.hidden_states is not None
        assert isinstance(output.hidden_states, tuple)

        # Explicit False overrides config
        output = model(input_ids, output_hidden_states=False)
        assert output.hidden_states is None

    def test_causal_lm_hidden_states_with_labels(self, config, backend):
        """Test hidden states returned alongside loss computation."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids, labels=labels, output_hidden_states=True)
        assert output.loss is not None
        assert output.loss.ndim == 0
        assert output.hidden_states is not None
        assert output.hidden_states[0].shape == (batch_size, seq_len, config.hidden_size)

    def test_causal_lm_prepare_inputs_for_generation(self, config, backend):
        """Test prepare_inputs_for_generation returns full sequence."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        model = NemotronHForCausalLM(config, backend=backend)

        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        model_inputs = model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask)

        # Full sequence is always forwarded (no cache slicing)
        assert model_inputs["input_ids"].shape == (batch_size, seq_len)
        assert (model_inputs["attention_mask"] == attention_mask).all()

    def test_causal_lm_prepare_inputs_for_generation_uses_inputs_embeds_length(self, config, backend):
        """Inputs-embeds prefill should derive cache positions from the embed sequence length."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend).to(torch.bfloat16)

        batch_size, prompt_len = 2, 6
        input_ids = torch.empty(batch_size, 0, dtype=torch.long)
        inputs_embeds = torch.randn(batch_size, prompt_len, config.hidden_size, dtype=torch.bfloat16)
        attention_mask = torch.ones(batch_size, prompt_len)

        model_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        assert model_inputs["input_ids"] is None
        assert model_inputs["inputs_embeds"] is inputs_embeds
        torch.testing.assert_close(
            model_inputs["cache_position"],
            torch.arange(prompt_len, device=model_inputs["cache_position"].device),
        )
        assert model_inputs["causal_mask_mapping"]["full_attention"].shape == (batch_size, 1, prompt_len, prompt_len)

    def test_causal_lm_prepare_inputs_for_generation_trims_decode_cache_position(self, config, backend):
        """Decode steps should keep only the last cache position for Nemotron-v3's Mamba cache update."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend).to(torch.bfloat16)

        cache = NemotronHybridCache(hf_config, batch_size=1, dtype=torch.bfloat16, device=torch.device("cpu"))
        cache.has_previous_state = True

        model_inputs = model.prepare_inputs_for_generation(
            input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long),
            attention_mask=torch.ones(1, 8),
            past_key_values=cache,
            cache_position=torch.arange(8),
        )

        assert model_inputs["input_ids"].shape == (1, 1)
        torch.testing.assert_close(model_inputs["cache_position"], torch.tensor([7]))

    def test_causal_lm_generate(self, config, backend):
        """Test that .generate() produces token sequences of the requested length."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        # Wrap the mock config in a real PretrainedConfig so that GenerationMixin's
        # _prepare_generation_config() can call _get_generation_parameters().
        hf_config = PretrainedConfig(
            is_encoder_decoder=False,
            eos_token_id=1,
            pad_token_id=0,
        )
        # Copy all mock attributes onto the HF config so the model is built correctly.
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend)
        model = model.to(torch.bfloat16)
        model.eval()
        experts_implementation = model.get_experts_implementation()
        assert experts_implementation == {"": hf_config._experts_implementation}
        model.set_experts_implementation(experts_implementation)

        batch_size, prompt_len = 1, 4
        max_new_tokens = 3
        input_ids = torch.randint(2, config.vocab_size, (batch_size, prompt_len))

        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        assert output_ids.shape[0] == batch_size
        assert output_ids.shape[1] >= prompt_len
        assert output_ids.shape[1] <= prompt_len + max_new_tokens

    def test_dense_causal_lm_build_and_generate(self, config, backend):
        """Dense Nemotron-H end-to-end: the CausalLM builds with moe_config=None and no
        MTP, and .generate() produces tokens. Covers the issue #2004 inference path."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)
        # Make it dense: no 'moe' layers, and drop the MoE/expert fields entirely
        # (real dense configs like NVIDIA-Nemotron-3-Nano-4B-BF16 omit them).
        hf_config.layers_block_type = ["attention", "mlp"]
        for attr in (
            "n_routed_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
            "moe_intermediate_size",
            "norm_topk_prob",
            "moe_shared_expert_intermediate_size",
        ):
            if hasattr(hf_config, attr):
                delattr(hf_config, attr)

        model = NemotronHForCausalLM(hf_config, backend=backend).to(torch.bfloat16)
        model.eval()
        assert model.model.moe_config is None
        assert model.mtp is None

        batch_size, prompt_len, max_new_tokens = 1, 4, 3
        input_ids = torch.randint(2, config.vocab_size, (batch_size, prompt_len))
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        assert output_ids.shape[0] == batch_size
        assert prompt_len <= output_ids.shape[1] <= prompt_len + max_new_tokens


class TestNemotronV3KVCache:
    """Test NemotronHybridCache and cached generation."""

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

    def test_cache_creation(self, config):
        """Test that NemotronHybridCache initializes with correct shapes."""
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache

        batch_size = 2
        cache = NemotronHybridCache(config, batch_size, torch.bfloat16, torch.device("cpu"))

        assert cache.has_previous_state is False
        assert len(cache.conv_states) == config.num_hidden_layers
        assert len(cache.ssm_states) == config.num_hidden_layers
        assert cache.key_cache == []
        assert cache.value_cache == []

        # Check conv_state shape
        conv_dim = config.mamba_num_heads * config.mamba_head_dim + 2 * config.n_groups * config.ssm_state_size
        assert cache.conv_states[0].shape == (batch_size, conv_dim, config.conv_kernel)

        # Check ssm_state shape
        assert cache.ssm_states[0].shape == (
            batch_size,
            config.mamba_num_heads,
            config.mamba_head_dim,
            config.ssm_state_size,
        )

    def test_cache_kv_update(self, config):
        """Test attention K/V accumulation in cache."""
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache

        batch_size = 2
        cache = NemotronHybridCache(config, batch_size, torch.bfloat16, torch.device("cpu"))

        # Simulate first update (prefill)
        layer_idx = 0
        k1 = torch.randn(batch_size, config.num_key_value_heads, 4, config.head_dim)
        v1 = torch.randn(batch_size, config.num_key_value_heads, 4, config.head_dim)
        k_out, v_out = cache.update(k1, v1, layer_idx)
        assert k_out.shape == (batch_size, config.num_key_value_heads, 4, config.head_dim)
        assert cache.get_seq_length(layer_idx) == 4

        # Simulate second update (decode)
        k2 = torch.randn(batch_size, config.num_key_value_heads, 1, config.head_dim)
        v2 = torch.randn(batch_size, config.num_key_value_heads, 1, config.head_dim)
        k_out, v_out = cache.update(k2, v2, layer_idx)
        assert k_out.shape == (batch_size, config.num_key_value_heads, 5, config.head_dim)
        assert cache.get_seq_length(layer_idx) == 5

    def test_attention_with_cache_prefill_decode(self, config, backend):
        """Test correct shapes through prefill -> decode with attention cache."""
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Attention

        attn = NemotronV3Attention(config, backend=backend).to(torch.bfloat16)
        batch_size, prompt_len = 2, 4
        cache = NemotronHybridCache(config, batch_size, torch.bfloat16, torch.device("cpu"))

        # Prefill
        hidden = torch.randn(batch_size, prompt_len, config.hidden_size, dtype=torch.bfloat16)
        out = attn(hidden, past_key_values=cache, layer_idx=0)
        assert out.shape == (batch_size, prompt_len, config.hidden_size)
        assert cache.get_seq_length(0) == prompt_len

        # Decode
        hidden_decode = torch.randn(batch_size, 1, config.hidden_size, dtype=torch.bfloat16)
        out = attn(hidden_decode, past_key_values=cache, layer_idx=0)
        assert out.shape == (batch_size, 1, config.hidden_size)
        assert cache.get_seq_length(0) == prompt_len + 1

    def test_generate_with_cache(self, config, backend):
        """End-to-end .generate() with cache (attention+mlp config, CPU)."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend)
        model = model.to(torch.bfloat16)
        model.eval()

        batch_size, prompt_len, max_new_tokens = 1, 4, 3
        input_ids = torch.randint(2, config.vocab_size, (batch_size, prompt_len))

        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        assert output_ids.shape[0] == batch_size
        assert output_ids.shape[1] >= prompt_len
        assert output_ids.shape[1] <= prompt_len + max_new_tokens

    def test_generate_cache_vs_no_cache_deterministic(self, config, backend):
        """Verify greedy decode output matches between cached and uncached generation."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend)
        model = model.to(torch.bfloat16)
        model.eval()

        batch_size, prompt_len, max_new_tokens = 1, 4, 5
        input_ids = torch.randint(2, config.vocab_size, (batch_size, prompt_len))

        # Cached generation (default path via prepare_inputs_for_generation)
        output_cached = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        # Full-recompute reference: run the model manually step by step
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(generated, use_cache=False)
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == hf_config.eos_token_id:
                    break

        # Both should produce the same tokens
        min_len = min(output_cached.shape[1], generated.shape[1])
        assert torch.equal(output_cached[:, :min_len], generated[:, :min_len])

    def test_cache_get_seq_length_empty(self, config):
        """Test get_seq_length returns 0 when cache is empty."""
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache

        cache = NemotronHybridCache(config, 1, torch.bfloat16, torch.device("cpu"))
        assert cache.get_seq_length() == 0

    def test_cache_reorder(self, config):
        """Test reorder_cache for beam search."""
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache

        batch_size = 3
        cache = NemotronHybridCache(config, batch_size, torch.float32, torch.device("cpu"))

        # Add some KV data
        layer_idx = 0
        k = torch.randn(batch_size, config.num_key_value_heads, 2, config.head_dim)
        v = torch.randn(batch_size, config.num_key_value_heads, 2, config.head_dim)
        cache.update(k, v, layer_idx)

        # Set distinctive conv states
        cache.conv_states[0] = (
            torch.arange(batch_size).float().view(batch_size, 1, 1).expand_as(cache.conv_states[0]).clone()
        )

        # Reorder: swap batch elements
        beam_idx = torch.tensor([2, 0, 1])
        cache.reorder_cache(beam_idx)

        assert cache.key_cache[0].shape[0] == batch_size
        # Verify conv_states were reordered
        assert cache.conv_states[0][0, 0, 0].item() == 2.0
        assert cache.conv_states[0][1, 0, 0].item() == 0.0
        assert cache.conv_states[0][2, 0, 0].item() == 1.0


try:
    import mamba_ssm  # noqa: F401

    _has_mamba_ssm = True
except ImportError:
    _has_mamba_ssm = False

skip_if_no_mamba = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_mamba_ssm, reason="CUDA and mamba_ssm required for Mamba cache tests"
)


class TestNemotronV3MambaCacheGPU:
    """Test Mamba layer cache paths on GPU (prefill + decode)."""

    @pytest.fixture
    def config(self):
        # mamba_num_heads=8 ensures projection_size (552) is a multiple of 8,
        # which avoids causal_conv1d stride-alignment errors when seq_len==1
        # produces degenerate strides from in_proj's .split() slices.
        return MockNemotronV3Config(
            layers_block_type=["mamba", "attention"],
            num_hidden_layers=2,
            mamba_num_heads=8,
        )

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

    @skip_if_no_mamba
    def test_mamba_block_training_path(self, config, backend):
        """Test Mamba block forward without cache (training path A)."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block

        config.layers_block_type = ["mamba"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)
        block = block.to(torch.bfloat16).cuda()

        batch_size, seq_len = 2, 8
        hidden = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        out = block(hidden)
        assert out.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_mamba
    def test_mamba_mixer_prefill_decode(self, config, backend):
        """Test Mamba mixer prefill (path B) then decode (path C) with cache."""
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0).to(torch.bfloat16).cuda()

        batch_size, prompt_len = 2, 8
        cache = NemotronHybridCache(config, batch_size, torch.bfloat16, torch.device("cuda"))
        cache_position = torch.arange(prompt_len, device="cuda")

        # Prefill (path B)
        hidden = torch.randn(batch_size, prompt_len, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        out = mixer(hidden, past_key_values=cache, cache_position=cache_position)
        assert out.shape == (batch_size, prompt_len, config.hidden_size)

        # Verify SSM state was stored (non-zero after prefill)
        assert cache.ssm_states[0].abs().sum() > 0

        # Decode (path C)
        cache.has_previous_state = True
        hidden_decode = torch.randn(batch_size, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        cache_position_decode = torch.tensor([prompt_len], device="cuda")
        out_decode = mixer(hidden_decode, past_key_values=cache, cache_position=cache_position_decode)
        assert out_decode.shape == (batch_size, 1, config.hidden_size)

    @skip_if_no_mamba
    def test_hybrid_model_with_mamba_generate(self, config, backend):
        """End-to-end .generate() with hybrid mamba+attention config on GPU."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend)
        model = model.to(torch.bfloat16).cuda()
        model.eval()

        batch_size, prompt_len, max_new_tokens = 1, 4, 3
        input_ids = torch.randint(2, config.vocab_size, (batch_size, prompt_len), device="cuda")

        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        assert output_ids.shape[0] == batch_size
        assert output_ids.shape[1] >= prompt_len
        assert output_ids.shape[1] <= prompt_len + max_new_tokens

    # @skip_if_no_mamba
    @pytest.mark.skip
    def test_hybrid_model_generate_with_inputs_embeds_matches_manual_decode(self, config, backend):
        """Cached generate(inputs_embeds=...) should match full-recompute decoding."""
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend)
        model = model.to(torch.bfloat16).cuda()
        model.eval()

        batch_size, prompt_len, max_new_tokens = 1, 4, 5
        input_ids = torch.randint(2, config.vocab_size, (batch_size, prompt_len), device="cuda")
        inputs_embeds = model.model.embed_tokens(input_ids).to(torch.bfloat16)
        attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device="cuda")

        output_cached = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(generated, use_cache=False)
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == hf_config.eos_token_id:
                    break

        expected_new_tokens = generated[:, prompt_len:]
        min_len = min(output_cached.shape[1], expected_new_tokens.shape[1])
        assert torch.equal(output_cached[:, :min_len], expected_new_tokens[:, :min_len])

    @skip_if_no_mamba
    def test_hybrid_mamba_cache_deterministic(self, config, backend):
        """Verify cached generation is deterministic for hybrid mamba+attention.

        Note: we do NOT compare cached vs uncached (use_cache=False) because
        the mamba mixer uses different CUDA kernels for the cached path
        (causal_conv1d + selective_state_update) vs the uncached path
        (mamba_split_conv1d_scan_combined).  These are mathematically
        equivalent but not bit-identical in bf16, causing token divergence.
        """
        from transformers import PretrainedConfig

        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        hf_config = PretrainedConfig(is_encoder_decoder=False, eos_token_id=1, pad_token_id=0)
        for attr, val in vars(config).items():
            setattr(hf_config, attr, val)

        model = NemotronHForCausalLM(hf_config, backend=backend)
        model = model.to(torch.bfloat16).cuda()
        model.eval()

        batch_size, prompt_len, max_new_tokens = 1, 4, 5
        input_ids = torch.randint(2, config.vocab_size, (batch_size, prompt_len), device="cuda")

        # Run generate twice — both should produce identical tokens
        output1 = model.generate(input_ids.clone(), max_new_tokens=max_new_tokens, do_sample=False)
        output2 = model.generate(input_ids.clone(), max_new_tokens=max_new_tokens, do_sample=False)

        assert torch.equal(output1, output2)


class TestNemotronV3ModelWithMoE:
    """Test NemotronV3Model with MoE layers."""

    @pytest.fixture
    def config(self):
        config = MockNemotronV3Config()
        config.layers_block_type = ["moe", "attention"]
        return config

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            dispatcher="torch",
            fake_balanced_gate=True,  # Use fake balanced gate for deterministic testing
            enable_hf_state_dict_adapter=False,
        )

    def test_moe_model_init(self, config, backend):
        """Test model with MoE layer initializes correctly."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        # First layer should be MoE
        assert model.layers["0"].block_type == "moe"
        assert hasattr(model.layers["0"].mixer, "experts")
        assert hasattr(model.layers["0"].mixer, "shared_experts")

    def test_e_score_correction_bias_stays_fp32_after_dtype_cast(self, config, backend):
        """Nemotron v3 router correction bias must stay fp32 under bf16 storage casts."""
        from nemo_automodel.components.models.common.utils import cast_model_to_dtype
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM, NemotronV3Model

        backend = BackendConfig(
            linear=backend.linear,
            attn=backend.attn,
            rms_norm=backend.rms_norm,
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )
        base_model = NemotronV3Model(config, backend=backend)
        original_bias = torch.tensor([1.001, -2.003, 0.3333, 17.125], dtype=torch.float32)
        base_model.layers["0"].mixer.gate.e_score_correction_bias.copy_(original_bias)
        cast_model_to_dtype(base_model, torch.bfloat16)

        base_gate = base_model.layers["0"].mixer.gate
        assert base_model.embed_tokens.weight.dtype == torch.bfloat16
        assert base_model.layers["0"].mixer.experts.gate_and_up_projs.dtype == torch.bfloat16
        assert base_gate.e_score_correction_bias.dtype == torch.float32
        assert torch.equal(base_gate.e_score_correction_bias, original_bias)

        model = NemotronHForCausalLM(config, backend=backend)
        model.model.layers["0"].mixer.gate.e_score_correction_bias.copy_(original_bias)
        cast_model_to_dtype(model, torch.bfloat16)

        gate = model.model.layers["0"].mixer.gate
        state_dict = model.state_dict()
        assert model.model.embed_tokens.weight.dtype == torch.bfloat16
        assert model.model.layers["0"].mixer.experts.gate_and_up_projs.dtype == torch.bfloat16
        assert gate.e_score_correction_bias.dtype == torch.float32
        assert torch.equal(gate.e_score_correction_bias, original_bias)
        assert state_dict["model.layers.0.mixer.gate.e_score_correction_bias"].dtype == torch.float32
        assert torch.equal(state_dict["model.layers.0.mixer.gate.e_score_correction_bias"], original_bias)

    @skip_if_no_gpu
    def test_moe_model_forward(self, config, backend):
        """Test MoE model forward pass."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)
        model = model.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_moe_uses_relu2_activation(self, config, backend):
        """Test that MoE experts use relu2 activation."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        assert model.moe_config.expert_activation == "relu2"

        # Experts should use non-gated activation (up_projs instead of gate_and_up_projs)
        experts = model.layers["0"].mixer.experts
        # Non-gated experts have different weight shapes
        # gate_and_up_projs for non-gated has shape [n_experts, dim, inter_dim] instead of [n_experts, dim, 2*inter_dim]
        assert experts.gate_and_up_projs.shape[2] == config.moe_intermediate_size

    def test_moe_shared_experts_relu2(self, config, backend):
        """Test that shared experts use relu2 activation."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model

        model = NemotronV3Model(config, backend=backend)

        shared_experts = model.layers["0"].mixer.shared_experts

        # Shared experts should not have gate_proj (relu2 is non-gated)
        assert shared_experts.gate_proj is None
        assert shared_experts.up_proj is not None
