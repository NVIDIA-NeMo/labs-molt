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

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.model import Block, Qwen3MoeForCausalLM, Qwen3MoeModel
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, MoE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def qwen_config():
    return Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=5000.0,
        router_aux_loss_coef=0.01,
        use_sliding_window=False,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.01,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
    )


class TestBlock:
    def test_block_initializes_moe_by_default(self, qwen_config, moe_config, backend_config):
        block = Block(layer_idx=0, config=qwen_config, moe_config=moe_config, backend=backend_config)

        assert isinstance(block.self_attn, object)
        assert isinstance(block.mlp, MoE)
        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")

    def test_block_non_moe_layer_uses_mlp(self, qwen_config, backend_config):
        qwen_config.num_experts = 0
        block = Block(layer_idx=0, config=qwen_config, moe_config=MagicMock(), backend=backend_config)

        assert isinstance(block.mlp, MLP)

    def test_forward_pass_calls_attention_and_mlp(self, qwen_config, backend_config, device):
        block = Block(layer_idx=0, config=qwen_config, moe_config=magic_moe_config(qwen_config), backend=backend_config)
        block = block.to(device)

        batch, seq_len = 2, 4
        x = torch.randn(batch, seq_len, qwen_config.hidden_size, device=device)
        freqs_cis = torch.randn(batch, seq_len, qwen_config.head_dim, device=device)

        with (
            patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn,
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp,
        ):
            out = block(x, freqs_cis=freqs_cis)

        assert out.shape == x.shape
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()

    def test_forward_builds_padding_mask_from_attention(self, qwen_config, backend_config, device):
        block = Block(layer_idx=0, config=qwen_config, moe_config=magic_moe_config(qwen_config), backend=backend_config)
        block = block.to(device)

        x = torch.randn(1, 3, qwen_config.hidden_size, device=device)
        freqs_cis = torch.randn(1, 3, qwen_config.head_dim, device=device)
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool, device=device)

        with (
            patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn,
            patch.object(block, "_mlp", return_value=torch.zeros_like(x)) as mock_mlp,
        ):
            block(x, freqs_cis=freqs_cis, attention_mask=attention_mask)

        mock_attn.assert_called_once()
        _, kwargs = mock_mlp.call_args
        padding_mask = kwargs.get("padding_mask")
        assert padding_mask is not None
        torch.testing.assert_close(padding_mask, attention_mask.logical_not())

    def test_mlp_wrapper_handles_mlp_instance(self, qwen_config, backend_config):
        block = Block(layer_idx=0, config=qwen_config, moe_config=magic_moe_config(qwen_config), backend=backend_config)
        block.mlp = MLP(dim=qwen_config.hidden_size, inter_dim=qwen_config.intermediate_size, backend="torch")
        x = torch.randn(2, 4, qwen_config.hidden_size).to(torch.bfloat16)

        out = block._mlp(x, padding_mask=None)

        assert out.shape == x.shape

    def test_init_weights_resets_sublayers(self, qwen_config, backend_config):
        block = Block(layer_idx=0, config=qwen_config, moe_config=magic_moe_config(qwen_config), backend=backend_config)

        with (
            patch.object(block.input_layernorm, "reset_parameters") as mock_in,
            patch.object(block.post_attention_layernorm, "reset_parameters") as mock_post,
            patch.object(block.self_attn, "init_weights") as mock_attn,
            patch.object(block.mlp, "init_weights") as mock_mlp,
        ):
            block.init_weights(torch.device("cpu"))

        mock_in.assert_called_once()
        mock_post.assert_called_once()
        mock_attn.assert_called_once()
        mock_mlp.assert_called_once()


class TestQwen3MoeModel:
    def test_model_initialization_sets_components(self, qwen_config, backend_config):
        model = Qwen3MoeModel(qwen_config, backend=backend_config)

        assert model.config == qwen_config
        assert model.backend == backend_config
        assert len(model.layers) == qwen_config.num_hidden_layers
        assert model.embed_tokens.num_embeddings == qwen_config.vocab_size
        assert model.rotary_emb.head_dim == qwen_config.head_dim

    def test_forward_runs_all_layers(self, qwen_config, backend_config):
        model = Qwen3MoeModel(qwen_config, backend=backend_config)

        batch, seq_len = 2, 5
        input_ids = torch.randint(0, qwen_config.vocab_size, (batch, seq_len))
        freqs_mock = MagicMock(return_value=(1.0, torch.ones(qwen_config.head_dim // 2)))

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq", freqs_mock):
            with patch.object(
                Block, "forward", side_effect=lambda *_, **__: torch.randn(batch, seq_len, qwen_config.hidden_size)
            ) as mock_block:
                out = model(input_ids)

        assert out.shape == (batch, seq_len, qwen_config.hidden_size)
        assert mock_block.call_count == qwen_config.num_hidden_layers

    def test_forward_accepts_position_ids(self, qwen_config, backend_config):
        model = Qwen3MoeModel(qwen_config, backend=backend_config)
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, qwen_config.vocab_size, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0)

        with patch.object(
            model.rotary_emb,
            "_compute_concentration_and_inv_freq",
            return_value=(1.0, torch.ones(qwen_config.head_dim // 2)),
        ):
            with patch.object(Block, "forward", return_value=torch.zeros(batch, seq_len, qwen_config.hidden_size)):
                out = model(input_ids, position_ids=position_ids)

        assert out.shape == (batch, seq_len, qwen_config.hidden_size)

    def test_init_weights_updates_embeddings_and_layers(self, qwen_config, backend_config):
        model = Qwen3MoeModel(qwen_config, backend=backend_config)
        original = model.embed_tokens.weight.clone()

        with (
            patch.object(model.norm, "reset_parameters") as mock_norm,
            patch.object(Block, "init_weights") as mock_layer_init,
        ):
            model.init_weights(torch.device("cpu"))

        mock_norm.assert_called_once()
        assert not torch.equal(model.embed_tokens.weight, original)
        assert mock_layer_init.call_count == qwen_config.num_hidden_layers


class TestQwen3MoeForCausalLM:
    def test_forward_returns_logits(self, qwen_config, backend_config, device):
        model = Qwen3MoeForCausalLM(qwen_config, backend=backend_config)
        model = model.to(device)

        batch, seq_len = 2, 6
        input_ids = torch.randint(0, qwen_config.vocab_size, (batch, seq_len), device=device)

        with patch.object(
            model.model,
            "forward",
            return_value=torch.randn(batch, seq_len, qwen_config.hidden_size, device=device).to(torch.bfloat16),
        ):
            out = model(input_ids)

        logits = getattr(out, "logits", out)
        assert logits.shape == (batch, seq_len, qwen_config.vocab_size)

    def test_initialize_weights_invokes_submodules(self, qwen_config, backend_config):
        model = Qwen3MoeForCausalLM(qwen_config, backend=backend_config)
        original = model.lm_head.weight.clone()

        with patch.object(model.model, "init_weights") as mock_init:
            model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        mock_init.assert_called_once()
        assert not torch.equal(model.lm_head.weight, original)
        assert model.lm_head.weight.dtype == torch.float32

    def test_state_dict_adapter_created_when_enabled(self, qwen_config, backend_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = Qwen3MoeForCausalLM(qwen_config, backend=backend_config)

        assert hasattr(model, "state_dict_adapter")


def magic_moe_config(config: Qwen3MoeConfig) -> MoEConfig:
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=0,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=config.router_aux_loss_coef,
        norm_topk_prob=config.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
    )


class TestQwen3MoeModelFromPretrainedAndExport:
    def test_from_pretrained_classmethod(self):
        """Ensure classmethod from_pretrained builds config then delegates to from_config."""
        cfg = Qwen3MoeConfig(
            vocab_size=128,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            intermediate_size=128,
            head_dim=16,
            num_experts=2,
            num_experts_per_tok=1,
        )

        with patch(
            "transformers.models.qwen3_moe.configuration_qwen3_moe.Qwen3MoeConfig.from_pretrained"
        ) as mock_from_pretrained:
            mock_from_pretrained.return_value = cfg

            with patch.object(
                Qwen3MoeForCausalLM, "from_config", wraps=Qwen3MoeForCausalLM.from_config
            ) as mock_from_config:
                model = Qwen3MoeForCausalLM.from_pretrained("qwen3/model")
                assert isinstance(model, Qwen3MoeForCausalLM)
                mock_from_pretrained.assert_called_once_with("qwen3/model")
                called_cfg = mock_from_config.call_args[0][0]
                assert called_cfg is cfg

    def test_modelclass_export_exists(self):
        """Ensure ModelClass pointer is defined and points to class."""
        from nemo_automodel.components.models.qwen3_moe import model as qwen_mod

        assert hasattr(qwen_mod, "ModelClass")
        assert qwen_mod.ModelClass is Qwen3MoeForCausalLM
