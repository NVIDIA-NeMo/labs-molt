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

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.attention.utils import postprocess_output_for_attn, preprocess_args_and_kwargs_for_attn
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.layers import (
    Qwen3MoeAttention,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def config():
    cfg = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        decoder_sparse_step=1,
        moe_intermediate_size=64,
        num_experts=0,
        num_experts_per_tok=1,
        router_aux_loss_coef=0.0,
        attention_dropout=0.0,
    )
    cfg.head_dim = 16
    return cfg


@pytest.fixture
def sdpa_backend():
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
def te_backend():
    return BackendConfig(
        linear="torch",
        attn="te",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class TestPreprocessForAttn:
    def test_te_backend_without_mask_keeps_layout(self, te_backend):
        q = torch.randn(2, 4, 2, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        q_out, k_out, v_out, kwargs = preprocess_args_and_kwargs_for_attn(q, k, v, attention_mask=None, attn_impl=te_backend.attn)

        torch.testing.assert_close(q_out, q)
        torch.testing.assert_close(k_out, k)
        torch.testing.assert_close(v_out, v)
        assert kwargs == {"window_size": (-1, 0)}

    def test_te_backend_with_mask_builds_padding_kwargs(self, te_backend):
        q = torch.randn(1, 3, 2, 4)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

        _, _, _, kwargs = preprocess_args_and_kwargs_for_attn(q, k, v, attention_mask=attention_mask, attn_impl=te_backend.attn)

        assert kwargs["attn_mask_type"] == "padding_causal"
        assert kwargs["window_size"] == (-1, 0)
        mask = kwargs["attention_mask"]
        assert mask.shape == (1, 1, 1, 3)
        expected = attention_mask.logical_not().unsqueeze(1).unsqueeze(2)
        torch.testing.assert_close(mask, expected)

    def test_sdpa_backend_transposes_qkv(self, sdpa_backend):
        q = torch.randn(2, 5, 3, 6)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        q_out, k_out, v_out, kwargs = preprocess_args_and_kwargs_for_attn(q, k, v, attention_mask=None, attn_impl=sdpa_backend.attn)

        assert q_out.shape == (2, 3, 5, 6)
        assert k_out.shape == (2, 3, 5, 6)
        assert v_out.shape == (2, 3, 5, 6)
        assert kwargs["is_causal"] is True
        assert kwargs.get("attn_mask") is None
        assert set(kwargs.keys()) <= {"is_causal", "attn_mask"}


class TestPostprocessFromAttn:
    def test_sdpa_backend_transposes_back(self, sdpa_backend):
        x = torch.randn(2, 4, 6, 8)

        out = postprocess_output_for_attn(x, sdpa_backend.attn)

        assert out.shape == (2, 6, 4, 8)
        torch.testing.assert_close(out.transpose(1, 2), x)

    def test_other_backend_returns_input(self, te_backend):
        x = torch.randn(1, 2, 3, 4)
        out = postprocess_output_for_attn(x, te_backend.attn)
        torch.testing.assert_close(out, x)


class TestQwen3MoeAttention:
    def test_initialization_populates_projections(self, config, sdpa_backend):
        attention = Qwen3MoeAttention(config, sdpa_backend)

        assert attention.num_heads == config.num_attention_heads
        assert attention.num_kv_heads == config.num_key_value_heads
        assert attention.head_dim == config.head_dim
        assert attention.q_proj.in_features == config.hidden_size
        assert attention.q_proj.out_features == config.num_attention_heads * config.head_dim
        assert attention.k_proj.out_features == config.num_key_value_heads * config.head_dim
        assert attention.o_proj.in_features == config.num_attention_heads * config.head_dim
        assert attention.o_proj.out_features == config.hidden_size

    def test_forward_shape_is_preserved(self, config, sdpa_backend):
        attention = Qwen3MoeAttention(config, sdpa_backend)
        batch_size, seq_len = 2, 5
        hidden = torch.randn(batch_size, seq_len, config.hidden_size).to(torch.bfloat16)
        freqs_cis = torch.randn(batch_size, seq_len, config.head_dim)

        fake_attn = torch.zeros(batch_size, config.num_attention_heads, seq_len, config.head_dim)
        attention.attn_func = MagicMock(return_value=fake_attn.to(torch.bfloat16))

        with patch("nemo_automodel.components.models.qwen3_moe.layers.apply_rotary_emb_qk", side_effect=lambda q, k, *_, **__: (q, k)):
            out = attention(hidden, freqs_cis=freqs_cis)

        assert out.shape == (batch_size, seq_len, config.hidden_size)
        attention.attn_func.assert_called_once()

    def test_forward_passes_preprocessed_kwargs(self, config, sdpa_backend):
        attention = Qwen3MoeAttention(config, sdpa_backend)
        batch, seq_len = 1, 3
        hidden = torch.randn(batch, seq_len, config.hidden_size).to(torch.bfloat16)
        freqs_cis = torch.randn(batch, seq_len, config.head_dim)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)

        fake_attn = torch.zeros(batch, config.num_attention_heads, seq_len, config.head_dim).to(torch.bfloat16)
        attention.attn_func = MagicMock(return_value=fake_attn.to(torch.bfloat16))

        with patch("nemo_automodel.components.models.qwen3_moe.layers.apply_rotary_emb_qk", side_effect=lambda q, k, *_, **__: (q, k)):
            attention(hidden, freqs_cis=freqs_cis, attention_mask=attention_mask)

        _, kwargs = attention.attn_func.call_args
        assert kwargs.get("is_causal") is True

    def test_forward_applies_rotary_embedding(self, config, sdpa_backend):
        attention = Qwen3MoeAttention(config, sdpa_backend)
        batch_size, seq_len = 1, 2
        hidden = torch.randn(batch_size, seq_len, config.hidden_size).to(torch.bfloat16)
        freqs_cis = torch.randn(batch_size, seq_len, config.head_dim)

        attention.attn_func = MagicMock(
            return_value=torch.zeros(batch_size, config.num_attention_heads, seq_len, config.head_dim).to(torch.bfloat16)
        )

        with patch("nemo_automodel.components.models.qwen3_moe.layers.apply_rotary_emb_qk") as mock_rotary:
            mock_rotary.side_effect = lambda q, k, *_, **__: (q, k)
            attention(hidden, freqs_cis=freqs_cis)

        assert mock_rotary.call_count == 1

    def test_init_weights_resets_norms_and_linears(self, config, sdpa_backend):
        attention = Qwen3MoeAttention(config, sdpa_backend)

        with patch("torch.nn.init.trunc_normal_") as mock_trunc, \
            patch.object(attention.q_norm, "reset_parameters") as mock_q_reset, \
            patch.object(attention.k_norm, "reset_parameters") as mock_k_reset:
            attention.init_weights(torch.device("cpu"), init_std=0.05)

        assert mock_trunc.call_count == 4
        mock_q_reset.assert_called_once()
        mock_k_reset.assert_called_once()

    def test_forward_with_te_backend_supports_attention_mask(self, config, te_backend):
        batch, seq_len = 1, 3
        fake_out = torch.zeros(batch, seq_len, config.num_attention_heads, config.head_dim)
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=fake_out.to(torch.bfloat16))
        with patch(
            "nemo_automodel.components.models.qwen3_moe.layers.initialize_attn_module_and_func",
            return_value=(fake_module, fake_func),
        ):
            attention = Qwen3MoeAttention(config, te_backend)

        hidden = torch.randn(batch, seq_len, config.hidden_size).to(torch.bfloat16)
        freqs_cis = torch.randn(batch, seq_len, config.head_dim)
        attention_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

        with patch("nemo_automodel.components.models.qwen3_moe.layers.apply_rotary_emb_qk", side_effect=lambda q, k, *_, **__: (q, k)):
            attention(hidden, freqs_cis=freqs_cis, attention_mask=attention_mask)

        _, kwargs = attention.attn_func.call_args
        assert "attention_mask" in kwargs
        mask = kwargs["attention_mask"]
        assert mask.shape == (batch, 1, 1, seq_len)

    def test_softmax_scale_matches_head_dim(self, config, sdpa_backend):
        attention = Qwen3MoeAttention(config, sdpa_backend)
        with patch("torch.nn.functional.scaled_dot_product_attention", return_value=torch.zeros(1)) as mock_sdpa:
            dummy = torch.zeros(1)
            attention.attn_func(dummy, dummy, dummy)
            _, call_kwargs = mock_sdpa.call_args
            scale = call_kwargs.get("scale")
        assert scale is not None
        assert math.isclose(scale, config.head_dim ** -0.5, rel_tol=1e-6)
