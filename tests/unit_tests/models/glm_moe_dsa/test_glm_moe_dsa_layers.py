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

from unittest.mock import patch

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa.layers import (
    GlmMoeDsaIndexer,
    GlmMoeDsaMLA,
    _apply_index_rope_half_split,
    _rotate_activation,
    _to_additive_key_mask,
)


@pytest.fixture
def config():
    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        qk_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        mlp_layer_types=["dense", "dense"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        torch_dtype="float32",
    )


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


def _freqs(batch: int, seq_len: int, rope_dim: int) -> torch.Tensor:
    angles = torch.zeros(batch, seq_len, rope_dim // 2, dtype=torch.float32)
    return torch.polar(torch.ones_like(angles), angles)


def test_apply_index_rope_half_split_bshd_matches_rotate_half_formula():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    theta = torch.tensor([[[torch.pi / 2, 0.0]]])
    freqs_cis = torch.polar(torch.ones_like(theta), theta)

    out = _apply_index_rope_half_split(x, freqs_cis, "bshd")

    expected = torch.tensor([[[[-3.0, 2.0, 1.0, 4.0]]]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_apply_index_rope_half_split_thd_matches_rotate_half_formula():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    theta = torch.tensor([[torch.pi / 2, 0.0]])
    freqs_cis = torch.polar(torch.ones_like(theta), theta)

    out = _apply_index_rope_half_split(x, freqs_cis, "thd")

    expected = torch.tensor([[[-3.0, 2.0, 1.0, 4.0]]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_to_additive_key_mask_converts_bool_and_keep_masks_without_inf():
    neg = torch.finfo(torch.bfloat16).min

    bool_mask = torch.tensor([[True, False]])
    bool_out = _to_additive_key_mask(bool_mask, torch.bfloat16)
    torch.testing.assert_close(bool_out, torch.tensor([[0.0, neg]], dtype=torch.bfloat16))

    keep_mask = torch.tensor([[1, 0]])
    keep_out = _to_additive_key_mask(keep_mask, torch.bfloat16)
    torch.testing.assert_close(keep_out, torch.tensor([[0.0, neg]], dtype=torch.bfloat16))
    assert not torch.isinf(keep_out).any()


def test_to_additive_key_mask_preserves_existing_additive_mask():
    additive = torch.tensor([[0.0, -42.0]])
    out = _to_additive_key_mask(additive, torch.float32)
    torch.testing.assert_close(out, additive)


def test_rotate_activation_converts_to_bfloat16():
    x = torch.randn(2, 8, dtype=torch.float32)

    out = _rotate_activation(x)

    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16


class TestGlmMoeDsaIndexer:
    def test_initialization_uses_reference_shapes_and_layernorm_eps(self, config, sdpa_backend):
        indexer = GlmMoeDsaIndexer(config, sdpa_backend)

        assert indexer.num_heads == config.index_n_heads
        assert indexer.head_dim == config.index_head_dim
        assert indexer.qk_nope_head_dim == config.index_head_dim - config.qk_rope_head_dim
        assert isinstance(indexer.k_norm, torch.nn.LayerNorm)
        assert indexer.k_norm.eps == 1e-6

    def test_forward_bshd_masks_future_and_padding_keys(self, config, sdpa_backend):
        indexer = GlmMoeDsaIndexer(config, sdpa_backend)
        seq_len = 4
        x = torch.randn(1, seq_len, config.hidden_size)
        q_resid = torch.randn(1, seq_len, config.q_lora_rank)
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

        topk = indexer(x, q_resid, _freqs(1, seq_len, config.qk_rope_head_dim), attention_mask=attention_mask)

        assert topk.shape == (1, seq_len, config.index_topk)
        assert topk[0, 0, 0] == 0
        assert torch.all(topk[0, 1] <= 1)
        assert torch.all(topk[0, 2] <= 2)
        assert torch.all(topk[0, 2] != seq_len - 1)

    def test_forward_thd_clamps_topk_to_sequence_length(self, config, sdpa_backend):
        config.index_topk = 16
        indexer = GlmMoeDsaIndexer(config, sdpa_backend)
        seq_len = 3
        x = torch.randn(seq_len, config.hidden_size)
        q_resid = torch.randn(seq_len, config.q_lora_rank)
        freqs_cis = _freqs(1, seq_len, config.qk_rope_head_dim).squeeze(0)

        topk = indexer(x, q_resid, freqs_cis)

        assert topk.shape == (seq_len, seq_len)
        assert topk[0, 0] == 0

    def test_init_weights_resets_projections_and_norm(self, config, sdpa_backend):
        indexer = GlmMoeDsaIndexer(config, sdpa_backend)

        with (
            patch("torch.nn.init.trunc_normal_") as trunc_normal,
            patch.object(indexer.k_norm, "reset_parameters") as reset_norm,
        ):
            indexer.init_weights(init_std=0.01)

        assert trunc_normal.call_count == 3
        reset_norm.assert_called_once()


class TestGlmMoeDsaMLA:
    def test_initialization_uses_upstream_latent_norm_epsilon(self, config, sdpa_backend):
        config.rms_norm_eps = 1e-5

        mla = GlmMoeDsaMLA(config, sdpa_backend)

        assert mla.q_a_layernorm.eps == 1e-6
        assert mla.kv_a_layernorm.eps == 1e-6

    def test_initialization_can_skip_local_indexer_for_shared_layers(self, config, sdpa_backend):
        full = GlmMoeDsaMLA(config, sdpa_backend, skip_topk=False)
        shared = GlmMoeDsaMLA(config, sdpa_backend, skip_topk=True)

        assert isinstance(full.indexer, GlmMoeDsaIndexer)
        assert shared.indexer is None

    def test_sparse_mask_bshd_is_causal_padding_aware_and_finite(self, config, sdpa_backend):
        mla = GlmMoeDsaMLA(config, sdpa_backend)
        topk_indices = torch.tensor([[[0, 1], [0, 1], [1, 2], [2, 3]]])
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        neg = torch.finfo(torch.bfloat16).min

        sparse_mask = mla._build_sparse_mask(
            topk_indices,
            seq_len=4,
            qkv_format="bshd",
            bsz=1,
            n_heads=1,
            dtype=torch.bfloat16,
            attention_mask=attention_mask,
            union_across_batches=False,
        )

        assert sparse_mask.shape == (1, 1, 4, 4)
        assert not torch.isinf(sparse_mask).any()
        assert sparse_mask[0, 0, 2, 2] == 0
        assert sparse_mask[0, 0, 2, 3] == neg
        assert sparse_mask[0, 0, 0, 1] == neg

    def test_sparse_mask_union_across_batches_keeps_any_selected_past_key(self, config, sdpa_backend):
        mla = GlmMoeDsaMLA(config, sdpa_backend)
        topk_indices = torch.tensor(
            [
                [[0], [0], [0]],
                [[0], [1], [2]],
            ]
        )

        sparse_mask = mla._build_sparse_mask(
            topk_indices,
            seq_len=3,
            qkv_format="bshd",
            bsz=2,
            n_heads=2,
            dtype=torch.float32,
            union_across_batches=True,
        )

        assert sparse_mask.shape == (1, 2, 3, 3)
        assert sparse_mask[0, 0, 1, 0] == 0
        assert sparse_mask[0, 0, 1, 1] == 0
        assert sparse_mask[0, 0, 1, 2] == torch.finfo(torch.float32).min

    def test_sparse_mask_thd_expands_heads_and_applies_causality(self, config, sdpa_backend):
        mla = GlmMoeDsaMLA(config, sdpa_backend)
        topk_indices = torch.tensor([[0, 1], [0, 1], [1, 2]])

        sparse_mask = mla._build_sparse_mask(
            topk_indices,
            seq_len=3,
            qkv_format="thd",
            n_heads=2,
            dtype=torch.float32,
        )

        assert sparse_mask.shape == (1, 2, 3, 3)
        assert sparse_mask[0, 0, 2, 2] == 0
        assert sparse_mask[0, 0, 0, 1] == torch.finfo(torch.float32).min

    def test_shared_forward_requires_previous_topk(self, config, sdpa_backend):
        mla = GlmMoeDsaMLA(config, sdpa_backend, skip_topk=True)
        x = torch.randn(1, 3, config.hidden_size)

        with pytest.raises(ValueError, match="Shared DSA layers"):
            mla(x, _freqs(1, 3, config.qk_rope_head_dim))

    def test_shared_forward_reuses_previous_topk_and_returns_it(self, config, sdpa_backend):
        mla = GlmMoeDsaMLA(config, sdpa_backend, skip_topk=True)
        seq_len = 3
        x = torch.randn(1, seq_len, config.hidden_size)
        prev_topk = torch.tensor([[[0, 0], [0, 1], [1, 2]]])

        out, returned_topk = mla(
            x,
            _freqs(1, seq_len, config.qk_rope_head_dim),
            prev_topk_indices=prev_topk,
            return_topk_indices=True,
        )

        assert out.shape == x.shape
        torch.testing.assert_close(returned_topk, prev_topk)

    def test_init_weights_skips_absent_indexer_for_shared_layer(self, config, sdpa_backend):
        mla = GlmMoeDsaMLA(config, sdpa_backend, skip_topk=True)

        with patch("torch.nn.init.trunc_normal_") as trunc_normal:
            mla.init_weights(torch.device("cpu"), init_std=0.01)

        assert trunc_normal.call_count == 5
