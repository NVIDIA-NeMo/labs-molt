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

"""Tied/untied alias tests for the Qwen3.5 dense family (TieSupport.BOTH).

Supported dense Qwen3.5 checkpoints are mixed by size (0.8B/2B/4B tied,
9B/27B + Qwen3.6-27B untied), so both classes declare BOTH.

The VLM class is the important regression guard: its ``__init__`` replaces
``self.model.language_model`` with ``Qwen3_5DenseTextBackbone`` *after* HF's
``post_init`` tied ``lm_head`` to the original embedding, orphaning that alias.
Its ``tie_weights()`` re-ties to the active backbone embedding when tie=True.

Runs on CPU (torch backends, no TE / DeepEP).
"""

import torch.nn as nn
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5.model import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


def _backend() -> BackendConfig:
    """CPU-friendly backend: plain torch kernels, no fused RoPE."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


def _tiny_text_config(tie_word_embeddings: bool) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["full_attention"],
        attn_implementation="eager",
        torch_dtype="float32",
        tie_word_embeddings=tie_word_embeddings,
    )


def _tiny_vlm_config(tie_word_embeddings: bool) -> Qwen3_5Config:
    text_config = _tiny_text_config(tie_word_embeddings)
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
    )
    return Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        tie_word_embeddings=tie_word_embeddings,
    )


class TestQwen3_5CausalTieWeights:
    def test_tied_shares_lm_head_storage(self):
        model = Qwen3_5ForCausalLM(_tiny_text_config(tie_word_embeddings=True), backend=_backend())
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_untied_has_separate_lm_head(self):
        model = Qwen3_5ForCausalLM(_tiny_text_config(tie_word_embeddings=False), backend=_backend())
        assert model.lm_head.weight is not model.model.embed_tokens.weight


class TestQwen3_5ConditionalGenerationTieWeights:
    def test_tied_reties_to_active_backbone_after_swap(self):
        model = Qwen3_5ForConditionalGeneration(_tiny_vlm_config(tie_word_embeddings=True), backend=_backend())
        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight

    def test_tie_weights_restores_post_swap_alias(self):
        model = Qwen3_5ForConditionalGeneration(_tiny_vlm_config(tie_word_embeddings=True), backend=_backend())
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
        assert model.lm_head.weight is not model.model.language_model.embed_tokens.weight

        model.tie_weights()

        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight

    def test_untied_has_separate_lm_head(self):
        model = Qwen3_5ForConditionalGeneration(_tiny_vlm_config(tie_word_embeddings=False), backend=_backend())
        assert model.lm_head.weight is not model.model.language_model.embed_tokens.weight
