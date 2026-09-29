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

"""Tied/untied alias tests for Qwen2ForCausalLM (TieSupport.BOTH).

Qwen2 ships both tied (0.5B-3B) and untied (7B/72B) checkpoints, so it declares
BOTH and honors the flag either way. These pin the explicit ``tie_weights()``
override (added because HF's base machinery does not reliably tie this custom
model from the dict-shaped ``_tied_weights_keys`` under transformers v5).
"""

from transformers import Qwen2Config

from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM


def _tiny_qwen2_config(tie_word_embeddings: bool) -> Qwen2Config:
    return Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        tie_word_embeddings=tie_word_embeddings,
    )


def test_qwen2_ties_lm_head_when_config_requests_tied_embeddings():
    model = Qwen2ForCausalLM(_tiny_qwen2_config(tie_word_embeddings=True))
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_qwen2_leaves_lm_head_untied_when_config_requests_untied_embeddings():
    model = Qwen2ForCausalLM(_tiny_qwen2_config(tie_word_embeddings=False))
    assert model.lm_head.weight is not model.model.embed_tokens.weight
