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

"""CPU guard test: separate-head models reject tie_word_embeddings=True.

The reject guard runs at the very start of ``__init__`` (before any device- or
kernel-dependent construction), so this is CPU-safe even though the full
qwen3_moe model build requires a GPU. qwen3_moe stands in for the whole
untied-default family, which declares ``tie_word_embeddings_support =
TieSupport.UNTIED_ONLY`` and is validated by
``reject_unsupported_tie_word_embeddings``.
"""

import pytest
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM


def _tiny_config(tie_word_embeddings: bool) -> Qwen3MoeConfig:
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
        tie_word_embeddings=tie_word_embeddings,
    )


def test_qwen3_moe_rejects_tied_word_embeddings():
    """Constructing with tie_word_embeddings=True raises a clear error before model build."""
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", experts="torch", dispatcher="torch")
    with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=True"):
        Qwen3MoeForCausalLM(_tiny_config(tie_word_embeddings=True), backend=backend)
