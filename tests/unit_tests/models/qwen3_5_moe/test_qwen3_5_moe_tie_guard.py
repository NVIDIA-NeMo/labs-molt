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

"""CPU-safe tests for the Qwen3.5-MoE causal-LM embedding-tie policy."""

import pytest

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM


def test_qwen3_5_moe_causal_lm_rejects_tied_word_embeddings():
    """The separate-head architecture rejects tying before allocating the model."""
    config = Qwen3_5MoeTextConfig(tie_word_embeddings=True)

    with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=True"):
        Qwen3_5MoeForCausalLM(config)
