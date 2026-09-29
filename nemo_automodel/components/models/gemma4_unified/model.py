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

"""NeMo wrapper for the Hugging Face Gemma4 Unified model."""

from dataclasses import dataclass

from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration as HFGemma4UnifiedForConditionalGeneration,
)

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.gemma4_unified.state_dict_adapter import (
    Gemma4UnifiedStateDictAdapter,
)


class Gemma4UnifiedForConditionalGeneration(HFCheckpointingMixin, HFGemma4UnifiedForConditionalGeneration):
    """Gemma4 Unified with NeMo checkpoint key conversion."""

    tie_word_embeddings_support: TieSupport = TieSupport.TIED_ONLY

    @dataclass(frozen=True)
    class ModelCapabilities:
        """No custom parallelism implementations are provided by this wrapper."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.state_dict_adapter = Gemma4UnifiedStateDictAdapter()

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        """Tie the output head to Gemma4 Unified's text input embedding."""
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
