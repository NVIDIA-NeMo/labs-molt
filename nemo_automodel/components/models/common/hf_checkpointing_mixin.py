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

"""HuggingFace-compatible checkpointing mixin for NeMo Automodel.

This module provides a mixin class that gives models HuggingFace-compatible
save_pretrained() and from_pretrained() methods while using NeMo's checkpointing
infrastructure internally.

Key design principle: We do NOT override state_dict() or load_state_dict().
PyTorch's DCP expects these to behave like standard nn.Module methods.
HF format conversions happen only in save_pretrained() and from_pretrained() via Checkpointer.

Checkpointer is passed explicitly (dependency injection) - no global state.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizerBase

    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

logger = logging.getLogger(__name__)


class HFCheckpointingMixin:
    """Mixin providing HF-compatible API using NeMo's checkpointing infrastructure.

    Provides save_pretrained() and from_pretrained() methods that use Checkpointer
    for unified distributed/async support with HF format conversion.

    Key design: We do NOT override state_dict() or load_state_dict() because
    PyTorch's DCP expects these to behave like standard nn.Module methods.

    For PreTrainedModel subclasses:
    - super().from_pretrained() handles: downloads, quantization config, meta device init
    - Checkpointer.load_base_model() handles: actual weight loading with format conversion

    For nn.Module subclasses (no parent from_pretrained):
    - Falls back to manual config loading + Checkpointer
    """

    def save_pretrained(
        self,
        save_directory: str,
        checkpointer: Optional["Checkpointer"] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> None:
        """Save model in HF-compatible format using Checkpointer infrastructure.

        Supports distributed saving, sharding, and async checkpointing.

        Args:
            save_directory: Output path
            checkpointer: Checkpointer instance. Uses self._checkpointer if not provided.
            tokenizer: Optional tokenizer to save alongside model
            **kwargs: Additional arguments, including ``peft_config`` and
                ``is_final_checkpoint``. Direct callers that do not have recipe
                step-scheduler context default ``is_final_checkpoint`` to
                ``False``.
        """
        if checkpointer is None:
            raise ValueError("No checkpointer provided. Please pass the `checkpointer` argument.")

        # Use Checkpointer.save_model() which handles:
        # - ModelState.state_dict()
        # - _maybe_adapt_state_dict_to_hf() conversion
        # - Distributed/sharded saving via DCP
        # - Consolidated HF safetensors output
        # - Async checkpointing if enabled
        checkpointer.save_model(
            model=self,
            weights_path=save_directory,
            peft_config=kwargs.get("peft_config", None),
            tokenizer=tokenizer,
            is_final_checkpoint=kwargs.get("is_final_checkpoint", False),
        )
