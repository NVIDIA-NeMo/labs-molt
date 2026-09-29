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

"""Unit tests for HFCheckpointingMixin."""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin


class SimpleModelWithMixin(HFCheckpointingMixin, nn.Module):
    """Simple model using the mixin for testing."""

    def __init__(self):
        nn.Module.__init__(self)
        self.linear = nn.Linear(32, 32)

    def forward(self, x):
        return self.linear(x)


class TestHFCheckpointingMixinSavePretrained:
    """Tests for save_pretrained() method."""

    def test_save_pretrained_requires_checkpointer(self):
        """Test that save_pretrained() raises error without checkpointer."""
        model = SimpleModelWithMixin()

        with pytest.raises(ValueError, match="No checkpointer provided"):
            model.save_pretrained("/tmp/test")

    def test_save_pretrained_uses_checkpointer(self):
        """Test that save_pretrained() uses Checkpointer.save_model()."""
        model = SimpleModelWithMixin()
        mock_checkpointer = MagicMock()

        model.save_pretrained("/tmp/test", checkpointer=mock_checkpointer)

        mock_checkpointer.save_model.assert_called_once_with(
            model=model,
            weights_path="/tmp/test",
            peft_config=None,
            tokenizer=None,
            is_final_checkpoint=False,
        )

    def test_save_pretrained_passes_peft_config(self):
        """Test that save_pretrained() passes peft_config from kwargs."""
        model = SimpleModelWithMixin()
        mock_checkpointer = MagicMock()
        peft_config = {"type": "lora"}

        model.save_pretrained("/tmp/test", checkpointer=mock_checkpointer, peft_config=peft_config)

        mock_checkpointer.save_model.assert_called_once_with(
            model=model,
            weights_path="/tmp/test",
            peft_config=peft_config,
            tokenizer=None,
            is_final_checkpoint=False,
        )

    def test_save_pretrained_passes_tokenizer(self):
        """Test that save_pretrained() passes tokenizer."""
        model = SimpleModelWithMixin()
        mock_checkpointer = MagicMock()
        mock_tokenizer = MagicMock()

        model.save_pretrained("/tmp/test", checkpointer=mock_checkpointer, tokenizer=mock_tokenizer)

        mock_checkpointer.save_model.assert_called_once_with(
            model=model,
            weights_path="/tmp/test",
            peft_config=None,
            tokenizer=mock_tokenizer,
            is_final_checkpoint=False,
        )

    def test_save_pretrained_passes_is_final_checkpoint(self):
        """Test that save_pretrained() forwards explicit final checkpoint state."""
        model = SimpleModelWithMixin()
        mock_checkpointer = MagicMock()

        model.save_pretrained("/tmp/test", checkpointer=mock_checkpointer, is_final_checkpoint=True)

        mock_checkpointer.save_model.assert_called_once_with(
            model=model,
            weights_path="/tmp/test",
            peft_config=None,
            tokenizer=None,
            is_final_checkpoint=True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
