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
from transformers import AutoModelForCausalLM, Qwen2Config, set_seed

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen2.model import Qwen2Attention

set_seed(42)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# Tiny Qwen2 config for testing
TINY_DEFAULT_QWEN2_CONFIG = dict(
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
)


def test_quack_rope_reports_missing_dependency():
    config = Qwen2Config(**TINY_DEFAULT_QWEN2_CONFIG)
    with (
        patch(
            "nemo_automodel.components.models.qwen2.model.safe_import_from",
            return_value=(False, None),
        ),
        pytest.raises(ImportError, match="quack-kernels"),
    ):
        Qwen2Attention(config, layer_idx=0, backend=BackendConfig(rope="quack"))


def _create_checkpoint(config_kwargs, tmpdir):
    """Create a tiny HF Qwen2 checkpoint in the given directory.

    Args:
        config_kwargs: Dict of Qwen2Config keyword arguments.
        tmpdir: Directory (str or Path) to save the checkpoint into.

    Returns:
        str path to the checkpoint directory.
    """
    tmpdir = str(tmpdir)
    config = Qwen2Config(**config_kwargs)
    config.save_pretrained(tmpdir)
    model = AutoModelForCausalLM.from_config(config)
    for param in model.parameters():
        # Reinitialize trivially constant parameters (e.g., norm weight=all 1s, bias=all 0s)
        if param.data.unique().numel() == 1:
            param.data.normal_(mean=0, std=0.1)
    model.save_pretrained(tmpdir)
    return tmpdir


class TestQwen2Model:
    @pytest.fixture(scope="class", autouse=True)
    def _tiny_checkpoint(self, tmp_path_factory):
        """Create a tiny HF Qwen2 checkpoint shared across tests (auto-cleaned by pytest)."""
        self.__class__.tiny_qwen2_checkpoint = _create_checkpoint(
            TINY_DEFAULT_QWEN2_CONFIG, tmp_path_factory.mktemp("qwen2_ckpt")
        )


