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

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.api import CheckpointException
from transformers import AutoModelForSeq2SeqLM, BartConfig, PretrainedConfig, T5Config

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig


@pytest.fixture(params=["t5", "bart"])
def model_config(request) -> PretrainedConfig:
    if request.param == "t5":
        return T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            d_kv=8,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            dropout_rate=0.0,
        )
    return BartConfig(
        vocab_size=32,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=16,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )


@pytest.fixture
def checkpointer(tmp_path: Path) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path / "cache"),
            model_repo_id="test/shared-parameters",
            model_save_format="safetensors",
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


def test_hf_checkpoint_can_keep_only_the_lm_head_alias(
    tmp_path: Path, model_config: PretrainedConfig, checkpointer: Checkpointer
) -> None:
    """A saved alias absent from ModelState's initial destinations can still supply all shared embeddings."""
    reference = AutoModelForSeq2SeqLM.from_config(model_config)
    reference.save_pretrained(tmp_path / "model")
    checkpoint = load_file(tmp_path / "model" / "model.safetensors")
    source_name = "shared.weight" if model_config.model_type == "t5" else "model.shared.weight"
    checkpoint["lm_head.weight"] = checkpoint.pop(source_name)
    save_file(checkpoint, tmp_path / "model" / "model.safetensors")

    model = AutoModelForSeq2SeqLM.from_config(model_config)
    checkpointer.load_model(model, str(tmp_path / "model"), is_init_step=True)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("missing_parameter", ["shared", "independent", "untied_alias"])
def test_shared_alias_handling_does_not_hide_missing_weights(
    tmp_path: Path, model_config: PretrainedConfig, checkpointer: Checkpointer, missing_parameter: str
) -> None:
    """Only aliases of the same live parameter may be omitted from DCP destinations."""
    reference = AutoModelForSeq2SeqLM.from_config(model_config)
    reference.save_pretrained(tmp_path / "model")
    checkpoint = load_file(tmp_path / "model" / "model.safetensors")
    model = AutoModelForSeq2SeqLM.from_config(model_config)
    if missing_parameter == "untied_alias":
        # Leave the config's tying declaration intact, but make the encoder embedding independent.
        model.get_encoder().embed_tokens = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)
    else:
        source_name = "shared.weight" if model_config.model_type == "t5" else "model.shared.weight"
        missing_name = (
            source_name if missing_parameter == "shared" else next(key for key in checkpoint if key != source_name)
        )
        del checkpoint[missing_name]
        save_file(checkpoint, tmp_path / "model" / "model.safetensors")

    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict"):
        checkpointer.load_model(model, str(tmp_path / "model"), is_init_step=True)
