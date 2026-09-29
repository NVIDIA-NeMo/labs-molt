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

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling.configuration import InklingConfig, InklingTextConfig
from nemo_automodel.components.models.inkling.layers import InklingMoE
from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration
from nemo_automodel.components.models.inkling.state_dict_adapter import InklingStateDictAdapter

from .parity_check_inkling import build_tiny_config


def test_inkling_config_registered_with_auto_config():
    assert CONFIG_MAPPING["inkling_mm_model"] is InklingConfig

    cfg = AutoConfig.for_model("inkling_mm_model", architectures=["InklingForConditionalGeneration"])

    assert isinstance(cfg, InklingConfig)
    assert cfg.model_type == "inkling_mm_model"
    assert cfg.text_config.model_type == "inkling_text"


def test_inkling_checkpoint_aliases_and_placeholder_defaults():
    cfg = InklingConfig.from_dict(
        {
            "model_type": "inkling_mm_model",
            "image_token_id": None,
            "audio_token_id": None,
            "text_config": {"hidden_size": 64, "num_hidden_layers": 2},
            "vision_config": {"n_layers": 4, "n_channels": 5, "decoder_dmodel": 64},
            "audio_config": {"n_mel_bins": 8, "mel_vocab_size": 16, "decoder_dmodel": 64},
        }
    )

    assert cfg.image_token_id == 200054
    assert cfg.audio_token_id == 200053
    assert cfg.vision_config.num_hidden_layers == 4
    assert cfg.vision_config.num_channels == 5
    assert cfg.vision_config.text_hidden_size == 64
    assert cfg.audio_config.text_hidden_size == 64


def test_inkling_architecture_instantiates_from_local_config():
    assert ModelRegistry.has_custom_model("InklingForConditionalGeneration")
    backend = BackendConfig(linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    model = InklingForConditionalGeneration.from_config(build_tiny_config(), backend=backend)
    assert isinstance(model, InklingForConditionalGeneration)
    assert model.config.model_type == "inkling_mm_model"


@pytest.mark.parametrize(
    "hidden_size,expert_width,dense_width",
    [(4096, 2048, 16384), (6144, 3072, 24576)],
    ids=["inkling-small", "inkling"],
)
def test_inkling_raw_checkpoint_expert_shapes(
    tmp_path: Path, hidden_size: int, expert_width: int, dense_width: int
) -> None:
    # Published checkpoints use intermediate_size for experts, not dense layers.
    checkpoint_config = {
        "model_type": "inkling_mm_model",
        "architectures": ["InklingForConditionalGeneration"],
        "text_config": {
            "hidden_size": hidden_size,
            "n_routed_experts": 256,
            "intermediate_size": expert_width,
            "dense_intermediate_size": dense_width,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))
    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, InklingConfig)
    assert config.text_config.intermediate_size == dense_width
    assert config.text_config.moe_intermediate_size == expert_width

    backend = BackendConfig(linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    with torch.device("meta"):
        mlp = InklingMoE(config.text_config, backend)
    adapter = InklingStateDictAdapter(config.text_config, mlp.moe_config, backend)
    weights = adapter.to_hf(
        {f"model.language_model.layers.10.mlp.{key}": value for key, value in mlp.state_dict().items()}
    )
    assert weights["model.llm.layers.10.mlp.experts.w13_weight"].shape == (256, 2 * expert_width, hidden_size)
    assert weights["model.llm.layers.10.mlp.experts.w2_weight"].shape == (256, hidden_size, expert_width)

    config.save_pretrained(tmp_path / "roundtrip")
    reloaded = AutoConfig.from_pretrained(tmp_path / "roundtrip")
    assert reloaded.text_config.intermediate_size == dense_width
    assert reloaded.text_config.moe_intermediate_size == expert_width


@pytest.mark.parametrize(
    "kwargs,dense_width,expert_width",
    [
        ({}, 24576, 3072),
        ({"intermediate_size": 128}, 128, 3072),
        ({"intermediate_size": 128, "moe_intermediate_size": 32}, 128, 32),
        ({"dense_intermediate_size": 96, "moe_intermediate_size": 32}, 96, 32),
        ({"intermediate_size": 64, "dense_intermediate_size": 128, "moe_intermediate_size": 32}, 128, 32),
    ],
    ids=["defaults", "dense-only", "normalized", "tiny-config", "explicit-expert-width"],
)
def test_inkling_expert_width_defaults_and_overrides(
    kwargs: dict[str, int], dense_width: int, expert_width: int
) -> None:
    config = InklingTextConfig(**kwargs)
    assert config.intermediate_size == dense_width
    assert config.moe_intermediate_size == expert_width
