# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for Qwen3.8-Flash-Next checkpoint configuration and registry wiring."""

from __future__ import annotations

import json

import pytest
from transformers import AutoConfig

from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig,
    Qwen3_8_FlashNextTextConfig,
    Qwen3_8_FlashNextVisionConfig,
)

_LAYER_TYPES = ["full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention" for layer_idx in range(48)]

_CHECKPOINT_CONFIG = {
    "architectures": ["Qwen3_8_FlashNextForConditionalGeneration"],
    "image_token_id": 248056,
    "language_model_only": False,
    "model_type": "qwen3_8_flash_next",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 248044,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "heads_per_ngram": 8,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 128,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 4,
        "initializer_range": 0.02,
        "layer_types": _LAYER_TYPES,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "make_ngram_vocab_size_divisible_by": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen3_8_flash_next_text",
        "moe_intermediate_size": 640,
        "mtp": {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "mtp_use_hidden_state_from_layer": None,
            "num_hidden_layers": 1,
            "rope_theta": 10000000,
        },
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "ngram_size": 3,
        "ngram_vocab_size_base": 20000000,
        "num_attention_heads": 24,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "num_hidden_layers": 48,
        "num_key_value_heads": 2,
        "output_gate_type": "sigmoid",
        "output_router_logits": False,
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "ple_conv_kernel_size": 4,
        "ple_embed_dim": 2560,
        "ple_layer_ids": [2],
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": 640,
        "split_ngram_parts": 128,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 248320,
    },
    "tie_word_embeddings": False,
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_8_flash_next",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
}


def test_checkpoint_style_auto_config_resolves_local_nested_configs(tmp_path) -> None:
    payload = dict(_CHECKPOINT_CONFIG)
    payload["auto_map"] = {"AutoConfig": "configuration_qwen3_8_flash_next.Qwen3_8_FlashNextConfig"}
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)

    assert type(config) is Qwen3_8_FlashNextConfig
    assert type(config.text_config) is Qwen3_8_FlashNextTextConfig
    assert type(config.vision_config) is Qwen3_8_FlashNextVisionConfig
    assert config.architectures == ["Qwen3_8_FlashNextForConditionalGeneration"]
    assert config.image_token_id == 248056
    assert config.video_token_id == 248057
    assert config.vision_start_token_id == 248053
    assert config.vision_end_token_id == 248054

    text_config = config.text_config
    assert text_config.hc_count == 4
    assert text_config.hc_lowrank == 320
    assert text_config.ple_layer_ids == [2]
    assert text_config.indexer_budget == 2048
    assert text_config.indexer_compress_ratio == 4
    assert text_config.linear_num_key_heads == 16
    assert text_config.linear_num_value_heads == 48
    assert text_config.num_experts == 512
    assert text_config.num_experts_per_tok == 10
    assert text_config.mtp["layer_types"] == ["full_attention"]
    assert text_config.rope_parameters["mrope_section"] == [11, 11, 10]
    assert text_config.rope_scaling == _CHECKPOINT_CONFIG["text_config"]["rope_parameters"]
    assert text_config.rope_theta == 10000000

    vision_config = config.vision_config
    assert vision_config.depth == 27
    assert vision_config.hidden_size == 1152
    assert vision_config.out_hidden_size == 2560
    assert vision_config.patch_size == 16
    assert vision_config.temporal_patch_size == 2
    assert vision_config.spatial_merge_size == 2

    serialized = config.to_dict()
    for key, value in _CHECKPOINT_CONFIG["text_config"].items():
        assert serialized["text_config"][key] == value
    for key, value in _CHECKPOINT_CONFIG["vision_config"].items():
        assert serialized["vision_config"][key] == value


def test_custom_config_registry_covers_outer_and_text_model_types() -> None:
    assert resolve_custom_config_cls("qwen3_8_flash_next") is Qwen3_8_FlashNextConfig
    assert resolve_custom_config_cls("qwen3_8_flash_next_text") is Qwen3_8_FlashNextTextConfig
    assert _CUSTOM_CONFIG_REGISTRATIONS["qwen3_8_flash_next"] == (
        "nemo_automodel.components.models.qwen3_8_flash_next.config",
        "Qwen3_8_FlashNextConfig",
    )
    assert _CUSTOM_CONFIG_REGISTRATIONS["qwen3_8_flash_next_text"] == (
        "nemo_automodel.components.models.qwen3_8_flash_next.config",
        "Qwen3_8_FlashNextTextConfig",
    )


def test_architecture_registry_points_to_qwen3_8_flash_next_model() -> None:
    assert MODEL_ARCH_MAPPING["Qwen3_8_FlashNextForConditionalGeneration"] == (
        "nemo_automodel.components.models.qwen3_8_flash_next.model",
        "Qwen3_8_FlashNextForConditionalGeneration",
    )


def test_text_config_derives_hybrid_and_ple_metadata() -> None:
    config = Qwen3_8_FlashNextTextConfig(
        hidden_size=64,
        num_hidden_layers=8,
        full_attention_interval=4,
        hc_count=4,
        ple_layer_ids=[2],
        ple_conv_kernel_size=4,
        ngram_size=3,
    )

    assert config.layer_types is None
    assert config.layers_block_type == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert config.linear_layer_ids == [0, 1, 2, 4, 5, 6]
    assert config.full_attention_layer_ids == [3, 7]
    assert config.short_conv_layer_ids == [1]
    assert config.short_conv_state_shape == (256, 9)
    assert config.ngram_context_len == 2
    assert config.hc_mult == 4


def test_sglang_compatible_defaults_do_not_enable_ple() -> None:
    config = Qwen3_8_FlashNextTextConfig(num_hidden_layers=4)

    assert config.ple_layer_ids == []
    assert config.ple_embed_dim == config.hidden_size
    assert config.short_conv_layer_ids == []
    assert config.short_conv_state_shape is None
    assert config.ngram_context_len == 0
    assert config.rope_parameters == {}
    assert config.rope_scaling == {}
    assert config.rope_theta == 10000.0


def test_legacy_flat_text_config_is_preserved() -> None:
    config = Qwen3_8_FlashNextConfig(
        hidden_size=96,
        num_hidden_layers=4,
        ple_layer_ids=[2],
        split_ngram_parts=16,
    )

    assert config.text_config.hidden_size == 96
    assert config.text_config.num_hidden_layers == 4
    assert config.text_config.ple_layer_ids == [2]
    assert config.text_config.split_ngram_parts == 16
    assert config.hidden_size == 96
    assert config.num_hidden_layers == 4
    assert config.split_ngram_parts == 16


def test_nested_config_instances_and_serialization_are_preserved() -> None:
    text_config = Qwen3_8_FlashNextTextConfig(num_hidden_layers=4, ple_layer_ids=[2])
    vision_config = Qwen3_8_FlashNextVisionConfig(depth=2)
    config = Qwen3_8_FlashNextConfig(text_config=text_config, vision_config=vision_config)

    assert config.text_config is text_config
    assert config.vision_config is vision_config

    payload = config.to_dict()
    assert payload["model_type"] == "qwen3_8_flash_next"
    assert payload["text_config"]["model_type"] == "qwen3_8_flash_next_text"
    assert payload["text_config"]["num_hidden_layers"] == 4
    assert payload["vision_config"]["model_type"] == "qwen3_8_flash_next"
    assert payload["vision_config"]["depth"] == 2


def test_nested_split_count_wins_over_stale_top_level_copy() -> None:
    payload = dict(_CHECKPOINT_CONFIG)
    payload["split_ngram_parts"] = 64

    config = Qwen3_8_FlashNextConfig(**payload)

    assert config.text_config.split_ngram_parts == 128
    assert "split_ngram_parts" not in config.__dict__


def test_invalid_hyperconnection_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="hc_count > 1"):
        Qwen3_8_FlashNextTextConfig(hc_count=1)


def test_qsa_attention_has_no_public_query_chunk_or_rematerialization_config() -> None:
    serialized = Qwen3_8_FlashNextTextConfig().to_dict()

    assert "qsa_attention_query_chunk_size" not in serialized
    assert "qsa_attention_activation_recompute" not in serialized


def test_legacy_qwen4_exp_checkpoint_config_resolves_via_autoconfig(tmp_path) -> None:
    """Immutable pre-rename checkpoint dumps must resolve to the renamed classes."""
    import json

    from transformers import AutoConfig

    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING
    from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextLegacyConfig

    checkpoint_config = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"model_type": "qwen4_exp_text", "hidden_size": 8},
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))

    from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig

    resolved = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(resolved, Qwen3_8_FlashNextLegacyConfig)
    assert isinstance(resolved.text_config, Qwen3_8_FlashNextTextConfig)
    assert resolved.text_config.hidden_size == 8
    assert resolved.model_type == "qwen4_exp"
    assert resolved.architectures == ["Qwen4ExpForConditionalGeneration"]
    module_path, class_name = MODEL_ARCH_MAPPING["Qwen4ExpForConditionalGeneration"]
    assert class_name == "Qwen3_8_FlashNextForConditionalGeneration"
