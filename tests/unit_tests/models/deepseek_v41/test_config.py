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

"""Checkpoint boundary and schedule validation for DeepSeek-V4.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from nemo_automodel._transformers.registry import resolve_custom_config_cls
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)


def test_checkpoint_nested_configs_and_unknown_metadata_roundtrip(tmp_path: Path) -> None:
    payload = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "transformers_version": "5.6.0",
        "dtype": "bfloat16",
        "image_token_id": 129264,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "num_hidden_layers": 40,
            "hidden_size": 5120,
            "rms_norm_eps": 1e-20,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "checkpoint_note": "preserve upstream metadata",
        },
        "vision_config": {
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": 32,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "patch_size": 14,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)

    assert isinstance(config.text_config, DeepseekV41TextConfig)
    assert isinstance(config.vision_config, DeepseekV41VisionConfig)
    assert config.get_text_config() is config.text_config
    assert config.architectures == ["DeepseekV41ForCausalLM"]
    assert config.dtype is torch.bfloat16
    assert config.text_config.checkpoint_note == "preserve upstream metadata"
    config.save_pretrained(tmp_path / "saved")
    reloaded = DeepseekV41Config.from_pretrained(tmp_path / "saved")
    serialized = reloaded.to_dict()
    for key in (
        "architectures",
        "image_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "quantization_config",
    ):
        assert serialized[key] == payload[key]
    for name in ("text_config", "vision_config"):
        for key, expected in payload[name].items():
            assert serialized[name][key] == expected


def test_released_defaults_preserve_precision_and_architecture() -> None:
    config = DeepseekV41Config()
    text = config.text_config
    assert (text.vocab_size, text.hidden_size, text.moe_intermediate_size) == (129280, 5120, 2304)
    assert (text.num_attention_heads, text.head_dim, text.qk_rope_head_dim) == (64, 512, 64)
    assert (text.q_lora_rank, text.o_lora_rank, text.o_groups) == (1280, 1024, 8)
    assert (text.n_routed_experts, text.n_shared_experts, text.num_experts_per_tok) == (384, 1, 6)
    assert (text.scoring_func, text.routed_scaling_factor, text.swiglu_limit) == ("sqrtsoftplus", 1.5, 10.0)
    assert (text.rms_norm_eps, text.hc_eps, text.hc_mult, text.hc_sinkhorn_iters) == (1e-20, 1e-6, 4, 20)
    assert (text.index_n_heads, text.index_head_dim, text.index_topk, text.sliding_window) == (32, 128, 512, 128)
    assert (text.candidate_source_layer_id, text.candidate_topk_blocks, text.candidate_block_size) == (20, 2048, 8)
    assert (text.engram_max_ngram_size, text.engram_n_heads, text.engram_head_dim) == (4, 8, 256)
    assert (text.engram_vocab_size, text.engram_compressed_vocab_size, text.engram_pad_token_id) == (16000000, 99092, 2)
    assert (text.num_nextn_predict_layers, text.dspark_block_size, text.dspark_noise_token_id) == (3, 5, 128799)
    assert (text.dspark_markov_rank, text.dspark_n_routed_experts, text.dspark_num_experts_per_tok) == (256, 128, 3)
    assert text.dspark_target_layer_ids == [37, 38, 39]
    expected_rope = {
        "rope_type": "yarn",
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    }
    # Transformers normalizes the legacy RoPE dictionary by adding rope_theta.
    for key, expected in expected_rope.items():
        assert text.rope_scaling[key] == expected
    assert not text.tie_word_embeddings
    assert not config.tie_word_embeddings
    assert "quantization_config" not in config.to_dict()


def test_pretrained_prefix_retains_full_source_and_hash_identities(tmp_path: Path) -> None:
    prefix = DeepseekV41TextConfig(num_hidden_layers=4)
    assert prefix.compress_ratios == [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
    assert prefix.kv_source_layer_ids == [2, 8, 14, 20]
    assert prefix.index_source_layer_ids == [2, 8, 14, 20, 24, 28, 32, 36]
    assert prefix.engram_layer_ids == [1, 14]
    assert prefix.engram_num_embeddings == [384006168, 384016682]
    DeepseekV41Config(text_config=prefix).save_pretrained(tmp_path)
    restored = DeepseekV41TextConfig.from_pretrained(tmp_path)
    assert restored.num_hidden_layers == 4
    assert restored.kv_source_layer_ids == prefix.kv_source_layer_ids
    assert restored.engram_layer_ids == prefix.engram_layer_ids


def test_tiny_schedule_supports_full_reuse_and_reindex_modes() -> None:
    ratios = [0, 2, 2, 1, 1]
    config = DeepseekV41TextConfig(
        hidden_size=16,
        num_hidden_layers=5,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        sliding_window=2,
        compress_ratios=ratios,
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 4],
        candidate_source_layer_id=3,
        candidate_block_size=2,
        candidate_topk_blocks=2,
        engram_layer_ids=[],
        dtype="float32",
    )
    ratios[1] = 0
    assert config.compress_ratios == [0, 2, 2, 1, 1]
    assert config.engram_layer_ids == []
    assert config.engram_num_embeddings == []
    assert config.dtype is torch.float32


def test_config_instances_remain_typed_and_defaults_do_not_alias() -> None:
    text = DeepseekV41TextConfig(num_hidden_layers=4)
    vision = DeepseekV41VisionConfig(num_hidden_layers=0)
    outer = DeepseekV41Config(text_config=text, vision_config=vision)
    assert outer.text_config is text
    assert outer.vision_config is vision
    text.compress_ratios[2] = 1
    text.rope_scaling["factor"] = 2
    other = DeepseekV41TextConfig()
    assert other.compress_ratios[2] == 2
    assert other.rope_scaling["factor"] == 16


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"head_dim": 32}, "qk_rope_head_dim"),
        ({"qk_rope_head_dim": 3}, "qk_rope_head_dim"),
        ({"num_attention_heads": 63}, "o_groups"),
        ({"num_key_value_heads": 2}, "num_key_value_heads"),
        ({"n_routed_experts": 4}, "num_experts_per_tok"),
        ({"n_shared_experts": 2}, "n_shared_experts"),
        ({"rms_norm_eps": float("nan")}, "rms_norm_eps"),
        ({"hc_eps": 0}, "hc_eps"),
        ({"attention_dropout": 1.0}, "attention_dropout"),
        ({"compress_ratios": [0, 0]}, "cover every active"),
        ({"compress_ratios": [0, 0, -1] + [1] * 37}, "non-negative"),
        ({"kv_source_layer_ids": [2, 2, 20]}, "strictly increasing"),
        ({"kv_source_layer_ids": [2, 8, 14, 99]}, "covered by compress_ratios"),
        ({"kv_source_layer_ids": [3, 8, 14, 20]}, "Every KV source"),
        ({"index_source_layer_ids": [0, 2, 8, 14, 20]}, "SWA-only"),
        ({"kv_source_layer_ids": [8, 14, 20]}, "preceding KV source"),
        ({"candidate_source_layer_id": 24}, "Full-mode"),
        ({"candidate_topk_blocks": 0}, "must be positive"),
        ({"engram_num_embeddings": [123]}, "one table row count"),
        ({"engram_num_embeddings": [123, 0]}, "positive integer row counts"),
        ({"engram_max_ngram_size": 1}, "at least 2"),
        ({"engram_pad_token_id": -1}, "within the token vocabulary"),
    ],
)
def test_invalid_text_config_fails_before_model_allocation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DeepseekV41TextConfig(**overrides)


def test_compression_ratio_transition_requires_a_new_owner() -> None:
    with pytest.raises(ValueError, match="layer 3 needs a preceding KV source"):
        DeepseekV41TextConfig(
            num_hidden_layers=4,
            compress_ratios=[0, 2, 2, 1],
            kv_source_layer_ids=[1],
            index_source_layer_ids=[1],
            candidate_source_layer_id=-1,
            engram_layer_ids=[],
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_size": 1000}, "divisible by 4"),
        ({"num_hidden_layers": -1}, "non-negative"),
        ({"patch_size": 0}, "patch_size"),
        ({"max_wh_ratio": 0.5}, "max_wh_ratio"),
    ],
)
def test_invalid_vision_dimensions_fail_at_config_boundary(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DeepseekV41VisionConfig(**overrides)


@pytest.mark.parametrize("field", ["text_config", "vision_config"])
def test_untyped_runtime_objects_are_rejected_at_hf_boundary(field: str) -> None:
    with pytest.raises(TypeError, match=field):
        DeepseekV41Config(**{field: object()})


def test_custom_config_registry_resolves_every_checkpoint_model_type() -> None:
    assert resolve_custom_config_cls("deepseek_v41") is DeepseekV41Config
    assert resolve_custom_config_cls("deepseek_v41_text") is DeepseekV41TextConfig
    assert resolve_custom_config_cls("deepseek_v41_vision") is DeepseekV41VisionConfig


def test_tokenizer_build_requires_a_checkpoint_source() -> None:
    with pytest.raises(ValueError, match="checkpoint source or an explicit tokenizer"):
        DeepseekV41Config().build_tokenizer()


def test_tokenizer_build_preserves_checkpoint_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    received = {}

    def from_pretrained(source: str, **kwargs: Any) -> PreTrainedTokenizerFast:
        received.update(source=source, **kwargs)
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", from_pretrained)
    config = DeepseekV41Config(name_or_path="local/checkpoint", _commit_hash="pinned-revision")
    assert config.build_tokenizer() is tokenizer
    assert received == {
        "source": "local/checkpoint",
        "revision": "pinned-revision",
        "trust_remote_code": False,
        "use_fast": True,
    }
