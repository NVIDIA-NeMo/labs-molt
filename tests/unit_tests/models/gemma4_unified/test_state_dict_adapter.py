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

from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Gemma4UnifiedConfig

from nemo_automodel._transformers.model_init import _init_model
from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.models.gemma4_unified.model import Gemma4UnifiedForConditionalGeneration
from nemo_automodel.components.models.gemma4_unified.state_dict_adapter import Gemma4UnifiedStateDictAdapter

GEMMA4_UNIFIED_KEY_PAIRS = [
    ("embed_vision.patch_ln1.weight", "vision_embedder.patch_ln1.weight"),
    ("embed_vision.patch_dense.bias", "vision_embedder.patch_dense.bias"),
    ("embed_vision.patch_ln2.weight", "vision_embedder.patch_ln2.weight"),
    ("embed_vision.pos_embedding", "vision_embedder.pos_embedding"),
    ("embed_vision.pos_norm.bias", "vision_embedder.pos_norm.bias"),
    (
        "embed_vision.multimodal_embedder.embedding_projection.weight",
        "embed_vision.embedding_projection.weight",
    ),
    (
        "model.embed_vision.multimodal_embedder.embedding_projection.weight",
        "model.embed_vision.embedding_projection.weight",
    ),
]


class TestGemma4UnifiedStateDictAdapter:
    @pytest.mark.parametrize(("model_key", "hf_key"), GEMMA4_UNIFIED_KEY_PAIRS)
    def test_round_trip(self, model_key, hf_key):
        adapter = Gemma4UnifiedStateDictAdapter()
        tensor = torch.ones(1)

        exported = adapter.to_hf({model_key: tensor})

        assert exported == {hf_key: tensor}
        assert adapter.from_hf(exported) == {model_key: tensor}
        assert exported[hf_key] is tensor

    def test_leaves_unrelated_keys_untouched(self):
        adapter = Gemma4UnifiedStateDictAdapter()
        state_dict = {"model.layers.0.self_attn.q_proj.weight": torch.ones(1), "lm_head.weight": torch.zeros(1)}

        assert adapter.to_hf(state_dict) == state_dict
        assert adapter.from_hf(state_dict) == state_dict

    def test_to_hf_drops_excluded_keys(self):
        tensor = torch.ones(1)
        state_dict = {
            "embed_vision.patch_ln1.weight": tensor,
            "model.layers.0.self_attn._extra_state": torch.zeros(1),
        }

        exported = Gemma4UnifiedStateDictAdapter().to_hf(state_dict, exclude_key_regex=r".*_extra_state.*")

        assert exported == {"vision_embedder.patch_ln1.weight": tensor}

    def test_rejects_export_key_collision(self):
        state_dict = {
            "embed_vision.embedding_projection.weight": torch.ones(1),
            "embed_vision.multimodal_embedder.embedding_projection.weight": torch.zeros(1),
        }

        with pytest.raises(ValueError, match="key collision"):
            Gemma4UnifiedStateDictAdapter().to_hf(state_dict)


def _tiny_config() -> Gemma4UnifiedConfig:
    return Gemma4UnifiedConfig(
        architectures=["Gemma4UnifiedForConditionalGeneration"],
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "sliding_window": 8,
            "layer_types": ["full_attention"],
            "global_head_dim": 8,
            "num_global_key_value_heads": 1,
        },
        vision_config={
            "patch_size": 2,
            "pooling_kernel_size": 1,
            "mm_embed_dim": 8,
            "mm_posemb_size": 4,
            "output_proj_dims": 8,
        },
        audio_config=None,
        boi_token_id=3,
        eoi_token_id=4,
        image_token_id=5,
        video_token_id=6,
        boa_token_id=7,
        eoa_token_index=8,
        audio_token_id=9,
    )


class TestGemma4UnifiedCustomModel:
    def test_registry_uses_model_specific_wrapper(self):
        assert MODEL_ARCH_MAPPING["Gemma4UnifiedForConditionalGeneration"] == (
            "nemo_automodel.components.models.gemma4_unified.model",
            "Gemma4UnifiedForConditionalGeneration",
        )

    def test_init_uses_custom_model_with_adapter(self):
        is_custom_model, model = _init_model(object(), _tiny_config(), None, "auto", None, False)

        assert is_custom_model is True
        assert isinstance(model, Gemma4UnifiedForConditionalGeneration)
        assert isinstance(model.state_dict_adapter, Gemma4UnifiedStateDictAdapter)
        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight

        exported = model.state_dict_adapter.to_hf(model.state_dict())
        assert "model.vision_embedder.patch_ln1.weight" in exported
        assert "model.embed_vision.patch_ln1.weight" not in exported


def test_checkpointer_save_resume_and_consolidated_export_use_hf_keys(tmp_path):
    """Exercise the existing adapter hooks across the complete checkpoint lifecycle."""

    class TinyGemma4Unified(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_vision = torch.nn.Module()
            self.embed_vision.patch_ln1 = torch.nn.Linear(2, 2, bias=False)
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            self.state_dict_adapter = Gemma4UnifiedStateDictAdapter()

    model = TinyGemma4Unified()
    expected_patch_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    expected_lm_head_weight = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
    with torch.no_grad():
        model.embed_vision.patch_ln1.weight.copy_(expected_patch_weight)
        model.lm_head.weight.copy_(expected_lm_head_weight)

    config = CheckpointingConfig(
        checkpoint_dir=str(tmp_path),
        model_save_format="safetensors",
        model_cache_dir=str(tmp_path / "cache"),
        model_repo_id="",
        save_consolidated=True,
    )
    with patch("torch.distributed.is_initialized", return_value=False):
        checkpointer = Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0)

    base_model_path = tmp_path / "base_model"
    base_model_path.mkdir()
    save_file(
        {
            "vision_embedder.patch_ln1.weight": expected_patch_weight,
            "lm_head.weight": expected_lm_head_weight,
        },
        base_model_path / "model.safetensors",
    )
    with torch.no_grad():
        model.embed_vision.patch_ln1.weight.zero_()
        model.lm_head.weight.zero_()

    checkpointer.load_model(
        model,
        str(base_model_path),
        is_init_step=True,
        key_mapping={r"^vision_embedder\.patch_ln1": "embed_vision.patch_ln1"},
    )

    torch.testing.assert_close(model.embed_vision.patch_ln1.weight, expected_patch_weight)
    torch.testing.assert_close(model.lm_head.weight, expected_lm_head_weight)

    checkpoint_path = tmp_path / "step_1"
    checkpointer.save_model(model, str(checkpoint_path))

    consolidated_files = list((checkpoint_path / "model" / "consolidated").glob("*.safetensors"))
    assert len(consolidated_files) == 1
    exported = load_file(consolidated_files[0])
    assert set(exported) == {"vision_embedder.patch_ln1.weight", "lm_head.weight"}
    torch.testing.assert_close(exported["vision_embedder.patch_ln1.weight"], expected_patch_weight)

    with torch.no_grad():
        model.embed_vision.patch_ln1.weight.zero_()
        model.lm_head.weight.zero_()

    checkpointer.load_model(model, str(checkpoint_path / "model"))

    torch.testing.assert_close(model.embed_vision.patch_ln1.weight, expected_patch_weight)
    torch.testing.assert_close(model.lm_head.weight, expected_lm_head_weight)
