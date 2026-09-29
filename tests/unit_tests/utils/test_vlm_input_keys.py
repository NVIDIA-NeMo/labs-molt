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

"""Tests for ``VLM_INPUT_KEYS`` umbrella in ``components/utils/model_utils.py``.

The umbrella centralizes multimodal kwarg names across known VLM families so the
recipe (and any other consumer) doesn't hardcode a per-model list.
"""

from __future__ import annotations

import importlib

from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS


def test_vlm_input_keys_is_tuple_of_strings():
    assert isinstance(VLM_INPUT_KEYS, tuple)
    assert len(VLM_INPUT_KEYS) > 0
    for k in VLM_INPUT_KEYS:
        assert isinstance(k, str)


def test_vlm_input_keys_no_duplicates():
    assert len(VLM_INPUT_KEYS) == len(set(VLM_INPUT_KEYS))


def test_vlm_input_keys_includes_input_ids():
    """The umbrella must include input_ids so the recipe can pass it through to
    the model's prepare step (which then swaps in inputs_embeds)."""
    assert "input_ids" in VLM_INPUT_KEYS


def test_vlm_input_keys_covers_nemotron_omni_modalities():
    """Nemotron-Omni image + video + sound keys must all be in the umbrella."""
    expected = {
        "pixel_values",
        "image_flags",
        "imgs_sizes",
        "pixel_values_videos",
        "sound_features",
        "sound_attention_mask",
    }
    missing = expected - set(VLM_INPUT_KEYS)
    assert not missing, f"missing Nemotron-Omni keys: {missing}"


def test_vlm_input_keys_covers_gemma4_modalities():
    """Gemma4 (PR #1914) keys must be in the umbrella."""
    expected = {"image_position_ids", "mm_token_type_ids"}
    missing = expected - set(VLM_INPUT_KEYS)
    assert not missing, f"missing Gemma4 keys: {missing}"


def test_vlm_input_keys_covers_qwen_kimi_mistral_grid_keys():
    """Qwen-VL/Kimi-VL/Mistral4 grid + sizes keys must be present."""
    expected = {"image_grid_hws", "image_grid_thw", "video_grid_thw", "image_sizes"}
    missing = expected - set(VLM_INPUT_KEYS)
    assert not missing, f"missing grid/size keys: {missing}"


def test_vlm_input_keys_covers_phi4mm_audio_keys():
    """Phi-4-MM audio keys present so future Phi-4-MM CP enablement just works."""
    expected = {"audio_input_values", "audio_attention_mask"}
    missing = expected - set(VLM_INPUT_KEYS)
    assert not missing, f"missing Phi-4-MM keys: {missing}"


def test_vlm_input_keys_importable_from_model_utils_module():
    """Any layer (recipe, distributed, datasets, tests) can import the umbrella."""
    mod = importlib.import_module("nemo_automodel.components.utils.model_utils")
    assert getattr(mod, "VLM_INPUT_KEYS", None) is VLM_INPUT_KEYS
