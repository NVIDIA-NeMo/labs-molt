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

"""Unit tests for the NemotronOmni state-dict adapter.

The adapter sits between the HF ``NemotronH_Nano_Omni_Reasoning_V3`` checkpoint
layout and the custom Automodel layout. The LLM portion is delegated to
``NemotronV3StateDictAdapter`` (already covered by its own tests), so we mock
that delegate here and focus on the vision/audio key remapping that this PR
introduces.
"""

from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.models.nemotron_omni.state_dict_adapter import (
    _VISION_PROJ_CUSTOM_TO_HF,
    _VISION_PROJ_HF_TO_CUSTOM,
    NemotronOmniStateDictAdapter,
)


@pytest.fixture
def adapter():
    """Adapter with the LLM delegate mocked out (LLM remap tested elsewhere)."""
    a = NemotronOmniStateDictAdapter.__new__(NemotronOmniStateDictAdapter)
    a.config = MagicMock()
    a.llm_config = MagicMock()
    a.moe_config = MagicMock()
    a.backend = MagicMock()
    a.dtype = torch.bfloat16
    a.vision_uses_native_radio = False
    # Echo input keys through unchanged — lets us verify *only* the
    # NemotronOmni-side key handling without depending on NemotronV3 internals.
    a._llm_adapter = MagicMock()
    a._llm_adapter.from_hf.side_effect = lambda sd, **kwargs: dict(sd)
    a._llm_adapter.to_hf.side_effect = lambda sd, **kwargs: dict(sd)
    a._llm_adapter.convert_single_tensor_to_hf.side_effect = lambda fqn, tensor, **kwargs: [(fqn, tensor)]
    return a


# ---------------------------------------------------------------------------
# from_hf — HF layout -> Automodel layout
# ---------------------------------------------------------------------------


def test_from_hf_remaps_vision_projector(adapter):
    """``mlp1.{0,1,3}.weight`` should become ``vision_projector.{norm,linear1,linear2}.weight``."""
    hf_sd = {
        "mlp1.0.weight": torch.zeros(8),
        "mlp1.1.weight": torch.zeros(16, 8),
        "mlp1.3.weight": torch.zeros(4, 16),
    }
    out = adapter.from_hf(dict(hf_sd))

    assert "vision_projector.norm.weight" in out
    assert "vision_projector.linear1.weight" in out
    assert "vision_projector.linear2.weight" in out
    # No HF-style mlp1.* keys leak through.
    assert not any(k.startswith("mlp1.") for k in out)


def test_from_hf_passes_vision_model_through(adapter):
    """``vision_model.*`` keys (the RADIO encoder) are emitted unchanged."""
    hf_sd = {
        "vision_model.radio_model.model.blocks.0.norm1.weight": torch.zeros(2),
        "vision_model.radio_model.input_conditioner.norm_mean": torch.zeros(3),
    }
    out = adapter.from_hf(dict(hf_sd))
    assert set(out) == set(hf_sd)


def test_from_hf_strips_sound_encoder_inner_prefix(adapter):
    """``sound_encoder.encoder.*`` -> ``sound_encoder.*`` (Parakeet wrapper unwrap)."""
    hf_sd = {
        "sound_encoder.encoder.layers.0.weight": torch.zeros(4),
    }
    out = adapter.from_hf(dict(hf_sd))
    assert "sound_encoder.layers.0.weight" in out
    assert "sound_encoder.encoder.layers.0.weight" not in out


def test_from_hf_passes_sound_projection_through(adapter):
    """``sound_projection.*`` keys are identical between HF and custom layouts."""
    hf_sd = {
        "sound_projection.norm.weight": torch.zeros(8),
        "sound_projection.linear1.weight": torch.zeros(16, 8),
        "sound_projection.linear2.weight": torch.zeros(4, 16),
    }
    out = adapter.from_hf(dict(hf_sd))
    assert set(out) == set(hf_sd)


def test_from_hf_routes_llm_through_v3_adapter_with_prefix(adapter):
    """``language_model.*`` keys are stripped, delegated, then re-prefixed."""
    hf_sd = {
        "language_model.backbone.embeddings.weight": torch.zeros(2),
        "language_model.lm_head.weight": torch.zeros(2),
    }
    out = adapter.from_hf(dict(hf_sd))

    # The mock delegate echoes input — keys come back with the prefix re-added.
    assert "language_model.backbone.embeddings.weight" in out
    assert "language_model.lm_head.weight" in out
    # Delegate was called with stripped keys.
    args, _ = adapter._llm_adapter.from_hf.call_args
    delegated = args[0]
    assert "backbone.embeddings.weight" in delegated
    assert "lm_head.weight" in delegated
    assert not any(k.startswith("language_model.") for k in delegated)


def test_view_loaded_llm_keys_restore_outer_prefix(adapter):
    adapter._llm_adapter.view_loaded_native_keys = {
        "model.layers.0.mixer.experts.gate_and_up_projs",
        "model.layers.0.mixer.experts.down_projs",
    }

    assert adapter.view_loaded_native_keys == {
        "language_model.model.layers.0.mixer.experts.gate_and_up_projs",
        "language_model.model.layers.0.mixer.experts.down_projs",
    }


# ---------------------------------------------------------------------------
# to_hf — Automodel layout -> HF layout (round-trip)
# ---------------------------------------------------------------------------


def test_to_hf_reverses_vision_projector_remap(adapter):
    custom_sd = {
        "vision_projector.norm.weight": torch.zeros(8),
        "vision_projector.linear1.weight": torch.zeros(16, 8),
        "vision_projector.linear2.weight": torch.zeros(4, 16),
    }
    out = adapter.to_hf(dict(custom_sd))
    assert "mlp1.0.weight" in out
    assert "mlp1.1.weight" in out
    assert "mlp1.3.weight" in out
    assert not any(k.startswith("vision_projector.") for k in out)


def test_to_hf_re_adds_sound_encoder_inner_prefix(adapter):
    custom_sd = {"sound_encoder.layers.0.weight": torch.zeros(4)}
    out = adapter.to_hf(dict(custom_sd))
    assert "sound_encoder.encoder.layers.0.weight" in out


def test_round_trip_vision_projector(adapter):
    """from_hf -> to_hf yields the original HF keys for the vision projector."""
    hf_sd = {
        "mlp1.0.weight": torch.tensor([1.0, 2.0]),
        "mlp1.1.weight": torch.tensor([[3.0]]),
        "mlp1.3.weight": torch.tensor([[4.0]]),
    }
    custom = adapter.from_hf(dict(hf_sd))
    back = adapter.to_hf(custom)
    assert set(back) == set(hf_sd)
    for k in hf_sd:
        assert torch.equal(back[k], hf_sd[k])


def test_round_trip_sound_encoder(adapter):
    """from_hf -> to_hf preserves ``sound_encoder.encoder.*`` keys."""
    hf_sd = {"sound_encoder.encoder.layers.3.bias": torch.tensor([0.5])}
    custom = adapter.from_hf(dict(hf_sd))
    back = adapter.to_hf(custom)
    assert set(back) == set(hf_sd)


def test_to_hf_exclude_key_regex(adapter):
    """``exclude_key_regex`` should filter out matching keys before remap."""
    custom_sd = {
        "vision_projector.norm.weight": torch.zeros(2),
        "sound_projection.norm.weight": torch.zeros(2),
    }
    out = adapter.to_hf(dict(custom_sd), exclude_key_regex=r"sound_projection\..*")
    assert "mlp1.0.weight" in out
    assert "sound_projection.norm.weight" not in out


# ---------------------------------------------------------------------------
# convert_single_tensor_to_hf — single-key path used by streaming save
# ---------------------------------------------------------------------------


def test_convert_single_tensor_vision_projector(adapter):
    pairs = adapter.convert_single_tensor_to_hf("vision_projector.linear1.weight", torch.zeros(2, 2))
    assert pairs == [("mlp1.1.weight", pairs[0][1])]


def test_convert_single_tensor_sound_encoder(adapter):
    pairs = adapter.convert_single_tensor_to_hf("sound_encoder.layers.0.weight", torch.zeros(2))
    new_fqn, _ = pairs[0]
    assert new_fqn == "sound_encoder.encoder.layers.0.weight"


def test_convert_single_tensor_pass_through(adapter):
    """``vision_model.*`` and ``sound_projection.*`` keep their original FQNs."""
    pairs_v = adapter.convert_single_tensor_to_hf("vision_model.radio_model.x", torch.zeros(1))
    assert pairs_v[0][0] == "vision_model.radio_model.x"

    pairs_s = adapter.convert_single_tensor_to_hf("sound_projection.norm.weight", torch.zeros(1))
    assert pairs_s[0][0] == "sound_projection.norm.weight"


def test_convert_single_tensor_llm_delegates(adapter):
    """LLM keys are stripped, delegated, then re-prefixed."""
    pairs = adapter.convert_single_tensor_to_hf("language_model.lm_head.weight", torch.zeros(1))
    assert pairs[0][0] == "language_model.lm_head.weight"
    args, _ = adapter._llm_adapter.convert_single_tensor_to_hf.call_args
    assert args[0] == "lm_head.weight"


# ---------------------------------------------------------------------------
# Internal mapping table sanity
# ---------------------------------------------------------------------------


def test_mapping_tables_are_inverses():
    """``_VISION_PROJ_CUSTOM_TO_HF`` is the inverse of ``_VISION_PROJ_HF_TO_CUSTOM``."""
    for hf_key, custom_key in _VISION_PROJ_HF_TO_CUSTOM.items():
        assert _VISION_PROJ_CUSTOM_TO_HF[custom_key] == hf_key
    assert len(_VISION_PROJ_HF_TO_CUSTOM) == len(_VISION_PROJ_CUSTOM_TO_HF)


# ---------------------------------------------------------------------------
# Native RadioModel remap (NemotronH_Omni_Reasoning_V3 / Super-scale omni)
# ---------------------------------------------------------------------------


@pytest.fixture
def native_adapter():
    """Adapter configured for a checkpoint whose vision tower is the native RadioModel."""
    a = NemotronOmniStateDictAdapter.__new__(NemotronOmniStateDictAdapter)
    a.config = MagicMock()
    a.llm_config = MagicMock()
    a.moe_config = MagicMock()
    a.backend = MagicMock()
    a.dtype = torch.bfloat16
    a.vision_uses_native_radio = True
    a._llm_adapter = MagicMock()
    a._llm_adapter.from_hf.side_effect = lambda sd, **kwargs: dict(sd)
    a._llm_adapter.to_hf.side_effect = lambda sd, **kwargs: dict(sd)
    a._llm_adapter.convert_single_tensor_to_hf.side_effect = lambda fqn, tensor, **kwargs: [(fqn, tensor)]
    return a


def test_from_hf_renames_legacy_radio_keys_to_native(native_adapter):
    """Legacy ``radio_model.model.blocks``/``patch_generator`` keys remap to the native tree."""
    hf_sd = {
        "vision_model.radio_model.model.blocks.0.norm1.weight": torch.zeros(2),
        "vision_model.radio_model.model.blocks.0.attn.proj.weight": torch.zeros(2, 2),
        "vision_model.radio_model.model.patch_generator.cls_token.token": torch.zeros(1),
        "vision_model.radio_model.input_conditioner.norm_mean": torch.zeros(3),
    }
    out = native_adapter.from_hf(dict(hf_sd))

    assert "vision_model.encoder.layer.0.norm1.weight" in out
    assert "vision_model.encoder.layer.0.attention.output.dense.weight" in out
    assert "vision_model.embeddings.cls_register_token" in out
    assert "vision_model.input_conditioner.norm_mean" in out
    assert not any("radio_model" in k for k in out)


def test_from_hf_splits_fused_qkv_for_native_radio(native_adapter):
    """A fused ``attn.qkv`` tensor splits into three native query/key/value tensors."""
    qkv = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)  # 3 chunks of 3 rows each
    hf_sd = {"vision_model.radio_model.model.blocks.0.attn.qkv.weight": qkv}
    out = native_adapter.from_hf(dict(hf_sd))

    q_key = "vision_model.encoder.layer.0.attention.attention.query.weight"
    k_key = "vision_model.encoder.layer.0.attention.attention.key.weight"
    v_key = "vision_model.encoder.layer.0.attention.attention.value.weight"
    assert q_key in out and k_key in out and v_key in out
    assert torch.equal(out[q_key], qkv[0:3])
    assert torch.equal(out[k_key], qkv[3:6])
    assert torch.equal(out[v_key], qkv[6:9])


def test_native_radio_round_trip(native_adapter):
    """from_hf -> to_hf recovers the original legacy keys, shapes, and values (incl. fused qkv)."""
    hf_sd = {
        "vision_model.radio_model.model.blocks.0.norm1.weight": torch.randn(4),
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight": torch.randn(9, 4),
        "vision_model.radio_model.model.patch_generator.pos_embed": torch.randn(2, 4),
    }
    custom_sd = native_adapter.from_hf(dict(hf_sd))
    round_tripped = native_adapter.to_hf(dict(custom_sd))

    assert set(round_tripped) == set(hf_sd)
    for key, value in hf_sd.items():
        assert torch.equal(round_tripped[key], value)


def test_supports_low_memory_dcp_load_false_for_native_radio(native_adapter):
    """Native RadioModel's fused-qkv split requires a materializing conversion."""
    native_adapter._llm_adapter.supports_low_memory_dcp_load = True
    assert native_adapter.supports_low_memory_dcp_load is False


def test_convert_single_tensor_keeps_native_radio_keys(native_adapter):
    """Per-tensor conversion must not rename native RADIO keys to the legacy tree.

    The legacy layout needs query/key/value fused into ``attn.qkv``, which a single-tensor
    conversion cannot do; renaming without fusing yields keys no loader recognizes.
    Streaming consumers (NeMo-RL refit into vLLM) load the native per-shard names instead.
    """
    for fqn in (
        "vision_model.encoder.layer.0.attention.attention.query.weight",
        "vision_model.encoder.layer.0.attention.output.dense.weight",
        "vision_model.embeddings.patch_projection.weight",
        "vision_model.encoder.layer.0.norm1.weight",
    ):
        t = torch.zeros(2)
        assert native_adapter.convert_single_tensor_to_hf(fqn, t) == [(fqn, t)]


def test_to_hf_still_emits_legacy_radio_keys_for_native_radio(native_adapter):
    """Whole-state-dict export keeps the legacy (fused qkv) layout for checkpoint compatibility."""
    custom_sd = {
        "vision_model.encoder.layer.0.attention.attention.query.weight": torch.zeros(3, 4),
        "vision_model.encoder.layer.0.attention.attention.key.weight": torch.ones(3, 4),
        "vision_model.encoder.layer.0.attention.attention.value.weight": torch.full((3, 4), 2.0),
        "vision_model.encoder.layer.0.attention.output.dense.weight": torch.zeros(4, 4),
    }
    out = native_adapter.to_hf(dict(custom_sd))
    assert set(out) == {
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight",
        "vision_model.radio_model.model.blocks.0.attn.proj.weight",
    }
    assert out["vision_model.radio_model.model.blocks.0.attn.qkv.weight"].shape == (9, 4)


def test_super35_final_vision_norm_roundtrip(adapter):
    """Preserve both affine tensors under their original HF checkpoint names."""
    hf = {
        "vision_projector.vision_final_layernorm.weight": torch.randn(8),
        "vision_projector.vision_final_layernorm.bias": torch.randn(8),
    }
    converted = adapter.from_hf(dict(hf))
    restored = adapter.to_hf(dict(converted))
    assert set(converted) == set(hf) == set(restored)
    for key in hf:
        assert torch.equal(restored[key], hf[key])
