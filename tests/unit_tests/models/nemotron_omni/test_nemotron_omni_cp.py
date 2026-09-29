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

"""Tests for the CP enablement on ``NemotronOmniForConditionalGeneration``.

Covers:
  - ``prepare_model_inputs_for_cp`` returns a dict containing ``inputs_embeds``
    with the expected shape and image/video/sound token positions filled.
  - ``prepare_model_inputs_for_cp`` is a sharder-only hook: it returns the CP
    sharder without entering the LLM body (embed / vision splice / shard happen
    in the model's own forward).
  - ``forward(inputs_embeds=...)`` skips the multimodal scatter block.

The model is constructed via ``object.__new__`` + minimal stub submodules so we
do not load the 30B real model.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniForConditionalGeneration,
)

IMG_TOKEN_ID = 18
SOUND_TOKEN_ID = 27
HIDDEN = 8


class _StubEmbedding(nn.Module):
    """nn.Embedding-equivalent that always returns a small constant per id."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.hidden = hidden

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Deterministic: each id maps to a vector of (id+1)/100.
        out = (ids.float().unsqueeze(-1) + 1.0) / 100.0
        return out.expand(*ids.shape, self.hidden).clone()


class _StubLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._embed = _StubEmbedding()
        self.captured: dict = {}

    def get_input_embeddings(self):
        return self._embed

    def forward(self, **kwargs):
        # Capture the (post-embed, post-splice, post-CP-shard) inputs_embeds the
        # wrapper forward hands the LM so tests can assert on it.
        self.captured = kwargs
        return SimpleNamespace(logits=kwargs.get("inputs_embeds"), hidden_states=None)


def _make_omni_stub(*, with_sound_encoder: bool = True):
    """Construct a NemotronOmniForConditionalGeneration with only the attrs the
    CP path touches. Vision/video/sound encoders are stubs producing constants."""
    self = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(self)
    self.img_context_token_id = IMG_TOKEN_ID
    self.sound_context_token_id = SOUND_TOKEN_ID
    self.language_model = _StubLanguageModel()

    # extract_feature returns a tensor of constant 9.0 with shape [N_tiles, K, H]
    # where the model will reshape to (-1, H) and scatter onto img positions.
    def _extract_feature(pixel_values):
        # pixel_values: [N_tiles, C, H, W] -> emit one feature per tile.
        n_tiles = pixel_values.shape[0]
        return torch.full((n_tiles, 1, HIDDEN), 9.0)

    def _extract_feature_dynamic(pixel_values, imgs_sizes):
        n = pixel_values.shape[0]
        return torch.full((1, n, HIDDEN), 7.0)

    def _extract_video_feature(pixel_values_videos):
        n = pixel_values_videos.shape[0]
        return torch.full((1, n, HIDDEN), 5.0)

    def _extract_sound_feature(features, attention_mask):
        n = features.shape[0]
        return torch.full((n, 1, HIDDEN), 3.0)

    self.extract_feature = _extract_feature
    self.extract_feature_dynamic = _extract_feature_dynamic
    self.extract_video_feature = _extract_video_feature
    self.extract_sound_feature = _extract_sound_feature
    self.sound_encoder = nn.Identity() if with_sound_encoder else None
    return self


# -----------------------------------------------------------------------------
# prepare_model_inputs_for_cp
# -----------------------------------------------------------------------------


class _FakeCPMesh:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


def _forward_embeds(model, **forward_kwargs):
    """Run the wrapper forward and return the inputs_embeds it handed the LM."""
    model(**forward_kwargs)
    return model.language_model.captured["inputs_embeds"]


def test_prepare_model_inputs_for_cp_is_sharder_only():
    """The CP hook is sharder-only: it consumes nothing and returns a
    ContextParallelSharder; embed + splice + shard run inside forward."""
    from nemo_automodel.components.distributed.context_parallel.sharder import (
        ContextParallelSharder,
        round_robin_local_indices,
        shard_batch_aux_only,
    )

    model = _make_omni_stub()
    input_ids = torch.tensor([[1, IMG_TOKEN_ID, IMG_TOKEN_ID, 4]])
    batch = {"input_ids": input_ids, "pixel_values": torch.zeros(2, 3, 4, 4), "image_flags": torch.tensor([[1], [1]])}
    out = model.prepare_model_inputs_for_cp(batch)

    assert set(out) == {"cp_sharder"}
    sharder = out["cp_sharder"]
    assert isinstance(sharder, ContextParallelSharder)
    assert sharder.shard_batch is shard_batch_aux_only
    assert sharder.local_token_global_indices is round_robin_local_indices
    # nothing consumed: input_ids / media stay for the forward
    assert batch["input_ids"] is input_ids and "pixel_values" in batch


def test_prepare_model_inputs_for_cp_records_global_media_masks_for_thd():
    """Packed THD defers sharding and records global image/sound placeholder masks."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[1, IMG_TOKEN_ID, SOUND_TOKEN_ID, 4]])

    out = model.prepare_model_inputs_for_cp({"input_ids": input_ids, "qkv_format": "thd"})

    assert set(out) == {"_nemotron_omni_global_image_mask", "_nemotron_omni_global_sound_mask"}
    assert out["_nemotron_omni_global_image_mask"].tolist() == [False, True, False, False]
    assert out["_nemotron_omni_global_sound_mask"].tolist() == [False, False, True, False]


def test_select_local_media_features_maps_global_placeholders_to_local_rows(monkeypatch):
    """The local THD token order selects the matching global media-feature rows."""

    def _partition_indices(
        cu_seqlens: torch.Tensor,
        total_tokens: int,
        cp_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        """Return a fixed Tensor of shape [local_tokens] for one global THD sequence."""
        assert cu_seqlens.tolist() == [0, 8]
        assert (total_tokens, cp_size, cp_rank) == (8, 2, 0)
        return torch.tensor([0, 1, 6, 7])

    monkeypatch.setitem(
        sys.modules,
        "transformer_engine_torch",
        SimpleNamespace(thd_get_partitioned_indices=_partition_indices),
    )
    features = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    global_mask = torch.tensor([False, True, True, False, False, True, True, False])
    local_selected = torch.tensor([False, True, True, False])

    local = NemotronOmniForConditionalGeneration._select_local_media_features(
        features,
        global_mask,
        local_selected,
        torch.tensor([0, 8], dtype=torch.int32),
        cp_size=2,
        cp_rank=0,
    )

    torch.testing.assert_close(local, torch.tensor([[10.0], [40.0]]))


def test_forward_text_only_embeds():
    """No multimodal inputs -> embeds are just embed_tokens(input_ids)."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[5, 6, 7]])
    out = _forward_embeds(model, input_ids=input_ids)
    expected = model.language_model.get_input_embeddings()(input_ids)
    assert torch.equal(out, expected)


def test_forward_image_scatter_at_placeholder_positions():
    """Image positions in input_ids must receive the vit feature value (9.0)."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[1, IMG_TOKEN_ID, IMG_TOKEN_ID, 4]])
    out = _forward_embeds(
        model, input_ids=input_ids, pixel_values=torch.zeros(2, 3, 4, 4), image_flags=torch.tensor([[1], [1]])
    )
    assert out.shape == (1, 4, HIDDEN)
    assert torch.allclose(out[0, 1], torch.full((HIDDEN,), 9.0))
    assert torch.allclose(out[0, 2], torch.full((HIDDEN,), 9.0))
    expected_pos0 = (torch.tensor([1.0]) + 1.0) / 100.0
    expected_pos3 = (torch.tensor([4.0]) + 1.0) / 100.0
    assert torch.allclose(out[0, 0], expected_pos0.expand(HIDDEN))
    assert torch.allclose(out[0, 3], expected_pos3.expand(HIDDEN))


def test_forward_dynamic_res_takes_priority_over_static():
    """When imgs_sizes is provided, the dynamic-res branch handles vision."""
    model = _make_omni_stub()
    out = _forward_embeds(
        model,
        input_ids=torch.tensor([[1, IMG_TOKEN_ID, 3]]),
        pixel_values=torch.zeros(1, 3, 8, 8),
        imgs_sizes=torch.tensor([[8, 8]]),
    )
    # extract_feature_dynamic stub returns 7.0; extract_feature returns 9.0
    assert torch.allclose(out[0, 1], torch.full((HIDDEN,), 7.0))


def test_forward_video_scatter_at_img_token_positions():
    """Video features scatter at the img_context_token_id positions."""
    model = _make_omni_stub()
    out = _forward_embeds(
        model,
        input_ids=torch.tensor([[1, IMG_TOKEN_ID, IMG_TOKEN_ID, 4]]),
        pixel_values_videos=torch.zeros(2, 3, 4, 4),
    )
    assert torch.allclose(out[0, 1], torch.full((HIDDEN,), 5.0))
    assert torch.allclose(out[0, 2], torch.full((HIDDEN,), 5.0))


def test_forward_sound_scatter_at_sound_token():
    model = _make_omni_stub()
    out = _forward_embeds(
        model,
        input_ids=torch.tensor([[SOUND_TOKEN_ID, 2, SOUND_TOKEN_ID]]),
        sound_features=torch.zeros(2, 4, 16),
        sound_attention_mask=torch.ones(2, 4),
    )
    assert torch.allclose(out[0, 0], torch.full((HIDDEN,), 3.0))
    assert torch.allclose(out[0, 2], torch.full((HIDDEN,), 3.0))
    expected_pos1 = (torch.tensor([2.0]) + 1.0) / 100.0
    assert torch.allclose(out[0, 1], expected_pos1.expand(HIDDEN))


def test_forward_sound_skipped_when_no_sound_encoder():
    """If model.sound_encoder is None (sound disabled), the sound branch is a no-op."""
    model = _make_omni_stub(with_sound_encoder=False)
    input_ids = torch.tensor([[SOUND_TOKEN_ID, 2, SOUND_TOKEN_ID]])
    out = _forward_embeds(model, input_ids=input_ids, sound_features=torch.zeros(2, 4, 16))
    expected = model.language_model.get_input_embeddings()(input_ids)
    assert torch.equal(out, expected)


def test_forward_cp_shards_embedded_sequence():
    """With a CP mesh installed, forward embeds+splices the full sequence then
    keeps this rank's round-robin chunk pair, so the LM sees the local shard."""
    sdpa_before = F.scaled_dot_product_attention
    model = _make_omni_stub()
    model.cp_mesh = _FakeCPMesh(2)
    input_ids = torch.tensor([[1, IMG_TOKEN_ID, IMG_TOKEN_ID, 4, 5, 6, 7, 8]])  # len 8 == multiple of 2*cp
    out = _forward_embeds(
        model, input_ids=input_ids, pixel_values=torch.zeros(2, 3, 4, 4), image_flags=torch.tensor([[1], [1]])
    )
    assert out.shape == (1, 4, HIDDEN)  # 8 // cp_size(2) = 4 local tokens
    assert F.scaled_dot_product_attention is sdpa_before


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# prepare_model_inputs_for_cp (sharder-only hook)
# -----------------------------------------------------------------------------


def test_prepare_model_inputs_for_cp_is_sharder_only_and_skips_lm(monkeypatch):
    """The sharder-only CP hook returns just the CP sharder and never touches the
    LLM body (embed / vision splice / shard happen in the model's own forward)."""
    model = _make_omni_stub()

    def _llm_must_not_run(*args, **kwargs):
        raise AssertionError("language_model should NOT be called by the CP hook")

    model.language_model.forward = _llm_must_not_run

    out = model.prepare_model_inputs_for_cp(
        {
            "input_ids": torch.tensor([[1, IMG_TOKEN_ID, 3]]),
            "pixel_values": torch.zeros(1, 3, 4, 4),
            "image_flags": torch.tensor([[1]]),
        }
    )
    # Sharder-only hook: returns the CP sharder, does not embed or call the LM.
    assert isinstance(out, dict)
    assert set(out) == {"cp_sharder"}


def test_forward_inputs_embeds_skips_multimodal_scatter_block():
    """If caller passes inputs_embeds (the post-CP-shard path), forward should
    NOT call extract_feature etc. — the embeds are already correct."""
    model = _make_omni_stub()
    sentinel = []
    orig_extract = model.extract_feature

    def _spy(pixel_values):
        sentinel.append(pixel_values)
        return orig_extract(pixel_values)

    model.extract_feature = _spy

    # Mock LM to swallow the call so forward can complete
    def _fake_llm(input_ids=None, inputs_embeds=None, **kw):
        return SimpleNamespace(logits=inputs_embeds, loss=None, hidden_states=None)

    model.language_model.forward = _fake_llm
    model.language_model.__call__ = _fake_llm  # nn.Module.__call__ wraps forward

    pre_built = torch.randn(1, 3, HIDDEN)
    model.forward(
        inputs_embeds=pre_built,
        pixel_values=torch.zeros(1, 3, 4, 4),  # provided but should be IGNORED
        image_flags=torch.tensor([[1]]),
    )
    # extract_feature must NOT have been called
    assert sentinel == [], "extract_feature should be skipped when inputs_embeds is supplied"
