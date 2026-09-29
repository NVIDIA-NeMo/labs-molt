# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Qwen3.5 dense VLM CP pre-embedding.

``Qwen3_5ForConditionalGeneration.prepare_model_inputs_for_cp`` builds
full-sequence multimodal embeddings and mRoPE positions *before* context-parallel
sharding (the CP linear-attention layers recover the dense token order
internally via the DualChunkSwap layout, so no seq_index is threaded here).

Instantiating the full VL model is expensive, so we build a barebones instance
via ``__new__`` and stub the heavy submodules (visual encoder, rope-index helper,
embedding table).
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.models.qwen3_5")

import nemo_automodel.components.models.qwen3_5.model as qwen3_5_model_module
from nemo_automodel.components.models.qwen3_5.model import (
    HFQwen3_5Model,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)


def _build_model(*, rope_index=None, image_token_id=None, video_token_id=None, vision_start_token_id=None):
    """Build a barebones Qwen3_5ForConditionalGeneration with stubbed deps."""
    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)

    hidden = 4

    def _embed(input_ids):
        # Deterministic embeddings: [B, S, H] where each value mirrors the token id.
        return input_ids.unsqueeze(-1).expand(*input_ids.shape, hidden).float()

    # Instance attribute shadows the class method.
    model.get_input_embeddings = lambda: _embed

    inner = types.SimpleNamespace()
    inner.visual = None  # no rotary_pos_emb attribute -> hasattr() is False

    def _default_rope_index(input_ids, **kwargs):
        # [3, B, S] mRoPE positions + a rope_delta per row.
        b, s = input_ids.shape
        pos = torch.arange(s).view(1, 1, s).expand(3, b, s).contiguous()
        return pos, torch.zeros(b, 1)

    inner.get_rope_index = rope_index or _default_rope_index
    inner.rope_deltas = None
    model.model = inner

    model.config = types.SimpleNamespace(
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
    )
    return model


class TestPrepareModelInputsForCP:
    def test_requires_input_ids(self):
        model = _build_model()
        with pytest.raises(ValueError, match="requires input_ids"):
            model.prepare_model_inputs_for_cp({"input_ids": None})

    def test_returns_sharder_and_positions_only(self):
        """Sharder-only hook: no inputs_embeds (the forward embeds), full mRoPE
        positions returned for the aux shard, mm_token_type_ids consumed."""
        from nemo_automodel.components.distributed.context_parallel.sharder import (
            ContextParallelSharder,
            round_robin_local_indices,
            shard_batch_aux_only,
        )

        model = _build_model()
        out = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[5, 6, 7, 8]])})

        assert "inputs_embeds" not in out  # embedding happens in forward now
        sharder = out["cp_sharder"]
        assert isinstance(sharder, ContextParallelSharder)
        assert sharder.shard_batch is shard_batch_aux_only
        assert sharder.local_token_global_indices is round_robin_local_indices
        # position_ids came from get_rope_index (mRoPE [3, B, S]); aux shard slices it.
        assert out["position_ids"].shape == (3, 1, 4)
        # mm_token_type_ids is consumed by get_rope_index -> returned as a None marker.
        assert out["mm_token_type_ids"] is None
        assert model.model.rope_deltas is not None

    def test_input_ids_and_media_not_consumed(self):
        """input_ids stays in the batch for the forward's in-forward embed+splice."""
        model = _build_model()
        out = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[5, 6, 7, 8]])})
        assert "input_ids" not in out  # not returned as a None marker -> stays in batch

    def test_existing_position_ids_not_recomputed(self):
        called = {"count": 0}

        def _rope(input_ids, **kwargs):
            called["count"] += 1
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope)
        pos = torch.arange(4).view(1, 4)
        out = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[5, 6, 7, 8]]), "position_ids": pos})

        assert called["count"] == 0, "get_rope_index must not run when position_ids provided"
        assert out["position_ids"] is pos

    def test_image_grid_hws_promoted_to_thw(self):
        """image_grid_hws of shape [N, 2] is promoted to [N, 3] for get_rope_index and
        written back to the batch (so the forward's embed path reads the T/H/W grid)."""
        captured = {}

        def _rope(input_ids, **kwargs):
            captured.update(kwargs)
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope)
        image_grid_hws = torch.tensor([[2, 2]])  # [N, 2]
        out = model.prepare_model_inputs_for_cp(
            {
                "input_ids": torch.tensor([[5, 6, 7, 8]]),
                "image_grid_hws": image_grid_hws,
            }
        )
        assert captured["image_grid_thw"].tolist() == [[1, 2, 2]]
        # promoted grid is returned for the forward; the raw hws key is dropped.
        assert out["image_grid_thw"].tolist() == [[1, 2, 2]]
        assert out["image_grid_hws"] is None

    def test_image_grid_hws_already_thw_passes_through(self):
        """image_grid_hws already shaped [N, 3] is used as-is (no temporal column prepended)."""
        captured = {}

        def _rope(input_ids, **kwargs):
            captured.update(kwargs)
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope)
        image_grid_hws = torch.tensor([[1, 2, 2]])  # [N, 3]
        model.prepare_model_inputs_for_cp(
            {
                "input_ids": torch.tensor([[5, 6, 7, 8]]),
                "image_grid_hws": image_grid_hws,
            }
        )
        assert captured["image_grid_thw"].tolist() == [[1, 2, 2]]

    def test_mm_token_type_ids_synthesized_from_token_ids(self):
        """When get_rope_index accepts mm_token_type_ids, it is built from image/video token ids."""
        captured = {}

        def _rope(input_ids, *, image_grid_thw=None, video_grid_thw=None, attention_mask=None, mm_token_type_ids=None):
            captured["mm_token_type_ids"] = mm_token_type_ids
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope, image_token_id=6, video_token_id=8)
        model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[5, 6, 7, 8]])})

        # token 6 -> image (1), token 8 -> video (2), others 0.
        assert captured["mm_token_type_ids"].tolist() == [[0, 1, 0, 2]]


class TestPopStagedVlmMedia:
    def test_drops_orphaned_image_media(self):
        model = _build_model(image_token_id=99, video_token_id=98, vision_start_token_id=97)
        kwargs = {
            "pixel_values": torch.randn(4, 8),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }

        pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw = model._pop_staged_vlm_media(
            torch.tensor([[10, 11, 12]]),
            kwargs,
        )

        assert pixel_values is None
        assert pixel_values_videos is None
        assert image_grid_thw is None
        assert video_grid_thw is None

    def test_keeps_image_media_when_placeholder_exists(self):
        model = _build_model(image_token_id=99, video_token_id=98, vision_start_token_id=97)
        pixel_values_in = torch.randn(4, 8)
        image_grid_in = torch.tensor([[1, 4, 4]])
        kwargs = {
            "pixel_values": pixel_values_in,
            "image_grid_thw": image_grid_in,
        }

        pixel_values, _, image_grid_thw, _ = model._pop_staged_vlm_media(
            torch.tensor([[10, 99, 12]]),
            kwargs,
        )

        assert pixel_values is pixel_values_in
        assert image_grid_thw is image_grid_in


class TestEmbedAndSpliceForCP:
    """The in-forward embed + vision splice (moved out of the CP hook)."""

    def test_image_features_scattered_into_embeds(self):
        """pixel_values path: image features replace image-token embeddings via masked_scatter."""
        model = _build_model(image_token_id=99)

        # Visual with a rotary_pos_emb so the device-move branch is exercised.
        moved = {}
        model.model.visual = types.SimpleNamespace(
            rotary_pos_emb=types.SimpleNamespace(to=lambda dev: moved.setdefault("dev", dev))
        )

        feat = torch.full((1, 4), 8.0)  # one image token, hidden=4
        model.model.get_image_features = lambda pixel_values, image_grid_thw=None, return_dict=True: (
            types.SimpleNamespace(pooler_output=[feat])
        )

        def _mask(input_ids, *, inputs_embeds=None, image_features=None, video_features=None):
            image_mask = (input_ids == 99).unsqueeze(-1).expand_as(inputs_embeds)
            return image_mask, torch.zeros_like(image_mask)

        model.model.get_placeholder_mask = _mask

        ids = torch.tensor([[5, 99, 7]])
        emb = model._embed_and_splice_for_cp(
            ids,
            pixel_values=torch.zeros(1, 3, 2, 2),
            pixel_values_videos=None,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            video_grid_thw=None,
        )
        assert torch.allclose(emb[0, 1], torch.full((4,), 8.0))  # image token overwritten
        assert torch.allclose(emb[0, 0], torch.full((4,), 5.0))  # text token untouched
        assert moved["dev"] is not None  # visual.rotary_pos_emb.to(...) ran

    def test_video_features_scattered_into_embeds(self):
        """pixel_values_videos path: video features replace video-token embeddings."""
        model = _build_model(video_token_id=88)

        feat = torch.full((1, 4), 3.0)  # one video token, hidden=4
        model.model.get_video_features = lambda pixel_values_videos, video_grid_thw=None, return_dict=True: (
            types.SimpleNamespace(pooler_output=[feat])
        )

        def _mask(input_ids, *, inputs_embeds=None, image_features=None, video_features=None):
            video_mask = (input_ids == 88).unsqueeze(-1).expand_as(inputs_embeds)
            return torch.zeros_like(video_mask), video_mask

        model.model.get_placeholder_mask = _mask

        ids = torch.tensor([[5, 88, 7]])
        emb = model._embed_and_splice_for_cp(
            ids,
            pixel_values=None,
            pixel_values_videos=torch.zeros(1, 3, 2, 2),
            image_grid_thw=None,
            video_grid_thw=torch.tensor([[1, 2, 2]]),
        )
        assert torch.allclose(emb[0, 1], torch.full((4,), 3.0))  # video token overwritten
        assert torch.allclose(emb[0, 2], torch.full((4,), 7.0))  # text token untouched

    @pytest.mark.parametrize("is_video", [False, True])
    def test_active_vision_frame_shard_routes_exact_visual_tower(self, monkeypatch, is_video):
        """Active CP sharding must bypass the replicated HF feature helpers."""
        model = _build_model()
        visual = types.SimpleNamespace(dtype=torch.bfloat16)
        model.model.visual = visual
        model.model.get_image_features = lambda *args, **kwargs: pytest.fail("replicated image path used")
        model.model.get_video_features = lambda *args, **kwargs: pytest.fail("replicated video path used")

        pixel_values = torch.randn(8, 12, dtype=torch.float32)
        grid_thw = torch.tensor([[1, 2, 4]], dtype=torch.long)
        expected = torch.randn(2, 4)
        captured = {}

        def _fake_distribute(visual_arg, pixel_arg, grid_arg):
            captured.update(visual=visual_arg, pixel=pixel_arg, grid=grid_arg)
            return types.SimpleNamespace(pooler_output=expected)

        monkeypatch.setattr(qwen3_5_model_module, "cp_vision_frame_sharding_active", lambda: True)
        monkeypatch.setattr(qwen3_5_model_module, "maybe_distribute_visual", _fake_distribute)

        actual = model._encode_vision_for_cp(pixel_values, grid_thw, is_video=is_video)

        assert actual is expected
        assert captured["visual"] is visual
        assert captured["grid"] is grid_thw
        assert captured["pixel"].dtype == visual.dtype
        torch.testing.assert_close(captured["pixel"].float(), pixel_values, atol=5e-3, rtol=5e-3)


class TestQwen3_5ModelForward:
    def _build_inner_model(self):
        model = Qwen3_5Model.__new__(Qwen3_5Model)
        nn.Module.__init__(model)
        model.visual = types.SimpleNamespace(rotary_pos_emb=types.SimpleNamespace(to=lambda device: None))
        model.get_input_embeddings = lambda: lambda input_ids: pytest.fail("media forward should keep input_ids")
        return model

    def test_media_forward_keeps_input_ids_for_hf_placeholder_mask(self, monkeypatch):
        model = self._build_inner_model()
        captured = {}
        sentinel = object()

        def _fake_hf_forward(self, **kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(HFQwen3_5Model, "forward", _fake_hf_forward)

        input_ids = torch.tensor([[5, 99, 7]])
        pixel_values = torch.randn(4, 8)
        out = model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )

        assert out is sentinel
        assert captured["input_ids"] is input_ids
        assert captured["inputs_embeds"] is None
        assert captured["pixel_values"] is pixel_values

    def test_media_forward_accepts_hidden_states_as_input_ids(self, monkeypatch):
        model = self._build_inner_model()
        captured = {}
        sentinel = object()

        def _fake_hf_forward(self, **kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(HFQwen3_5Model, "forward", _fake_hf_forward)

        hidden_states = torch.randn(1, 3, 4)
        out = model.forward(
            input_ids=hidden_states,
            pixel_values=torch.randn(4, 8),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )

        assert out is sentinel
        assert captured["input_ids"] is None
        assert captured["inputs_embeds"] is hidden_states


class _FakeCPMesh:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class TestPipelineStageMetas:
    """get_pipeline_stage_metas: CP shards stage outputs; cp=1 stays symmetric."""

    def _model(self, *, cp_size, lm_head):
        model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
        nn.Module.__init__(model)
        model.config = types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=8, vocab_size=32))
        model.lm_head = nn.Linear(8, 32, bias=False) if lm_head else None
        model.cp_mesh = _FakeCPMesh(cp_size) if cp_size > 1 else None
        return model

    def test_cp_shards_stage_outputs(self):
        model = self._model(cp_size=2, lm_head=True)
        ins, outs = model.get_pipeline_stage_metas(is_first=True, microbatch_size=1, seq_len=6, dtype=torch.float32)
        # first stage consumes the FULL token ids; output is the local shard (pad 6->8, //2 = 4)
        assert ins[0].shape == (1, 6) and ins[0].dtype == torch.long
        assert outs[0].shape == (1, 4, 32)  # last stage (lm_head) -> vocab

    def test_cp_middle_stage_local_hidden(self):
        model = self._model(cp_size=2, lm_head=False)
        ins, outs = model.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=6, dtype=torch.float32)
        assert ins[0].shape == (1, 4, 8)  # local hidden in
        assert outs[0].shape == (1, 4, 8)  # local hidden out (no lm_head)

    def test_cp1_symmetric_matches_default(self):
        model = self._model(cp_size=1, lm_head=True)
        ins, outs = model.get_pipeline_stage_metas(is_first=True, microbatch_size=2, seq_len=5, dtype=torch.float32)
        assert ins[0].shape == (2, 5) and ins[0].dtype == torch.long
        assert outs[0].shape == (2, 5, 32)  # full length, no CP shard
