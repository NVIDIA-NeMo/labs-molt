# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the tile-based vision path on mixed-resolution batches.

`nemotron_omni_collate_fn` cannot stack images of differing resolutions, so it
hands `forward()` a *list* of per-image tensors instead of a dense
[num_tiles, C, H, W] tensor. That happens for any variable-resolution dataset
(e.g. CORD-V2) as soon as `local_batch_size > 1`. The vision tower and LM are
mocked so these tests stay CPU-only and don't need RADIO weights.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniForConditionalGeneration,
    VisionProjector,
)


class _StubVisionModel(nn.Module):
    """Dense-only vision tower, matching HF RADIO: it accepts a single
    [B, C, H, W] tensor and returns (H//p)*(W//p) patch features per image."""

    def __init__(self, patch_size: int, c_feat: int):
        super().__init__()
        self.patch_size = patch_size
        self.c_feat = c_feat
        self.dummy = nn.Parameter(torch.zeros(1))
        self.received_shapes: list[tuple[int, int, int]] = []
        # extract_feature reads the patch size off this attribute chain.
        self.radio_model = SimpleNamespace(
            model=SimpleNamespace(patch_generator=SimpleNamespace(patch_size=patch_size))
        )

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), "RADIO only accepts a dense tensor"
        b, _, h, w = x.shape
        self.received_shapes.append((b, h, w))
        length = (h // self.patch_size) * (w // self.patch_size)
        # Fill with a per-resolution marker so callers can verify which image's
        # features landed in which slot. Values stay exact in bfloat16.
        return SimpleNamespace(features=torch.full((b, length, self.c_feat), _marker(h, w), dtype=torch.float32))


def _marker(h: int, w: int) -> float:
    """Small, bfloat16-exact value identifying an (h, w) image."""
    return float((h // 8) * 10 + (w // 8))


class _IdentityProjector(VisionProjector):
    def __init__(self):
        nn.Module.__init__(self)
        self.vision_final_layernorm = None
        self.downsample_ratio = 0.5
        self.norm = nn.Identity()
        self.linear1 = nn.Identity()
        self.activation = nn.Identity()
        self.linear2 = nn.Identity()


class _StubLM(nn.Module):
    """Mocks just enough of NemotronV3ForCausalLM for forward() to run."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.captured_inputs_embeds: torch.Tensor | None = None
        self.embed = nn.Embedding(32, hidden_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, *, inputs_embeds=None, **kwargs):
        self.captured_inputs_embeds = inputs_embeds.detach().clone()
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(loss=None, logits=inputs_embeds)


def _make_model_stub(*, patch_size=2, downsample_ratio=0.5, c_feat=16, img_token_id=18, hidden=64):
    """Bare NemotronOmniForConditionalGeneration with only the attributes the
    tile-based path touches. Skips the heavy __init__ (RADIO + LM)."""
    self = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(self)
    self.patch_size = patch_size
    self.downsample_ratio = downsample_ratio
    self.img_context_token_id = img_token_id
    self.ps_version = "v2"
    self.vision_model = _StubVisionModel(patch_size, c_feat)
    self.vision_projector = _IdentityProjector()
    self.language_model = _StubLM(hidden_size=hidden)
    return self


def _num_tokens(h: int, w: int, patch_size: int = 2, downsample_ratio: float = 0.5) -> int:
    """Image-slot embeddings produced for an (h, w) image by the tile path."""
    return int((h // patch_size) * downsample_ratio) * int((w // patch_size) * downsample_ratio)


# ---------------------------------------------------------------------------
# extract_feature
# ---------------------------------------------------------------------------


def test_extract_feature_accepts_list_of_variable_shaped_tensors():
    """The collate fn emits a list when resolutions differ; extract_feature must
    run the (dense-only) vision tower once per image and keep the per-image
    token counts, which depend on resolution."""
    model = _make_model_stub()

    pixel_values = [torch.zeros(3, 8, 8), torch.zeros(3, 8, 16), torch.zeros(3, 16, 8)]
    out = model.extract_feature(pixel_values)

    assert isinstance(out, list) and len(out) == 3
    assert model.vision_model.received_shapes == [(1, 8, 8), (1, 8, 16), (1, 16, 8)]
    assert [tuple(o.shape) for o in out] == [
        (1, _num_tokens(8, 8), 64),
        (1, _num_tokens(8, 16), 64),
        (1, _num_tokens(16, 8), 64),
    ]
    assert all(o.dtype == torch.bfloat16 for o in out)


def test_extract_feature_dense_input_is_unchanged():
    """Uniform-resolution batches still take the stacked-tensor path."""
    model = _make_model_stub()

    out = model.extract_feature(torch.zeros(2, 3, 8, 8))

    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (2, _num_tokens(8, 8), 64)
    # A single dense call, not one per image.
    assert model.vision_model.received_shapes == [(2, 8, 8)]


def test_extract_feature_list_restores_train_mode():
    """The vision tower is force-evaled for deterministic spectral reparam; the
    original mode must be restored, on the list path too."""
    model = _make_model_stub()
    model.vision_model.train()

    model.extract_feature([torch.zeros(3, 8, 8), torch.zeros(3, 8, 16)])

    assert model.vision_model.training, "train mode should be restored"


# ---------------------------------------------------------------------------
# forward() tile-based branch
# ---------------------------------------------------------------------------


def test_forward_tile_branch_handles_mixed_resolution_list():
    """Regression: `vit_batch_size = pixel_values.shape[0]` raised
    AttributeError ('list' object has no attribute 'shape') for any
    variable-resolution batch with local_batch_size > 1."""
    img, txt = 18, 5
    model = _make_model_stub(img_token_id=img, hidden=64)

    sizes = [(8, 8), (8, 16), (16, 8)]
    n_img_tokens = sum(_num_tokens(h, w) for h, w in sizes)
    assert n_img_tokens == 20

    # 2 samples; image slots split 12 / 8 across the batch, padded with text.
    seq = 14
    input_ids = torch.full((2, seq), txt, dtype=torch.long)
    input_ids[0, 1:13] = img
    input_ids[1, 2:10] = img
    assert int((input_ids == img).sum()) == n_img_tokens

    pixel_values = [torch.zeros(3, h, w) for h, w in sizes]
    image_flags = torch.ones(len(sizes), 1, dtype=torch.long)

    text_mask = input_ids != img
    pre_scatter = model.language_model.get_input_embeddings()(input_ids)
    expected_text = pre_scatter[text_mask].clone()

    model(input_ids=input_ids, pixel_values=pixel_values, image_flags=image_flags)

    final = model.language_model.captured_inputs_embeds
    assert final is not None
    assert final.shape == (2, seq, 64)
    # Each image was run separately at its own resolution.
    assert model.vision_model.received_shapes == [(1, 8, 8), (1, 8, 16), (1, 16, 8)]
    # Text positions untouched.
    torch.testing.assert_close(final[text_mask], expected_text.to(final.dtype))
    # Image slots hold each image's features, in order and un-truncated. A
    # partial scatter is silently caught-and-warned in forward(), so assert on
    # the values rather than just "something changed".
    expected_img = torch.cat([torch.full((_num_tokens(h, w) * 64,), _marker(h, w)) for h, w in sizes])
    torch.testing.assert_close(final[~text_mask].flatten().float(), expected_img)


def test_forward_tile_branch_list_respects_image_flags():
    """image_flags=0 marks padding images; on the list path those must be
    dropped per image rather than by indexing a stacked tensor."""
    img, txt = 18, 5
    model = _make_model_stub(img_token_id=img, hidden=64)

    # Only the first two images are real: 4 + 8 = 12 image slots.
    sizes = [(8, 8), (8, 16), (16, 8)]
    image_flags = torch.tensor([[1], [1], [0]], dtype=torch.long)
    n_img_tokens = _num_tokens(8, 8) + _num_tokens(8, 16)

    seq = 14
    input_ids = torch.full((1, seq), txt, dtype=torch.long)
    input_ids[0, 1 : 1 + n_img_tokens] = img
    assert int((input_ids == img).sum()) == 12

    pixel_values = [torch.zeros(3, h, w) for h, w in sizes]

    text_mask = input_ids != img
    pre_scatter = model.language_model.get_input_embeddings()(input_ids)
    expected_text = pre_scatter[text_mask].clone()

    model(input_ids=input_ids, pixel_values=pixel_values, image_flags=image_flags)

    final = model.language_model.captured_inputs_embeds
    assert final.shape == (1, seq, 64)
    torch.testing.assert_close(final[text_mask], expected_text.to(final.dtype))
    # Only the two flagged images contribute; the (16, 8) marker must not appear.
    expected_img = torch.cat([torch.full((_num_tokens(h, w) * 64,), _marker(h, w)) for h, w in sizes[:2]])
    torch.testing.assert_close(final[~text_mask].flatten().float(), expected_img)


def test_forward_tile_branch_single_image_list_synthesizes_one_flag():
    """A one-image variable-resolution pack works when image_flags is absent."""
    img, txt = 18, 5
    model = _make_model_stub(img_token_id=img, hidden=64)

    input_ids = torch.tensor([[txt, img, img, img, img, txt]])
    pixel_values = [torch.zeros(3, 8, 8)]

    model(input_ids=input_ids, pixel_values=pixel_values)

    final = model.language_model.captured_inputs_embeds
    assert final is not None
    assert final.shape == (1, 6, 64)
    torch.testing.assert_close(final[0, 1:5].float(), torch.full((4, 64), _marker(8, 8)))
