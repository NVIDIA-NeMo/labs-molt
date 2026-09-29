# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for NemotronOmni's dynamic-resolution vision path.

Covers the vLLM-aligned `extract_feature_dynamic` / `_pixel_shuffle_dynamic_res`
helpers and the `imgs_sizes` branch of `forward()`. The vision tower and LM are
mocked so these tests stay CPU-only and don't need RADIO weights.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniForConditionalGeneration,
    VisionProjector,
)


class _StubVisionModel(nn.Module):
    """Vision model whose `forward(img)` records the (h, w) it received and
    returns features sized to `(h//patch) * (w//patch)` patches."""

    def __init__(self, patch_size: int, c_feat: int):
        super().__init__()
        self.patch_size = patch_size
        self.c_feat = c_feat
        self.dummy = nn.Parameter(torch.zeros(1))  # ensures .parameters() yields a dtype
        self.received_shapes: list[tuple[int, int]] = []

    def forward(self, img: torch.Tensor):
        b, _, h, w = img.shape
        assert b == 1, "extract_feature_dynamic should crop to one image at a time"
        self.received_shapes.append((h, w))
        num_patches = (h // self.patch_size) * (w // self.patch_size)
        # Encode (h, w) into the features so callers can verify per-image
        # routing without relying on order.
        feats = torch.full((1, num_patches, self.c_feat), float(h * 1000 + w), dtype=torch.float32)
        return SimpleNamespace(features=feats)


class _IdentityProjector(VisionProjector):
    def __init__(self):
        nn.Module.__init__(self)
        self.vision_final_layernorm = None
        self.downsample_ratio = 0.5
        self.norm = nn.Identity()
        self.linear1 = nn.Identity()
        self.activation = nn.Identity()
        self.linear2 = nn.Identity()


def _make_model_stub(*, patch_size=2, downsample_ratio=0.5, c_feat=16, img_token_id=18):
    """Build a bare NemotronOmniForConditionalGeneration with only the
    attributes the dynamic-res path touches set up. Skips the heavy __init__
    (RADIO + LM construction)."""
    self = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(self)
    self.patch_size = patch_size
    self.downsample_ratio = downsample_ratio
    self.img_context_token_id = img_token_id
    self.ps_version = "v2"
    self.vision_model = _StubVisionModel(patch_size, c_feat)
    self.vision_projector = _IdentityProjector()
    return self


# ---------------------------------------------------------------------------
# _pixel_shuffle_dynamic_res
# ---------------------------------------------------------------------------


def test_pixel_shuffle_dynamic_res_splits_per_image():
    """Per-image split lengths must match (h//p) * (w//p) for each image, and
    pixel_shuffle must shrink spatial dims by `downsample_ratio` while inflating
    channels by 1/scale^2."""
    model = _make_model_stub(patch_size=2, downsample_ratio=0.5, c_feat=16)

    # Two images: 4x4 patches and 2x6 patches at patch_size=2 → input HxW (8x8)
    # and (4x12) → seq lengths 16 and 12.
    sizes = [(8, 8), (4, 12)]
    seq_lens = [(8 // 2) * (8 // 2), (4 // 2) * (12 // 2)]
    assert seq_lens == [16, 12]
    total = sum(seq_lens)
    x = torch.arange(total * 16, dtype=torch.float32).reshape(1, total, 16)

    out = model._pixel_shuffle_dynamic_res(x, sizes)

    # downsample_ratio=0.5 → spatial dims halved per side, channels x4.
    # Image 1: (4, 4) → (2, 2) flat=4, c=64. Image 2: (2, 6) → (1, 3) flat=3, c=64.
    expected_total = 4 + 3
    assert out.shape == (1, expected_total, 64)


def test_pixel_shuffle_dynamic_res_rejects_mismatched_sizes():
    """If `sum(seq_lens)` doesn't match the input length, torch.split raises —
    catching upstream bugs that pass wrong imgs_sizes."""
    model = _make_model_stub(patch_size=2, downsample_ratio=0.5, c_feat=16)
    x = torch.zeros(1, 16, 16)
    # (4,4) at patch=2 → seq_len 4, not 16
    with pytest.raises(RuntimeError):
        model._pixel_shuffle_dynamic_res(x, [(4, 4)])


# ---------------------------------------------------------------------------
# extract_feature_dynamic
# ---------------------------------------------------------------------------


def test_extract_feature_dynamic_crops_padded_input_to_real_size():
    """pixel_values is padded to per-batch max (H, W). extract_feature_dynamic
    must crop each image back to its real (h, w) before calling the ViT — the
    HF RADIO can't accept packed/variable inputs."""
    model = _make_model_stub(patch_size=2, downsample_ratio=0.5, c_feat=16)

    # Padded batch: 2 images, both materialized as 12x12 even though true sizes
    # differ. Channel count = 3.
    pixel_values = torch.zeros(2, 3, 12, 12)
    sizes = torch.tensor([[8, 8], [4, 12]], dtype=torch.long)

    out = model.extract_feature_dynamic(pixel_values, sizes)

    # The vision_model must have been called exactly twice with the cropped sizes.
    assert model.vision_model.received_shapes == [(8, 8), (4, 12)]

    # Output should be the projector(pixel_shuffle(...)) of concatenated features.
    # With downsample_ratio=0.5 and patch_size=2:
    #   img 1: (4,4) → (2,2) flat=4
    #   img 2: (2,6) → (1,3) flat=3
    assert out.shape == (1, 7, 64)
    # The stub features encode (h*1000 + w); the dynamic path casts to bfloat16
    # internally so check via float() equality on a low-precision range.
    assert out.dtype == torch.bfloat16


def test_extract_feature_dynamic_accepts_list_sizes():
    """imgs_sizes may arrive as a tensor or as a list of tuples."""
    model = _make_model_stub(patch_size=2, downsample_ratio=0.5, c_feat=16)

    pixel_values = torch.zeros(1, 3, 8, 8)
    out_list = model.extract_feature_dynamic(pixel_values, [(8, 8)])
    out_tensor = model.extract_feature_dynamic(pixel_values, torch.tensor([[8, 8]], dtype=torch.long))
    assert out_list.shape == out_tensor.shape


def test_extract_feature_dynamic_restores_train_mode():
    """The vision tower is force-evaled during feature extraction; train mode
    must be restored on exit when it was originally training."""
    model = _make_model_stub()
    model.vision_model.train()
    assert model.vision_model.training

    model.extract_feature_dynamic(torch.zeros(1, 3, 8, 8), [(8, 8)])

    assert model.vision_model.training, "train mode should be restored"


# ---------------------------------------------------------------------------
# forward() dynamic-resolution branch
# ---------------------------------------------------------------------------


class _StubLM(nn.Module):
    """Mocks just enough of NemotronV3ForCausalLM for forward() to run."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.captured_inputs_embeds: torch.Tensor | None = None
        self.embed = nn.Embedding(32, hidden_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        *,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Capture the post-injection embeddings so the test can assert on them.
        self.captured_inputs_embeds = inputs_embeds.detach().clone()
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(loss=None, logits=inputs_embeds)


def test_forward_dynamic_res_branch_fills_image_slots():
    """When pixel_values + imgs_sizes are passed, forward() must:
    - take the dynamic-res branch (not the tile-based one);
    - call extract_feature_dynamic with the padded pixel_values + imgs_sizes;
    - replace exactly the <image> token slots with vit_embeds (channels = LM hidden).
    """
    # LM hidden must equal c_feat * (1/scale)^2 because the identity projector
    # passes pixel-shuffled features straight through. patch=2, scale=0.5, c=16 → 64.
    hidden = 64
    model = _make_model_stub(patch_size=2, downsample_ratio=0.5, c_feat=16, img_token_id=18)
    model.language_model = _StubLM(hidden_size=hidden)

    # Sequence: 2 batch x 6 tokens; image tokens scattered.
    img = 18
    txt = 5
    input_ids = torch.tensor(
        [[txt, img, img, img, img, txt], [img, img, img, txt, txt, txt]],
        dtype=torch.long,
    )
    text_mask = input_ids != img

    # Two images, padded to 12x12. img 1 real size 8x8, img 2 real size 4x12.
    # After dynamic pixel-shuffle:
    #   img 1: (4,4) patch grid → (2,2) flat=4 → 4 image-slot embeddings
    #   img 2: (2,6) patch grid → (1,3) flat=3 → 3 image-slot embeddings
    # Total = 7 image-token positions in input_ids, which matches.
    pixel_values = torch.zeros(2, 3, 12, 12)
    imgs_sizes = torch.tensor([[8, 8], [4, 12]], dtype=torch.long)

    # Pre-compute the *expected* text embeddings (what embed_tokens will emit
    # for the text positions before any scatter touches them).
    pre_scatter = model.language_model.get_input_embeddings()(input_ids)
    expected_text = pre_scatter[text_mask].clone()

    # Don't pass inputs_embeds — forward should compute them internally then
    # apply the dynamic-res scatter.  (Caller-supplied inputs_embeds is the CP
    # path: post-prepare embeds, forward must NOT re-scatter.)
    model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        imgs_sizes=imgs_sizes,
    )

    final = model.language_model.captured_inputs_embeds
    assert final is not None
    assert final.shape == (2, 6, hidden)

    # Text positions must equal the unsalted embed_tokens output.
    torch.testing.assert_close(final[text_mask], expected_text.to(final.dtype))

    # Image positions must have been replaced by vit_embeds (≠ embed_tokens).
    assert not torch.allclose(final[~text_mask].float(), pre_scatter[~text_mask])


def test_forward_prefers_dynamic_res_over_image_flags_when_both_present():
    """If imgs_sizes is set, the dynamic-res branch wins regardless of image_flags."""
    hidden = 64
    model = _make_model_stub(patch_size=2, downsample_ratio=0.5, c_feat=16, img_token_id=18)
    model.language_model = _StubLM(hidden_size=hidden)

    # Spy on extract_feature_dynamic vs extract_feature.
    calls = {"dynamic": 0, "tile": 0}
    orig_dynamic = model.extract_feature_dynamic
    orig_tile = model.extract_feature

    def spy_dynamic(pv, sz):
        calls["dynamic"] += 1
        return orig_dynamic(pv, sz)

    def spy_tile(pv):
        calls["tile"] += 1
        return orig_tile(pv)

    model.extract_feature_dynamic = spy_dynamic
    model.extract_feature = spy_tile

    input_ids = torch.tensor([[5, 18, 18, 18, 18]], dtype=torch.long)
    pixel_values = torch.zeros(1, 3, 8, 8)
    imgs_sizes = torch.tensor([[8, 8]], dtype=torch.long)
    image_flags = torch.tensor([[1]], dtype=torch.long)

    # Don't pass inputs_embeds — caller-supplied embeds is the CP path which
    # skips the multimodal scatter; this test wants the scatter to fire.
    model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        imgs_sizes=imgs_sizes,
        image_flags=image_flags,
    )

    assert calls["dynamic"] == 1
    assert calls["tile"] == 0
