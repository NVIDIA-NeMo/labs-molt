# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Verify Super VL's checkpoint-owned norm before spatial pixel shuffle."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniForConditionalGeneration,
    VisionProjector,
)


class _PatchFeatures(nn.Module):
    """Expose deterministic patch features without constructing RADIO."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 1
        self.features = nn.Parameter(torch.randn(1, 24, 4) + 3.0)
        self.radio_model = SimpleNamespace(
            model=SimpleNamespace(
                patch_generator=SimpleNamespace(patch_size=1, embedder=nn.Identity(), video_embedder=nn.Identity())
            )
        )

    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        """Return test patch features.

        Args:
            images: Tensor of shape [batch, channels, height, width].

        Returns:
            Namespace with features of shape [batch, patches, hidden].
        """
        return SimpleNamespace(features=self.features)


@pytest.mark.parametrize("path", ["dense", "dynamic", "video"])
def test_final_vision_norm_matches_reference_before_shuffle_and_backward(path: str) -> None:
    """Non-square patch grids match HF's norm placement and its gradients."""
    torch.manual_seed(123)
    model = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.vision_model = _PatchFeatures()
    model.vision_projector = VisionProjector(4, 12, 8, vision_final_layernorm_eps=1e-6).to(torch.bfloat16)
    model.downsample_ratio = 0.5
    model.ps_version = "v2"
    model.patch_size = 1
    model.video_temporal_patch_dim = 2
    with torch.no_grad():
        model.vision_projector.vision_final_layernorm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        model.vision_projector.vision_final_layernorm.bias.copy_(torch.tensor([-0.5, 0.25, 0.0, 0.5]))
    ref_projector = copy.deepcopy(model.vision_projector)
    ref_features = model.vision_model.features.detach().clone().requires_grad_()
    norm = ref_projector.vision_final_layernorm
    normalized = F.layer_norm(ref_features.to(torch.bfloat16), (4,), norm.weight, norm.bias, norm.eps).reshape(
        1, 4, 6, 4
    )
    # Explicit spatial 2x2 neighborhoods, independent of production reshape logic.
    shuffled = torch.cat(
        [normalized[:, 0::2, 0::2], normalized[:, 0::2, 1::2], normalized[:, 1::2, 0::2], normalized[:, 1::2, 1::2]],
        dim=-1,
    ).reshape(1, 6, 16)
    expected = ref_projector(shuffled)
    images = torch.zeros(1, 3, 4, 6)
    if path == "dense":
        actual = model._extract_feature_dense(images)
    elif path == "dynamic":
        actual = model.extract_feature_dynamic(images, [(4, 6)])
    else:
        actual = model.extract_video_feature(images.repeat(2, 1, 1, 1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(model.vision_model.features.grad, ref_features.grad, rtol=0, atol=0)
    for name, param in model.vision_projector.named_parameters():
        torch.testing.assert_close(param.grad, dict(ref_projector.named_parameters())[name].grad, rtol=0, atol=0)


def test_legacy_projector_has_no_additional_checkpoint_parameters() -> None:
    """Non-Super projectors retain their checkpoint schema."""
    projector = VisionProjector(4, 12, 8)
    assert projector.vision_final_layernorm is None
    assert set(projector.state_dict()) == {"norm.weight", "linear1.weight", "linear2.weight"}
