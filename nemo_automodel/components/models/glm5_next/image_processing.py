# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-only backport of the GLM-5.3 dynamic patch processor."""

from __future__ import annotations

import math

import torch
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling, SizeDict
from transformers.processing_utils import ImagesKwargs

from nemo_automodel.shared.import_utils import safe_import

_, tvF = safe_import(
    "torchvision.transforms.v2.functional",
    msg="GLM-5.3 image preprocessing requires torchvision. Install the `vlm` extra.",
)


class Glm5NextImageProcessorKwargs(ImagesKwargs, total=False):
    """Additional dynamic resize and patchification options."""

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    min_image_tokens: int
    max_image_tokens: int


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 16,
    max_pixels: int = 8000,
) -> tuple[int, int]:
    """Return an aligned H/W canvas inside the configured token budget."""
    pixels_per_token = temporal_factor * factor**2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value: int) -> int:
        return math.ceil(value / factor) * factor

    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    target_height, target_width = align(height), align(width)
    budget = aligned_frames * target_height * target_width
    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        target_height, target_width = align(max(1, math.ceil(height * scale))), align(max(1, math.ceil(width * scale)))
        budget = aligned_frames * target_height * target_width
    if budget > max_pixels:
        if max_pixels < aligned_frames * factor**2:
            raise ValueError(f"max pixel budget {max_pixels} cannot hold one aligned patch")
        low, high = 1, height
        target_height = target_width = factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height, candidate_width = align(content_height), align(content_width)
            if aligned_frames * candidate_height * candidate_width <= max_pixels:
                target_height, target_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
    return target_height, target_width


class Glm5NextImageProcessor(TorchvisionBackend):
    """Dynamically resize, normalize and flatten GLM-5.3 image patches."""

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"longest_edge": 1}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    patch_expand_factor = 1
    min_image_tokens = 16
    max_image_tokens = 8000
    valid_kwargs = Glm5NextImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def resize(
        self,
        images: torch.Tensor,
        resample,
        factor: int,
        temporal_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """Aspect-preserving resize followed by right/bottom zero padding."""
        del kwargs
        height, width = images.shape[-2:]
        target_height, target_width = smart_resize(
            temporal_factor,
            height,
            width,
            temporal_factor,
            factor,
            min_image_tokens,
            max_image_tokens,
        )
        pixels_per_token = temporal_factor * factor**2
        scale = min(target_height / height, target_width / width)
        if temporal_factor * height * width >= pixels_per_token * min_image_tokens:
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))
        if (content_height, content_width) != (height, width):
            images = super().resize(images, SizeDict(height=content_height, width=content_width), resample=resample)
        return tvF.pad(images, [0, 0, target_width - content_width, target_height - content_height], fill=0)

    @staticmethod
    def patchify(
        images: torch.Tensor,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple[torch.Tensor, int, int]:
        """Flatten block-major duplicated temporal patches."""
        batch, channels, height, width = images.shape
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = images.reshape(
            batch,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        ).permute(0, 2, 5, 3, 6, 1, 4, 7)
        patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(batch, grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size)
        )
        return patches, grid_h, grid_w

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        patch_expand_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        disable_grouping: bool | None,
        return_tensors,
        **kwargs,
    ) -> BatchFeature:
        """Implement TorchvisionBackend's grouped preprocessing contract."""
        del size, kwargs
        grouped, indices = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized = {}
        for shape, stacked in grouped.items():
            if do_resize:
                stacked = self.resize(
                    stacked,
                    resample,
                    patch_size * merge_size * patch_expand_factor,
                    temporal_patch_size,
                    min_image_tokens,
                    max_image_tokens,
                )
            resized[shape] = stacked
        images = reorder_images(resized, indices)
        grouped, indices = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed, grids = {}, {}
        for shape, stacked in grouped.items():
            stacked = self.rescale_and_normalize(
                stacked,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            patches, grid_h, grid_w = self.patchify(stacked, patch_size, merge_size, temporal_patch_size)
            processed[shape] = patches
            grids[shape] = [[1, grid_h, grid_w]] * len(stacked)
        images = reorder_images(processed, indices)
        image_grids = reorder_images(grids, indices)
        pixel_values = images[0] if len(images) == 1 else torch.cat(images, dim=0)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": torch.tensor(image_grids)},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """Return the number of unmerged vision patches for one source image."""
        values = images_kwargs or {}
        patch_size = values.get("patch_size", self.patch_size)
        merge_size = values.get("merge_size", self.merge_size)
        target_h, target_w = smart_resize(
            self.temporal_patch_size,
            height,
            width,
            self.temporal_patch_size,
            patch_size * merge_size,
            values.get("min_image_tokens", self.min_image_tokens),
            values.get("max_image_tokens", self.max_image_tokens),
        )
        return (target_h // patch_size) * (target_w // patch_size)


__all__ = ["Glm5NextImageProcessor", "smart_resize"]
