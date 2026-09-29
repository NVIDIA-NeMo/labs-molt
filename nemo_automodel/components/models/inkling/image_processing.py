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

"""Native image preprocessing for Inkling's hierarchical vision tower."""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput, PILImageResampling, SizeDict, validate_preprocess_arguments
from transformers.processing_utils import ImagesKwargs, Unpack
from transformers.utils import TensorType
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


class InklingImageProcessorKwargs(ImagesKwargs, total=False):
    """Additional Inkling image preprocessing arguments."""

    rescale_image_frac: float | None
    rescale_image_max_upscaled_long_edge: int | None


def _divide_to_patches(image: torch.Tensor, patch_size: int) -> list[torch.Tensor]:
    """Divide a channels-first image into possibly incomplete square patches.

    Args:
        image: Tensor of shape ``[channels, height, width]``.
        patch_size: Height and width of each patch.

    Returns:
        Tensors of shape ``[channels, patch_height, patch_width]`` in row-major order.
    """
    height, width = image.shape[-2:]
    num_rows = (height + patch_size - 1) // patch_size
    num_columns = width // patch_size + 1
    return [
        image[..., row * patch_size : (row + 1) * patch_size, column * patch_size : (column + 1) * patch_size]
        for row in range(num_rows)
        for column in range(num_columns)
    ]


class InklingImageProcessor(TorchvisionBackend):
    """Convert images into Inkling's padded spatiotemporal patch layout."""

    resample = PILImageResampling.LANCZOS
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    default_to_square = False
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    size = {"height": 40, "width": 40}
    rescale_image_frac = None
    rescale_image_max_upscaled_long_edge = 2048
    valid_kwargs = InklingImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[InklingImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(
        self,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | tuple[float, ...] | None = None,
        image_std: float | tuple[float, ...] | None = None,
        do_resize: bool | None = None,
        size: SizeDict | None = None,
        do_center_crop: bool | None = None,
        crop_size: SizeDict | None = None,
        resample: PILImageResampling | int | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate model-specific image preprocessing settings."""
        del kwargs
        if do_resize is False:
            raise ValueError("do_resize cannot be False for Inkling")
        if size is None or size.height != size.width:
            raise ValueError(f"Inkling requires a square patch size, got {size}")
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[ImagesKwargs],
    ) -> BatchFeature:
        """Convert image inputs into vision-tower patches.

        Args:
            images: One image or a batch in a supported PIL, NumPy, or tensor layout.
            **kwargs: Standard Transformers image preprocessing overrides.

        Returns:
            A batch containing ``pixel_values`` with shape
            ``[patches, time=2, height, width, channels]`` and ``num_patches``
            with shape ``[images]``.
        """
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        resample: PILImageResampling | int | None,
        rescale_image_frac: float | None,
        rescale_image_max_upscaled_long_edge: int | None,
        return_tensors: str | TensorType | None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Process normalized channels-first image tensors.

        Args:
            images: Tensors of shape ``[channels, height, width]``.
            size: Square patch dimensions.
            do_rescale: Whether to multiply pixels by ``rescale_factor``.
            rescale_factor: Pixel rescaling multiplier.
            do_normalize: Whether to normalize channels.
            image_mean: Per-channel normalization means.
            image_std: Per-channel normalization standard deviations.
            resample: Resize interpolation mode.
            rescale_image_frac: Optional multiplier for the long image edge.
            rescale_image_max_upscaled_long_edge: Maximum long edge when upscaling.
            return_tensors: Requested output tensor framework.
            **kwargs: Unused common backend arguments.

        Returns:
            A batch containing ``pixel_values`` with shape
            ``[patches, time=2, height, width, channels]`` and ``num_patches``
            with shape ``[images]``.
        """
        del kwargs
        per_image_patches: list[torch.Tensor] = []
        patch_counts: list[int] = []
        for image in images:
            if rescale_image_frac is not None:
                height, width = image.shape[-2:]
                long_edge = max(height, width)
                target_long_edge = long_edge * rescale_image_frac
                if rescale_image_max_upscaled_long_edge is not None:
                    target_long_edge = min(target_long_edge, max(rescale_image_max_upscaled_long_edge, long_edge))
                ratio = target_long_edge / long_edge
                if ratio != 1.0:
                    new_size = SizeDict(
                        height=math.floor(height * ratio + 0.5),
                        width=math.floor(width * ratio + 0.5),
                    )
                    image = self.resize(image, new_size, resample=resample)

            patches = [patch.float() for patch in _divide_to_patches(image, size.height)]
            patches = self.pad(patches, pad_size=SizeDict(height=size.height, width=size.width), fill_value=-1.0)
            patch_tensor = torch.stack(patches, dim=0)
            patch_tensor = self.rescale_and_normalize(
                patch_tensor,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            patch_tensor = patch_tensor[..., None].repeat(1, 1, 1, 1, 2)
            per_image_patches.append(patch_tensor)
            patch_counts.append(patch_tensor.shape[0])

        pixel_values = torch.cat(per_image_patches, dim=0).permute(0, 4, 2, 3, 1)
        return BatchFeature(
            data={"pixel_values": pixel_values, "num_patches": torch.tensor(patch_counts)},
            tensor_type=return_tensors,
        )


__all__ = ["InklingImageProcessor"]
