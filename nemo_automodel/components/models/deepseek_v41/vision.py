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

"""Reuse the unchanged DeepSeek vision encoder with the V4.1 nested config."""

from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.vision import DeepseekV4VisionAligner, DeepseekV4VisionTransformer

from .config import DeepseekV41Config


def _vision_config(config: DeepseekV41Config) -> DeepseekV4Config:
    """Translate the typed V4.1 config into the existing vision implementation's config."""
    vision = config.vision_config
    return DeepseekV4Config(
        hidden_size=config.text_config.hidden_size,
        dtype=config.dtype,
        vision_n_layers=vision.num_hidden_layers,
        vision_dim=vision.hidden_size,
        vision_n_heads=vision.num_attention_heads,
        vision_inter_dim=vision.intermediate_size,
        vision_patch_size=vision.patch_size,
        vision_rope_theta=vision.rope_theta,
        vision_downsample_ratio=vision.downsample_ratio,
        vision_max_n_token=vision.max_image_tokens,
        vision_min_pixels=vision.min_pixels,
        vision_max_wh_ratio=vision.max_wh_ratio,
    )


class DeepseekV41VisionTransformer(DeepseekV4VisionTransformer):
    """The released full-attention 2D-RoPE ViT, configured by nested V4.1 fields.

    Args:
        config: Top-level V4.1 configuration containing the vision dimensions.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__(_vision_config(config))


class DeepseekV41VisionAligner(DeepseekV4VisionAligner):
    """The released padded spatial merger and GELU projection into text width.

    Args:
        config: Top-level V4.1 configuration containing vision and text widths.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__(_vision_config(config))
