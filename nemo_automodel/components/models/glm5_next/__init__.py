# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native AutoModel support for ``zai-org/GLM-5.3-Flash``."""

from nemo_automodel.components.models.glm5_next.config import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration

ModelClass = Glm5NextForConditionalGeneration

__all__ = [
    "Glm5NextConfig",
    "Glm5NextForConditionalGeneration",
    "Glm5NextTextConfig",
    "Glm5NextVisionConfig",
]
