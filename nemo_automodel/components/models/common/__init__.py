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

"""Common/shared components for model implementations."""

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import (
    HAVE_DEEP_EP,
    HAVE_TE,
    BackendConfig,
    CudaGraphConfig,
    compute_lm_head_logits,
    get_rope_config,
    initialize_linear_module,
    initialize_rms_norm_module,
)

__all__ = [
    # HF checkpointing mixin
    "HFCheckpointingMixin",
    # Backend utilities
    "HAVE_TE",
    "HAVE_DEEP_EP",
    "BackendConfig",
    "CudaGraphConfig",
    "get_rope_config",
    "initialize_rms_norm_module",
    "initialize_linear_module",
    # lm_head projection
    "compute_lm_head_logits",
]
