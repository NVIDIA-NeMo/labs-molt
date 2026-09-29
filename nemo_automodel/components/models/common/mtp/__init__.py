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

"""Shared scaffolding for Multi-Token Prediction (MTP) auxiliary heads.

MTP follows the DeepSeek-V3 design (Liu et al., 2024). Each MTP "depth"
predicts one additional future token; per depth the input is rolled left by
one position, fused with the previous-depth hidden state, and passed through
an inner block before producing logits via the shared LM head.

Components in this package are model-agnostic. Model-specific glue (building
the inner block out of the model's own decoder layers, wiring HF state-dict
keys) lives in the model's own package.
"""

from nemo_automodel.components.models.common.mtp.mtp import (
    MTPConfig,
    MTPContextParallelInputs,
    MTPModule,
    get_mtp_loss_scaling_factor,
    prepare_mtp_context_parallel_inputs,
    roll_tensor,
    shift_packed_tensor,
)

__all__ = [
    "MTPConfig",
    "MTPContextParallelInputs",
    "MTPModule",
    "get_mtp_loss_scaling_factor",
    "prepare_mtp_context_parallel_inputs",
    "roll_tensor",
    "shift_packed_tensor",
]
