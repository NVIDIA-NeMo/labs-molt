# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
Transformers compatibility patches.

Runtime monkey-patch for apex's FusedRMSNorm which does not support bfloat16.
Call `patch_t5_layer_norm()` before loading any T5 models when running in bf16.
"""

from __future__ import annotations

import logging

import torch

_logger = logging.getLogger(__name__)


def patch_t5_layer_norm() -> None:
    """Replace apex's FusedRMSNorm with a native T5LayerNorm in the T5 module.

    Apex's FusedRMSNorm doesn't support bfloat16, but the native T5LayerNorm
    handles it correctly by upcasting to fp32 internally for numerical stability.
    This must be called before loading any T5 models.

    This function is idempotent and safe to call multiple times.
    """
    try:
        from transformers.models.t5 import modeling_t5

        if not (getattr(modeling_t5.T5LayerNorm, "__module__", "") or "").startswith("apex"):
            return

        class _NativeT5LayerNorm(torch.nn.Module):
            """RMS norm (no bias, no mean subtraction) that works with bf16."""

            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                return self.weight * hidden_states

        modeling_t5.T5LayerNorm = _NativeT5LayerNorm
        _logger.info("Replaced apex FusedRMSNorm with native T5LayerNorm for bf16 compatibility")
    except ImportError:
        pass
