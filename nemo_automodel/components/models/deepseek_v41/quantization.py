# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Reference cache QAT formats with straight-through activation gradients."""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F


def _power_of_two_ceiling(value: torch.Tensor) -> torch.Tensor:
    """Round positive normal FP32 values up to an exactly represented power of two.

    Args:
        value: Positive FP32 tensor of arbitrary shape.

    Returns:
        FP32 tensor of the same shape with integer powers of two, matching the
        official IEEE754 exponent/mantissa operation without logarithm rounding.
    """
    bits = value.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    increment = (bits & ((1 << 23) - 1)) != 0
    return ((exponent + increment.to(torch.int32)) << 23).view(torch.float32)


class _CacheQuantization(torch.autograd.Function):
    """Quantize cache values in forward and pass the activation gradient unchanged."""

    @staticmethod
    def forward(ctx: object, values: torch.Tensor, block_size: int, format: str) -> torch.Tensor:
        """Round independent channel groups without modifying the input.

        Args:
            ctx: Autograd context; no tensors need saving for the straight-through gradient.
            values: Tensor of shape [..., channels], with arbitrary leading dimensions.
            block_size: Consecutive channels sharing a quantization scale.
            format: fp8, mxfp4 (E8M0 scale), or nvfp4 (E4M3 scale).

        Returns:
            Independently stored tensor of shape [..., channels] with input dtype.
        """
        channels = values.shape[-1]
        grouped = F.pad(values.float(), (0, -channels % block_size)).unflatten(-1, (-1, block_size))
        maximum = grouped.abs().amax(-1, keepdim=True)
        if format == "fp8":
            scale = _power_of_two_ceiling(maximum.clamp_min(1e-4) * (1 / 448))
            rounded = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        else:
            if format == "nvfp4":
                scale = (maximum.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
            else:
                scale = _power_of_two_ceiling(maximum.clamp_min(6 * 2**-126) * (1 / 6))
            normalized = (grouped / scale).clamp(-6, 6)
            magnitude = normalized.abs().contiguous()
            midpoints = magnitude.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
            levels = magnitude.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
            lower = torch.bucketize(magnitude, midpoints)
            tie = magnitude == midpoints[lower.clamp_max(6)]
            selected = lower + (tie & (lower % 2 == 1)).to(lower.dtype)
            rounded = torch.copysign(levels[selected], normalized)
        return (rounded * scale).flatten(-2)[..., :channels].to(values.dtype)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Pass the gradient for [..., channels] activations through quantization.

        Args:
            ctx: Unused forward context.
            gradient: Tensor of shape [..., channels] from the cache consumer.

        Returns:
            Gradient of shape [..., channels], followed by no gradients for the
            integer group size and format name. The gradient storage is reused.
        """
        return gradient, None, None


def quantize_cache(values: torch.Tensor, *, format: Literal["fp8", "mxfp4", "nvfp4"], block_size: int) -> torch.Tensor:
    """Apply the released model's quantize/dequantize cache representation.

    Args:
        values: Tensor of shape [..., channels], with arbitrary leading dimensions.
            Complete groups are required by the released kernels; a final partial
            group is zero-padded internally for scaled unit-test configurations.
        format: fp8 for SWA KV, mxfp4 for index Q/K, nvfp4 for compressed KV.
        block_size: Channels per scale: 32 for SWA/indexer, 16 for compressed KV.

    Returns:
        Tensor of shape [..., channels], in the original dtype, with a
        straight-through gradient. Neither input storage nor its aliases are mutated.
    """
    if format not in ("fp8", "mxfp4", "nvfp4") or block_size <= 0:
        raise ValueError("Cache quantization requires fp8/mxfp4/nvfp4 and a positive block_size")
    return _CacheQuantization.apply(values, block_size, format)
