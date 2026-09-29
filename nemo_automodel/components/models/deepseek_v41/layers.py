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

"""Single-pass mHC and modality-aware routing for DeepSeek V4.1.

The coefficient handoff follows section 2.4.1 of the official technical report:
each sublayer consumes the pre-mix produced by the preceding sublayer.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sinkhorn_normalize
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig


class DeepseekV41RMSNorm(nn.Module):
    """Normalize in FP32 and multiply the scale before casting the result."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the released reference's RMS normalization.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., hidden], with the input dtype.
        """
        value = hidden_states.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * value).to(hidden_states.dtype)


class DeepseekV41Mix(NamedTuple):
    """FP32 coefficients: pre/post [batch, sequence, streams], comb [batch, sequence, streams, streams]."""

    pre: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


class DeepseekV41HyperConnection(nn.Module):
    """Predict one sublayer's residual mixing coefficients in FP32."""

    def __init__(
        self, config: DeepseekV41TextConfig, *, sinkhorn_backend: Literal["torch", "tilelang"] = "torch"
    ) -> None:
        super().__init__()
        if sinkhorn_backend not in ("torch", "tilelang"):
            raise ValueError("DeepSeek V4.1 mHC supports 'torch' or 'tilelang' Sinkhorn backends")
        self.sinkhorn_backend = sinkhorn_backend
        self.streams = config.hc_mult
        self.iterations = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        width = self.streams * (self.streams + 2)
        self.fn = nn.Parameter(torch.empty(width, self.streams * config.hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(width, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.reset_parameters(config.initializer_range)

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02) -> None:
        """Initialize all coefficients before checkpoint-free execution."""
        nn.init.normal_(self.fn, std=std)
        self.base.zero_()
        self.scale.fill_(1)

    def forward(self, hidden_states: torch.Tensor) -> DeepseekV41Mix:
        """Predict coefficients, preserving projection-before-RMS arithmetic.

        Args:
            hidden_states: Tensor of shape [batch, sequence, streams, hidden].

        Returns:
            Coefficients with pre/post tensors of shape [batch, sequence, streams]
            and comb of shape [batch, sequence, streams, streams]. All use FP32.
        """
        flat = hidden_states.flatten(2).float()
        mixes = F.linear(flat, self.fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        streams = self.streams
        scales = torch.cat(
            (self.scale[0].expand(streams), self.scale[1].expand(streams), self.scale[2].expand(streams * streams))
        )
        # The released kernel fuses the affine multiply/add before sigmoid and
        # Sinkhorn. Separate operations round its FP32 coefficients differently;
        # these differences can survive the subsequent BF16 stream collapse.
        logits = torch.addcmul(self.base, mixes, scales)
        pre = torch.sigmoid(logits[..., :streams]) + self.eps
        post = 2 * torch.sigmoid(logits[..., streams : 2 * streams])
        comb = dsv4_sinkhorn_normalize(
            logits[..., 2 * streams :].unflatten(-1, (streams, streams)),
            backend=self.sinkhorn_backend,
            repeat=self.iterations,
            eps=self.eps,
        )
        return DeepseekV41Mix(pre, post, comb)

    @staticmethod
    def collapse(hidden_states: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse streams using the preceding sublayer's coefficients.

        Args:
            hidden_states: Tensor of shape [batch, sequence, streams, hidden].
            pre_mix: FP32 tensor of shape [batch, sequence, streams].

        Returns:
            Tensor of shape [batch, sequence, hidden], with the input dtype.
        """
        return (pre_mix.unsqueeze(-1) * hidden_states.float()).sum(2).to(hidden_states.dtype)

    @staticmethod
    def expand(output: torch.Tensor, residual: torch.Tensor, mix: DeepseekV41Mix) -> torch.Tensor:
        """Mix the sublayer output and residual in the source's coefficient orientation.

        Args:
            output: Tensor of shape [batch, sequence, hidden].
            residual: Tensor of shape [batch, sequence, streams, hidden].
            mix: FP32 pre/post tensors of shape [batch, sequence, streams] and
                comb of shape [batch, sequence, input_streams, output_streams].

        Returns:
            Tensor of shape [batch, sequence, streams, hidden], with output's dtype.
        """
        update = mix.post.unsqueeze(-1) * output.unsqueeze(-2)
        mixed = (mix.comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)
        return (update + mixed).to(output.dtype)
