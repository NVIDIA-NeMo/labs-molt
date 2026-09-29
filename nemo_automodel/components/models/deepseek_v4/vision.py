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

"""Vision encoder and aligner used by DeepSeek-V4-Flash-Vision-Exp."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def get_vision_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the reference 2D rotary table.

    Returns:
        Cosine and sine tensors with layout ``[n_h * n_w, 1, dim]``. The
        first half of ``dim`` encodes row position and the second half encodes
        column position. Rotary arithmetic remains in fp32.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply DeepSeek's half-split 2D rotary embedding.

    Args:
        x: Query or key tensor with layout ``[patches, heads, head_dim]``.
        cos: Cosine table with layout ``[patches, 1, head_dim / 2]``.
        sin: Sine table with layout ``[patches, 1, head_dim / 2]``.

    Returns:
        Rotated tensor with the same layout and dtype as ``x``.
    """
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    """Reference RMSNorm with an fp32 scale parameter."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize an input of layout ``[..., vision_dim]``."""
        dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * normalized).to(dtype)


class DeepseekV4VisionPatchEmbed(nn.Module):
    """Linear embedding of flattened RGB patches."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        patch_dim = 3 * int(config.vision_patch_size) ** 2
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.proj = nn.Linear(patch_dim, int(config.vision_dim), bias=True, dtype=dtype)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Project ``[patches, 3, patch_h, patch_w]`` to ``[patches, vision_dim]``."""
        return self.proj(patches.flatten(1))


class DeepseekV4VisionAttention(nn.Module):
    """Full bidirectional patch attention with 2D RoPE."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.n_heads = int(config.vision_n_heads)
        vision_dim = int(config.vision_dim)
        if vision_dim % self.n_heads != 0:
            raise ValueError("vision_dim must be divisible by vision_n_heads")
        self.head_dim = vision_dim // self.n_heads
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.wqkv = nn.Linear(vision_dim, 3 * vision_dim, bias=True, dtype=dtype)
        self.wo = nn.Linear(vision_dim, vision_dim, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Attend over one image.

        Args:
            x: Patch states with layout ``[patches, vision_dim]``.
            cos: Cosine table with layout ``[patches, 1, head_dim / 2]``.
            sin: Sine table with layout ``[patches, 1, head_dim / 2]``.

        Returns:
            Attention output with layout ``[patches, vision_dim]``.
        """
        n_patches = x.size(0)
        q, k, v = (tensor.view(n_patches, self.n_heads, self.head_dim) for tensor in self.wqkv(x).chunk(3, dim=-1))
        q = apply_vision_rotary(q, cos, sin)
        k = apply_vision_rotary(k, cos, sin)
        output = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(output.transpose(0, 1).reshape(n_patches, -1))


class DeepseekV4VisionMLP(nn.Module):
    """Bias-free SwiGLU MLP used by each vision block."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        vision_dim = int(config.vision_dim)
        inter_dim = int(config.vision_inter_dim)
        self.w1 = nn.Linear(vision_dim, 2 * inter_dim, bias=False, dtype=dtype)
        self.w2 = nn.Linear(inter_dim, vision_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[patches, vision_dim]`` back to ``[patches, vision_dim]``."""
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    """Pre-norm attention and MLP residual block."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.norm1 = DeepseekV4VisionRMSNorm(int(config.vision_dim))
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = DeepseekV4VisionRMSNorm(int(config.vision_dim))
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Transform patch states with layout ``[patches, vision_dim]``."""
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4VisionTransformer(nn.Module):
    """DeepSeek ViT: full attention over one image with 2D RoPE."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.rope_dim = int(config.vision_dim) // int(config.vision_n_heads) // 2
        self.rope_theta = float(config.vision_rope_theta)
        self.patch_embed = DeepseekV4VisionPatchEmbed(config)
        self.blocks = nn.ModuleList([DeepseekV4VisionBlock(config) for _ in range(int(config.vision_n_layers))])
        self.norm = DeepseekV4VisionRMSNorm(int(config.vision_dim))

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Encode one image.

        Args:
            patches: Normalized RGB patches with layout
                ``[n_h * n_w, 3, patch_size, patch_size]``.
            n_h: Number of patch rows.
            n_w: Number of patch columns.

        Returns:
            Encoded patches with layout ``[n_h * n_w, vision_dim]``.
        """
        if patches.shape[0] != n_h * n_w:
            raise ValueError(f"Expected {n_h * n_w} patches for a {n_h}x{n_w} grid, got {patches.shape[0]}")
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta, device=x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def init_weights(self, init_std: float) -> None:
        """Initialize all checkpoint-free vision parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, DeepseekV4VisionRMSNorm):
                nn.init.ones_(module.weight)


class DeepseekV4VisionAligner(nn.Module):
    """Spatially downsample ViT patches and project them into the LLM width."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.downsample_ratio = int(config.vision_downsample_ratio)
        in_dim = int(config.vision_dim) * self.downsample_ratio**2
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.w1 = nn.Linear(in_dim, int(config.hidden_size), bias=True, dtype=dtype)
        self.w2 = nn.Linear(int(config.hidden_size), int(config.hidden_size), bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Downsample encoded patch states.

        Args:
            x: ViT output with layout ``[n_h * n_w, vision_dim]``.
            n_h: Number of patch rows.
            n_w: Number of patch columns.

        Returns:
            LLM image embeddings with layout
            ``[ceil(n_h / ratio) * ceil(n_w / ratio), hidden_size]``.
        """
        if x.shape[0] != n_h * n_w:
            raise ValueError(f"Expected {n_h * n_w} patch states for a {n_h}x{n_w} grid, got {x.shape[0]}")
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))

    def init_weights(self, init_std: float) -> None:
        """Initialize all checkpoint-free aligner parameters."""
        for module in (self.w1, self.w2):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            nn.init.zeros_(module.bias)


__all__ = [
    "DeepseekV4VisionAligner",
    "DeepseekV4VisionTransformer",
    "DeepseekV4VisionRMSNorm",
    "apply_vision_rotary",
    "get_vision_cos_sin",
]
