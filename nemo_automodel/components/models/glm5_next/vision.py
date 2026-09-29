# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 image encoder with checkpoint-compatible module names."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.glm5_next.config import Glm5NextVisionConfig
from nemo_automodel.components.models.glm5_next.layers import Glm5NextRMSNorm
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _vision_position_ids(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Return block-major H/W positions ``[vision_tokens, 2]``."""
    positions = []
    for temporal, height, width in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(
            torch.arange(height, device=grid_thw.device),
            torch.arange(width, device=grid_thw.device),
            indexing="ij",
        )
        shape = (height // merge_size, merge_size, width // merge_size, merge_size)
        hpos = hpos.reshape(shape).transpose(1, 2).flatten()
        wpos = wpos.reshape(shape).transpose(1, 2).flatten()
        positions.append(torch.stack((hpos, wpos), dim=-1).repeat(temporal, 1))
    return torch.cat(positions, dim=0)


def _vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Return per-frame attention boundaries ``[segments + 1]``."""
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))


class Glm5NextVisionPatchEmbed(nn.Module):
    """Conv3d patch projection from ``[patches, C*T*P*P]`` to vision hidden."""

    def __init__(self, config: Glm5NextVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
            dtype=dtype,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Project flattened patches to ``[vision_tokens, vision_hidden]``."""
        patches = pixel_values.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(patches.to(self.proj.weight.dtype)).view(patches.shape[0], -1)


class Glm5NextVisionRotaryEmbedding(nn.Module):
    """Two-axis rotary frequencies used by the vision transformer."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Map H/W ids ``[tokens, 2]`` to frequencies ``[tokens, head_dim/2]``."""
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Glm5NextVisionAttention(nn.Module):
    """Bidirectional per-image attention over ``[vision_tokens, hidden]``."""

    def __init__(self, config: Glm5NextVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.qkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.q_norm = Glm5NextRMSNorm(self.head_dim, config.rms_norm_eps, dtype)
        self.k_norm = Glm5NextRMSNorm(self.head_dim, config.rms_norm_eps, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Attend within each cu-seqlens segment and return ``[tokens, hidden]``."""
        sequence = hidden_states.shape[0]
        query, key, value = self.qkv(hidden_states).view(sequence, 3, self.num_heads, self.head_dim).unbind(1)
        query, key = self.q_norm(query), self.k_norm(key)
        cos, sin = (item.unsqueeze(-2).float() for item in position_embeddings)
        query_float, key_float = query.float(), key.float()
        query = (query_float * cos + _rotate_half(query_float) * sin).to(query.dtype)
        key = (key_float * cos + _rotate_half(key_float) * sin).to(key.dtype)
        query, key, value = (item.transpose(0, 1).unsqueeze(0) for item in (query, key, value))
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_chunks, k_chunks, v_chunks = (torch.split(item, lengths, dim=2) for item in (query, key, value))
        output = [
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scaling,
            )
            for q, k, v in zip(q_chunks, k_chunks, v_chunks)
        ]
        output = torch.cat(output, dim=2).squeeze(0).transpose(0, 1).reshape(sequence, -1)
        return self.proj(output)


class Glm5NextVisionMLP(nn.Module):
    """Clamped SwiGLU vision feed-forward block."""

    def __init__(self, config: Glm5NextVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform ``[tokens, vision_hidden]`` with clamped SwiGLU."""
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(-self.swiglu_limit, self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionBlock(nn.Module):
    """Pre-norm bidirectional vision transformer block."""

    def __init__(self, config: Glm5NextVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.norm1 = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.norm2 = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.attn = Glm5NextVisionAttention(config, dtype)
        self.mlp = Glm5NextVisionMLP(config, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Transform ``[vision_tokens, hidden]`` without cross-image attention."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Glm5NextVisionPatchMerger(nn.Module):
    """Post-downsample projection in the text hidden dimension."""

    def __init__(self, config: Glm5NextVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        dim = config.out_hidden_size
        context = config.projection_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.post_projection_norm = nn.LayerNorm(dim, dtype=dtype)
        self.gate_proj = nn.Linear(dim, context, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(dim, context, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(context, dim, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project downsampled image tokens ``[tokens, text_hidden]``."""
        hidden_states = F.gelu(self.post_projection_norm(self.proj(hidden_states)))
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(-self.swiglu_limit, self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


@dataclass
class Glm5NextVisionOutput:
    """Raw and merged vision token representations."""

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


class Glm5NextVisionModel(nn.Module):
    """Patch encoder returning image features in the language hidden size."""

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        dtype = get_dtype(getattr(config, "torch_dtype", getattr(config, "dtype", None)), torch.bfloat16)
        self.patch_embed = Glm5NextVisionPatchEmbed(config, dtype)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Glm5NextVisionBlock(config, dtype) for _ in range(config.depth)])
        self.post_layernorm = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
            dtype=dtype,
        )
        self.merger = Glm5NextVisionPatchMerger(config, dtype)

    @property
    def dtype(self) -> torch.dtype:
        """Return the patch embedding dtype."""
        return self.patch_embed.proj.weight.dtype

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> Glm5NextVisionOutput:
        """Encode flattened patches using grid metadata ``[images, (t,h,w)]``."""
        positions = _vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = _vision_cu_seqlens(grid_thw)
        hidden_states = self.patch_embed(pixel_values)
        rotary = self.rotary_pos_emb(positions)
        embedding = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)
        hidden_states = self.post_layernorm(hidden_states)
        merge = self.spatial_merge_size
        downsampled = hidden_states.view(-1, merge, merge, hidden_states.shape[-1]).permute(0, 3, 1, 2)
        downsampled = self.downsample(downsampled).view(-1, self.config.out_hidden_size)
        return Glm5NextVisionOutput(last_hidden_state=downsampled, pooler_output=self.merger(downsampled))

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize vision parameters without materializing outside ``buffer_device``."""
        with buffer_device:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, (nn.LayerNorm, Glm5NextRMSNorm)):
                    module.reset_parameters()


__all__ = ["Glm5NextVisionModel", "Glm5NextVisionOutput"]
