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

"""Native MuseGlimmer vision path copied from the canonical Transformers implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer

from nemo_automodel.components.models.muse_glimmer.config import MuseGlimmerConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the hidden dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical split-half vision RoPE in float32."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Return one packed-attention segment per frame."""
    dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=dtype), (1, 0), value=0)


def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Build canonical block-major two-dimensional vision positions."""
    device = grid_thw.device
    position_ids = []
    for t, h, w in grid_thw.tolist():
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        block_shape = (
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def get_vision_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy the canonical window-attention permutation and segment lengths."""
    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size**2

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w

    indices = torch.cat(window_index, dim=0).to(grid_thw.device)
    cu_seqlens = torch.tensor(cu_window_seqlens, device=grid_thw.device, dtype=torch.int32)
    return indices, torch.unique_consecutive(cu_seqlens)


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy the checkpoint's grid-sample-equivalent position interpolation."""
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device
    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = (torch.arange(h, device=device).float() + 0.5) * (side / h) - 0.5
        w_grid = (torch.arange(w, device=device).float() + 0.5) * (side / w) - 0.5
        h_floor = torch.floor(h_grid).long()
        w_floor = torch.floor(w_grid).long()
        h_ceil = h_floor + 1
        w_ceil = w_floor + 1
        h_frac = h_grid - h_floor.float()
        w_frac = w_grid - w_floor.float()

        h_floor_valid = (h_floor >= 0) & (h_floor <= side - 1)
        h_ceil_valid = (h_ceil >= 0) & (h_ceil <= side - 1)
        w_floor_valid = (w_floor >= 0) & (w_floor <= side - 1)
        w_ceil_valid = (w_ceil >= 0) & (w_ceil <= side - 1)
        h_floor = h_floor.clamp(0, side - 1)
        h_ceil = h_ceil.clamp(0, side - 1)
        w_floor = w_floor.clamp(0, side - 1)
        w_ceil = w_ceil.clamp(0, side - 1)

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            (
                (1 - h_frac)[:, None] * (1 - w_frac)[None, :] * (h_floor_valid[:, None] & w_floor_valid[None, :])
            ).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :] * (h_floor_valid[:, None] & w_ceil_valid[None, :])).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :] * (h_ceil_valid[:, None] & w_floor_valid[None, :])).flatten(),
            (h_frac[:, None] * w_frac[None, :] * (h_ceil_valid[:, None] & w_ceil_valid[None, :])).flatten(),
        ]

        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
        for index in range(4):
            idx_parts[index].append(corner_indices[index][reorder])
            weight_parts[index].append(corner_weights[index][reorder])

    indices = torch.stack([torch.cat(parts) for parts in idx_parts])
    weights = torch.stack([torch.cat(parts) for parts in weight_parts])
    return indices, weights


class MuseGlimmerVisionRotaryEmbedding(nn.Module):
    """Canonical independent-frequency two-axis vision RoPE."""

    def __init__(self, config) -> None:
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        spatial_dim = head_dim // 2
        theta = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim))
        # AutoModel constructs the module on meta, so the checkpoint adapter
        # materializes this deterministic HF buffer during native loading.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].to(device=x.device, dtype=torch.float32)
        inv_freq = inv_freq.expand(position_ids.shape[0], -1, 1)
        w_ids = position_ids[:, :, 0][:, None, :].float()
        h_ids = position_ids[:, :, 1][:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freq_h = (inv_freq @ h_ids).transpose(1, 2)
            freq_w = (inv_freq @ w_ids).transpose(1, 2)
            freq = torch.cat([freq_w, freq_h, freq_w, freq_h], dim=-1)
            cos = freq.cos()
            sin = freq.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


class MuseGlimmerVisionAttention(nn.Module):
    """Canonical packed bidirectional vision attention."""

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.o_proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query = self.q_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
        key = self.k_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
        value = self.v_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
        query = query.transpose(2, 1)
        key = key.transpose(2, 1)
        value = value.transpose(2, 1)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [torch.split(tensor, lengths, dim=2) for tensor in (query, key, value)]
        outputs = [
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self.scaling)
            .transpose(1, 2)
            .contiguous()
            for q, k, v in zip(*splits)
        ]
        return self.o_proj(torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous())


class MuseGlimmerVisionMLP(nn.Module):
    """Canonical vision MLP."""

    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class MuseGlimmerVisionBlock(GradientCheckpointingLayer):
    """Canonical vision encoder layer."""

    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = MuseGlimmerVisionAttention(config)
        self.mlp = MuseGlimmerVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.ln_1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.ln_2(hidden_states))


class MuseGlimmerVisionEncoder(nn.Module):
    """Canonical processor-patch vision encoder."""

    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        vision_config = config.vision_config
        self.config = vision_config
        patch_dim = vision_config.patch_temporal * 3 * vision_config.patch_size**2
        self.conv1_linear = nn.Linear(patch_dim, vision_config.hidden_size, bias=False)
        self.positional_embedding_vlm = nn.Parameter(
            torch.zeros(vision_config.pos_emb_height * vision_config.pos_emb_width, vision_config.hidden_size)
        )
        self.rotary_emb = MuseGlimmerVisionRotaryEmbedding(vision_config)
        self.ln_pre = nn.LayerNorm(vision_config.hidden_size, eps=vision_config.layer_norm_eps)
        self.transformer = nn.ModuleList(
            [MuseGlimmerVisionBlock(vision_config) for _ in range(vision_config.num_hidden_layers)]
        )
        self.ln_post = nn.LayerNorm(vision_config.hidden_size, eps=vision_config.layer_norm_eps)

    def _pixel_shuffle(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        factor = self.config.merge_size
        dim = hidden_states.shape[-1]
        output = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            n_tokens = t * h * w
            chunk = hidden_states[offset : offset + n_tokens]
            n_out_per_frame = (h // factor) * (w // factor)
            permutation = torch.arange(h * w, device=hidden_states.device)
            permutation = permutation.view(h // factor, factor, w // factor, factor)
            permutation = permutation.permute(0, 2, 1, 3).reshape(-1)
            if t > 1:
                frame_offsets = (torch.arange(t, device=hidden_states.device) * h * w).view(t, 1)
                permutation = (permutation.unsqueeze(0) + frame_offsets).reshape(-1)
            downsampled = chunk[permutation]
            downsampled = downsampled.view(t * n_out_per_frame, factor * factor, dim)
            downsampled = downsampled.permute(0, 2, 1).contiguous()
            output.append(downsampled.view(t * n_out_per_frame, dim * factor * factor))
            offset += n_tokens
        return torch.cat(output, dim=0)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            spatial_merge_size=1,
            window_size=self.config.pos_emb_height * self.config.patch_size,
            patch_size=self.config.patch_size,
        )

        batch_sequence_len = pixel_values.shape[0]
        target_dtype = self.conv1_linear.weight.dtype
        patch_embeds = self.conv1_linear(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(-2).squeeze(-1).reshape(batch_sequence_len, -1)
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.config.pos_emb_height,
            spatial_merge_size=1,
        )
        pos_embeds = (self.positional_embedding_vlm[bilinear_indices] * bilinear_weights[:, :, None]).sum(0)
        hidden_states = self.ln_pre(embeddings + pos_embeds.to(embeddings.dtype))
        hidden_states = hidden_states[window_index, :]

        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1)
        position_ids = position_ids.flip(-1) + 1
        position_ids = position_ids[None, window_index, :]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens_mapping = {
            "full_attention": cu_seqlens,
            "window_attention": cu_window_seqlens,
        }
        for index, block in enumerate(self.transformer):
            hidden_states = block(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens_mapping[self.config.layer_types[index]],
            )

        hidden_states = hidden_states[torch.argsort(window_index), :]
        hidden_states = self.ln_post(hidden_states)
        return self._pixel_shuffle(hidden_states, grid_thw)


class MuseGlimmerVisionAdapter(nn.Module):
    """Canonical two-layer visual adapter."""

    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.out_hidden_size, config.projector_hidden_size, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.c_proj = nn.Linear(config.projector_hidden_size, config.projector_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.c_proj(self.act(self.c_fc(x))))
