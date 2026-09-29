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

import functools
import math

import torch


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    If cos/sin have fewer dimensions than x (due to partial_rotary_factor < 1.0),
    only the first rotary_dim dimensions of x are rotated, and the rest are passed through.

    Args:
        x: Input tensor (..., head_dim)
        cos: Cosine tensor (..., rotary_dim // 2)
        sin: Sine tensor (..., rotary_dim // 2)
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    # Handle partial rotary embeddings
    # cos/sin have dimension rotary_dim//2, so full rotary_dim is cos.shape[-1] * 2
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim < x.shape[-1]:
        # Only apply rotary to the first rotary_dim dimensions
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2, x_pass), dim=-1)
    else:
        # Standard full rotary embeddings
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        partial_rotary_factor: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        # Compute the actual dimension to apply RoPE to (may be less than head_dim)
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    @functools.cache
    @torch.no_grad()
    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071

        Uses rotary_dim instead of head_dim to support partial rotary embeddings.
        """
        freq = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=self.device) / self.rotary_dim
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0  # YaRN concentration

            d_half = self.rotary_dim / 2
            # NTK by parts
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[1]
        cos, sin = self._compute_cos_sin(num_tokens)

        query = apply_rotary_emb(query, cos, sin)

        key = apply_rotary_emb(key, cos, sin)
        return query, key


def apply_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    format: str = "bshd",
    rope_fusion: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    concentration: float | None = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        freqs_cis: Frequency tensor. Format depends on rope_fusion:
                   - If rope_fusion=True: [angles, angles] for TE fused rope
                   - If rope_fusion=False: [cos, sin] with concentration applied
        format: QKV format ("bshd" or "thd").
        rope_fusion: If True, use TE fused rope. If False, use non-fused rope.
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.
        cp_size: Context parallelism size.
        cp_rank: Context parallelism rank.

    Returns:
        Tuple of (q, k) with rotary embeddings applied.
    """
    if rope_fusion:
        from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

        q = apply_rotary_pos_emb(
            q, freqs_cis, tensor_format=format, fused=True, cu_seqlens=cu_seqlens, cp_size=cp_size, cp_rank=cp_rank
        )
        k = apply_rotary_pos_emb(
            k, freqs_cis, tensor_format=format, fused=True, cu_seqlens=cu_seqlens, cp_size=cp_size, cp_rank=cp_rank
        )
        if concentration is not None:
            q = q * concentration
            k = k * concentration
        return q, k
    else:
        cos, sin = freqs_cis.split(freqs_cis.shape[-1] // 2, dim=-1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k


@torch.no_grad()
def position_ids_to_freqs_cis(
    rotary_emb: RotaryEmbedding,
    position_ids: torch.Tensor,
    qkv_format: str = "bshd",
    for_fused_rope: bool = True,
    cp_size: int = 1,
) -> torch.Tensor:
    if qkv_format == "thd":
        if for_fused_rope:
            # shape[0] is T for 1D [T] or batch for 2D [B, T].
            # For THD with PP schedule, position_ids may be [1, T] — use shape[-1].
            seq_len = position_ids.shape[-1] if position_ids.ndim >= 2 else position_ids.shape[0]
            position_ids = torch.arange(seq_len * cp_size, device=position_ids.device, dtype=torch.int32).unsqueeze(0)
        else:
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)

    concentration, inv_freq = rotary_emb._compute_concentration_and_inv_freq()
    inv_freq = inv_freq.to(device=position_ids.device, dtype=torch.float32)
    # angles: (B, T, D/2)
    angles = torch.einsum("bt,d->btd", position_ids.to(dtype=torch.float32), inv_freq)

    if for_fused_rope:
        # TE fused rope expects [angles, angles]
        freqs_cis = torch.cat((angles, angles), dim=-1)
    else:
        # Non-fused rope expects [cos, sin] with concentration applied
        cos = torch.cos(angles) * concentration
        sin = torch.sin(angles) * concentration
        freqs_cis = torch.cat([cos, sin], dim=-1)

    if qkv_format == "thd":
        freqs_cis = freqs_cis.squeeze(0)
    else:
        freqs_cis = freqs_cis[0] if for_fused_rope else freqs_cis

    if for_fused_rope:
        freqs_cis = freqs_cis.reshape(freqs_cis.size(0), 1, 1, freqs_cis.size(1)).contiguous()

    return freqs_cis
