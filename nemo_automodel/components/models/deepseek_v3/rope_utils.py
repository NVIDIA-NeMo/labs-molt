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

import math

import torch


# Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modular_deepseek_v3.py#L78
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_freqs_cis(
    qk_rope_head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    rope_scaling: dict[str, float | int] | None,
) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        qk_rope_head_dim (int): Dimensionality of the rotary positional embeddings.
        max_seq_len (int): Maximum sequence length.
        original_seq_len (int): Original sequence length.
        beta_fast (int): Fast beta value for the exponential computation.
        beta_slow (int): Slow beta value for the exponential computation.
        rope_theta (float): Base value for the exponential computation.
        rope_factor (float): Factor value for the exponential computation.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = qk_rope_head_dim
    seqlen = max_seq_len
    base = rope_theta

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if rope_scaling is not None and all(
        map(lambda x: x in rope_scaling, ["factor", "beta_fast", "beta_slow", "original_max_position_embeddings"])
    ):
        factor = rope_scaling["factor"]
        beta_fast = rope_scaling["beta_fast"]
        beta_slow = rope_scaling["beta_slow"]
        original_seq_len = rope_scaling["original_max_position_embeddings"]
        if seqlen > original_seq_len:
            low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

    return freqs


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, qkv_format: str = "bshd", unsqueeze_dim: int | None = None
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    if qkv_format == "thd":
        x = x.unsqueeze(0)

    if unsqueeze_dim is not None:
        x = x.unsqueeze(unsqueeze_dim)

    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))  # [b, s, h, d] -> [b, s, h, d/2, 2]
    freqs_cis = freqs_cis.view(x.size(0), x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    y = y.to(dtype)

    if unsqueeze_dim is not None:
        y = y.squeeze(unsqueeze_dim)

    if qkv_format == "thd":
        y = y.squeeze(0)
    return y


def apply_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    format: str = "bshd",
    rope_fusion: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_fusion:
        from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

        q = apply_rotary_pos_emb(
            q,
            freqs_cis,
            tensor_format=format,
            interleaved=True,
            fused=True,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        k = apply_rotary_pos_emb(
            k,
            freqs_cis,
            tensor_format=format,
            interleaved=True,
            fused=True,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        return q, k
    else:
        q = apply_rotary_emb(q, freqs_cis, qkv_format=format)
        k = apply_rotary_emb(k, freqs_cis, qkv_format=format)
        return q, k


@torch.no_grad()
def freqs_cis_from_position_ids(
    position_ids: torch.Tensor,
    freqs: torch.Tensor,
    qkv_format: str = "bshd",
    for_fused_rope: bool = False,
    cp_size: int = 1,
) -> torch.Tensor:
    if qkv_format == "thd":
        if for_fused_rope:
            # For fused rope with thd, use sequential positions
            position_ids = torch.arange(
                position_ids.shape[0] * cp_size, device=position_ids.device, dtype=torch.int32
            ).unsqueeze(0)
        else:
            # For non-fused thd, ensure 2D
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

    freqs = freqs.to(device=position_ids.device)

    # Compute angles: (B, T, D/2) - same as original implementation
    # angles = torch.matmul(position_ids.unsqueeze(-1).float(), freqs.unsqueeze(0))
    angles = torch.einsum("bt,d->btd", position_ids.to(dtype=torch.float32), freqs)

    if for_fused_rope:
        if qkv_format == "thd":
            freqs_cis = angles.squeeze(0)
        else:
            # For bshd, take first batch (assumes uniform positions)
            freqs_cis = angles[0]

        # TE fused rope expects [angles, angles] format
        # freqs_cis = torch.cat((angles, angles), dim=-1)
        freqs_cis = torch.stack((freqs_cis.view(-1, 1), freqs_cis.view(-1, 1)), dim=-1).view(freqs_cis.shape[0], -1)

        # Reshape to (T, 1, 1, D) for TE fused rope
        freqs_cis = freqs_cis.reshape(freqs_cis.size(0), 1, 1, freqs_cis.size(1)).contiguous()
    else:
        # Return complex exponentials for non-fused rope (original behavior)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        if qkv_format == "thd":
            freqs_cis = freqs_cis.squeeze(0)

    return freqs_cis
