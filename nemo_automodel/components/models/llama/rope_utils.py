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

"""Rotary Position Embedding utilities for Llama and Qwen2 models.

This module provides RoPE implementation following HuggingFace's architecture.

API:
    rotary_emb = RotaryEmbedding(config)
    cos, sin = rotary_emb(x, position_ids)  # Returns (cos, sin) tuple
    q, k = apply_rotary_pos_emb(q, k, cos, sin)  # Applies RoPE

Supports both:
- LlamaConfig: uses config.rope_theta and config.rope_scaling
- Qwen2Config: uses config.rope_parameters["rope_theta"] and config.rope_parameters

Note: gpt_oss and deepseek_v3 have their own specialized rope_utils.py
with model-specific optimizations (YaRN, MLA, etc.).
"""

import math

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: Query tensor ``[B, Hq, S, D]`` in BSHD attention layout or
            ``[T, Hq, D]`` in packed THD layout. ``B`` is batch, ``S`` is
            sequence, ``T`` is total local tokens, ``Hq`` is query heads, and
            ``D`` is head dimension.
        k: Key tensor ``[B, Hkv, S, D]`` or ``[T, Hkv, D]``, where ``Hkv`` is
            key/value heads.
        cos: Cosine embeddings ``[B, S, D]`` or ``[T, D]``.
        sin: Sine embeddings ``[B, S, D]`` or ``[T, D]``.

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_quack(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    quack_apply_rotary_emb,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Llama-style RoPE with QuACK's fused rotary kernel.

    QuACK consumes BSHD tensors and half-width cosine/sine tables, whereas the
    HuggingFace contract uses BHSD tensors and duplicates the table across the
    full head dimension. Batch size one uses QuACK's in-place path. Larger
    batches are rotated independently so batch-specific position IDs remain
    correct rather than assuming every sequence uses the same positions.
    """

    rotary_dim = cos.shape[-1]
    rotary_dim_half = rotary_dim // 2

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x_bshd = x.permute(0, 2, 1, 3)
        if x_bshd.shape[0] == 1:
            out = quack_apply_rotary_emb(
                x_bshd,
                cos[0, :, :rotary_dim_half],
                sin[0, :, :rotary_dim_half],
                inplace=True,
            )
        else:
            out = torch.cat(
                [
                    quack_apply_rotary_emb(
                        x_bshd[batch_idx : batch_idx + 1],
                        cos[batch_idx, :, :rotary_dim_half],
                        sin[batch_idx, :, :rotary_dim_half],
                    )
                    for batch_idx in range(x_bshd.shape[0])
                ],
                dim=0,
            )
        return out.permute(0, 2, 1, 3)

    return _apply(q), _apply(k)


def apply_rotary_pos_emb_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE using TE's fused kernel.

    Args:
        q: Query tensor ``[B, Hq, S, D]`` or packed ``[T, Hq, D]``.
        k: Key tensor ``[B, Hkv, S, D]`` or packed ``[T, Hkv, D]``.
        freqs_cis: Global raw-angle table ``[S, 1, 1, D]`` in TE format.
        cu_seqlens: Packed-document cumulative lengths ``[N + 1]``. Required
            for THD, where ``N`` is the number of packed documents.
        cp_size: Number of context-parallel ranks partitioning ``T``.
        cp_rank: Rank within the context-parallel mesh.

    Returns:
        Rotated tensors with the same local shapes and storage ownership as
        ``q`` and ``k``.
    """
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb as te_apply_rope

    if q.ndim == 3:
        if cu_seqlens is None:
            raise ValueError("Packed THD fused RoPE requires cu_seqlens.")
        q = te_apply_rope(
            q,
            freqs_cis,
            tensor_format="thd",
            fused=True,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        k = te_apply_rope(
            k,
            freqs_cis,
            tensor_format="thd",
            fused=True,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        return q, k

    # TE expects bshd format: [batch, seq, heads, head_dim]
    q_bshd = q.permute(0, 2, 1, 3)
    k_bshd = k.permute(0, 2, 1, 3)
    q_bshd = te_apply_rope(q_bshd, freqs_cis, tensor_format="bshd", fused=True)
    k_bshd = te_apply_rope(k_bshd, freqs_cis, tensor_format="bshd", fused=True)
    return q_bshd.permute(0, 2, 1, 3), k_bshd.permute(0, 2, 1, 3)


def _get_rope_config(config) -> tuple[float, dict]:
    """Extract rope parameters from config (handles both Llama and Qwen2 formats).

    Returns:
        Tuple of (rope_theta, rope_scaling_dict)
    """
    # Qwen2 uses config.rope_parameters, Llama uses config.rope_theta + config.rope_scaling
    if hasattr(config, "rope_parameters") and config.rope_parameters:
        rope_params = config.rope_parameters
        base = rope_params.get("rope_theta", 10000.0)
        rope_scaling = rope_params  # rope_parameters contains scaling info too
    else:
        base = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", {}) or {}
    return base, rope_scaling


def _compute_default_inv_freq(config, device: torch.device | None = None) -> tuple[torch.Tensor, float]:
    """Computes inverse frequencies for standard RoPE."""
    base, _ = _get_rope_config(config)
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(dim * partial_rotary_factor)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim))
    return inv_freq, 1.0


def _compute_llama3_inv_freq(config, device: torch.device | None = None) -> tuple[torch.Tensor, float]:
    """Computes inverse frequencies for Llama3-style RoPE with smooth interpolation.

    Branch logic (matches HF _compute_llama3_parameters):
      - Long wavelength  (low freq,  wavelen > low_freq_wavelen)  → scale by factor
      - Short wavelength (high freq, wavelen < high_freq_wavelen) → unchanged
      - Medium band → smooth interpolation
    """
    inv_freq, _ = _compute_default_inv_freq(config, device)

    _, rope_scaling = _get_rope_config(config)
    factor = rope_scaling.get("factor", 1.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings", config.max_position_embeddings)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # Long wavelen (low freq) → scale; short wavelen (high freq) → unchanged
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # Medium band: smooth interpolation
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama, 1.0


class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding module for Llama and Qwen2 models.

    Returns (cos, sin) tuple for use with apply_rotary_pos_emb.

    Usage:
        rotary_emb = RotaryEmbedding(config)
        cos, sin = rotary_emb(x, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    """

    inv_freq: torch.Tensor

    def __init__(self, config, device: torch.device | None = None, rope_fusion: bool = False):
        super().__init__()
        self.max_seq_len_cached = 0
        self.rope_fusion = rope_fusion
        self.dtype = getattr(config, "torch_dtype", None) or torch.float32

        # ``default`` and ``llama3`` have local fast-path implementations. Every
        # other schedule (``yarn``, ``linear``, ``dynamic``, ``longrope``, ...)
        # is delegated to transformers' canonical ``ROPE_INIT_FUNCTIONS`` so the
        # frequencies and attention scaling match HuggingFace exactly.
        rope_functions = {
            "default": _compute_default_inv_freq,
            "llama3": _compute_llama3_inv_freq,
        }

        # Determine rope_type and compute inv_freq
        _, rope_scaling = _get_rope_config(config)
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))

        # Resolve the compute function. An unrecognised ``rope_type`` previously
        # fell back silently to the llama3 NTK schedule, which yields wrong
        # frequencies for YaRN/linear/dynamic configs -- a latent bug, e.g. for
        # an EAGLE dense draft that inherits a YaRN target's ``rope_scaling``.
        # Delegate such types to transformers instead of guessing.
        compute_fn = rope_functions.get(rope_type)
        if compute_fn is None:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            compute_fn = ROPE_INIT_FUNCTIONS.get(rope_type)
            if compute_fn is None:
                supported = sorted(set(rope_functions) | set(ROPE_INIT_FUNCTIONS))
                raise ValueError(f"Unsupported RoPE rope_type={rope_type!r}; expected one of {supported}.")
        inv_freq, self.attention_scaling = compute_fn(config, device)

        # Keep the config + compute fn so ``_build_cache`` can recompute ``inv_freq``
        # in float32. ``LlamaForCausalLM.__init__`` casts the whole model via
        # ``self.to(config.torch_dtype)``, and ``nn.Module.to`` rounds floating-point
        # buffers -- so the ``inv_freq`` buffer registered below is downcast to bf16.
        # Building the cos/sin tables from that bf16-rounded buffer degrades RoPE
        # precision relative to HuggingFace (which keeps ``inv_freq`` in float32) and
        # surfaces as a large logit divergence when a checkpoint is reloaded in
        # vanilla HF. Recomputing from config keeps the tables float32-accurate
        # regardless of the model's parameter dtype.
        self._rope_config = config
        self._inv_freq_compute_fn = compute_fn

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cache", None, persistent=False)
        self.register_buffer("_sin_cache", None, persistent=False)
        self.register_buffer("_freqs_cache", None, persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        """Build cos/sin cache in config dtype for positions [0, seq_len)."""
        self.max_seq_len_cached = seq_len

        # Compute in float32 for precision, then convert to target dtype.
        # Recompute inv_freq from config instead of reading ``self.inv_freq``: the
        # registered buffer is rounded to the model dtype (e.g. bf16) by the
        # model-wide ``.to()`` in ``LlamaForCausalLM.__init__``, and upcasting a
        # bf16 buffer back to float32 cannot recover the lost precision. See __init__.
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq, _ = self._inv_freq_compute_fn(self._rope_config, device)
        inv_freq = inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, head_dim]

        self._cos_cache = (emb.cos() * self.attention_scaling).to(self.dtype)
        self._sin_cache = (emb.sin() * self.attention_scaling).to(self.dtype)
        if self.rope_fusion:
            # TE fused rope expects raw angles in [seq, 1, 1, head_dim] format
            self._freqs_cache = emb.to(self.dtype).unsqueeze(1).unsqueeze(1).contiguous()

    def _ensure_cache(self, seq_len: int, device: torch.device) -> None:
        """Build or grow the cos/sin cache so it covers positions ``[0, seq_len)``."""
        if self._cos_cache is None or seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, device)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        qkv_format: str = "bshd",
        cp_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for the given positions.

        In the non-fused path ``cos`` / ``sin`` are gathered by the *values* in
        ``position_ids``, so non-contiguous positions receive the correct rotary
        phase: EAGLE TTT depth offsets (``arange(seq_len) + step_idx``), packed
        sequences, and context parallelism all pass
        ``position_ids != arange(seq_len)``. The earlier implementation returned
        ``cos_cache[:seq_len]``, which keyed only on the sequence *length* and
        silently ignored the position values. For the common
        ``position_ids == arange(seq_len)`` case the gather is numerically
        identical to that slice.

        The fused TE path (``rope_fusion=True``) consumes raw angles indexed by
        sequence position and assumes contiguous ``[0, seq_len)`` positions, so
        it keeps the legacy contiguous slice and does NOT honor non-contiguous
        ``position_ids``.

        Args:
            x: Hidden states ``[B, S, H]`` or packed ``[T, H]``; used for
                device placement. ``H`` is hidden size.
            position_ids: Position IDs ``[B, S]`` or packed local IDs ``[T]``.
            qkv_format: ``"bshd"`` for padded batches or ``"thd"`` for a
                packed total-token layout.
            cp_size: Context-parallel size. For fused THD RoPE, the cached raw
                frequency table spans the global token count ``T * cp_size``.

        Returns:
            Cosine and sine tensors ``[B, S, D]`` or ``[T, D]``. When fused
            RoPE is enabled, the tuple also carries a global raw-frequency
            table ``[S, 1, 1, D]`` as its third item.
        """
        if qkv_format not in ("bshd", "thd"):
            raise ValueError(f"Unsupported qkv_format={qkv_format!r}; expected 'bshd' or 'thd'.")

        if self.rope_fusion:
            # The fused TE kernel indexes raw angles by sequence position and
            # assumes contiguous ``[0, seq_len)`` positions, so it only needs the
            # cache sized to ``seq_len`` -- and must avoid the ``position_ids``
            # host-device sync below on this default GPU training path.
            seq_len = position_ids.shape[-1]
            global_seq_len = seq_len * cp_size if qkv_format == "thd" else seq_len
            self._ensure_cache(global_seq_len, x.device)
            if qkv_format == "thd":
                flat_position_ids = position_ids.reshape(-1)
                cos = self._cos_cache[flat_position_ids]
                sin = self._sin_cache[flat_position_ids]
                return cos, sin, self._freqs_cache[:global_seq_len]
            cos = self._cos_cache[:seq_len].unsqueeze(0).expand(position_ids.shape[0], -1, -1)
            sin = self._sin_cache[:seq_len].unsqueeze(0).expand(position_ids.shape[0], -1, -1)
            return cos, sin, self._freqs_cache[:seq_len]

        # Non-fused: gather per-position. Size the cache to the highest requested
        # position -- ``position_ids`` can exceed its own length (e.g. EAGLE TTT
        # passes ``arange(seq_len) + k``), so ``seq_len`` alone is not a safe
        # upper bound. This ``.item()`` sync only runs on the non-fused path.
        needed = max(int(position_ids.max().item()) + 1, position_ids.shape[-1])
        self._ensure_cache(needed, x.device)
        # Gather per-position; identical to ``cos_cache[:seq_len]`` for arange.
        cos = self._cos_cache[position_ids]
        sin = self._sin_cache[position_ids]
        return cos, sin


# Aliases for HuggingFace compatibility
RotaryEmbedding = LlamaRotaryEmbedding
Qwen2RotaryEmbedding = LlamaRotaryEmbedding


__all__ = [
    "RotaryEmbedding",
    "LlamaRotaryEmbedding",
    "Qwen2RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_fused",
    "apply_rotary_pos_emb_quack",
]
