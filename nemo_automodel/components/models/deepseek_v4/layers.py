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

"""DeepSeek V4 Attention Layer.

Architecture (from official inference/model.py):

Q path:
  x  -> wq_a [hidden -> q_lora_rank]
     -> q_norm (RMSNorm)
     -> wq_b  [q_lora_rank -> n_heads * head_dim]
     -> reshape [n_heads, head_dim]
     -> per-head RMSNorm  (q_norm applied per-head in official code)
     -> apply_rotary_emb on last rope_head_dim dims

KV path (K = V, single latent):
  x  -> wkv   [hidden -> head_dim]        # single KV head, K = V = kv
     -> kv_norm (RMSNorm on head_dim)
     -> apply_rotary_emb on last rope_head_dim dims
  K = V = kv  (one latent vector serves both key and value)

Output path (grouped):
  o [bsz, seq, n_heads, head_dim]
    -> reshape [bsz, seq, n_groups, n_heads_per_group * head_dim]
    -> wo_a einsum per group: [n_heads_per_group * head_dim] -> [o_lora_rank]
    -> reshape [bsz, seq, n_groups * o_lora_rank]
    -> wo_b [n_groups * o_lora_rank -> hidden]

attn_sink: learnable per-head scalar bias added to attention-sink position score.

HC (Hyper-Connections):
  Each Block maintains hc_mult=4 copies of the hidden state.
  hc_pre  reduces [bsz, seq, hc_mult, dim] -> [bsz, seq, dim] via Sinkhorn mixing.
  hc_post expands [bsz, seq, dim] -> [bsz, seq, hc_mult, dim].
  See ``DeepseekV4HyperConnection.compute_weights`` and
  ``optimized_kernels.dsv4_sinkhorn_normalize`` for the torch reference and
  optional TileKernels Sinkhorn path.

Compress-ratio attention (Compressor + Indexer) is wired into
DeepseekV4Attention.forward for layers with compress_ratio > 0.
All layers share the same sliding-window causal mask on the local KV path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.cp import (
    dsv4_cp_all_gather,
    dsv4_cp_all_gather_metadata,
    dsv4_cp_enabled,
    dsv4_cp_rank,
    dsv4_cp_size,
)
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
    build_dsv4_sparse_topk_indices,
    dsv4_indexer_scores,
    dsv4_sinkhorn_normalize,
    dsv4_sparse_attention,
)
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _full_tensor_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.clone()


def _dsv4_kernel_backend(backend: BackendConfig) -> str:
    """Use TileLang DSV4 sparse kernels only when the attention backend requests them."""
    return "tilelang" if backend.attn == "tilelang" else "torch"


def _dsv4_sinkhorn_backend(backend: BackendConfig) -> str:
    """Use optional TileKernels sinkhorn when available, otherwise torch fallback."""
    return "auto" if backend.attn == "tilelang" else "torch"


def _rms_norm_last_dim(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS-normalize the last dim without materializing an ``x.square()`` tensor."""
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


# ---------------------------------------------------------------------------
# DeepSeek V4 attention + compressor + indexer + rotary embedding, ported
# verbatim from HuggingFace transformers PR 45616 (Arthur Zucker, "Add
# DeepSeek V4") at
#   transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
# with two adjustments:
#   1) Rotary helper ``apply_rotary_pos_emb`` and ``repeat_kv`` are inlined
#      so we do not depend on HF's transformers-version-specific rotary API.
#   2) The ``DeepseekV4Cache`` integration is replaced with a minimal
#      training-only shim — KAutomodel training never carries a KV cache,
#      so ``accumulate_windows`` / ``update_pool`` are pass-throughs on a
#      per-forward scratch dict.  The KV-cache path is left for a future
#      inference port.
# The compressor + indexer modules are only constructed when a layer's
# ``compress_ratio`` is non-zero; sparse attention and indexer execution are
# selected through ``BackendConfig``.
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (Llama / GPT-NeoX style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_partial_rope_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int
) -> torch.Tensor:
    """Interleaved RoPE on the last ``rope_head_dim`` dims of ``x`` (pairs are
    ``(2k, 2k+1)``).  Matches the DeepSeek inference reference's complex-mul
    formulation in ``dsv4flash/inference/model.py:apply_rotary_emb``: the
    released DSV4-Flash weights were trained with this layout, NOT the
    Llama-style ``rotate_half`` layout HF transformers PR 45616/45643 still
    uses (pairs ``(d, d+rd/2)``).

    Args:
        x: ``[..., rope_head_dim]`` (or larger trailing dim with rope on the
            last ``rope_head_dim`` slice).  Typical attention-layout shapes:
            ``[B, H, S, D]`` for q/k or ``[B, 1, S, D]`` for shared-KV.
        cos, sin: shape ``[B, S, rope_head_dim]`` produced by the Llama-style
            ``cat([freqs, freqs], -1)`` rotary; we take the first half which
            contains the unique per-pair frequencies (the second half is a
            duplicate that the Llama-style helper needs and we don't).
        rope_head_dim: Must be even.

    Inverse rotation: pass ``-sin`` instead of ``sin`` (caller's
    responsibility — same as our existing inverse-rope call site).
    """
    rd = rope_head_dim
    half = rd // 2
    nope, rope = x[..., :-rd], x[..., -rd:]
    # Pair-reshape last dim: [..., rd] -> [..., rd/2, 2]
    rope_pairs = rope.float().unflatten(-1, (-1, 2))
    a, b = rope_pairs[..., 0], rope_pairs[..., 1]  # [..., rd/2]
    c = cos[..., :half].float()
    s = sin[..., :half].float()
    # Broadcast c/s up to ``a``'s rank by inserting a head dim before S.
    while c.ndim < a.ndim:
        c = c.unsqueeze(1)
        s = s.unsqueeze(1)
    out = torch.empty_like(x)
    if nope.shape[-1] > 0:
        out[..., :-rd] = nope
    out_pairs = out[..., -rd:].unflatten(-1, (-1, 2))
    out_pairs[..., 0] = a * c - b * s
    out_pairs[..., 1] = a * s + b * c
    return out


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of transformers.models.llama.modeling_llama.apply_rotary_pos_emb."""
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Port of transformers.models.llama.modeling_llama.repeat_kv."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _yarn_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    import math

    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _yarn_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int) -> tuple[int, int]:
    import math

    low = math.floor(_yarn_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_yarn_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(min_v: float, max_v: float, dim: int, device=None) -> torch.Tensor:
    if min_v == max_v:
        max_v += 0.001
    linear = (torch.arange(dim, dtype=torch.float32, device=device) - min_v) / (max_v - min_v)
    return torch.clamp(linear, 0, 1)


class DeepseekV4RotaryEmbedding(nn.Module):
    """V4 rotary embedding.  Produces ``(cos, sin)`` sized to ``qk_rope_head_dim``
    (via ``partial_rotary_factor = qk_rope_head_dim / head_dim``), matching HF.

    YaRN: when ``rope_scaling`` is a YaRN-typed dict
    (``{"type": "yarn", "factor": F, "original_max_position_embeddings": L0,
    "beta_fast": ..., "beta_slow": ...}``), modify ``inv_freq`` per
    ``dsv4flash/inference/model.py:precompute_freqs_cis`` — frequency
    interpolation with a smooth linear ramp between beta_fast/beta_slow
    correction dims.  Used by the compress-rope (theta=160000) on layers
    with ``compress_ratio > 0``.  The main rope (theta=10000, used only on
    sliding-window layers) gets ``rope_scaling=None`` because the reference
    builds it with ``original_seq_len=0`` for those layers.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        partial_rotary_factor: float,
        attention_scaling: float = 1.0,
        device: torch.device | None = None,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        if rope_scaling and str(rope_scaling.get("type", "")).lower() == "yarn":
            factor = float(rope_scaling.get("factor", 1.0))
            orig = int(rope_scaling.get("original_max_position_embeddings", 0))
            beta_fast = float(rope_scaling.get("beta_fast", 32))
            beta_slow = float(rope_scaling.get("beta_slow", 1))
            if orig > 0 and factor > 0:
                low, high = _yarn_correction_range(beta_fast, beta_slow, dim, rope_theta, orig)
                smooth = 1.0 - _yarn_linear_ramp(low, high, dim // 2, device=inv_freq.device)
                inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth
        self.attention_scaling = attention_scaling
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force fp32 for numerical stability.
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear (HF PR 45616 port).

    ``weight`` parameter has the standard ``nn.Linear`` shape
    ``[out_features, in_features_per_group]`` so quantizers keyed on
    ``nn.Linear.weight`` still find it; ``forward`` does per-group bmm.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_groups, in_features_per_group]
        batch_shape = x.shape[:-2]
        d_in = x.shape[-1]
        out_per_group = self.out_features // self.n_groups
        w = self.weight.view(self.n_groups, out_per_group, d_in)
        x = x.reshape(-1, self.n_groups, d_in).permute(1, 0, 2)
        y = torch.bmm(x, w.transpose(-1, -2)).permute(1, 0, 2)
        return y.reshape(*batch_shape, self.n_groups, out_per_group)


class DeepseekV4TrainCache:
    """Training-only cache shim mirroring the three methods ``DeepseekV4Compressor``
    / ``DeepseekV4Indexer`` call on ``DeepseekV4Cache``.

    KAutomodel training forward is stateless — we never persist KV or compressor
    windows across calls.  Each ``DeepseekV4Attention.forward`` creates a fresh
    cache instance, which holds per-layer scratch dicts for the duration of the
    call.  When a full window hasn't accumulated yet we return an empty tensor
    and let the downstream code handle it.
    """

    def __init__(self):
        self.compressor_state: list[dict] = []
        self.indexer_state: list[dict] = []

    def _branch_state(self, state_key: str, layer_idx: int) -> dict:
        store = getattr(self, state_key, None)
        if store is None:
            store = []
            setattr(self, state_key, store)
        while len(store) <= layer_idx:
            store.append({"buffer_kv": None, "buffer_gate": None, "pooled": None})
        return store[layer_idx]

    def accumulate_windows(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        layer_idx: int,
        state_key: str,
        ratio: int,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        state = self._branch_state(state_key, layer_idx)
        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        if buf_kv is not None and buf_kv.shape[1]:
            kv = torch.cat([buf_kv, kv], dim=1)
            gate = torch.cat([buf_gate, gate], dim=1)
        usable = (kv.shape[1] // ratio) * ratio
        state["buffer_kv"] = kv[:, usable:]
        state["buffer_gate"] = gate[:, usable:]
        pool_base = max(0, start_pos) - (buf_kv.shape[1] if buf_kv is not None else 0)
        return kv[:, :usable], gate[:, :usable], pool_base

    def update_pool(self, new_pooled: torch.Tensor, layer_idx: int, state_key: str) -> torch.Tensor:
        state = self._branch_state(state_key, layer_idx)
        pool = state["pooled"]
        if new_pooled.shape[1] > 0:
            pool = new_pooled if pool is None else torch.cat([pool, new_pooled], dim=1)
            state["pooled"] = pool
        if pool is None:
            pool = new_pooled.new_zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]))
        return pool


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int) -> torch.Tensor:
    """Split ``x`` along its last dim into nope (first) and rope (last
    ``rope_head_dim``) slices, rotate only the rope slice with INTERLEAVED
    pair-RoPE (pairs ``(2k, 2k+1)``), concat back.

    The DSV4-Flash released checkpoint uses interleaved RoPE end-to-end
    (see ``dsv4flash/inference/model.py:apply_rotary_emb`` — complex
    multiplication on ``view_as_complex`` of pairs).  HF transformers PR
    45616 / PR 45643 ship a Llama-style ``rotate_half`` here instead, which
    pairs ``(d, d+rd/2)``.  Same algebra but a different dim-to-frequency
    mapping — the released weights expect the interleaved layout, so the
    Llama-style helper produces wrong activations on the released checkpoint
    (verified empirically: kv_post_rope cosine drops from 0.9999 to 0.866
    after one block under Llama-style; matches at >0.999 under interleaved).
    """
    return _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim)


def _overlap_transform(
    tensor: torch.Tensor,
    head_dim: int,
    fill_value: float,
    window_seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape ``[B, S, ratio, 2*head_dim]`` -> ``[B, S, 2*ratio, head_dim]`` with the
    cross-window overlap from the DeepSeek inference reference (``Compressor.overlap_transform``
    in ``dsv4flash/inference/model.py:307-314``).

    Window N consumes:
      * positions ``[ratio:]`` of the new tensor: the **second half** of the feature dim
        of window N (current block).
      * positions ``[:ratio]`` of the new tensor: the **first half** of the feature dim
        of window N-1 (previous block, i.e. the overlap into the past).

    Window 0 has no previous block, so its ``[:ratio]`` slice is left at ``fill_value``
    (``0`` for the kv tensor, ``-inf`` for the score tensor so softmax masks it out).
    """
    b, s, ratio, _ = tensor.shape
    new = tensor.new_full((b, s, 2 * ratio, head_dim), fill_value)
    current = tensor[:, :, :, head_dim:]
    previous = tensor[:, :-1, :, :head_dim]
    if window_seq_ids is not None:
        valid_current = window_seq_ids.to(device=tensor.device).unsqueeze(-1).unsqueeze(-1) > 0
        current = torch.where(valid_current, current, torch.full_like(current, fill_value))
    new[:, :, ratio:] = current
    if window_seq_ids is None:
        new[:, 1:, :ratio] = previous
    else:
        seq_ids = window_seq_ids.to(device=tensor.device)
        same_previous_doc = (seq_ids[:, 1:] > 0) & (seq_ids[:, 1:] == seq_ids[:, :-1])
        previous = torch.where(
            same_previous_doc.unsqueeze(-1).unsqueeze(-1),
            previous,
            torch.full_like(previous, fill_value),
        )
        new[:, 1:, :ratio] = previous
    return new


def _query_positions(
    position_ids: torch.Tensor | None,
    *,
    batch: int,
    seq_len: int,
    device: torch.device,
    cp_group=None,
) -> torch.Tensor:
    if position_ids is not None:
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        return position_ids.to(device=device, dtype=torch.long)

    start = dsv4_cp_rank(cp_group) * seq_len if dsv4_cp_enabled(cp_group) else 0
    return torch.arange(start, start + seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)


def _overlap_transform_with_cp(
    tensor: torch.Tensor,
    head_dim: int,
    fill_value: float,
    cp_group,
    window_seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if not dsv4_cp_enabled(cp_group):
        return _overlap_transform(tensor, head_dim, fill_value, window_seq_ids=window_seq_ids)

    local_windows = tensor.shape[1]
    full = dsv4_cp_all_gather(tensor, dim=1, cp_group=cp_group)
    full_seq_ids = dsv4_cp_all_gather_metadata(window_seq_ids, dim=1, cp_group=cp_group)
    full = _overlap_transform(full, head_dim, fill_value, window_seq_ids=full_seq_ids)
    start = dsv4_cp_rank(cp_group) * local_windows
    return full[:, start : start + local_windows, :, :]


def _compressed_window_metadata(
    *,
    seq_ids: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    ready_len: int,
    ratio: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if seq_ids is None or position_ids is None or ready_len <= 0:
        return None, None
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if seq_ids.dim() == 1:
        seq_ids = seq_ids.unsqueeze(0)
    seq_ids = seq_ids[:, :ready_len].to(dtype=torch.long)
    position_ids = position_ids[:, :ready_len].to(device=seq_ids.device, dtype=torch.long)
    n_windows = ready_len // ratio
    if n_windows <= 0:
        empty = seq_ids.new_zeros((seq_ids.shape[0], 0))
        return empty, empty
    seq_w = seq_ids[:, : n_windows * ratio].view(seq_ids.shape[0], n_windows, ratio)
    pos_w = position_ids[:, : n_windows * ratio].view(seq_ids.shape[0], n_windows, ratio)
    first_seq = seq_w[:, :, 0]
    same_seq = (seq_w == first_seq.unsqueeze(-1)).all(dim=-1) & (first_seq > 0)
    window_seq_ids = torch.where(same_seq, first_seq, torch.zeros_like(first_seq))
    window_positions = torch.where(same_seq, pos_w[:, :, 0] // ratio, torch.zeros_like(first_seq))
    return window_seq_ids, window_positions


def _pool_windows(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    head_dim: int,
    overlap: bool = False,
    cp_group=None,
    window_seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax-gated sum-pool over ``ratio`` consecutive tokens.

    Non-overlap mode (HF PR 45616 layout, ratio==128 in V4-Flash):
      Input  ``kv``/``gate`` of shape ``[B, length, head_dim]``.
      Reshape to ``[B, length/ratio, ratio, head_dim]`` and pool over the ``ratio`` axis.

    Overlap mode (DeepSeek inference reference layout, ratio==4 in V4-Flash):
      Input  ``kv``/``gate`` of shape ``[B, length, 2*head_dim]`` (``wkv``/``wgate``
      project to ``2*head_dim`` so each window can carry both its own kv and a
      half-overlap into the next window).
      Reshape to ``[B, length/ratio, ratio, 2*head_dim]``, apply :func:`_overlap_transform`
      to remap to ``[B, length/ratio, 2*ratio, head_dim]``, then pool over the ``2*ratio``
      axis.  Each compressed token thus aggregates ``2*ratio = 8`` raw tokens — the
      ``ratio`` tokens of the current window plus the ``ratio`` tokens of the previous
      window — giving smoother compression boundaries that the released checkpoint
      was trained under.

    HF PR 45616 omits the overlap path entirely; the released DSV4-Flash safetensors
    have ``ape``/``wkv``/``wgate`` shapes that only match the overlap layout (``[ratio,
    2*head_dim]`` and ``[2*head_dim, hidden]``), so we must support it here to load
    the released weights.
    """
    coff = 2 if overlap else 1
    feat = coff * head_dim
    batch, length, _ = kv.shape
    n_windows = length // ratio
    kv_w = kv.view(batch, n_windows, ratio, feat)
    gate_w = gate.view(batch, n_windows, ratio, feat) + ape
    if overlap:
        kv_w = _overlap_transform_with_cp(
            kv_w,
            head_dim,
            fill_value=0.0,
            cp_group=cp_group,
            window_seq_ids=window_seq_ids,
        )
        gate_w = _overlap_transform_with_cp(
            gate_w,
            head_dim,
            fill_value=float("-inf"),
            cp_group=cp_group,
            window_seq_ids=window_seq_ids,
        )
        all_masked = torch.isneginf(gate_w).all(dim=2, keepdim=True)
        gate_w = torch.where(all_masked, torch.zeros_like(gate_w), gate_w)
    return (kv_w * gate_w.softmax(dim=2)).sum(dim=2)


def _rope_pool_positions(
    pool_length: int, pool_base: int, ratio: int, device: torch.device, batch: int
) -> torch.Tensor:
    return (torch.arange(pool_length, device=device) * ratio + pool_base).unsqueeze(0).expand(batch, -1)


def build_causal_padding_mask(
    attention_mask: torch.Tensor | None,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    """Build a 4D additive causal+padding (+optional sliding-window) mask
    compatible with ``eager_attention_with_sink``.

    Mirrors HF's ``create_sliding_window_causal_mask`` (used in
    ``DeepseekV4Model.forward``): each query at position ``i`` attends only to
    keys at positions ``[max(0, i - sliding_window + 1), i]``.  The DSV4-Flash
    weights were trained with this banding on every layer, so dropping it makes
    the softmax see a different distribution than training and degrades loss.

    Inputs:
        attention_mask: 2D ``[B, S]`` tensor with 1=valid, 0=padding (HF convention),
            or already-4D additive mask, or ``None``.
        sliding_window: if not None, mask out keys further back than this many
            positions from the query (in addition to causal masking).
    Returns:
        ``[B, 1, S, S]`` additive mask of ``dtype`` (0 where keep, large negative
        where mask).
    """
    min_value = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    causal = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    causal = torch.triu(causal, diagonal=1)
    if sliding_window is not None and sliding_window > 0:
        # Mask k_pos < q_pos - (sliding_window - 1)  →  diagonal = -(window - 1)
        # tril at that diagonal keeps the lower-band; we need to MASK the lower
        # tail (older keys).  Build a "too old" mask: positions where (q - k) >= window.
        idx = torch.arange(seq_len, device=device)
        too_old = (idx.unsqueeze(0) - idx.unsqueeze(1)) >= sliding_window  # [k_pos, q_pos] ?
        # We want shape [q_pos, k_pos], so use [q_pos=row, k_pos=col]:
        too_old = (idx.unsqueeze(1) - idx.unsqueeze(0)) >= sliding_window
        causal = causal.masked_fill(too_old, min_value)
    causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
    if attention_mask is None:
        return causal.expand(batch_size, 1, seq_len, seq_len).contiguous()
    if attention_mask.dim() == 4:
        return attention_mask.to(dtype)
    if attention_mask.dim() == 2:
        # 1=valid, 0=padding.  Overwrite padded key columns with ``min_value``
        # instead of ADDING a second ``min_value`` plane: a cell that is masked
        # by BOTH the causal/sliding-window mask AND padding would otherwise sum
        # two ``min_value`` planes, which overflows to ``-inf`` in low precision
        # (bf16: -3.39e38 + -3.39e38 -> -inf).  An ``-inf`` logit poisons the
        # softmax backward (0 * inf -> NaN), producing nan grad_norm at step 0.
        # Masking via where keeps masked cells at exactly ``min_value`` and
        # matches the sibling builders (build_packed_causal_padding_mask,
        # build_dsv4_cp_causal_padding_mask).
        causal = causal.expand(attention_mask.shape[0], 1, seq_len, seq_len)
        is_pad_key = (attention_mask == 0).to(device).view(attention_mask.shape[0], 1, 1, seq_len)
        return torch.where(is_pad_key, torch.full((), min_value, dtype=dtype, device=device), causal).to(dtype)
    raise ValueError(f"Unsupported attention_mask rank: {attention_mask.dim()}")


def build_packed_causal_padding_mask(
    seq_lens: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build a 4D additive block-causal mask from packed-sequence lengths."""
    if seq_lens.dim() == 1:
        seq_lens = seq_lens.unsqueeze(0)
    seq_lens = seq_lens.to(device=device, dtype=torch.long)
    seq_lens = torch.where(seq_lens > 0, seq_lens, torch.zeros((), device=device, dtype=torch.long))

    batch_size = seq_lens.shape[0]
    positions = torch.arange(seq_len, device=device, dtype=torch.long).expand(batch_size, -1)
    ends = seq_lens.cumsum(dim=-1)
    total = ends[:, -1:]
    doc_ids = torch.searchsorted(ends.contiguous(), positions.contiguous(), right=True) + 1
    doc_ids = torch.where(positions < total, doc_ids, torch.zeros_like(doc_ids))

    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    not_padding = doc_ids > 0
    idx = torch.arange(seq_len, device=device)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
    allowed = same_doc & causal.unsqueeze(0) & not_padding.unsqueeze(2) & not_padding.unsqueeze(1)
    if sliding_window is not None and sliding_window > 0:
        allowed = allowed & ((idx.unsqueeze(1) - idx.unsqueeze(0)) < sliding_window).unsqueeze(0)

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    return torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )


def eager_attention_with_sink(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention with per-head sink: appends an extra softmax column
    whose logit is ``module.sinks[h]`` and whose value-slot is zero.  Ported
    verbatim from HF PR 45616.
    """
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
    if hasattr(module, "sinks_param"):
        sinks = module.sinks_param(query)
    else:
        sinks = module.sinks
    sinks = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=torch.float32)[..., :-1]
    probs = F.dropout(probs, p=dropout, training=module.training).to(value_states.dtype)
    return torch.matmul(probs, value_states).transpose(1, 2).contiguous(), probs


def _build_indexer_topk_compressed_mask(
    attention_mask: torch.Tensor,
    indexer_topk: torch.Tensor,
    n_pooled: int,
) -> torch.Tensor:
    """Build the additive compressed-position mask for Indexer-selected pool IDs."""
    min_val = torch.finfo(attention_mask.dtype).min
    indexer_topk = indexer_topk.to(device=attention_mask.device)
    batch, seq_len = indexer_topk.shape[:2]
    valid = (indexer_topk >= 0) & (indexer_topk < n_pooled)  # [B, S, K]
    safe_idx = indexer_topk.clamp(min=0, max=max(n_pooled - 1, 0))
    indicator_counts = torch.zeros(
        (batch, seq_len, n_pooled),
        dtype=torch.int32,
        device=attention_mask.device,
    )
    indicator_counts.scatter_add_(-1, safe_idx, valid.to(torch.int32))
    indicator = indicator_counts > 0
    return torch.where(
        indicator,
        torch.zeros((), dtype=attention_mask.dtype, device=attention_mask.device),
        torch.full((), min_val, dtype=attention_mask.dtype, device=attention_mask.device),
    )  # [B, S, P]


class DeepseekV4FP32Parameter(nn.Module):
    """Callable holder for fp32 tensors that need their own FSDP unit."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(value.to(torch.float32))

    def forward(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        del reference
        return _full_tensor_if_dtensor(self.weight)


class DeepseekV4Indexer(nn.Module):
    """HF PR 45616 port.  Picks the top-k compressed positions per query when
    ``compress_ratio == 4``.  Owns its own pool at ``index_head_dim`` plus a
    query projection + weights_proj head-mixer.
    """

    def __init__(self, config: DeepseekV4Config, backend: BackendConfig | None = None):
        super().__init__()
        self.backend = backend or BackendConfig()
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.compress_ratio = 4
        # Indexer's pool is always at compress_ratio==4, which means overlap mode
        # (matching the released checkpoint's ``indexer.compressor.{ape,wkv,wgate}``
        # shapes of ``[ratio, 2*index_head_dim]`` / ``[2*index_head_dim, hidden_size]``).
        self.overlap = True
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        proj_dim = 2 * self.head_dim  # overlap mode
        self.wkv = nn.Linear(config.hidden_size, proj_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(config.hidden_size, proj_dim, bias=False, dtype=torch.float32)
        self.ape_param = DeepseekV4FP32Parameter(torch.zeros(self.compress_ratio, proj_dim, dtype=torch.float32))
        self.kv_norm = initialize_rms_norm_module(
            "torch_fp32", self.head_dim, eps=config.rms_norm_eps, dtype=model_dtype
        )
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    @property
    def ape(self) -> torch.Tensor:
        return self.ape_param()

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: DeepseekV4TrainCache,
        layer_idx: int,
        start_pos: int,
        position_ids: torch.Tensor | None = None,
        packed_seq_ids: torch.Tensor | None = None,
        cp_group=None,
    ) -> torch.LongTensor:
        input_dtype = hidden_states.dtype
        batch, seq_len, _ = hidden_states.shape
        cp_active = dsv4_cp_enabled(cp_group)
        hidden_states_fp32 = hidden_states.float()
        kv = self.wkv(hidden_states_fp32)
        gate = self.wgate(hidden_states_fp32)
        ready_kv, ready_gate, pool_base = cache.accumulate_windows(
            kv, gate, layer_idx, "indexer_state", self.compress_ratio, start_pos
        )
        window_seq_ids, window_positions = _compressed_window_metadata(
            seq_ids=packed_seq_ids,
            position_ids=position_ids,
            ready_len=ready_kv.shape[1],
            ratio=self.compress_ratio,
        )
        new_pooled = self.kv_norm(
            _pool_windows(
                ready_kv,
                ready_gate,
                self.ape_param(hidden_states_fp32),
                self.compress_ratio,
                self.head_dim,
                overlap=self.overlap,
                cp_group=cp_group,
                window_seq_ids=window_seq_ids,
            ).to(input_dtype)
        )
        if new_pooled.shape[1] > 0:
            if window_positions is not None:
                positions = (window_positions[:, : new_pooled.shape[1]] * self.compress_ratio).to(new_pooled.device)
            else:
                positions = _rope_pool_positions(
                    new_pooled.shape[1], pool_base, self.compress_ratio, new_pooled.device, new_pooled.shape[0]
                )
            cos, sin = rotary(new_pooled, positions)
            new_pooled = _apply_partial_rope(new_pooled.unsqueeze(1), cos, sin, self.rope_head_dim).squeeze(1)
        pooled_kv = cache.update_pool(new_pooled, layer_idx, "indexer_state")
        if cp_active:
            pooled_kv = dsv4_cp_all_gather(pooled_kv, dim=1, cp_group=cp_group)
            window_seq_ids = dsv4_cp_all_gather_metadata(window_seq_ids, dim=1, cp_group=cp_group)
            window_positions = dsv4_cp_all_gather_metadata(window_positions, dim=1, cp_group=cp_group)

        cos, sin = position_embeddings
        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim).transpose(1, 2)
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
        query_start = dsv4_cp_rank(cp_group) * seq_len if cp_active else 0
        query_total_len = dsv4_cp_size(cp_group) * seq_len if cp_active else None
        index_scores = dsv4_indexer_scores(
            q,
            pooled_kv,
            weights,
            compress_ratio=self.compress_ratio,
            softmax_scale=self.softmax_scale,
            backend=_dsv4_kernel_backend(self.backend),
            query_start=query_start,
            query_total_len=query_total_len,
        )
        q_positions = _query_positions(position_ids, batch=batch, seq_len=seq_len, device=hidden_states.device)
        packed_allowed = None
        if packed_seq_ids is not None and window_seq_ids is not None and window_positions is not None:
            q_seq = packed_seq_ids.to(device=hidden_states.device, dtype=torch.long)
            if q_seq.dim() == 1:
                q_seq = q_seq.unsqueeze(0)
            threshold = ((q_positions + 1) // self.compress_ratio).unsqueeze(-1)
            packed_allowed = (
                (q_seq.unsqueeze(-1) > 0)
                & (q_seq.unsqueeze(-1) == window_seq_ids.to(hidden_states.device).unsqueeze(1))
                & (window_positions.to(hidden_states.device).unsqueeze(1) < threshold)
            )
            index_scores = torch.where(
                packed_allowed,
                index_scores,
                torch.full((), float("-inf"), dtype=index_scores.dtype, device=index_scores.device),
            )
        else:
            valid_end = ((q_positions + 1) // self.compress_ratio).unsqueeze(-1)
            pooled_pos = torch.arange(pooled_kv.shape[1], device=hidden_states.device).view(1, 1, -1)
            index_scores = torch.where(
                pooled_pos < valid_end,
                index_scores,
                torch.full((), float("-inf"), dtype=index_scores.dtype, device=index_scores.device),
            )
        topk = min(self.index_topk, pooled_kv.shape[1])
        topk_indices = index_scores.topk(topk, dim=-1).indices
        if packed_allowed is not None:
            safe_topk = topk_indices.clamp(min=0, max=max(packed_allowed.shape[-1] - 1, 0))
            selected_allowed = torch.gather(packed_allowed, dim=-1, index=safe_topk)
            topk_indices = torch.where(selected_allowed, topk_indices, torch.full_like(topk_indices, -1))
        return topk_indices


class DeepseekV4Compressor(nn.Module):
    """HF PR 45616 port.  Long-range KV branch.  Pools ``compress_ratio`` tokens
    into one compressed KV; when ``ratio == 4`` the Indexer narrows the pool.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        head_dim: int,
        backend: BackendConfig | None = None,
    ):
        super().__init__()
        self.backend = backend or BackendConfig()
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        # Overlap mode (compress_ratio==4) doubles the feature dim of wkv / wgate /
        # ape — the released DSV4-Flash checkpoint was trained that way to give each
        # compressed token cross-window context. Non-overlap mode (compress_ratio==128)
        # keeps a flat head_dim. ``kv_norm`` always normalizes over ``head_dim`` because
        # the overlap_transform inside ``_pool_windows`` collapses 2*head_dim → head_dim
        # before the norm runs.
        self.overlap = compress_ratio == 4
        coff = 2 if self.overlap else 1
        proj_dim = coff * head_dim
        self.wkv = nn.Linear(config.hidden_size, proj_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(config.hidden_size, proj_dim, bias=False, dtype=torch.float32)
        self.ape_param = DeepseekV4FP32Parameter(torch.zeros(compress_ratio, proj_dim, dtype=torch.float32))
        self.kv_norm = initialize_rms_norm_module("torch_fp32", head_dim, eps=config.rms_norm_eps, dtype=model_dtype)
        self.indexer: DeepseekV4Indexer | None = (
            DeepseekV4Indexer(config, backend=self.backend) if compress_ratio == 4 else None
        )
        self._hca_param_sync_group = None

    @property
    def ape(self) -> torch.Tensor:
        return self.ape_param()

    def _set_hca_param_sync_group(self, process_group) -> None:
        self._hca_param_sync_group = process_group

    def _compute_fsdp_group_has_complete_hca_window(
        self,
        local_has_complete_hca_window: bool,
        device: torch.device,
    ) -> bool:
        process_group = self._hca_param_sync_group
        if process_group is None or not dist.is_available() or not dist.is_initialized():
            return local_has_complete_hca_window
        if dist.get_world_size(group=process_group) == 1:
            return local_has_complete_hca_window

        # A single int max-reduce tells short ranks whether any peer in the same
        # HCA parameter sync group has a complete HCA window. It is a small
        # synchronization point, but it avoids using a broader or world group.
        flag = torch.tensor(int(local_has_complete_hca_window), device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=process_group)
        return bool(flag.item())

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor | None,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: DeepseekV4TrainCache,
        layer_idx: int,
        start_pos: int,
        enable_hca_fsdp_graph_alignment: bool = False,
        position_ids: torch.Tensor | None = None,
        packed_seq_ids: torch.Tensor | None = None,
        cp_group=None,
    ) -> tuple[torch.Tensor, torch.LongTensor | None, torch.Tensor | None, torch.Tensor | None]:
        input_dtype = hidden_states.dtype
        batch, seq_len, _ = hidden_states.shape
        cp_active = dsv4_cp_enabled(cp_group)
        hidden_states_fp32 = hidden_states.float()
        kv = self.wkv(hidden_states_fp32)
        gate = self.wgate(hidden_states_fp32)
        ready_kv, ready_gate, pool_base = cache.accumulate_windows(
            kv, gate, layer_idx, "compressor_state", self.compress_ratio, start_pos
        )
        window_seq_ids, window_positions = _compressed_window_metadata(
            seq_ids=packed_seq_ids,
            position_ids=position_ids,
            ready_len=ready_kv.shape[1],
            ratio=self.compress_ratio,
        )
        local_has_complete_hca_window = ready_kv.shape[1] > 0
        fsdp_group_has_complete_hca_window = (
            self._compute_fsdp_group_has_complete_hca_window(local_has_complete_hca_window, kv.device)
            if enable_hca_fsdp_graph_alignment
            else local_has_complete_hca_window
        )
        needs_masked_synthetic_hca_window = (
            enable_hca_fsdp_graph_alignment
            and self.indexer is None
            and fsdp_group_has_complete_hca_window
            and not local_has_complete_hca_window
            and 0 < kv.shape[1] < self.compress_ratio
        )
        if needs_masked_synthetic_hca_window:
            # A mixed short/long HCA parameter sync group needs every rank to
            # create the same compressor autograd edges. All-short groups keep
            # the original no-HCA path so optimizer semantics stay at
            # grad=None. Zero-length local HCA inputs are outside this narrow
            # training fix.
            pad_len = self.compress_ratio - kv.shape[1]
            pad_shape = (batch, pad_len, kv.shape[-1])
            ready_kv = torch.cat([kv, kv.new_zeros(pad_shape)], dim=1)
            ready_gate = torch.cat([gate, gate.new_zeros(pad_shape)], dim=1)
            pool_base = max(0, start_pos)
            window_seq_ids = None
            window_positions = None
        new_pooled = self.kv_norm(
            _pool_windows(
                ready_kv,
                ready_gate,
                self.ape_param(hidden_states_fp32),
                self.compress_ratio,
                self.head_dim,
                overlap=self.overlap,
                cp_group=cp_group,
                window_seq_ids=window_seq_ids,
            ).to(input_dtype)
        )
        if window_positions is not None:
            positions = (window_positions[:, : new_pooled.shape[1]] * self.compress_ratio).to(new_pooled.device)
        else:
            positions = _rope_pool_positions(
                new_pooled.shape[1], pool_base, self.compress_ratio, new_pooled.device, batch
            )
        cos, sin = rotary(new_pooled, positions)
        new_pooled = _apply_partial_rope(new_pooled.unsqueeze(1), cos, sin, self.rope_head_dim).squeeze(1)
        pooled = cache.update_pool(new_pooled, layer_idx, "compressor_state").unsqueeze(1)
        if cp_active:
            pooled = dsv4_cp_all_gather(pooled, dim=2, cp_group=cp_group)
            window_seq_ids = dsv4_cp_all_gather_metadata(window_seq_ids, dim=1, cp_group=cp_group)
            window_positions = dsv4_cp_all_gather_metadata(window_positions, dim=1, cp_group=cp_group)

        # Indexer narrows the attended compressed positions per query.  The
        # caller (DSV4Attention) is responsible for turning ``indexer_topk``
        # into an additive attention mask; we do NOT pre-gather here.  The
        # earlier per-query ``torch.gather`` produced an
        # ``[B, 1, S*topk, D]`` tensor that, when concatenated to ``full_kv``
        # and run through dense attention with ``F.pad(value=0.0)``, let
        # every query attend to every other query's gathered slice — a
        # silent non-causal leak (verified empirically: layer 2 attention
        # output cosine-vs-reference jumps from 0.81 to 0.99+ once we move
        # to mask-driven sparse semantics).
        #
        # ``indexer_topk`` follows the reference contract from
        # ``dsv4flash/inference/model.py:472-475``: shape ``[B, S, K]`` with
        # entries that are either valid pool indices in ``[0, P_total)``
        # or ``-1`` for "do not attend" (masked by causality).
        indexer_topk: torch.LongTensor | None = None
        if self.indexer is not None:
            raw_topk = self.indexer(
                hidden_states,
                q_residual,
                rotary,
                position_embeddings,
                cache,
                layer_idx,
                start_pos,
                position_ids=position_ids,
                packed_seq_ids=packed_seq_ids,
                cp_group=cp_group,
            )
            q_positions = _query_positions(position_ids, batch=batch, seq_len=seq_len, device=raw_topk.device)
            if packed_seq_ids is not None and window_seq_ids is not None and window_positions is not None:
                q_seq = packed_seq_ids.to(device=raw_topk.device, dtype=torch.long)
                if q_seq.dim() == 1:
                    q_seq = q_seq.unsqueeze(0)
                safe_topk = raw_topk.clamp(min=0, max=max(window_seq_ids.shape[-1] - 1, 0))
                selected_seq = torch.gather(
                    window_seq_ids.to(raw_topk.device).unsqueeze(1).expand(-1, seq_len, -1), -1, safe_topk
                )
                selected_pos = torch.gather(
                    window_positions.to(raw_topk.device).unsqueeze(1).expand(-1, seq_len, -1),
                    -1,
                    safe_topk,
                )
                threshold = ((q_positions + 1) // self.compress_ratio).unsqueeze(-1)
                valid = (raw_topk >= 0) & (q_seq.unsqueeze(-1) > 0) & (selected_seq == q_seq.unsqueeze(-1))
                valid = valid & (selected_pos < threshold)
                indexer_topk = torch.where(valid, raw_topk, torch.full_like(raw_topk, -1))
            else:
                threshold = ((q_positions + 1) // self.compress_ratio).unsqueeze(-1)
                causal_invalid = raw_topk >= threshold
                indexer_topk = torch.where(causal_invalid, torch.full_like(raw_topk, -1), raw_topk)
        return pooled, indexer_topk, window_seq_ids, window_positions


# ---------------------------------------------------------------------------
# HC (Hyper-Connections) — ported verbatim from HuggingFace transformers
# PR 45616 (Arthur Zucker, "Add DeepSeek V4").  Source-of-truth reference:
# ``transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py``
# classes ``DeepseekV4HyperConnection`` (lines 613-670) and
# ``DeepseekV4HyperHead`` (lines 673-690).
#
# The previous pure-torch port (mean-pool / softmax-comb) had three silent
# divergences vs HF: (a) ``comb`` used row-softmax where HF uses ``sigmoid``,
# (b) ``post`` had a ``2 *`` prefactor + missing ``+eps``, (c) the mixer was
# wrapped in ``torch.no_grad()``.  Rather than patch those line-by-line,
# swap wholesale to the HF classes so future HF updates flow through cleanly
# via the adapter's state-dict rename rules.
# ---------------------------------------------------------------------------


class DeepseekV4HyperConnection(nn.Module):
    """Per-site HyperConnection mixer (attention or FFN).  Ported from
    ``transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py``
    class ``DeepseekV4HyperConnection``.

    Owns ``fn`` (packed linear), ``base`` (bias), and ``scale`` (scalar
    per-head gains).  ``compute_weights`` produces three mixer tensors:

      - ``pre``   [B, S, H]       : sigmoid-gated collapse weights
      - ``post``  [B, S, H]       : sigmoid-gated expand weights
      - ``comb``  [B, S, H, H]    : doubly-stochastic combination matrix
                                    from Sinkhorn-normalising sigmoid gates

    All math runs in fp32 regardless of the outer cast policy; parameters
    cast themselves via ``.float()`` on each forward.  HF lists these params
    in ``_keep_in_fp32_modules_strict`` — the KAutomodel adapter does the
    same via submodule-name matching.
    """

    def __init__(
        self,
        hc_mult: int,
        hidden_size: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        rms_norm_eps: float,
        sinkhorn_backend: str = "torch",
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = rms_norm_eps
        self.sinkhorn_backend = sinkhorn_backend
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize HyperConnection parameters using the DeepSeek-V4 reference scheme.

        Args:
            init_std: Standard deviation for the ``fn`` weight initialization.
        """
        nn.init.normal_(self.fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def compute_weights(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_streams.flatten(start_dim=2).float()  # [B, S, H*D]
        # HC mixer params are kept in fp32 for Sinkhorn stability — cast defensively.
        mix = torch.nn.functional.linear(_rms_norm_last_dim(flat, self.norm_eps), self.fn.float())  # [B, S, (2+H)*H]
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)
        hc = self.hc_mult

        # ``pre`` and ``post`` have DIFFERENT formulas in the released DSV4-Flash
        # (see ``dsv4flash/inference/kernel.py:hc_split_sinkhorn_kernel`` 391-394):
        #   pre  = sigmoid(...) + eps     range (eps, 1+eps]
        #   post = 2 * sigmoid(...)       range (0, 2)  — NO +eps, AND a 2x prefactor
        # HF transformers PR 45616 / 45643 treats post identically to pre (sigmoid
        # + eps), which makes ``post`` half the magnitude the released weights
        # were trained against — verified empirically on the parity test
        # (auto post std = 0.5x ref post std before this fix).
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc].float()) + self.hc_eps
        post = 2.0 * torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc].float())

        # ``comb`` uses softmax(dim=-1) on raw logits + eps, then sinkhorn.  HF
        # uses sigmoid + eps + sinkhorn — also a divergence from the reference
        # kernel.  Reference (kernel.py:395-413):
        #   1. comb_logit = mix * scale + base
        #   2. row_softmax(dim=-1) + eps   (numerically stable, NOT sigmoid)
        #   3. col-norm / sum(dim=-2)
        #   4. for sinkhorn_iters - 1: row-norm / sum(dim=-1) ; col-norm / sum(dim=-2)
        comb_logit = (
            mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[2 * hc :].view(hc, hc).float()
        )
        comb = dsv4_sinkhorn_normalize(
            comb_logit,
            backend=self.sinkhorn_backend,
            repeat=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )
        return pre, post, comb

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compute_weights(hidden_streams)


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse before the shared RMSNorm + ``lm_head``.
    Ported from ``modular_deepseek_v4.py`` class ``DeepseekV4HyperHead``.

    Sigmoid-weighted sum over the ``hc_mult`` streams (no Sinkhorn).  Used
    once at the end of ``DeepseekV4Model.forward`` to go from
    ``[B, S, H, D]`` back to ``[B, S, D]``.
    """

    def __init__(self, hc_mult: int, hidden_size: int, hc_eps: float, rms_norm_eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.norm_eps = rms_norm_eps
        self.eps = hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden_size, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize HyperHead parameters using the DeepSeek-V4 reference scheme.

        Args:
            init_std: Standard deviation for the ``hc_fn`` weight initialization.
        """
        nn.init.normal_(self.hc_fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2).float()
        mixes = torch.nn.functional.linear(_rms_norm_last_dim(flat, self.norm_eps), self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


# ---------------------------------------------------------------------------
# DeepseekV4Attention — port of HF PR 45616's DeepseekV4Attention with:
#   - no inheritance from DeepseekV3Attention (we don't want the HF
#     PreTrainedModel scaffolding);
#   - ``position_embeddings`` passed in from DeepseekV4Model as a ``(cos, sin)``
#     tuple produced by a matching ``DeepseekV4RotaryEmbedding`` (plus a
#     separate rotary / position_embeddings pair for the compressor path);
#   - ``past_key_values`` always ``None`` on the training path; the compressor
#     / indexer use a per-forward ``DeepseekV4TrainCache`` shim that behaves
#     the same as HF's ``DeepseekV4Cache`` within a single call;
#   - explicit backend dispatch for SDPA, the dense torch reference, and the
#     optional DeepSeek V4 sparse-attention kernels.
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """Sliding-window attention + Compressor + Indexer + attention sink.

    Single-head KV (``num_key_value_heads=1``), grouped low-rank output via
    :class:`DeepseekV4GroupedLinear`.  ``compress_ratio == 0`` layers skip
    the compressor / indexer and run pure SWA.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int, backend: BackendConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.layer_idx = layer_idx
        self.compress_ratio = int(config.compress_ratios[layer_idx]) if config.compress_ratios else 0
        self.num_heads = config.num_attention_heads
        # Single KV head broadcast to all attention heads (``num_key_value_groups == num_heads``).
        self.num_key_value_groups = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.sliding_window = int(getattr(config, "sliding_window", 128) or 128)
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = initialize_rms_norm_module(
            "torch_fp32", config.q_lora_rank, eps=config.rms_norm_eps, dtype=model_dtype
        )
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = initialize_rms_norm_module(
            "torch_fp32", self.head_dim, eps=config.rms_norm_eps, dtype=model_dtype
        )
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks_param = DeepseekV4FP32Parameter(torch.zeros(self.num_heads, dtype=torch.float32))

        self.compressor = (
            DeepseekV4Compressor(config, self.compress_ratio, self.head_dim, backend=self.backend)
            if self.compress_ratio
            else None
        )

    @property
    def sinks(self) -> torch.Tensor:
        return self.sinks_param()

    def setup_cp_attention(self, cp_mesh) -> None:
        """Model-owned context-parallel hook, called by ``moe.parallelizer.apply_cp``.

        DSV4 runs Miles-style CP (contiguous query shard + all-gathered K/V), so there
        is no TE ``DotProductAttention`` to configure -- we just record the CP process
        group that ``forward`` uses to all-gather K/V across CP ranks.
        """
        self._cp_group = cp_mesh.get_group()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_embeddings_compress: tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_compress: nn.Module | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cp_group = kwargs.get("_dsv4_cp_group") or getattr(self, "_cp_group", None)
        cp_active = dsv4_cp_enabled(cp_group)
        packed_seq_ids = kwargs.get("packed_seq_ids")
        vision_token_types = kwargs.get("vision_token_types")
        if packed_seq_ids is not None and packed_seq_ids.dim() == 1:
            packed_seq_ids = packed_seq_ids.unsqueeze(0)
        batch, seq_len = hidden_states.shape[:2]
        effective_start_pos = start_pos
        if cp_active:
            effective_start_pos = start_pos + dsv4_cp_rank(cp_group) * seq_len
        query_positions = _query_positions(
            position_ids,
            batch=batch,
            seq_len=seq_len,
            device=hidden_states.device,
            cp_group=cp_group,
        )
        if packed_seq_ids is not None:
            linear_start = start_pos + (dsv4_cp_rank(cp_group) * seq_len if cp_active else 0)
            sparse_query_positions = (
                torch.arange(linear_start, linear_start + seq_len, device=hidden_states.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch, -1)
            )
        else:
            sparse_query_positions = query_positions
        # IMPORTANT: for compress_ratio>0 layers the released DSV4-Flash uses
        # the compress-rope (theta=160000 + YaRN) for the MAIN attention Q/KV
        # too, NOT just for the compressor sub-module.  Reference at
        # ``dsv4flash/inference/model.py:476-501`` builds ``self.freqs_cis``
        # with ``compress_rope_theta`` whenever ``compress_ratio != 0``.  The
        # caller passes both ``position_embeddings`` (theta=10000, no YaRN)
        # and ``position_embeddings_compress`` (theta=160000, YaRN); we pick
        # the right one here based on compress_ratio.
        if self.compress_ratio and position_embeddings_compress is not None:
            cos, sin = position_embeddings_compress
        else:
            cos, sin = position_embeddings

        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_norm(self.wkv(hidden_states)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)

        # Per-head, non-learnable rsqrt on Q before RoPE (matches reference
        # ``dsv4flash/inference/model.py:498``; missing from HF PR 45616).
        q = _rms_norm_last_dim(q, self.config.rms_norm_eps)

        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim)
        kv = _apply_partial_rope(kv, cos, sin, self.rope_head_dim)

        full_kv = dsv4_cp_all_gather(kv, dim=2, cp_group=cp_group) if cp_active else kv
        vanilla_key_len = full_kv.shape[2]
        n_pooled = 0
        indexer_topk: torch.LongTensor | None = None

        if self.compressor is not None:
            assert rotary_compress is not None and position_embeddings_compress is not None, (
                "DeepseekV4Attention: compressor enabled but no rotary_compress / "
                "position_embeddings_compress supplied by the Block/Model."
            )
            cache = DeepseekV4TrainCache()
            pooled, indexer_topk, pooled_seq_ids, pooled_positions = self.compressor(
                hidden_states,
                q_residual=q_residual,
                rotary=rotary_compress,
                position_embeddings=position_embeddings_compress,
                cache=cache,
                layer_idx=self.layer_idx,
                start_pos=effective_start_pos,
                # Only DeepSeek-V4 HCA (ratio 128) needs this FSDP graph alignment.
                # Ratio 4 uses the Indexer/SCA path and is outside this fix.
                # Direct attention calls without an explicit mask keep the
                # original behavior since synthetic windows must be maskable.
                # The normal DeepSeek-V4 model path builds this mask upstream.
                enable_hca_fsdp_graph_alignment=(
                    self.training and attention_mask is not None and self.compress_ratio == 128
                ),
                position_ids=position_ids,
                packed_seq_ids=packed_seq_ids,
                cp_group=cp_group,
            )
            n_pooled = pooled.shape[2]
            full_kv = torch.cat([full_kv, pooled], dim=2)

            # Extend the additive 4D attention mask with a per-query
            # compressed-position mask so dense attention reproduces the
            # reference's ``sparse_attn`` semantics (per-query topk_idxs +
            # causality on the compressed pool).
            #
            # * compress_ratio == 4 (Indexer present): mask=0 only at the
            #   pool positions selected by ``indexer_topk`` for that query,
            #   -inf elsewhere.  ``-1`` entries in ``indexer_topk`` are
            #   already causally-masked by Compressor.
            # * compress_ratio > 4 (no Indexer, e.g. 128): every query q can
            #   attend to compressed position p iff ``p < (q+1) // ratio``
            #   (matches ``get_compress_topk_idxs`` in
            #   ``dsv4flash/inference/model.py:289-296``).
            if attention_mask is not None and n_pooled > 0:
                min_val = torch.finfo(attention_mask.dtype).min
                if indexer_topk is not None:
                    # OR-aggregate via scatter_add_: clamping -1 to 0 collides
                    # with valid index 0, and scatter_'s duplicate-index order
                    # is unspecified, so use count-then-threshold instead.
                    compressed_mask = _build_indexer_topk_compressed_mask(attention_mask, indexer_topk, n_pooled)
                    if packed_seq_ids is not None and pooled_seq_ids is not None and pooled_positions is not None:
                        q_seq = packed_seq_ids.to(device=full_kv.device, dtype=torch.long)
                        threshold = ((query_positions + 1) // self.compress_ratio).unsqueeze(-1)
                        allowed = (
                            (q_seq.unsqueeze(-1) > 0)
                            & (q_seq.unsqueeze(-1) == pooled_seq_ids.to(full_kv.device).unsqueeze(1))
                            & (pooled_positions.to(full_kv.device).unsqueeze(1) < threshold)
                        )
                        compressed_mask = torch.where(
                            allowed,
                            compressed_mask,
                            torch.full((), min_val, dtype=attention_mask.dtype, device=full_kv.device),
                        )
                else:
                    threshold = (query_positions + 1) // self.compress_ratio
                    if packed_seq_ids is not None and pooled_seq_ids is not None and pooled_positions is not None:
                        q_seq = packed_seq_ids.to(device=full_kv.device, dtype=torch.long)
                        allowed = (
                            (q_seq.unsqueeze(-1) > 0)
                            & (q_seq.unsqueeze(-1) == pooled_seq_ids.to(full_kv.device).unsqueeze(1))
                            & (pooled_positions.to(full_kv.device).unsqueeze(1) < threshold.unsqueeze(-1))
                        )
                    else:
                        p_pos = torch.arange(n_pooled, device=full_kv.device)
                        allowed = p_pos.view(1, 1, n_pooled) < threshold.unsqueeze(-1)  # [B, S, P]
                    compressed_mask = torch.where(
                        allowed,
                        torch.zeros((), dtype=attention_mask.dtype, device=full_kv.device),
                        torch.full((), min_val, dtype=attention_mask.dtype, device=full_kv.device),
                    )
                compressed_mask = compressed_mask.unsqueeze(1)  # [B, 1, S, P]
                attention_mask = torch.cat([attention_mask, compressed_mask], dim=-1)

        # If a caller supplied a 4D mask shorter than full_kv but no compressor
        # ran (shouldn't happen, but kept for defense), fall back to neutral pad.
        if attention_mask is not None and full_kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(attention_mask, (0, full_kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attn_backend = _dsv4_kernel_backend(self.backend)
        if attn_backend == "tilelang":
            topk_idxs = build_dsv4_sparse_topk_indices(
                batch_size=batch,
                seq_len=seq_len,
                key_len=full_kv.shape[2],
                window_size=self.sliding_window,
                device=full_kv.device,
                attention_mask=attention_mask,
                compress_ratio=self.compress_ratio,
                compressed_topk=indexer_topk,
                n_pooled=n_pooled,
                vanilla_key_len=vanilla_key_len,
                q_positions=sparse_query_positions,
                vision_token_types=vision_token_types,
                max_image_tokens=int(getattr(self.config, "vision_max_n_token", 0) or 0),
            )
            attn_output = dsv4_sparse_attention(
                q.transpose(1, 2).contiguous(),
                full_kv.squeeze(1).contiguous(),
                self.sinks_param(q),
                topk_idxs,
                self.scaling,
                backend=attn_backend,
            )
            attn_weights = None
        else:
            attn_output, attn_weights = eager_attention_with_sink(
                self,
                q,
                full_kv,
                full_kv,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )
            # eager_attention_with_sink returns [B, S, H, D] (already transposed).

        # Inverse RoPE on the attention output (same (cos, -sin) conjugate pattern
        # HF uses).  Reference: modular_deepseek_v4.py:607.
        attn_output = _apply_partial_rope(attn_output.transpose(1, 2), cos, -sin, self.rope_head_dim).transpose(1, 2)

        grouped = attn_output.reshape(batch, seq_len, -1).view(batch, seq_len, self.config.o_groups, -1)
        return self.wo_b(self.wo_a(grouped).flatten(2)), attn_weights

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.wq_a, self.wq_b, self.wkv, self.wo_b, self.wo_a):
            if hasattr(linear, "weight"):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.kv_norm):
            norm.reset_parameters()
        nn.init.zeros_(self.sinks_param.weight)
        if self.compressor is not None:
            for mod in self.compressor.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.trunc_normal_(mod.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.compressor.ape_param.weight)
            if self.compressor.indexer is not None:
                nn.init.zeros_(self.compressor.indexer.ape_param.weight)
