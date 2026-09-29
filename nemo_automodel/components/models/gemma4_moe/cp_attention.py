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

"""Gemma4-specific context-parallel attention helpers."""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, replace
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from nemo_automodel.components.attention.ffpa_attention import (
    _FFPA_HEAD_DIM,
    _ffpa_dense_bwd,
    _ffpa_dense_fwd,
    _ffpa_varlen_bwd,
    _ffpa_varlen_fwd,
    _ffpa_varlen_ready,
)
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_accumulated_grad_guard as _patch_fsdp_accumulated_grad_guard,
)

logger = logging.getLogger(__name__)
_GEMMA4_CP_FLEX_RING_OK_LOGGED = False


_patch_fsdp_accumulated_grad_guard()


@dataclass(frozen=True)
class CPRingAttentionContext:
    """Inputs for Gemma4 manual ring CP attention (built by the run_cp_manual_attention seam)."""

    module: torch.nn.Module
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    cp_mesh: Any
    cp_group: Any
    cp_size: int
    cp_rank: int
    seq_local: int
    seq_full: int
    seq_global_start: int
    attn_mask: Any
    dropout_p: float
    is_causal: bool
    scale: Any
    enable_gqa: bool
    kwargs: dict[str, Any]
    metadata: dict[str, torch.Tensor | None]
    metadata_seq_dims: dict[str, int]
    use_vision_bidirectional: bool = False


def gemma4_vision_group_ids(mm_token_type_ids: torch.Tensor) -> torch.Tensor:
    """Return per-image-block ids for Gemma4 vision tokens, or -1 for text/padding."""
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    return torch.where(is_vision, group_ids, torch.full_like(group_ids, -1))


def _config_uses_vision_bidirectional(attention_module: torch.nn.Module) -> bool:
    """Whether the rank-uniform module config enables bidirectional vision attention."""
    return getattr(getattr(attention_module, "config", None), "use_bidirectional_attention", None) == "vision"


# create_block_mask (~7.6ms/call) depends only on attention type + sequence geometry +
# position metadata, so all same-type layers in a step build an identical mask. Cache on the
# position scalars (not tensor storage -- the CUDA allocator recycles addresses, so a
# data_ptr key could alias stale content) and clear it when the batch's metadata object
# changes, holding that object as the generation token so its data_ptr can't be recycled live.
_BLOCK_MASK_CACHE: dict = {}
_BLOCK_MASK_GEN: list = [None, None]  # [data_ptr, held-tensor]


def _block_mask_set_generation(gen_tensor) -> None:
    """Reset the per-step block-mask cache when a new batch (new metadata) arrives."""
    ptr = None if gen_tensor is None else gen_tensor.data_ptr()
    if ptr != _BLOCK_MASK_GEN[0]:
        _BLOCK_MASK_CACHE.clear()
        _BLOCK_MASK_GEN[0] = ptr
        _BLOCK_MASK_GEN[1] = gen_tensor  # pin the storage for the duration of the step


def _cached_block_mask(key, build):
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = build()
    if len(_BLOCK_MASK_CACHE) >= 256:  # bound the gen=None (metadata-free) varying-seqlen case
        _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
    _BLOCK_MASK_CACHE[key] = mask
    return mask


def _compiled_flex_attention(attention_module: torch.nn.Module):
    compiled = getattr(attention_module, "_gemma4_cp_compiled_flex_attn", None)
    if compiled is None:
        from torch.nn.attention.flex_attention import flex_attention

        compiled = torch.compile(flex_attention, dynamic=True)
        attention_module._gemma4_cp_compiled_flex_attn = compiled
    return compiled


@contextlib.contextmanager
def _duck_shape_disabled():
    """Locally disable flex duck-shape specialization for the wrapped flex call.

    Otherwise the compiled flex kernel guards on incidental dim-equalities and recompiles on
    every new sequence length. Dynamo reads ``use_duck_shape`` at (re)trace time inside the
    flex call, so scoping it to the call window avoids mutating the process-global fx config.
    """
    from torch.fx.experimental import _config as _fx_config

    prev = _fx_config.use_duck_shape
    _fx_config.use_duck_shape = False
    try:
        yield
    finally:
        _fx_config.use_duck_shape = prev


def _base_gemma4_cp_mask(attention_module: torch.nn.Module, ctx: Any, q_idx, kv_idx, kv_global_start: int = 0):
    q_global_idx = q_idx + ctx.seq_global_start
    kv_global_idx = kv_idx + kv_global_start
    if not ctx.is_causal:
        allowed = torch.ones_like(q_global_idx >= kv_global_idx)
    else:
        allowed = kv_global_idx <= q_global_idx

    sliding_window = getattr(attention_module, "sliding_window", None)
    if sliding_window is not None:
        allowed = allowed & ((q_global_idx - kv_global_idx) < sliding_window)
    return allowed


def _metadata_like(metadata: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    return {name: torch.empty_like(value) if value is not None else None for name, value in metadata.items()}


def _detach_metadata(metadata: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    return {name: value.detach().contiguous() if value is not None else None for name, value in metadata.items()}


def _ring_exchange(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    cp_group: Any,
    cp_rank: int,
    cp_size: int,
) -> None:
    """One ring rotation step: send to ``cp_rank+1``, receive from ``cp_rank-1`` (p2p)."""
    _direct_exchange(
        tensors,
        cp_group=cp_group,
        cp_rank=cp_rank,
        send_cp_rank=(cp_rank + 1) % cp_size,
        recv_cp_rank=(cp_rank - 1) % cp_size,
    )


def _direct_exchange(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    cp_group: Any,
    cp_rank: int,
    send_cp_rank: int,
    recv_cp_rank: int,
) -> None:
    if not tensors:
        return

    ranks = torch.distributed.get_process_group_ranks(cp_group)
    send_dst = ranks[send_cp_rank]
    recv_src = ranks[recv_cp_rank]

    send_ops = [
        torch.distributed.P2POp(torch.distributed.isend, send_tensor.contiguous(), send_dst, cp_group)
        for send_tensor, _ in tensors
    ]
    recv_ops = [
        torch.distributed.P2POp(torch.distributed.irecv, recv_tensor, recv_src, cp_group) for _, recv_tensor in tensors
    ]
    ops = send_ops + recv_ops if cp_rank % 2 == 0 else recv_ops + send_ops
    for req in torch.distributed.batch_isend_irecv(ops):
        req.wait()


def _route_kv_grads_to_owners(
    grad_key_by_owner: dict[int, torch.Tensor],
    grad_value_by_owner: dict[int, torch.Tensor],
    *,
    cp_group: Any,
    cp_rank: int,
    cp_size: int,
    n_prior: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum each KV owner's dK/dV from every rank whose queries attended it.

    In the ring forward, rank ``r``'s queries attend the chunks owned by ranks
    ``r, r-1, ..., r-n_prior``, so backward produces a dK/dV contribution for each of
    those owners on rank ``r``. This sends each owner's contribution back (p2p) and sums
    them, returning the local ``(grad_key, grad_value)``. Shared by the Flex, FFPA-varlen
    and local-kernel ring backward passes. ``n_prior`` MUST match the forward's
    :func:`_ring_num_prior_chunks`; defaults to the full rotation (``cp_size - 1``).
    """
    if n_prior is None:
        n_prior = cp_size - 1
    grad_key = grad_key_by_owner[cp_rank].contiguous()
    grad_value = grad_value_by_owner[cp_rank].contiguous()
    for distance in range(1, n_prior + 1):
        send_owner = (cp_rank - distance) % cp_size
        recv_query_rank = (cp_rank + distance) % cp_size
        recv_grad_key = torch.empty_like(grad_key)
        recv_grad_value = torch.empty_like(grad_value)
        _direct_exchange(
            [
                (grad_key_by_owner[send_owner], recv_grad_key),
                (grad_value_by_owner[send_owner], recv_grad_value),
            ],
            cp_group=cp_group,
            cp_rank=cp_rank,
            send_cp_rank=send_owner,
            recv_cp_rank=recv_query_rank,
        )
        grad_key = grad_key + recv_grad_key
        grad_value = grad_value + recv_grad_value
    return grad_key, grad_value


def _merge_flex_chunk(
    out_acc: torch.Tensor | None,
    lse_acc: torch.Tensor | None,
    out_step: torch.Tensor,
    lse_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out_acc is None or lse_acc is None:
        return out_step, lse_step
    lse_next = torch.logaddexp(lse_acc, lse_step)
    old_scale = torch.exp(lse_acc - lse_next).unsqueeze(-1)
    new_scale = torch.exp(lse_step - lse_next).unsqueeze(-1)
    return out_acc * old_scale + out_step * new_scale, lse_next


def _run_gemma4_flex_chunk(
    attention_module: torch.nn.Module,
    ctx: Any,
    *,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    metadata_chunk: dict[str, torch.Tensor | None],
    kv_global_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    query = ctx.query
    orig_head_dim = query.shape[-1]
    padded_head_dim = 1 << (orig_head_dim - 1).bit_length()
    use_small_flex_blocks = padded_head_dim > 256
    flex_block_size = (32, 32) if use_small_flex_blocks else 128

    packed_seq_ids_q = ctx.metadata.get("_packed_seq_ids")
    packed_seq_ids_kv = metadata_chunk.get("_packed_seq_ids")
    padding_mask_q = ctx.metadata.get("padding_mask")
    padding_mask_kv = metadata_chunk.get("padding_mask")
    vision_group_ids_q = ctx.metadata.get("_gemma4_vision_group_ids")
    vision_group_ids_kv = metadata_chunk.get("_gemma4_vision_group_ids")
    if vision_group_ids_q is None and ctx.metadata.get("mm_token_type_ids") is not None:
        vision_group_ids_q = gemma4_vision_group_ids(ctx.metadata["mm_token_type_ids"])
    if vision_group_ids_kv is None and metadata_chunk.get("mm_token_type_ids") is not None:
        vision_group_ids_kv = gemma4_vision_group_ids(metadata_chunk["mm_token_type_ids"])

    sliding_window = getattr(attention_module, "sliding_window", None)
    config_uses_vision_bidir = _config_uses_vision_bidirectional(attention_module)
    use_vision_bidirectional = (
        sliding_window is not None
        and config_uses_vision_bidir
        and vision_group_ids_q is not None
        and vision_group_ids_kv is not None
    )

    empty_query_rows = None
    if packed_seq_ids_q is not None:
        empty_query_rows = packed_seq_ids_q <= 0
    if padding_mask_q is not None:
        padding_query_rows = padding_mask_q
        empty_query_rows = padding_query_rows if empty_query_rows is None else empty_query_rows | padding_query_rows

    try:
        from torch.nn.attention.flex_attention import create_block_mask

        if (
            use_vision_bidirectional
            or packed_seq_ids_q is not None
            or packed_seq_ids_kv is not None
            or padding_mask_q is not None
            or padding_mask_kv is not None
        ):

            def cp_mask(batch_idx, head_idx, q_idx, kv_idx):
                allowed = _base_gemma4_cp_mask(attention_module, ctx, q_idx, kv_idx, kv_global_start)
                if use_vision_bidirectional:
                    q_group = vision_group_ids_q[batch_idx, q_idx]
                    kv_group = vision_group_ids_kv[batch_idx, kv_idx]
                    same_vision_group = (q_group == kv_group) & (q_group >= 0)
                    allowed = allowed | same_vision_group
                if packed_seq_ids_q is not None and packed_seq_ids_kv is not None:
                    q_pack_id = packed_seq_ids_q[batch_idx, q_idx]
                    kv_pack_id = packed_seq_ids_kv[batch_idx, kv_idx]
                    allowed = allowed & (q_pack_id == kv_pack_id) & (q_pack_id > 0)
                    allowed = torch.where(q_pack_id <= 0, kv_idx == 0, allowed)
                if padding_mask_kv is not None:
                    kv_is_padding = padding_mask_kv[batch_idx, kv_idx]
                    allowed = allowed & ~kv_is_padding
                if padding_mask_q is not None:
                    q_is_padding = padding_mask_q[batch_idx, q_idx]
                    allowed = torch.where(q_is_padding, kv_idx == 0, allowed)
                return allowed

            block_mask_batch = query.shape[0]
        else:

            def cp_mask(batch_idx, head_idx, q_idx, kv_idx):
                return _base_gemma4_cp_mask(attention_module, ctx, q_idx, kv_idx, kv_global_start)

            block_mask_batch = None

        def _build_block_mask():
            return create_block_mask(
                cp_mask,
                B=block_mask_batch,
                H=None,
                Q_LEN=ctx.seq_local,
                KV_LEN=key_chunk.shape[2],
                device=query.device,
                BLOCK_SIZE=flex_block_size,
            )

        # Reset the cache when the batch changes (new metadata object => new step),
        # then key purely on the position scalars. Within a step the metadata content
        # is a function of position, so (layer-type, query/chunk geometry) uniquely
        # identifies the mask -- shared by all 42 same-type layers, both ring chunks,
        # and (since both pin the same captured metadata) forward and backward.
        gen_obj = None
        for _gen_key in ("mm_token_type_ids", "_packed_seq_ids", "padding_mask"):
            _cand = ctx.metadata.get(_gen_key)
            if _cand is not None:
                gen_obj = _cand
                break
        _block_mask_set_generation(gen_obj)
        block_mask_key = (
            getattr(attention_module, "sliding_window", None),
            bool(ctx.is_causal),
            int(ctx.seq_global_start),
            int(kv_global_start),
            int(ctx.seq_local),
            int(key_chunk.shape[2]),
            flex_block_size,
            block_mask_batch,
            bool(use_vision_bidirectional),
            query.device.type,
            query.device.index,
        )
        block_mask = _cached_block_mask(block_mask_key, _build_block_mask)

        query_for_flex = query
        key_for_flex = key_chunk
        value_for_flex = value_chunk
        flex_scale = ctx.scale
        if padded_head_dim != orig_head_dim:
            pad_len = padded_head_dim - orig_head_dim
            query_for_flex = F.pad(query_for_flex, (0, pad_len))
            key_for_flex = F.pad(key_for_flex, (0, pad_len))
            value_for_flex = F.pad(value_for_flex, (0, pad_len))
            if flex_scale is None:
                flex_scale = 1.0 / math.sqrt(orig_head_dim)

        flex_kwargs = {"block_mask": block_mask, "scale": flex_scale, "enable_gqa": ctx.enable_gqa}
        if use_small_flex_blocks:
            flex_kwargs["kernel_options"] = {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_M1": 32,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 32,
                "num_stages": 1,
                "num_warps": 4,
            }

        with _duck_shape_disabled():
            try:
                out, lse = _compiled_flex_attention(attention_module)(
                    query_for_flex.contiguous(), key_for_flex, value_for_flex, return_lse=True, **flex_kwargs
                )
            except TypeError as exc:
                if "kernel_options" in str(exc) and "kernel_options" in flex_kwargs:
                    flex_kwargs.pop("kernel_options")
                    out, lse = _compiled_flex_attention(attention_module)(
                        query_for_flex.contiguous(), key_for_flex, value_for_flex, return_lse=True, **flex_kwargs
                    )
                else:
                    raise

        if empty_query_rows is not None and empty_query_rows.any():
            out = out.masked_fill(empty_query_rows[:, None, :, None], 0)
        if padded_head_dim != orig_head_dim:
            out = out[..., :orig_head_dim]

        return out, lse, empty_query_rows, padded_head_dim
    except Exception as flex_err:
        raise RuntimeError(
            "Gemma4 CP ring requires FlexAttention for local-query/ring-KV attention. "
            f"FlexAttention failed for Q={tuple(query.shape)} K={tuple(key_chunk.shape)} "
            f"V={tuple(value_chunk.shape)} cp_rank={ctx.cp_rank} seq_local={ctx.seq_local} "
            f"kv_global_start={kv_global_start}."
        ) from flex_err


def _ring_num_prior_chunks(ctx: Any) -> int:
    """Number of *prior* KV chunks a rank must ring-collect (own chunk excluded).

    Global layers and vision-bidirectional layers take the full ``cp_size - 1`` rotation.
    A vision group can cross the sliding window and CP rank boundaries in either direction.
    For ordinary sliding-window layers a query reaches only ``window_left`` tokens back, so
    at most ``ceil(window_left/seq_local)`` preceding chunks hold any in-window key; collecting
    the rest just ships KV the mask discards. Forward collection and backward dK/dV routing
    both derive their hop count from here so they stay symmetric.
    """
    full = ctx.cp_size - 1
    attention_module = getattr(ctx, "module", None)
    window_left = getattr(attention_module, "sliding_window", None)
    if getattr(ctx, "use_vision_bidirectional", False):
        return full
    if not window_left:  # global / full-attention layer: needs the full rotation
        return full
    n_prior = math.ceil(int(window_left) / ctx.seq_local)
    return max(0, min(full, n_prior))


def _collect_ring_kv_chunks(ctx: Any) -> list[tuple[int, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]]:
    current_key = ctx.key.contiguous()
    current_value = ctx.value.contiguous()
    current_metadata = {name: value.contiguous() if value is not None else None for name, value in ctx.metadata.items()}
    current_owner = ctx.cp_rank
    chunks = []

    n_prior = _ring_num_prior_chunks(ctx)
    for step in range(n_prior + 1):
        chunks.append((current_owner, current_key, current_value, current_metadata))

        if step == n_prior:
            break

        recv_key = torch.empty_like(current_key)
        recv_value = torch.empty_like(current_value)
        recv_metadata = _metadata_like(current_metadata)
        exchange_tensors = [(current_key, recv_key), (current_value, recv_value)]
        exchange_tensors.extend(
            (current_metadata[name], recv_metadata[name])
            for name in sorted(current_metadata)
            if current_metadata[name] is not None
        )
        _ring_exchange(exchange_tensors, cp_group=ctx.cp_group, cp_rank=ctx.cp_rank, cp_size=ctx.cp_size)
        current_key = recv_key
        current_value = recv_value
        current_metadata = recv_metadata
        current_owner = (current_owner - 1) % ctx.cp_size

    return chunks


def _run_gemma4_cp_ring_attention_forward(attention_module: torch.nn.Module, ctx: Any) -> torch.Tensor:
    """Run Gemma4 local-query/ring-key CP attention forward with FlexAttention."""
    if ctx.dropout_p:
        raise NotImplementedError("Gemma4 FlexAttention ring CP does not support attention dropout.")

    out_acc = None
    lse_acc = None
    empty_query_rows = None
    padded_head_dim = ctx.query.shape[-1]

    for current_owner, current_key, current_value, current_metadata in _collect_ring_kv_chunks(ctx):
        kv_global_start = current_owner * ctx.seq_local
        out_step, lse_step, empty_query_rows, padded_head_dim = _run_gemma4_flex_chunk(
            attention_module,
            ctx,
            key_chunk=current_key,
            value_chunk=current_value,
            metadata_chunk=current_metadata,
            kv_global_start=kv_global_start,
        )
        out_acc, lse_acc = _merge_flex_chunk(out_acc, lse_acc, out_step, lse_step)

    if out_acc is None:
        raise RuntimeError("Gemma4 CP ring attention produced no output chunks.")
    if empty_query_rows is not None and empty_query_rows.any():
        out_acc = out_acc.masked_fill(empty_query_rows[:, None, :, None], 0)

    global _GEMMA4_CP_FLEX_RING_OK_LOGGED
    if not _GEMMA4_CP_FLEX_RING_OK_LOGGED:
        logger.info(
            "Gemma4 CP using compiled flex_attention p2p ring. Q=%s K_local=%s head_dim=%s->%s cp_rank=%s cp_size=%s",
            tuple(ctx.query.shape),
            tuple(ctx.key.shape),
            ctx.query.shape[-1],
            padded_head_dim,
            ctx.cp_rank,
            ctx.cp_size,
        )
        _GEMMA4_CP_FLEX_RING_OK_LOGGED = True
    return out_acc.to(ctx.query.dtype)


def _zero_if_none(grad: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(like) if grad is None else grad


# FFPA CuTeDSL ring path (head_dim=512 full-attention). Each ring KV chunk runs the dense
# kernel on the raw [B, H, S, D] tensors when its segment is dense-eligible (single document
# per row, optionally with a pad suffix -- zero gather/scatter), else the varlen THD path
# (gather -> _varlen_{fwd,bwd}_cute -> scatter, the only way to mask across documents within a
# shard). Other layers/batches (sliding-window, no kernel) use compiled FlexAttention.


def _ffpa_varlen_ring_available() -> bool:
    """Whether the FFPA CuTeDSL *varlen* ops are ready (CPU-test monkeypatch seam)."""
    return _ffpa_varlen_ready()


def _row_single_doc_prefix(row: list[int]) -> bool:
    """Row's real (``>0``) tokens are one document forming a contiguous prefix (pad, if any, is
    a trailing suffix) -- the shape a single-document shard takes after tail padding."""
    doc = None
    seen_pad = False
    for d in row:
        if d <= 0:
            seen_pad = True
        else:
            if seen_pad:
                return False  # a real token after padding -> not a clean prefix
            if doc is None:
                doc = d
            elif d != doc:
                return False  # a second document
    return doc is not None


def _row_single_doc_full(row: list[int]) -> bool:
    """Row is exactly one document with no padding (every position the same ``>0`` id)."""
    doc = row[0] if row else 0
    if doc <= 0:
        return False
    return all(d == doc for d in row)


def _build_packed_ring_segments(q_ids: torch.Tensor, k_ids: torch.Tensor) -> dict[str, Any] | None:
    """Per-document varlen segments shared between a query shard and one KV chunk.

    ``q_ids``/``k_ids`` are the ``[B, S]`` ``_packed_seq_ids`` (``0`` = pad, ``>0`` = doc id).
    For each doc in *both* shards, pair its query/key tokens into one segment; returns flat
    gather indices into ``B*S`` + int32 ``cu_seqlens`` (``cu_q[i]`` pairs ``cu_k[i]``), or
    ``None`` when no doc is shared.

    Dense-routing flags (consumed by :func:`_chunk_dense_eligible`): ``dense_local`` /
    ``dense_cross`` say whether the chunk can skip the THD gather/scatter and feed the raw
    ``[B, H, S, D]`` tensors to the dense FFPA kernel when used as the local (causal) or cross
    (non-causal) chunk; ``pad_rows`` is the ``[B, S]`` pad-query mask to zero after a dense run
    (``None`` when unpadded).
    """
    B, Sq = q_ids.shape
    Sk = k_ids.shape[1]
    device = q_ids.device
    q_index: list[int] = []
    k_index: list[int] = []
    cu_q: list[int] = [0]
    cu_k: list[int] = [0]
    max_q = 0
    max_k = 0
    # One device->host sync per shard (the id maps are tiny [B, S] int tensors);
    # everything below is plain Python so we never sync per document.
    q_ids_cpu = q_ids.tolist()
    k_ids_cpu = k_ids.tolist()
    for b in range(B):
        qrow = q_ids_cpu[b]
        krow = k_ids_cpu[b]
        k_by_doc: dict[int, list[int]] = {}
        for pos, d in enumerate(krow):
            if d > 0:
                k_by_doc.setdefault(d, []).append(pos + b * Sk)
        q_by_doc: dict[int, list[int]] = {}  # dict preserves first-occurrence (== ascending position) order
        for pos, d in enumerate(qrow):
            if d > 0:
                q_by_doc.setdefault(d, []).append(pos + b * Sq)
        for d, qpos in q_by_doc.items():
            kpos = k_by_doc.get(d)
            if not kpos:
                continue
            q_index.extend(qpos)
            k_index.extend(kpos)
            cu_q.append(cu_q[-1] + len(qpos))
            cu_k.append(cu_k[-1] + len(kpos))
            max_q = max(max_q, len(qpos))
            max_k = max(max_k, len(kpos))
    if not q_index:
        return None
    # Dense eligibility: a chunk can skip the THD detour only if it needs no cross-document
    # masking. The query shard must be one document per row, any pad confined to a trailing
    # suffix (``_row_single_doc_prefix``) -- enough for the local chunk, where k == q and causal
    # masking already hides the pad tail. A cross chunk has no causal mask to hide a second
    # document or pad in its KV, so it also needs the KV chunk to be that one document in full
    # (``_row_single_doc_full``).
    q_single_doc = all(_row_single_doc_prefix(row) for row in q_ids_cpu)
    k_single_doc = all(_row_single_doc_full(row) for row in k_ids_cpu)
    has_pad = any(d <= 0 for row in q_ids_cpu for d in row)
    return {
        "q_index": torch.tensor(q_index, dtype=torch.long, device=device),
        "k_index": torch.tensor(k_index, dtype=torch.long, device=device),
        "cu_q": torch.tensor(cu_q, dtype=torch.int32, device=device),
        "cu_k": torch.tensor(cu_k, dtype=torch.int32, device=device),
        "max_q": max_q,
        "max_k": max_k,
        "dense_local": q_single_doc,
        "dense_cross": q_single_doc and k_single_doc,
        "pad_rows": (q_ids <= 0) if has_pad else None,
    }


def _chunk_dense_eligible(seg: dict[str, Any], causal: bool) -> bool:
    """Select this chunk's dense-kernel eligibility for its role in the ring.

    A local chunk runs causally (``causal=True``), a cross chunk non-causally; the two precomputed
    flags from :func:`_build_packed_ring_segments` differ because only the cross chunk lacks a
    causal mask to hide a second document or a pad tail in its KV. Dense-eligible chunks still
    zero their pad query rows afterwards via ``seg["pad_rows"]``.
    """
    return seg["dense_local"] if causal else seg["dense_cross"]


# Ring segments depend only on the doc maps + ring geometry, not the Q/K/V values, and are
# identical forward and backward. Cache once per step (keyed on the q doc map's pinned
# data_ptr) so the ``.tolist()`` D->H sync + Python pairing in _build_packed_ring_segments
# runs once per ring chunk per step, not once per chunk per layer per pass.
_RING_SEGMENT_CACHE: dict = {}
_RING_SEGMENT_GEN: list = [None, None]


def _ring_segment_set_generation(gen_tensor: torch.Tensor) -> None:
    """Clear the ring-segment cache when a new batch (new q doc map) arrives."""
    ptr = gen_tensor.data_ptr()
    if ptr != _RING_SEGMENT_GEN[0]:
        _RING_SEGMENT_CACHE.clear()
        _RING_SEGMENT_GEN[0] = ptr
        _RING_SEGMENT_GEN[1] = gen_tensor  # pin the storage for the duration of the step


def _cached_ring_segments(
    q_ids: torch.Tensor, k_ids: torch.Tensor, *, cp_rank: int, owner: int, cp_size: int
) -> dict[str, Any] | None:
    """Per-step cache around :func:`_build_packed_ring_segments`.

    ``owner`` uniquely identifies the rotated k doc map within a step (each rank's
    shard is visited exactly once), so ``(cp_rank, owner, cp_size, B, Sq, Sk)``
    keyed under the current q-doc-map generation uniquely identifies the segment.
    ``None`` (no shared document) is cached too.
    """
    _ring_segment_set_generation(q_ids)
    key = (cp_rank, owner, cp_size, q_ids.shape[0], q_ids.shape[1], k_ids.shape[1])
    if key in _RING_SEGMENT_CACHE:
        return _RING_SEGMENT_CACHE[key]
    seg = _build_packed_ring_segments(q_ids, k_ids)
    if len(_RING_SEGMENT_CACHE) >= 256:
        _RING_SEGMENT_CACHE.pop(next(iter(_RING_SEGMENT_CACHE)))
    _RING_SEGMENT_CACHE[key] = seg
    return seg


def _gather_thd(t_bhsd: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D]`` -> packed ``[T, H, D]`` at flat ``b*S + pos`` indices."""
    B, H, S, D = t_bhsd.shape
    return t_bhsd.transpose(1, 2).reshape(B * S, H, D).index_select(0, flat_index)


def _scatter_thd(packed: torch.Tensor, flat_index: torch.Tensor, B: int, H: int, S: int) -> torch.Tensor:
    """Inverse of :func:`_gather_thd`: ``[T, H, D]`` -> ``[B, H, S, D]`` (0 elsewhere)."""
    D = packed.shape[-1]
    out = torch.zeros(B * S, H, D, dtype=packed.dtype, device=packed.device)
    out.index_copy_(0, flat_index, packed)
    return out.view(B, S, H, D).transpose(1, 2)


def _scatter_lse(lse_pack: torch.Tensor, flat_index: torch.Tensor, B: int, Hq: int, S: int) -> torch.Tensor:
    """Varlen LSE ``[Hq, T]`` -> dense ``[B, Hq, S]`` (``-inf`` for non-participating rows)."""
    flat = torch.full((B * S, Hq), float("-inf"), dtype=torch.float32, device=lse_pack.device)
    flat.index_copy_(0, flat_index, lse_pack.transpose(0, 1).to(torch.float32))
    return flat.view(B, S, Hq).permute(0, 2, 1).contiguous()


def _gather_lse(lse_dense: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    """Dense ``[B, Hq, S]`` -> packed ``[Hq, T]`` (inverse of :func:`_scatter_lse`)."""
    B, Hq, S = lse_dense.shape
    return lse_dense.permute(0, 2, 1).reshape(B * S, Hq).index_select(0, flat_index).transpose(0, 1).contiguous()


def _merge_ffpa_packed_chunk(
    out_acc: torch.Tensor | None,
    lse_acc: torch.Tensor | None,
    out_step: torch.Tensor,
    lse_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``-inf``-safe online-softmax merge: like :func:`_merge_flex_chunk` but a row where both
    inputs are ``-inf`` (pad query no chunk covered) keeps 0 instead of ``NaN``.

    On finite rows ``old_scale + new_scale == 1``, so the combine is a single ``lerp``; the only
    ``NaN`` is the both-``-inf`` row, which ``nan_to_num_`` zeroes (leaving its already-0 output).
    The fused ``lerp`` is ~2.5x cheaper than the explicit ``isneginf``/``where`` form.
    """
    if out_acc is None or lse_acc is None:
        return out_step, lse_step
    lse_next = torch.logaddexp(lse_acc, lse_step)
    new_scale = torch.exp(lse_step - lse_next).nan_to_num_(nan=0.0).unsqueeze(-1)
    return torch.lerp(out_acc, out_step, new_scale), lse_next


# A cross/straddle chunk shares only a small token subset (T << B*S), so scattering its packed
# result to a full [B, Hq, S, D] tensor + a full-tensor merge/add does O(B*S) work for a T-row
# effect. Like TransformerEngine's thd_out_correction / thd_grad_correction, we instead update
# in place at just the live rows (advanced-index on the permuted view), keeping the accumulator
# BSHD-contiguous so the dense path is untouched. Measured (H20Z, fp16, D=512): wins below
# T/(B*S) ~= 0.3 (2.3x at 0.05), loses above (strided in-place < contiguous scatter at full T),
# so gate at 0.25. The gather is deliberately left as transpose+reshape+index_select -- an
# advanced-index gather moves less data but is uniformly slower (coalescing wins).
_LIVE_ROW_MERGE_MAX_FRACTION = 0.25


def _use_live_row_merge(num_live: int, num_rows: int) -> bool:
    """True when the live-token subset is small enough for in-place updates to beat scatter."""
    return num_live <= _LIVE_ROW_MERGE_MAX_FRACTION * num_rows


def _merge_live_chunk_(
    out_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    out_pack: torch.Tensor,
    lse_pack: torch.Tensor,
    flat_index: torch.Tensor,
) -> None:
    """In-place online-softmax merge of one varlen chunk at its ``flat_index`` live rows only.

    Like :func:`_merge_ffpa_packed_chunk` but updates only the ``T`` live rows of the running
    ``[B, Hq, S, D]``/``[B, Hq, S]`` accumulators. The cheap ``lerp`` (no ``-inf`` guard) is safe:
    a cross chunk targets only real query tokens, all of which the local chunk already covered,
    so ``lse_acc`` is finite at every live row.
    """
    B, Hq, S, D = out_acc.shape
    out_view = out_acc.permute(0, 2, 1, 3)  # [B, S, Hq, D] view
    lse_view = lse_acc.permute(0, 2, 1)  # [B, S, Hq] view
    b_idx = torch.div(flat_index, S, rounding_mode="floor")
    pos = flat_index - b_idx * S
    acc_rows = out_view[b_idx, pos]  # [T, Hq, D]
    lse_step = lse_pack.transpose(0, 1)  # [T, Hq]
    lse_next = torch.logaddexp(lse_view[b_idx, pos], lse_step)
    new_scale = torch.exp(lse_step - lse_next).unsqueeze(-1)  # [T, Hq, 1]
    out_view[b_idx, pos] = torch.lerp(acc_rows, out_pack, new_scale)
    lse_view[b_idx, pos] = lse_next


def _grad_add_live_(grad_bhsd: torch.Tensor, grad_pack: torch.Tensor, flat_index: torch.Tensor) -> None:
    """Add ``grad_pack[T, H, D]`` into ``grad_bhsd[B, H, S, D]`` at the ``flat_index`` rows only.

    The in-place advanced-index update on the permuted view keeps ``grad_bhsd`` BSHD-contiguous
    (dense backward adds full tensors into it unchanged) and avoids the ``torch.zeros`` +
    full-tensor add of ``grad = grad + _scatter_thd(...)``. ``flat_index`` is duplicate-free, so
    the assign is a true add.
    """
    B, H, S, D = grad_bhsd.shape
    grad_view = grad_bhsd.permute(0, 2, 1, 3)  # [B, S, H, D] view
    b_idx = torch.div(flat_index, S, rounding_mode="floor")
    pos = flat_index - b_idx * S
    grad_view[b_idx, pos] += grad_pack


def _ffpa_varlen_forward_chunk(
    q_pack: torch.Tensor,
    k_pack: torch.Tensor,
    v_pack: torch.Tensor,
    seg: dict[str, Any],
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ring KV chunk via the FFPA varlen forward. Returns ``(out[T,Hq,D], lse[Hq,T])``."""
    return _ffpa_varlen_fwd(
        q_pack, k_pack, v_pack, seg["cu_q"], seg["cu_k"], seg["max_q"], seg["max_k"], scale=scale, causal=causal
    )


def _ffpa_varlen_backward_chunk(
    grad_out_pack: torch.Tensor,
    q_pack: torch.Tensor,
    k_pack: torch.Tensor,
    v_pack: torch.Tensor,
    out_pack: torch.Tensor,
    lse_pack: torch.Tensor,
    seg: dict[str, Any],
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-chunk FFPA varlen backward using the *global* merged out/lse (packed)."""
    return _ffpa_varlen_bwd(
        grad_out_pack,
        q_pack,
        k_pack,
        v_pack,
        out_pack,
        lse_pack,
        seg["cu_q"],
        seg["cu_k"],
        seg["max_q"],
        seg["max_k"],
        scale=scale,
        causal=causal,
    )


def _ring_use_ffpa_varlen(attention_module: torch.nn.Module, ctx: Any) -> bool:
    """Whether this ring attention call may use the FFPA *varlen* ring path.

    The decision gates collective p2p exchanges so it must be rank-uniform: it
    depends only on the per-layer config, the head_dim / dtype / scale / dropout, and
    *whether* ``_packed_seq_ids`` is present (a batch-level fact) -- never on
    per-rank slice content. This path *requires* ``_packed_seq_ids`` (the document
    map drives the varlen ``cu_seqlens``); Gemma4's manual CP batch always attaches
    one, so it is the sole FFPA path real CP training takes.

    Raises:
        NotImplementedError: If a statically eligible FFPA ring candidate requests nonzero attention dropout.
    """
    if not getattr(attention_module, "_gemma4_cp_use_ffpa", False):
        return False
    if not ctx.is_causal:
        return False
    if getattr(attention_module, "sliding_window", None) is not None:
        return False
    if ctx.metadata.get("_packed_seq_ids") is None:
        return False
    if ctx.query.shape[-1] != _FFPA_HEAD_DIM:
        return False
    if ctx.query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if ctx.scale is None:
        return False
    if ctx.dropout_p != 0.0:
        raise NotImplementedError(
            "Gemma4 FFPA ring CP does not support attention dropout: "
            f"ffpa-attn requires dropout_p=0.0, got {ctx.dropout_p!r}."
        )
    return _ffpa_varlen_ring_available()


def _run_gemma4_cp_ffpa_varlen_ring_forward(
    ctx: Any, *, seg_sink: dict[int, Any] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward of the FFPA ring: rotate K/V, per-chunk dense-or-varlen FFPA, online-merge.

    Returns ``(out_final[B,Hq,S,D] fp32, lse_final[B,Hq,S] fp32)``. ``owner > cp_rank`` chunks
    are causally skipped; chunks sharing no document contribute nothing; pad query rows stay 0.
    When ``seg_sink`` is given, each processed ``owner``'s segment is recorded (``owner -> seg``)
    so the backward reuses the same path choice and indices.
    """
    q = ctx.query
    B, Hq, S, _D = q.shape
    q_ids = ctx.metadata.get("_packed_seq_ids")
    if q_ids is None:
        raise RuntimeError("Gemma4 CP FFPA ring requires _packed_seq_ids in metadata.")
    # The local (causal, first) chunk is processed first; when it runs dense it covers every real
    # query row (pad rows stay 0, never -inf), so the cheap non-guarded merge is then safe for all
    # later chunks. Genuinely multi-document local shards keep the -inf-safe merge.
    local_seg = _cached_ring_segments(q_ids, q_ids, cp_rank=ctx.cp_rank, owner=ctx.cp_rank, cp_size=ctx.cp_size)
    local_dense = local_seg is not None and _chunk_dense_eligible(local_seg, True)
    merge = _merge_flex_chunk if local_dense else _merge_ffpa_packed_chunk
    out_acc = None
    lse_acc = None
    for owner, key_chunk, value_chunk, meta in _collect_ring_kv_chunks(ctx):
        if owner > ctx.cp_rank:
            continue
        seg = _cached_ring_segments(
            q_ids, meta.get("_packed_seq_ids"), cp_rank=ctx.cp_rank, owner=owner, cp_size=ctx.cp_size
        )
        if seg_sink is not None:
            seg_sink[owner] = seg
        if seg is None:
            continue
        causal = owner == ctx.cp_rank
        if _chunk_dense_eligible(seg, causal):
            out_step, lse_step = _ffpa_dense_fwd(q, key_chunk, value_chunk, scale=ctx.scale, causal=causal)
            out_step = out_step.to(torch.float32)
            pad_rows = seg["pad_rows"]
            if pad_rows is not None:
                # zero pad query rows (they attended garbage); mirrors Flex empty_query_rows
                out_step = out_step.masked_fill(pad_rows[:, None, :, None], 0)
            out_acc, lse_acc = merge(out_acc, lse_acc, out_step, lse_step)
            continue
        q_index = seg["q_index"]
        q_pack = _gather_thd(q, q_index)
        k_pack = _gather_thd(key_chunk, seg["k_index"])
        v_pack = _gather_thd(value_chunk, seg["k_index"])
        out_pack, lse_pack = _ffpa_varlen_forward_chunk(q_pack, k_pack, v_pack, seg, scale=ctx.scale, causal=causal)
        out_pack = out_pack.to(torch.float32)
        if out_acc is not None and _use_live_row_merge(q_index.numel(), B * S):
            # small cross/straddle subset -> in-place live-row merge (skips the full scatter+merge)
            _merge_live_chunk_(out_acc, lse_acc, out_pack, lse_pack, q_index)
        else:
            out_step = _scatter_thd(out_pack, q_index, B, Hq, S)
            lse_step = _scatter_lse(lse_pack, q_index, B, Hq, S)
            out_acc, lse_acc = merge(out_acc, lse_acc, out_step, lse_step)
    if out_acc is None or lse_acc is None:
        # All-pad query shard (no shared doc): return 0/-inf, never raise -- a one-rank
        # raise desyncs the ring p2p and hangs the job. Mirrors the Flex path.
        out_acc = torch.zeros(B, Hq, S, _D, dtype=torch.float32, device=q.device)
        lse_acc = torch.full((B, Hq, S), float("-inf"), dtype=torch.float32, device=q.device)
    return out_acc, lse_acc


class _Gemma4FFPAVarlenRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(autograd_ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, ring_ctx: Any):
        runtime_ctx = replace(ring_ctx, query=query, key=key, value=value)
        seg_by_owner: dict[int, Any] = {}
        out_final, lse_final = _run_gemma4_cp_ffpa_varlen_ring_forward(runtime_ctx, seg_sink=seg_by_owner)
        out = out_final.to(query.dtype)
        autograd_ctx.save_for_backward(query, key, value, out, lse_final)
        autograd_ctx.ring_segments = seg_by_owner
        autograd_ctx.ring_ctx = replace(
            ring_ctx,
            query=None,
            key=None,
            value=None,
            metadata=_detach_metadata(ring_ctx.metadata),
        )
        return out

    @staticmethod
    def backward(autograd_ctx, grad_output: torch.Tensor):
        query, key, value, out_final, lse_final = autograd_ctx.saved_tensors
        ring_ctx = autograd_ctx.ring_ctx
        cp_size = ring_ctx.cp_size
        cp_rank = ring_ctx.cp_rank
        B, Hq, S, _D = query.shape
        Hkv = key.shape[1]
        seg_by_owner = autograd_ctx.ring_segments

        collect_ctx = replace(ring_ctx, query=query, key=key, value=value)
        grad_query = torch.zeros_like(query, dtype=torch.float32)
        grad_key_by_owner = {owner: torch.zeros_like(key) for owner in range(cp_size)}
        grad_value_by_owner = {owner: torch.zeros_like(value) for owner in range(cp_size)}

        for owner, key_chunk, value_chunk, _meta in _collect_ring_kv_chunks(collect_ctx):
            if owner > cp_rank:
                continue
            seg = seg_by_owner.get(owner)
            if seg is None:
                continue
            causal = owner == cp_rank
            if _chunk_dense_eligible(seg, causal):
                # Dense backward fed the global out/lse (no gather/scatter). For the pad path,
                # zero grad_output's pad rows -- the forward forced those outputs to 0.
                grad_out_chunk = grad_output
                pad_rows = seg["pad_rows"]
                if pad_rows is not None:
                    grad_out_chunk = grad_output.masked_fill(pad_rows[:, None, :, None], 0)
                dq_step, dk_step, dv_step = _ffpa_dense_bwd(
                    grad_out_chunk,
                    query,
                    key_chunk,
                    value_chunk,
                    out_final,
                    lse_final,
                    scale=ring_ctx.scale,
                    causal=causal,
                )
                grad_query = grad_query + dq_step.to(torch.float32)
                grad_key_by_owner[owner] = grad_key_by_owner[owner] + dk_step.to(key.dtype)
                grad_value_by_owner[owner] = grad_value_by_owner[owner] + dv_step.to(value.dtype)
                continue
            q_index = seg["q_index"]
            k_index = seg["k_index"]
            q_pack = _gather_thd(query, q_index)
            k_pack = _gather_thd(key_chunk, k_index)
            v_pack = _gather_thd(value_chunk, k_index)
            out_pack = _gather_thd(out_final, q_index)
            grad_out_pack = _gather_thd(grad_output, q_index)
            lse_pack = _gather_lse(lse_final, q_index)
            dq_pack, dk_pack, dv_pack = _ffpa_varlen_backward_chunk(
                grad_out_pack, q_pack, k_pack, v_pack, out_pack, lse_pack, seg, scale=ring_ctx.scale, causal=causal
            )
            n_rows = B * S
            # small subset -> in-place live-row accumulate; larger -> scatter+add (see fwd note)
            if _use_live_row_merge(q_index.numel(), n_rows):
                _grad_add_live_(grad_query, dq_pack.to(torch.float32), q_index)
            else:
                grad_query = grad_query + _scatter_thd(dq_pack.to(torch.float32), q_index, B, Hq, S)
            if _use_live_row_merge(k_index.numel(), n_rows):
                _grad_add_live_(grad_key_by_owner[owner], dk_pack.to(key.dtype), k_index)
                _grad_add_live_(grad_value_by_owner[owner], dv_pack.to(value.dtype), k_index)
            else:
                grad_key_by_owner[owner] = grad_key_by_owner[owner] + _scatter_thd(
                    dk_pack.to(key.dtype), k_index, B, Hkv, S
                )
                grad_value_by_owner[owner] = grad_value_by_owner[owner] + _scatter_thd(
                    dv_pack.to(value.dtype), k_index, B, Hkv, S
                )

        # All-pad shard: grads stay 0; do NOT raise (the dK/dV p2p below is rank-uniform,
        # so a one-rank raise desyncs the ring). Forward returned 0 for these rows too.

        grad_key, grad_value = _route_kv_grads_to_owners(
            grad_key_by_owner,
            grad_value_by_owner,
            cp_group=ring_ctx.cp_group,
            cp_rank=cp_rank,
            cp_size=cp_size,
            n_prior=_ring_num_prior_chunks(ring_ctx),
        )
        return grad_query.to(query.dtype), grad_key, grad_value, None


class _Gemma4FlexRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(autograd_ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, ring_ctx: Any):
        runtime_ctx = replace(ring_ctx, query=query, key=key, value=value)
        out = _run_gemma4_cp_ring_attention_forward(ring_ctx.module, runtime_ctx)
        autograd_ctx.save_for_backward(query, key, value)
        autograd_ctx.ring_ctx = replace(
            ring_ctx,
            query=None,
            key=None,
            value=None,
            metadata=_detach_metadata(ring_ctx.metadata),
        )
        return out

    @staticmethod
    def backward(autograd_ctx, grad_output: torch.Tensor):
        query, key, value = autograd_ctx.saved_tensors
        ring_ctx = autograd_ctx.ring_ctx

        with torch.enable_grad():
            query_req = query.detach().requires_grad_(True)
            collect_ctx = replace(
                ring_ctx,
                query=query_req,
                key=key.detach(),
                value=value.detach(),
                metadata=_detach_metadata(ring_ctx.metadata),
            )
            chunks = _collect_ring_kv_chunks(collect_ctx)

            runtime_ctx = replace(collect_ctx, query=query_req)
            key_reqs = []
            value_reqs = []
            owners = []
            out_acc = None
            lse_acc = None
            empty_query_rows = None

            for owner, key_chunk, value_chunk, metadata_chunk in chunks:
                key_req = key_chunk.detach().requires_grad_(True)
                value_req = value_chunk.detach().requires_grad_(True)
                key_reqs.append(key_req)
                value_reqs.append(value_req)
                owners.append(owner)

                out_step, lse_step, empty_query_rows, _ = _run_gemma4_flex_chunk(
                    ring_ctx.module,
                    runtime_ctx,
                    key_chunk=key_req,
                    value_chunk=value_req,
                    metadata_chunk=metadata_chunk,
                    kv_global_start=owner * ring_ctx.seq_local,
                )
                out_acc, lse_acc = _merge_flex_chunk(out_acc, lse_acc, out_step, lse_step)

            if out_acc is None:
                raise RuntimeError("Gemma4 CP ring attention backward produced no output chunks.")
            if empty_query_rows is not None and empty_query_rows.any():
                out_acc = out_acc.masked_fill(empty_query_rows[:, None, :, None], 0)

            grad_targets = [query_req, *key_reqs, *value_reqs]
            grads = torch.autograd.grad(out_acc, grad_targets, grad_output, allow_unused=True)

        grad_query = _zero_if_none(grads[0], query)
        num_chunks = len(chunks)
        grad_key_by_owner = {owner: _zero_if_none(grads[1 + idx], key_reqs[idx]) for idx, owner in enumerate(owners)}
        grad_value_by_owner = {
            owner: _zero_if_none(grads[1 + num_chunks + idx], value_reqs[idx]) for idx, owner in enumerate(owners)
        }

        grad_key, grad_value = _route_kv_grads_to_owners(
            grad_key_by_owner,
            grad_value_by_owner,
            cp_group=ring_ctx.cp_group,
            cp_rank=ring_ctx.cp_rank,
            cp_size=ring_ctx.cp_size,
            n_prior=_ring_num_prior_chunks(ring_ctx),
        )
        return grad_query, grad_key, grad_value, None


class _Gemma4LocalKernelRingAttention(torch.autograd.Function):
    """Sliding-window CP ring driven by a local FlashAttention kernel instead of compiled flex.

    Reuses :class:`_Gemma4FlexRingAttention`'s p2p ring collection
    (:func:`_collect_ring_kv_chunks`) and dK/dV routing (:func:`_route_kv_grads_to_owners`)
    unchanged; only the per-shard compute differs: it concatenates the causal window
    neighborhood and issues one FlashAttention call
    (:func:`cp_local_ring.sliding_ring_compute_fa_fwd`) instead of per-chunk flex + softmax
    merge. Backward is no-recompute: the forward saves the kernel context (out, LSE, RNG) and
    the backward runs FlashAttention's own backward, then routes dK/dV to owners.
    """

    @staticmethod
    def forward(autograd_ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, ring_ctx: Any):
        from nemo_automodel.components.models.gemma4_moe.cp_local_ring import sliding_ring_compute_fa_fwd

        runtime_ctx = replace(ring_ctx, query=query, key=key, value=value)
        chunks = _collect_ring_kv_chunks(runtime_ctx)
        autograd_ctx.ring_ctx = replace(
            ring_ctx, query=None, key=None, value=None, metadata=_detach_metadata(ring_ctx.metadata)
        )
        out, autograd_ctx.fa_saved = sliding_ring_compute_fa_fwd(runtime_ctx, chunks)
        return out

    @staticmethod
    def backward(autograd_ctx, grad_output: torch.Tensor):
        from nemo_automodel.components.models.gemma4_moe.cp_local_ring import sliding_ring_compute_fa_bwd

        ring_ctx = autograd_ctx.ring_ctx
        grad_query, grad_key_by_owner, grad_value_by_owner = sliding_ring_compute_fa_bwd(
            autograd_ctx.fa_saved, grad_output
        )
        grad_key, grad_value = _route_kv_grads_to_owners(
            grad_key_by_owner,
            grad_value_by_owner,
            cp_group=ring_ctx.cp_group,
            cp_rank=ring_ctx.cp_rank,
            cp_size=ring_ctx.cp_size,
            n_prior=_ring_num_prior_chunks(ring_ctx),
        )
        return grad_query, grad_key, grad_value, None


def _run_gemma4_cp_ring_attention(attention_module: torch.nn.Module, ctx: Any) -> torch.Tensor:
    """Run Gemma4 local-query/ring-key CP attention.

    Full-attention (global) head_dim=512 layers run their per-chunk attention
    through the FFPA CuTeDSL kernel (``_ring_use_ffpa_varlen`` gate), choosing per
    chunk between the *dense* ``_fwd_cute`` path (single full document per row -- the
    unpacked / synthesized-single-document case -- on the raw ``[B, H, S, D]`` tensors,
    zero gather/scatter) and the *varlen* ``_varlen_fwd_cute`` path (genuinely packed /
    straddling shards, which need cross-document masking via THD ``cu_seqlens``). The
    manual CP batch always attaches a ``_packed_seq_ids`` map
    (``cp_batch._synthesize_single_document_seq_ids`` injects a trivial single-document
    one when the batch is not packed), so this is the path real CP training -- packed or
    unpacked -- always takes. Every other layer / batch (sliding-window, no kernel,
    wrong dtype/head_dim) keeps using compiled FlexAttention.
    """
    if _ring_use_ffpa_varlen(attention_module, ctx):
        return _Gemma4FFPAVarlenRingAttention.apply(ctx.query, ctx.key, ctx.value, ctx)
    # Sliding-window layers: optional FlashAttention-2 backend off compiled flex, fully
    # autograd (no analytical backward, no recompute). Only the plain causal+window+packed
    # case; vision-bidirectional batches stay on flex. The gate is rank-uniform because the
    # backend, layer type, model config, and full-sequence vision-input flag are identical
    # across CP ranks.
    sliding_backend = getattr(attention_module, "_gemma4_cp_sliding_backend", "flex")
    if (
        sliding_backend == "fa"
        and getattr(attention_module, "sliding_window", None) is not None
        and not getattr(ctx, "use_vision_bidirectional", False)
    ):
        return _Gemma4LocalKernelRingAttention.apply(ctx.query, ctx.key, ctx.value, ctx)
    return _Gemma4FlexRingAttention.apply(ctx.query, ctx.key, ctx.value, ctx)


def _gemma4_cp_manual_attention(
    attention_module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cp_mesh,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    kwargs,
) -> torch.Tensor:
    """Gemma4-owned manual ring CP attention entry.

    Plugs into context_parallel.utils' generic ``run_cp_manual_attention`` seam: receives the
    raw local (un-gathered) Q/K/V plus ``cp_mesh``, builds the ring context, and
    runs the p2p ring FlexAttention. K/V are rotated across CP ranks inside the
    ring autograd function -- they are never all-gathered.
    """
    group = cp_mesh.get_group()
    size = cp_mesh.size()
    cp_rank = torch.distributed.get_rank(group=group)
    seq_local = key.shape[2]
    seq_global_start = cp_rank * seq_local
    seq_full = seq_local * size
    if query.shape[1] != key.shape[1]:
        enable_gqa = True

    captured_metadata = getattr(attention_module, "_cp_manual_metadata", {})
    has_vision_tokens = bool(captured_metadata.get("_gemma4_has_vision_tokens", False))
    local_metadata = {name: value for name, value in captured_metadata.items() if name != "_gemma4_has_vision_tokens"}
    metadata_seq_dims = getattr(attention_module, "_cp_manual_metadata_seq_dims", {})
    ctx = CPRingAttentionContext(
        module=attention_module,
        query=query,
        key=key,
        value=value,
        cp_mesh=cp_mesh,
        cp_group=group,
        cp_size=size,
        cp_rank=cp_rank,
        seq_local=seq_local,
        seq_full=seq_full,
        seq_global_start=seq_global_start,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        kwargs=kwargs,
        metadata=local_metadata,
        metadata_seq_dims=metadata_seq_dims,
        use_vision_bidirectional=_config_uses_vision_bidirectional(attention_module) and has_vision_tokens,
    )
    return _run_gemma4_cp_ring_attention(attention_module, ctx)


def _install_gemma4_cp_ring_sdpa(attention_module: torch.nn.Module, cp_mesh) -> None:
    """Swap ``F.scaled_dot_product_attention`` -> Gemma4 ring CP attention on this module.

    Gemma4 owns its CP attention end-to-end (it does not use context_parallel.utils' generic CP
    SDPA hooks). It installs its own ``@torch._dynamo.disable`` SDPA wrapper -- on
    the inner attention module so it also fires during gradient-checkpointing
    recompute -- that runs the p2p ring FlexAttention. The per-forward attention
    kwargs the ring needs (mm_token_type_ids, packed-seq ids, padding/vision masks)
    are captured off the forward kwargs into ``_cp_manual_metadata`` here, since the
    swapped SDPA only receives Q/K/V.
    """
    import torch.nn.functional as F_module

    original_sdpa = F_module.scaled_dot_product_attention
    metadata_keys = getattr(attention_module, "_cp_manual_metadata_keys", ())

    @torch._dynamo.disable
    def _ring_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs
    ):
        return _gemma4_cp_manual_attention(
            attention_module,
            query,
            key,
            value,
            cp_mesh=cp_mesh,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            kwargs=kwargs,
        )

    def _pre_hook(module, args, kwargs):
        captured = {name: kwargs.pop(name, None) for name in metadata_keys}
        # Dense Gemma4 routes through HF's decoder layers, which do not thread the
        # CP/vision metadata down to self_attn kwargs (the MoE backend does). The
        # dense forward stashes it on the module as ``_cp_dense_metadata``; fall
        # back to that for any key the caller didn't pass so the ring can still
        # build the vision-bidirectional / packed masks. Persisted across the step
        # (not cleared by the post-hook) so it survives activation-checkpoint recompute.
        fallback = getattr(module, "_cp_dense_metadata", None)
        if fallback:
            for name in metadata_keys:
                if captured.get(name) is None:
                    captured[name] = fallback.get(name)
        module._cp_manual_metadata = captured
        # Own the CP mask handling (instead of relying on the generic
        # attach_context_parallel_hooks mask-strip): drop any full-sequence 4D
        # attention_mask -- invalid once the sequence is CP-sharded -- and force
        # the causal path, which the ring honors via run_cp_manual_attention.
        kwargs["attention_mask"] = None
        kwargs["is_causal"] = True
        F_module.scaled_dot_product_attention = _ring_sdpa
        return args, kwargs

    def _post_hook(module, inputs, output):
        module._cp_manual_metadata = {}
        F_module.scaled_dot_product_attention = original_sdpa

    attention_module._cp_uses_attention_hook = True
    attention_module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    attention_module.register_forward_hook(_post_hook, always_call=True)


def attach_gemma4_cp_ring_attention(
    attention_module: torch.nn.Module, *, use_ffpa: bool = False, sliding_backend: str = "flex"
) -> None:
    """Register Gemma4's model-owned p2p ring CP attention on a self-attention module.

    Declares the metadata keys the ring needs and exposes ``setup_cp_attention(cp_mesh)``
    -- the model-owned CP-attention seam the parallelizer calls (with the CP mesh)
    instead of context_parallel.utils' generic SDPA hooks. ``run_cp_manual_attention`` is also bound
    as the ring entry point.

    ``use_ffpa`` opts the (full-attention, head_dim=512) ring chunks into the FFPA
    CuTeDSL kernel; ``_ring_use_ffpa_varlen`` still verifies per-call eligibility, so
    this is a no-op for sliding-window layers, non-512 head_dim, wrong dtype, or when
    the FFPA kernel is unavailable.
    """
    attention_module._gemma4_cp_use_ffpa = bool(use_ffpa)
    # Sliding-window CP kernel: "flex" (default, compiled FlexAttention ring) | "fa"
    # (FlashAttention-2 local kernel over the contiguous ring, see cp_local_ring).
    attention_module._gemma4_cp_sliding_backend = str(sliding_backend).lower()
    attention_module._cp_manual_metadata_keys = (
        "mm_token_type_ids",
        "_packed_seq_ids",
        "padding_mask",
        "_gemma4_vision_group_ids",
        "_gemma4_has_vision_tokens",
    )
    attention_module._cp_manual_metadata_seq_dims = {
        "mm_token_type_ids": 1,
        "_packed_seq_ids": 1,
        "padding_mask": 1,
        "_gemma4_vision_group_ids": 1,
    }
    attention_module.run_cp_manual_attention = MethodType(_gemma4_cp_manual_attention, attention_module)
    attention_module.setup_cp_attention = MethodType(_install_gemma4_cp_ring_sdpa, attention_module)
