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

"""Dense and varlen block-diagonal attention kernels."""

from __future__ import annotations

import logging

import torch

from nemo_automodel.shared.import_utils import safe_import_from, safe_import_te

logger = logging.getLogger(__name__)

# Availability probes only -- the kernel symbols themselves are imported at the
# call sites so a per-call stub (tests) or lazy TE extension load keeps working.
HAS_FLASH_VARLEN, _ = safe_import_from("flash_attn", "flash_attn_varlen_func")
HAS_TE, _ = safe_import_te()

_CP_FLASH_DETERMINISTIC = False
_CP_FLASH_WARNED = False
_CP_FLASH_ENGAGED = False
_CP_VARLEN_SHAPE_LOGGED = False
_CP_FLASH_LONG_SEGMENT_WARNED = False
_CP_TE_DROPOUT_WARNED = False
_TE_DPA_CACHE = {}


def _varlen_backend_unavailable_reason(
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
) -> str | None:
    """Return a cheap, non-kernel-launching reason a varlen kernel cannot run.

    Needed-only KV exchange cannot safely discover these failures after a
    rank-local kernel call: an all-padding rank skips the call while a real-token
    rank may fail and otherwise diverge in collective order. The runtime calls
    this on every CP rank and reaches a small consensus before exchanging K/V.
    Shape/data-dependent kernel failures are still handled by a post-call
    consensus in :mod:`nemo_automodel.components.distributed.blockdiag_cp.runtime`.

    Args:
        backend: Varlen backend name (``"flash"`` or ``"te"``).
        dtype: Query/key/value dtype the kernel would run with.
        device: Device the kernel would run on.

    Returns:
        A human-readable reason string, or ``None`` when the kernel can run.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return f"varlen CP attention needs fp16/bf16, got {dtype}"
    if device.type != "cuda":
        return f"varlen CP attention requires CUDA, got device={device}"
    if backend == "flash":
        if not HAS_FLASH_VARLEN:
            return "flash_attn varlen kernel is unavailable"
        return None
    if backend == "te":
        if not HAS_TE:
            return "TransformerEngine varlen kernel is unavailable"
        return None
    return f"unsupported varlen backend={backend}"


def _varlen_metadata_unavailable_reason(
    meta: dict | None,
    *,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> str | None:
    """Validate every index consumed by a varlen CUDA kernel.

    FlashAttention trusts ``cu_seqlens`` and the caller-provided maxima. A bad
    terminal offset therefore does not reliably raise a Python exception; it can
    become an asynchronous illegal memory access and poison the whole CUDA
    context. Validate the compact, per-step metadata on the host before the
    first attention layer launches a kernel. The result is cached in ``meta``;
    the cache object is intentionally shared by the shallow metadata copies used
    by halo/A2A, so this costs one small device-to-host copy per step and shape,
    not once per layer.

    Args:
        meta: Per-step varlen metadata as produced by
            :func:`precompute_blockdiag_varlen_meta` (``cu_q``/``cu_k`` are 1D
            int32 device tensors of per-document cumulative offsets).
        query_len: Local query length ``L`` (rows this rank attends with).
        key_len: Key length ``S`` visible to this rank (full sequence for
            all-gather, needed range for halo/A2A).
        device: Device the ``cu_q``/``cu_k`` tensors must live on.

    Returns:
        A human-readable reason string when the metadata is unsafe, else ``None``.
    """
    if not isinstance(meta, dict):
        return "missing varlen metadata"

    try:
        n_real = int(meta["n_real"])
    except (KeyError, TypeError, ValueError) as exc:
        return f"invalid n_real ({type(exc).__name__})"

    query_len = int(query_len)
    key_len = int(key_len)
    if not 0 <= n_real <= query_len:
        return f"n_real={n_real} is outside query length {query_len}"
    if n_real == 0:
        return None

    required = ("s_first", "real_end", "cu_q", "cu_k", "max_q", "max_k")
    missing = [name for name in required if name not in meta]
    if missing:
        return f"missing varlen metadata fields: {missing}"

    try:
        s_first = int(meta["s_first"])
        real_end = int(meta["real_end"])
        max_q = int(meta["max_q"])
        max_k = int(meta["max_k"])
    except (TypeError, ValueError) as exc:
        return f"non-integral varlen scalar metadata ({type(exc).__name__})"

    signature = (
        n_real,
        query_len,
        key_len,
        s_first,
        real_end,
        max_q,
        max_k,
        str(device),
    )
    cache = meta.setdefault("_validation_cache", {})
    if signature in cache:
        return cache[signature]

    reason = None
    if not 0 <= s_first <= real_end <= key_len:
        reason = f"K/V slice [{s_first}, {real_end}) is outside key length {key_len}"
    else:
        cu_q = meta["cu_q"]
        cu_k = meta["cu_k"]
        for name, cu in (("cu_q", cu_q), ("cu_k", cu_k)):
            if not isinstance(cu, torch.Tensor):
                reason = f"{name} is not a tensor"
                break
            if cu.dtype != torch.int32:
                reason = f"{name} must be int32, got {cu.dtype}"
                break
            if cu.device != device:
                reason = f"{name} is on {cu.device}, expected {device}"
                break
            if cu.ndim != 1 or cu.numel() < 2:
                reason = f"{name} must be a 1D tensor with at least two entries"
                break
            if not cu.is_contiguous():
                reason = f"{name} must be contiguous"
                break
        if reason is None and cu_q.numel() != cu_k.numel():
            reason = f"cu_q/cu_k segment counts differ: {cu_q.numel() - 1} vs {cu_k.numel() - 1}"

    if reason is None:
        # One synchronized copy validates all offsets. Never let an unchecked
        # CUDA index tensor reach FlashAttention merely to save this per-step
        # microsecond-scale guard.
        q_offsets = meta["cu_q"].detach().cpu().tolist()
        k_offsets = meta["cu_k"].detach().cpu().tolist()
        q_lens = [b - a for a, b in zip(q_offsets, q_offsets[1:])]
        k_lens = [b - a for a, b in zip(k_offsets, k_offsets[1:])]
        # Retain a host-only forensic snapshot. If a CUDA kernel reports an
        # asynchronous illegal address at the following collective consensus,
        # the CUDA context is already poisoned and copying cu_seqlens there is
        # no longer possible. This compact snapshot lets that error name the
        # exact shape without adding another synchronization on the hot path.
        meta["_validation_snapshot"] = {
            "n_real": n_real,
            "query_len": query_len,
            "key_len": key_len,
            "s_first": s_first,
            "real_end": real_end,
            "max_q": max_q,
            "max_k": max_k,
            "first_q": int(meta.get("first_q", 0)),
            "first_k": int(meta.get("first_k", 0)),
            "max_tail": int(meta.get("max_tail", 0)),
            "cu_q": q_offsets,
            "cu_k": k_offsets,
        }
        if q_offsets[0] != 0 or k_offsets[0] != 0:
            reason = "cu_q and cu_k must start at zero"
        elif any(length <= 0 for length in q_lens + k_lens):
            reason = "cu_q/cu_k must be strictly increasing"
        elif q_offsets[-1] != n_real:
            reason = f"cu_q[-1]={q_offsets[-1]} does not equal n_real={n_real}"
        elif k_offsets[-1] != real_end - s_first:
            reason = f"cu_k[-1]={k_offsets[-1]} does not equal K/V slice length {real_end - s_first}"
        elif k_lens[0] < q_lens[0] or k_lens[1:] != q_lens[1:]:
            reason = "only the first local document may have more K/V than Q tokens"
        elif max_q != max(q_lens) or max_k != max(k_lens):
            reason = f"max_seqlen mismatch: supplied q/k={max_q}/{max_k}, actual={max(q_lens)}/{max(k_lens)}"

    cache[signature] = reason
    return reason


def _te_varlen_dpa(num_q_heads, num_kv_heads, head_dim, scale, device, dtype):
    """Cached TE DotProductAttention module for thd/varlen block-diagonal CP attention."""
    key = (num_q_heads, num_kv_heads, head_dim, round(float(scale), 10), str(device), str(dtype))
    m = _TE_DPA_CACHE.get(key)
    if m is None:
        from transformer_engine.pytorch import DotProductAttention

        # bottom_right: the local query rows are a SUFFIX of their document's keys
        # (the left-straddling doc has sk > sq), so causal must align to the
        # bottom-right corner -- matches flash's bottom-right and the dense
        # global-position causal. padding_causal (top-left) is WRONG here.
        m = DotProductAttention(
            num_attention_heads=num_q_heads,
            kv_channels=head_dim,
            num_gqa_groups=num_kv_heads,
            attn_mask_type="padding_causal_bottom_right",
            qkv_format="thd",
            softmax_scale=scale,
        ).to(device)
        _TE_DPA_CACHE[key] = m
    return m


def _flash_varlen_with_long_prefix_guard(
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    local_query_len: int,
    scale,
    dropout_p: float,
    meta: dict,
):
    """Run FlashAttention without an asymmetric-prefix varlen launch.

    In CP, only the first local document can have ``K > Q``: it may include a
    left halo from the preceding rank. Some FlashAttention builds have produced
    an asynchronous illegal access for long packs with this layout.
    Fixed-sequence Flash accepts the same bottom-right causal Q/K layout without
    consulting ``cu_seqlens``. Peel off just that one asymmetric segment, then
    process all remaining (ordinary ``K == Q``) documents in one varlen call.

    Normal packs stay on the original single-call path. A guarded pack adds at
    most one kernel launch, independent of its number of packed documents.

    Args:
        q_packed: Packed local queries ``[n_real, Hq, D]`` (``n_real`` = real
            local query tokens, ``Hq`` = query heads, ``D`` = head dim).
        k_packed: Packed keys ``[T_k, Hkv, D]`` covering the needed K/V range.
        v_packed: Packed values ``[T_k, Hkv, D]``; same layout as ``k_packed``.
        cu_q: Per-document cumulative query offsets, 1D int32 ``[n_docs + 1]``.
        cu_k: Per-document cumulative key offsets, 1D int32 ``[n_docs + 1]``.
        max_q: Maximum per-document query segment length.
        max_k: Maximum per-document key segment length.
        local_query_len: This rank's local sequence length (for diagnostics).
        scale: Softmax scale (``None`` -> kernel default ``D**-0.5``).
        dropout_p: Attention dropout probability.
        meta: Per-step metadata carrying ``first_q``/``first_k``/``max_tail``.

    Returns:
        Packed attention output ``[n_real, Hq, D]``.
    """
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    first_q = int(meta.get("first_q", 0))
    first_k = int(meta.get("first_k", 0))
    if first_q <= 0:
        first_q = int(cu_q[1].item())
    if first_k <= 0:
        first_k = int(cu_k[1].item())

    # Keep every symmetric segment on the original one-call fast path. Peel
    # off any left-straddling prefix, not just a >local-shard instance: the
    # extra launch is bounded to one per layer/rank, and this avoids relying
    # on an unknown size threshold inside the external FlashAttention build.
    if first_k <= first_q:
        return flash_attn_varlen_func(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=True,
            deterministic=_CP_FLASH_DETERMINISTIC,
        )

    global _CP_FLASH_LONG_SEGMENT_WARNED
    if not _CP_FLASH_LONG_SEGMENT_WARNED:
        logger.warning(
            "Flash varlen long-prefix guard engaged: first_q=%d first_k=%d local_len=%d; "
            "using fixed Flash for the boundary document and varlen Flash for the remaining documents",
            first_q,
            first_k,
            local_query_len,
        )
        _CP_FLASH_LONG_SEGMENT_WARNED = True

    first_out = flash_attn_func(
        q_packed[:first_q].unsqueeze(0),
        k_packed[:first_k].unsqueeze(0),
        v_packed[:first_k].unsqueeze(0),
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=True,
        deterministic=_CP_FLASH_DETERMINISTIC,
    ).squeeze(0)

    if cu_q.numel() == 2:
        return first_out

    tail_cu_q = (cu_q[1:] - first_q).contiguous()
    tail_cu_k = (cu_k[1:] - first_k).contiguous()
    tail_max = int(meta.get("max_tail", 0))
    if tail_max <= 0:
        tail_max = int((tail_cu_q[1:] - tail_cu_q[:-1]).max().item())
    tail_out = flash_attn_varlen_func(
        q_packed[first_q:],
        k_packed[first_k:],
        v_packed[first_k:],
        cu_seqlens_q=tail_cu_q,
        cu_seqlens_k=tail_cu_k,
        max_seqlen_q=tail_max,
        max_seqlen_k=tail_max,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=True,
        deterministic=_CP_FLASH_DETERMINISTIC,
    )
    return torch.cat((first_out, tail_out), dim=0)


def _cp_blockdiag_mask(
    doc_ids: torch.Tensor,
    row_offset: int,
    local_len: int,
    full_len: int,
    batch_size: int,
) -> torch.Tensor:
    """Per-document causal attention mask for block-diagonal CP, shape ``[B, 1, L, S]``.

    ``doc_ids`` is the full (all-rank, padded) per-position document index ``[B, S]``
    (0 == padding). Query rows are this rank's local positions
    ``[row_offset, row_offset+local_len)``; key columns span the full sequence. A
    query attends to a key iff they share a document, neither is padding, and the
    key is causally visible (global query position >= key position) -- identical to
    the block-causal mask a non-CP packed run would build for real tokens. The
    diagonal is always allowed so a query row is never fully masked (which the
    math/efficient SDPA backend turns into NaN); for padding rows this is a
    harmless self-edge whose output is dropped by the -100 labels.

    Args:
        doc_ids: Per-position document ids ``[B, S]`` or ``[S]`` (0 == padding),
            where ``B`` = batch and ``S`` = full padded sequence length.
        row_offset: Global position of this rank's first local query row.
        local_len: ``L``, the number of local query rows.
        full_len: ``S``, the number of key columns (full padded sequence).
        batch_size: ``B``, used to expand a 1D ``doc_ids``.

    Returns:
        Boolean allow-mask ``[B, 1, L, S]`` (True == may attend).
    """
    if doc_ids.dim() == 1:
        doc_ids = doc_ids.unsqueeze(0).expand(batch_size, -1)
    device = doc_ids.device
    L, S = local_len, full_len
    row_doc = doc_ids[:, row_offset : row_offset + L]  # [B, L]
    col_doc = doc_ids  # [B, S]
    same_doc = row_doc.unsqueeze(2) == col_doc.unsqueeze(1)  # [B, L, S]
    not_pad = (row_doc.unsqueeze(2) > 0) & (col_doc.unsqueeze(1) > 0)  # [B, L, S]
    row_pos = torch.arange(row_offset, row_offset + L, device=device).view(1, L, 1)
    col_pos = torch.arange(S, device=device).view(1, 1, S)
    causal = row_pos >= col_pos  # [1, L, S]
    # Always allow the diagonal (q_pos == k_pos) so every query attends to >=1 key even
    # in all-pad/empty rows -- prevents NaN/hang.
    self_diag = row_pos == col_pos  # [1, L, S]
    allow = (same_doc & not_pad & causal) | self_diag
    return allow.unsqueeze(1)  # [B, 1, L, S]


def _varlen_seg_for_rank(dids: torch.Tensor, row_offset: int, local_len: int, dev) -> dict | None:
    """Per-rank block-diagonal varlen segmentation for ONE packed sequence.

    Factoring the segmentation out of the per-layer hot path lets
    :func:`precompute_blockdiag_varlen_meta` run it once per step so the
    attention layers do zero GPU->CPU host syncs.

    Args:
        dids: Full (padded) per-position document id vector ``[S]`` (0 == pad).
        row_offset: Global position of this rank's first local query row.
        local_len: ``L``, the local query chunk length.
        dev: Device for the produced ``cu_q``/``cu_k`` tensors.

    Returns:
        The cu_seqlens / slice metadata dict for this rank's local query chunk
        ``[row_offset, row_offset + local_len)`` (keys ``n_real``, ``s_first``,
        ``real_end``, ``cu_q``, ``cu_k``, ``max_q``, ``max_k``, ``first_q``,
        ``first_k``, ``max_tail``), or ``None`` if the chunk is entirely padding.
    """
    local = dids[row_offset : row_offset + local_len]  # [L]
    n_real = int((local > 0).sum().item())
    if n_real == 0:
        return None
    real_dids = local[:n_real]  # padding is a tail -> real rows are the prefix

    # per-document local query segment lengths (run-length encode)
    if n_real == 1:
        seg_q = torch.ones(1, dtype=torch.long, device=dev)
    else:
        bnd = torch.nonzero(real_dids[1:] != real_dids[:-1], as_tuple=False).flatten() + 1
        edges = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=dev),
                bnd.to(torch.long),
                torch.tensor([n_real], dtype=torch.long, device=dev),
            ]
        )
        seg_q = edges[1:] - edges[:-1]

    # left-straddle: how far the first local doc extends before row_offset
    back = 0
    if row_offset > 0:
        d0 = local[0]
        prefix = dids[:row_offset]
        ne = torch.nonzero(prefix != d0, as_tuple=False)
        back = row_offset if ne.numel() == 0 else int(row_offset - 1 - ne.max().item())

    seg_k = seg_q.clone()
    seg_k[0] = seg_k[0] + back
    s_first = row_offset - back
    real_end = row_offset + n_real

    cu_q = torch.zeros(seg_q.numel() + 1, dtype=torch.int32, device=dev)
    cu_q[1:] = torch.cumsum(seg_q, 0).to(torch.int32)
    cu_k = torch.zeros(seg_k.numel() + 1, dtype=torch.int32, device=dev)
    cu_k[1:] = torch.cumsum(seg_k, 0).to(torch.int32)
    max_q = int(seg_q.max().item())
    max_k = int(seg_k.max().item())
    max_tail = int(seg_q[1:].max().item()) if seg_q.numel() > 1 else 0
    return {
        "n_real": n_real,
        "s_first": s_first,
        "real_end": real_end,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "max_q": max_q,
        "max_k": max_k,
        "first_q": int(seg_q[0].item()),
        "first_k": int(seg_k[0].item()),
        "max_tail": max_tail,
    }


def precompute_blockdiag_varlen_meta(doc_ids: torch.Tensor, row_offset: int, local_len: int, device) -> dict:
    """Precompute this rank's varlen segmentation once per step.

    The block-diagonal varlen cu_seqlens depend only on ``(doc_ids, row_offset,
    local_len)`` -- all step-constant -- yet rebuilding them inline on every
    attention layer (forward + AC recompute) costs 3-4 host syncs per rebuild.
    :func:`~nemo_automodel.components.distributed.blockdiag_cp.batch.make_cp_blockdiag_batch_and_ctx`
    calls this once and stashes the result in the CP state; ``_cp_blockdiag_varlen(meta=...)``
    then runs with zero ``.item()`` syncs.

    Assumes one packed sequence per rank (B == 1, the CP contract).

    Args:
        doc_ids: Per-position document ids ``[B, S]`` (row 0 used) or ``[S]``
            (0 == padding), covering the full padded sequence.
        row_offset: Global position of this rank's first local query row.
        local_len: ``L``, the local query chunk length.
        device: Device for the produced ``cu_q``/``cu_k`` tensors.

    Returns:
        A dict consumed directly by ``_cp_blockdiag_varlen``; ``{"n_real": 0}``
        for an all-padding chunk (the varlen path then takes its grad-preserving
        0-weighted touch).
    """
    dids = (doc_ids if doc_ids.dim() == 1 else doc_ids[0]).to(device=device, dtype=torch.long)
    seg = _varlen_seg_for_rank(dids, row_offset, local_len, device)
    return seg if seg is not None else {"n_real": 0}


def _cp_blockdiag_varlen(
    query,
    key_full,
    value_full,
    doc_ids,
    row_offset,
    scale=None,
    backend="flash",
    *,
    dropout_p=0.0,
    meta=None,
):
    """Block-diagonal CP attention via varlen (flash or TE) -- no dense ``[B,1,L,S]`` mask.

    Equivalent to ``_cp_blockdiag_mask`` + SDPA for real (non-padding) query rows.
    The sequence is sharded contiguously and packing puts padding (doc id 0) only
    as a contiguous tail, so the local query's real rows are a prefix ``[0, n_real)``.
    For each document the local rows touch, we emit a varlen segment: q segment =
    the local rows in that doc, k segment = ``[doc_start, last_local_q+1)`` (the only
    doc with ``sk > sq`` straddles the left boundary). Bottom-right causal alignment
    reproduces same-doc + global-causal exactly. Padding query rows are returned as
    zeros (their loss is masked by -100 labels).

    Args:
        query: Local query shard ``[B, Hq, L, D]`` (``B`` = batch, ``Hq`` = query
            heads, ``L`` = local sequence length, ``D`` = head dim).
        key_full: Keys ``[B, Hkv, S, D]`` covering the K/V range indexed by
            ``meta``'s ``[s_first, real_end)`` slice (full sequence for the
            all-gather path, needed range for halo/A2A).
        value_full: Values ``[B, Hkv, S, D]``; same layout as ``key_full``.
        doc_ids: Per-position document ids ``[B, S_full]`` or ``[S_full]``
            (0 == padding) on the full padded sequence.
        row_offset: Global position of this rank's first local query row.
        scale: Softmax scale (``None`` -> kernel default ``D**-0.5``).
        backend: ``"flash"`` (flash_attn_varlen_func) or ``"te"``
            (TransformerEngine DotProductAttention thd).
        dropout_p: Attention dropout probability.
        meta: Optional per-step segmentation from
            :func:`precompute_blockdiag_varlen_meta`; rebuilt inline when absent.

    Returns:
        Attention output ``[B, Hq, L, D]``, or ``None`` to signal "fall back to
        the dense path" when the requested varlen kernel path is unavailable or
        unsupported for the supplied inputs.
    """
    global _CP_FLASH_WARNED, _CP_TE_DROPOUT_WARNED
    if query.dtype not in (torch.float16, torch.bfloat16):
        if not _CP_FLASH_WARNED:
            logger.warning(
                "varlen CP attention needs fp16/bf16, got %s; reporting the unavailable varlen path to the caller",
                query.dtype,
            )
            _CP_FLASH_WARNED = True
        return None
    if backend == "te" and dropout_p > 0.0:
        if not _CP_TE_DROPOUT_WARNED:
            logger.warning(
                "TransformerEngine THD varlen attention does not support dropout_p=%s; "
                "reporting the unavailable varlen path so the caller preserves dropout via dense SDPA",
                dropout_p,
            )
            _CP_TE_DROPOUT_WARNED = True
        return None
    if backend == "flash" and not HAS_FLASH_VARLEN:
        if not _CP_FLASH_WARNED:
            logger.warning("flash_attn is unavailable; reporting the unavailable varlen path to the caller")
            _CP_FLASH_WARNED = True
        return None

    B, Hq, L, D = query.shape
    Hkv = key_full.shape[1]
    if doc_ids.dim() == 1:
        doc_ids = doc_ids.unsqueeze(0).expand(B, -1)
    dev = query.device
    out = torch.zeros_like(query)

    try:
        for b in range(B):
            # Use the per-step precomputed meta when available (B==1, the CP
            # contract) so the hot path does zero .item() host syncs; otherwise
            # rebuild inline for an unsupported multi-row/direct caller or when
            # metadata was not armed -- identical math.
            if meta is not None and B == 1:
                seg = meta
            else:
                seg = _varlen_seg_for_rank(doc_ids[b], row_offset, L, dev)
                if seg is None:
                    seg = {"n_real": 0}
            n_real = seg["n_real"]
            if n_real == 0:
                # All-padding local chunk: no kernel call. But the output MUST stay
                # attached to key_full/value_full, otherwise this CP rank skips the
                # all-gather's backward reduce_scatter while real-token ranks fire it
                # -> collective desync -> NCCL hang in backward (observed on cross-node
                # cp>1 where a whole rank-chunk lands in the pad tail; the dense path
                # never hangs because its self_diag diagonal always routes through
                # value_full). A 0-weighted touch keeps autograd symmetric.
                out[b] = out[b] + 0.0 * (key_full[b].sum() + value_full[b].sum()).to(out.dtype)
                continue  # output stays zeros numerically; grad path preserved

            s_first = seg["s_first"]
            real_end = seg["real_end"]
            cu_q = seg["cu_q"]
            cu_k = seg["cu_k"]
            max_q = seg["max_q"]
            max_k = seg["max_k"]

            metadata_reason = _varlen_metadata_unavailable_reason(
                seg,
                query_len=L,
                key_len=key_full.shape[2],
                device=dev,
            )
            if metadata_reason is not None:
                raise ValueError(f"unsafe varlen metadata: {metadata_reason}")

            if key_full.shape != value_full.shape:
                raise ValueError(f"K/V shapes differ: {tuple(key_full.shape)} vs {tuple(value_full.shape)}")
            if key_full.device != dev or value_full.device != dev:
                raise ValueError("Q/K/V must be on the same device")
            if key_full.dtype != query.dtype or value_full.dtype != query.dtype:
                raise ValueError("Q/K/V must have the same dtype")
            if Hq % Hkv != 0:
                raise ValueError(f"query heads {Hq} are not divisible by KV heads {Hkv}")
            if D != key_full.shape[-1] or D != value_full.shape[-1]:
                raise ValueError("Q/K/V head dimensions differ")

            global _CP_VARLEN_SHAPE_LOGGED
            if not _CP_VARLEN_SHAPE_LOGGED:
                logger.info(
                    "first varlen shape: backend=%s q_tokens=%d kv_tokens=%d segments=%d max_q=%d max_k=%d "
                    "heads=%d/%d head_dim=%d slice=[%d,%d)",
                    backend,
                    n_real,
                    real_end - s_first,
                    cu_q.numel() - 1,
                    max_q,
                    max_k,
                    Hq,
                    Hkv,
                    D,
                    s_first,
                    real_end,
                )
                _CP_VARLEN_SHAPE_LOGGED = True

            q_packed = query[b, :, :n_real, :].transpose(0, 1).contiguous()  # [n_real, Hq, D]
            k_packed = key_full[b, :, s_first:real_end, :].transpose(0, 1).contiguous()  # [n_real+back, Hkv, D]
            v_packed = value_full[b, :, s_first:real_end, :].transpose(0, 1).contiguous()

            if backend == "te":
                dpa = _te_varlen_dpa(
                    Hq,
                    Hkv,
                    D,
                    scale if scale is not None else D**-0.5,
                    dev,
                    query.dtype,
                )
                o = dpa(
                    q_packed,
                    k_packed,
                    v_packed,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_k,
                    max_seqlen_q=max_q,
                    max_seqlen_kv=max_k,
                )  # [n_real, Hq*D] or [n_real, Hq, D]
                o = o.view(n_real, Hq, D)
            else:
                o = _flash_varlen_with_long_prefix_guard(
                    q_packed,
                    k_packed,
                    v_packed,
                    cu_q=cu_q,
                    cu_k=cu_k,
                    max_q=max_q,
                    max_k=max_k,
                    local_query_len=L,
                    scale=scale,
                    dropout_p=dropout_p,
                    meta=seg,
                )  # [n_real, Hq, D]
            out[b, :, :n_real, :] = o.transpose(0, 1)
    except torch.cuda.OutOfMemoryError:
        # Never signal fallback on OOM: the dense-mask fallback path allocates
        # strictly more than the varlen kernel that just failed, so it is
        # doomed and misattributes the failure site. Surface the real OOM site.
        raise
    except Exception as e:
        if not _CP_FLASH_WARNED:
            logger.warning(
                "CP varlen backend '%s' failed (%s: %s); reporting failure to the caller",
                backend,
                type(e).__name__,
                str(e)[:100],
            )
            _CP_FLASH_WARNED = True
        return None

    global _CP_FLASH_ENGAGED
    if not _CP_FLASH_ENGAGED:
        logger.info("block-diagonal CP attention using VARLEN backend=%s (no dense [B,1,L,S] mask)", backend)
        _CP_FLASH_ENGAGED = True
    return out
