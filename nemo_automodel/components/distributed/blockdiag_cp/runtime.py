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

"""Block-diagonal SDPA routing and collective-safe fallback policy."""

from __future__ import annotations

import logging
import os

import torch

from nemo_automodel.components.distributed.blockdiag_cp import exchange, kernels
from nemo_automodel.components.distributed.blockdiag_cp import state as state_module

logger = logging.getLogger(__name__)
_KV_EXCHANGE_PATH_LOGGED = False
_KV_XNODE_LOGGED = False


def _cp_group_spans_nodes(group) -> bool:
    """True if the CP group spans more than one node.

    The needed-only exchanges (halo neighbor-p2p, a2a all-to-all-v) are verified when
    the CP group is node-local (cp_size <= GPUs/node, the common mesh layout where cp
    is the fast dim). Cross-node needed-only exchange is still fabric-sensitive, so
    the production default falls back to NCCL all-gather when the CP group spans nodes.
    Heuristic: cp_size > the local node's world size (LOCAL_WORLD_SIZE /
    NPROC_PER_NODE / visible GPUs).

    Set ``NEMO_CP_ALLOW_XNODE=1`` to force the needed-only path cross-node for targeted
    validation on a specific cluster.
    """
    if os.environ.get("NEMO_CP_ALLOW_XNODE") == "1":
        return False
    world = torch.distributed.get_world_size(group)
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPROC_PER_NODE") or 0)
    if local_ws <= 0:
        try:
            local_ws = torch.cuda.device_count()
        except Exception:
            local_ws = world
    spans = local_ws > 0 and world > local_ws
    global _KV_XNODE_LOGGED
    if spans and not _KV_XNODE_LOGGED:
        _KV_XNODE_LOGGED = True
        logger.warning(
            "needed-only KV exchange disabled: CP group (cp_size=%d) spans nodes "
            "(local_world_size=%d) -> using all-gather; node-local CP is unaffected.",
            world,
            local_ws,
        )
    return spans


def _resolve_cp_varlen_config(state: dict) -> tuple[str, str]:
    """Return ``(attn_backend, kv_exchange)`` for this step.

    Prefers the per-step snapshot threaded into ``state`` by the batch/ctx builder (so
    :func:`cp_blockdiag_sdpa` is a function of ``(qkv, state)`` on the production path),
    and falls back to the runtime configuration owned by
    :mod:`~nemo_automodel.components.distributed.blockdiag_cp.state` for callers that
    set the step state manually (e.g. parity tests).
    """
    return (
        state.get("attn_backend", state_module._CP_ATTN_BACKEND),
        state.get("kv_exchange", state_module._CP_KV_EXCHANGE),
    )


def _needed_only_preflight(
    state: dict,
    group,
    backend: str,
    query_dtype: torch.dtype,
    device: torch.device,
    *,
    query_len: int,
    key_len: int,
) -> tuple[bool, str | None]:
    """Collectively decide whether every CP rank can run the varlen kernel."""
    cache_key = (backend, str(query_dtype), str(device), int(query_len), int(key_len))
    cached = state.get("_needed_only_preflight")
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]

    local_reason = kernels._varlen_backend_unavailable_reason(backend, query_dtype, device)
    if local_reason is None:
        local_reason = kernels._varlen_metadata_unavailable_reason(
            state.get("varlen_meta"),
            query_len=query_len,
            key_len=key_len,
            device=device,
        )
    all_available = local_reason is None
    if torch.distributed.is_initialized() and torch.distributed.get_world_size(group) > 1:
        available = torch.tensor(int(all_available), dtype=torch.int32, device=device)
        torch.distributed.all_reduce(available, op=torch.distributed.ReduceOp.MIN, group=group)
        all_available = bool(available.item())

    reason = None
    if not all_available:
        reason = local_reason or "varlen kernel is unavailable on at least one CP rank"
    state["_needed_only_preflight"] = (cache_key, all_available, reason)
    return all_available, reason


def _needed_only_kernel_succeeded_on_all_ranks(out, group, device, *, diagnostic=None) -> bool:
    """Make a rank-local varlen result safe to act on collectively.

    In particular, an all-padding rank returns a zero tensor without invoking
    FlashAttention/TE. It must nevertheless learn when a peer's real-token
    kernel returned ``None`` before either rank advances to another collective.
    """
    succeeded = out is not None
    if torch.distributed.is_initialized() and torch.distributed.get_world_size(group) > 1:
        try:
            status = torch.tensor(int(succeeded), dtype=torch.int32, device=device)
        except Exception as exc:
            # CUDA launches are asynchronous: a bad attention kernel commonly
            # surfaces here rather than at its Python call. Include the global
            # rank and kernel metadata so the originating rank stays visible.
            try:
                global_rank = torch.distributed.get_rank()
            except Exception:
                global_rank = -1
            logger.exception(
                "CP_NEEDED_ONLY_ASYNC_ERROR global_rank=%d device=%s error=%r diagnostic=%r",
                global_rank,
                device,
                exc,
                diagnostic,
            )
            raise
        torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MIN, group=group)
        succeeded = bool(status.item())
    return succeeded


def _select_kv_exchange_path(
    state,
    group,
    doc_ids,
    local_len,
    device,
    offset,
    *,
    query_dtype: torch.dtype | None = None,
    dropout_p: float = 0.0,
):
    """Decide this step's KV-exchange path and WHY.

    Every downgrade to all-gather names its cause (mode, kernel, missing varlen meta,
    or cross-node topology) so a silent fall-through can't hide a misconfiguration.

    Args:
        state: The per-step CP state dict (memoizes the plan and preflight).
        group: The CP process group.
        doc_ids: Replicated per-position document ids ``[B, S_full]`` (0 == pad).
        local_len: Per-rank local sequence length ``L``.
        device: Device for plan tensors and the preflight all-reduce.
        offset: Global position of this rank's first local query row.
        query_dtype: Query dtype for the kernel preflight (skip when ``None``).
        dropout_p: Attention dropout probability.

    Returns:
        ``(path, plan, reason)`` where ``path`` is ``"halo"``/``"a2a"``/``"allgather"``,
        ``plan`` is the cached block-diagonal KV plan for halo/a2a (else ``None``),
        and ``reason`` names why the path was chosen.
    """
    attn_backend, kv_exchange = _resolve_cp_varlen_config(state)
    if kv_exchange not in ("halo", "a2a"):
        return "allgather", None, f"mode={kv_exchange}"
    if attn_backend not in ("flash", "te"):
        return "allgather", None, f"needed-only requires flash/te kernel, got {attn_backend}"
    if attn_backend == "te" and dropout_p > 0.0:
        return "allgather", None, "TE THD varlen dropout requires the dense fallback"
    if state.get("varlen_meta") is None:
        return "allgather", None, "no varlen_meta (dense / non-varlen step)"
    if _cp_group_spans_nodes(group):
        return "allgather", None, "CP group spans nodes (no GPU<->NIC P2P)"
    if query_dtype is not None:
        available, unavailable_reason = _needed_only_preflight(
            state,
            group,
            attn_backend,
            query_dtype,
            device,
            query_len=local_len,
            key_len=doc_ids.shape[-1],
        )
        if not available:
            return "allgather", None, f"needed-only preflight failed: {unavailable_reason}"
    plan = state.get("kv_plan")
    if plan is None:
        world = torch.distributed.get_world_size(group)
        plan = exchange._compute_blockdiag_kv_plan(doc_ids, world, local_len, device)
        plan["rank"] = offset // local_len if local_len else 0
        plan["world"] = world
        state["kv_plan"] = plan  # reused across layers + AC recompute this step
    if kv_exchange == "halo" and plan["use_halo"]:
        return "halo", plan, "neighbor p2p (doc<=2 ranks)"
    return "a2a", plan, "all-to-all-v (doc spans >2 ranks)"


def cp_blockdiag_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Block-diagonal context-parallel SDPA.

    Drop-in replacement for ``torch.nn.functional.scaled_dot_product_attention``
    while the context returned by
    :func:`~nemo_automodel.components.distributed.blockdiag_cp.batch.make_cp_blockdiag_batch_and_ctx`
    is active; a plain pass-through to stock SDPA otherwise. K/V are exchanged
    across the CP group (all-gather, or a needed-only halo/a2a exchange) and one
    local attention runs the local queries against the delivered keys with a
    per-document causal mask. The passed ``attn_mask`` / ``is_causal`` are ignored
    on the CP path -- masking is rebuilt from the document ids so packed sequences
    never attend across document boundaries.

    Args:
        query: This rank's LOCAL query shard ``[B, Hq, L, D]`` (``B`` = batch,
            ``Hq`` = query heads, ``L`` = local sequence length, ``D`` = head dim).
        key: This rank's LOCAL key shard ``[B, Hkv, L, D]``.
        value: This rank's LOCAL value shard ``[B, Hkv, L, D]``.
        attn_mask: Ignored on the CP path (forwarded to stock SDPA otherwise).
        dropout_p: Dropout probability.
        is_causal: Ignored on the CP path (forwarded to stock SDPA otherwise).
        scale: Softmax scale (``None`` -> ``D**-0.5``).
        enable_gqa: Grouped-query attention flag as passed by HF's sdpa path.
        **kwargs: Ignored; accepted for SDPA signature compatibility.

    Returns:
        Attention output ``[B, Hq, L, D]`` for this rank's local rows.
    """
    step_state = state_module._CP_BLOCKDIAG_STATE.get()
    if step_state is None:
        # No CP active (e.g. cp_size==1 fall-through) -- behave like stock SDPA.
        return _ORIGINAL_SDPA(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    state_module._CP_ATTN_FIRE_COUNT[0] += 1
    group = step_state["group"]
    doc_ids = step_state["doc_ids"]  # [B, S] full (padded) document ids; 0 == padding
    offset = step_state["row_offset"]  # global position of this rank's first local row
    seq_dim = 2
    attn_backend, _ = _resolve_cp_varlen_config(step_state)

    # Needed-only KV exchange (opt-in via kv_exchange='halo' or 'a2a') lets each rank
    # attend only local + required boundary K/V instead of the full all-gathered sequence.
    # The plan is computed once per step from replicated doc_ids and memoized in state.
    gm = step_state.get("varlen_meta")
    # Decide the KV-exchange path (needed-only halo/a2a vs full all-gather) explicitly,
    # logging why, including downgrades to all-gather for mode, kernel, missing meta, or
    # a cross-node CP group where needed-only exchange is disabled by default.
    path, plan, reason = _select_kv_exchange_path(
        step_state,
        group,
        doc_ids,
        query.shape[seq_dim],
        query.device,
        offset,
        query_dtype=query.dtype,
        dropout_p=dropout_p,
    )
    global _KV_EXCHANGE_PATH_LOGGED
    if not _KV_EXCHANGE_PATH_LOGGED:
        _KV_EXCHANGE_PATH_LOGGED = True
        # cp_size is informational; the all-gather path may run without a live process
        # group (e.g. the in-process parity tests pass group=None), so look it up safely.
        cp_sz = torch.distributed.get_world_size(group) if torch.distributed.is_initialized() else -1
        logger.info(
            "KV exchange path=%s (%s; cp_size=%d, local_len=%d)",
            path,
            reason,
            cp_sz,
            query.shape[seq_dim],
        )
    if path in ("halo", "a2a"):
        if path == "halo":
            out = exchange._blockdiag_halo_attention(
                query,
                key,
                value,
                doc_ids,
                group,
                plan,
                gm,
                offset,
                scale,
                attn_backend,
                dropout_p,
            )
        else:
            out = exchange._blockdiag_a2a_attention(
                query,
                key,
                value,
                doc_ids,
                group,
                plan,
                gm,
                offset,
                scale,
                attn_backend,
                dropout_p,
            )
        if _needed_only_kernel_succeeded_on_all_ranks(
            out,
            group,
            query.device,
            diagnostic=gm.get("_needed_only_diagnostic") if gm is not None else None,
        ):
            assert out is not None
            return out
        # The needed-only KV collective has already executed. An all-padding
        # rank may have produced a local zero while a peer's real-token kernel
        # failed, so the consensus above intentionally makes every rank raise.
        raise RuntimeError(
            f"needed-only CP {path} exchange completed, but the {attn_backend} "
            "varlen kernel failed on at least one CP rank; all ranks are "
            "refusing an unsafe post-exchange full-allgather fallback"
        )

    # Use one all-gather of stacked [K;V] instead of two separate collectives
    # (halves collective launch/latency; the differentiable cat + the single
    # reduce_scatter on backward split cleanly back into the K and V grads).
    n_kv_heads_local = key.shape[1]
    kv_full = exchange._AllGatherSeqDiff.apply(torch.cat((key, value), dim=1), group, seq_dim)
    key_full = kv_full[:, :n_kv_heads_local]
    value_full = kv_full[:, n_kv_heads_local:]

    if attn_backend in ("flash", "te"):
        out = kernels._cp_blockdiag_varlen(
            query,
            key_full,
            value_full,
            doc_ids,
            offset,
            scale,
            backend=attn_backend,
            dropout_p=dropout_p,
            meta=step_state.get("varlen_meta"),
        )
        if out is not None:
            return out
        # out is None -> varlen backend unavailable / unsupported: fall through to dense.

    # GQA: HF's sdpa path passes enable_gqa=True with UN-repeated K/V when the
    # attention_mask is None (which it is here -- we rebuild masking ourselves).
    # But the memory-efficient SDPA backend does not support enable_gqa, so with
    # our custom 4D mask (flash excluded) the dispatcher would silently fall back
    # to the MATH kernel and materialize the full [B, H_q, L, S] fp32 score matrix
    # (OOM at long context). Expand K/V to the query head count here and disable
    # enable_gqa so the efficient kernel handles the masked attention.
    n_q_heads = query.shape[1]
    n_kv_heads = key_full.shape[1]
    if enable_gqa and n_kv_heads != n_q_heads:
        n_rep = n_q_heads // n_kv_heads
        key_full = key_full.repeat_interleave(n_rep, dim=1)
        value_full = value_full.repeat_interleave(n_rep, dim=1)
        enable_gqa = False

    B = query.shape[0]
    L = query.shape[seq_dim]
    S = key_full.shape[seq_dim]

    allow = kernels._cp_blockdiag_mask(doc_ids, offset, L, S, B)  # [B, 1, L, S]

    return _ORIGINAL_SDPA(
        query,
        key_full,
        value_full,
        attn_mask=allow,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        enable_gqa=enable_gqa,
    )


# Captured once at import so the routed function can delegate when CP is inactive.
_ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention
