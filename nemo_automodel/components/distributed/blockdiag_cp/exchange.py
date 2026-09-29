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

"""Differentiable K/V collectives and needed-only exchange plans."""

from __future__ import annotations

import torch

from nemo_automodel.components.distributed.blockdiag_cp.kernels import _cp_blockdiag_varlen, _varlen_seg_for_rank


class _AllGatherSeqDiff(torch.autograd.Function):
    """All-gather a per-rank sequence shard along ``seq_dim`` over ``group``.

    Forward concatenates every rank's shard ``[B, H, L, D]`` in rank order
    (sequential sharding), producing the full sequence ``[B, H, L*world, D]``.
    Backward reduce-scatters the incoming gradient: each K/V position is read by
    every rank's queries, so its gradient is the sum over ranks of the per-rank
    grad slice; reduce-scatter(SUM) hands each rank the summed gradient for the
    shard it owns. All shards have equal length ``L`` (the sequence is padded to
    a multiple of the CP world size before sharding).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, seq_dim: int) -> torch.Tensor:
        """All-gather ``x`` (this rank's shard) into the full sequence along ``seq_dim``."""
        ctx.group = group
        ctx.seq_dim = seq_dim
        world = torch.distributed.get_world_size(group)
        ctx.world = world
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(world)]
        torch.distributed.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=seq_dim)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Reduce-scatter(SUM) the full-sequence gradient back to this rank's shard."""
        chunks = [c.contiguous() for c in grad_out.chunk(ctx.world, dim=ctx.seq_dim)]
        local = torch.empty_like(chunks[0])
        torch.distributed.reduce_scatter(local, chunks, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return local, None, None


def _compute_blockdiag_kv_plan(doc_ids_full: torch.Tensor, world: int, local_len: int, dev) -> dict:
    """Global left-halo exchange plan, computed from replicated ``doc_ids``.

    For each rank r the block-diagonal path needs the contiguous K/V range
    ``[s_first_r, real_end_r)`` = its boundary document's straddle (owned by ranks < r)
    plus its own local chunk. ``back_r = r*local_len - s_first_r`` is the straddle
    length. If ``back_r <= local_len`` for ALL ranks, every straddle fits entirely in
    the LEFT neighbor (rank r-1), so a single neighbor p2p suffices (``use_halo``). If
    any document spans >2 ranks (``back_r > local_len`` -- e.g. one doc across the whole
    sequence) the caller falls back to the general all-to-all-v. Every rank computes
    this from the same replicated ``doc_ids`` and therefore agrees on ``use_halo`` and
    on all send/recv counts without communication.

    Args:
        doc_ids_full: Replicated per-position document ids ``[B, S_full]`` (row 0
            used) or ``[S_full]`` (0 == padding) on the full padded sequence.
        world: CP world size.
        local_len: Per-rank local sequence length ``L`` (``S_full == world * L``).
        dev: Device used for the intermediate segmentation tensors.

    Returns:
        A plan dict with per-rank lists ``recv`` (tokens received from r-1 ==
        back_r), ``send`` (tokens sent to r+1 == back_{r+1}; ``send[-1]=0``,
        ``recv[0]=0``), ``nreal``, ``s_first``, ``real_end``, and the boolean
        ``use_halo``.
    """
    dids = doc_ids_full if doc_ids_full.dim() == 1 else doc_ids_full[0]
    backs, nreals, s_first, real_end = [], [], [], []
    for rr in range(world):
        seg = _varlen_seg_for_rank(dids, rr * local_len, local_len, dev)
        if seg is None:
            backs.append(0)
            nreals.append(0)
            s_first.append(rr * local_len)  # empty needed range
            real_end.append(rr * local_len)
        else:
            backs.append(rr * local_len - seg["s_first"])
            nreals.append(seg["n_real"])
            s_first.append(seg["s_first"])
            real_end.append(seg["real_end"])
    use_halo = all(b <= local_len for b in backs)
    send = backs[1:] + [0]  # rank rr sends back_{rr+1} to rr+1
    return {
        "use_halo": use_halo,
        "recv": backs,
        "send": send,
        "nreal": nreals,
        "s_first": s_first,
        "real_end": real_end,
    }


class _LeftHaloExchange(torch.autograd.Function):
    """Uniform-size single-step neighbor exchange for node-local halo mode.

    Every rank sends the last ``halo_size`` tokens of its chunk ``[B, C, L, D]`` to
    ``next_peer`` = (rank+1)%world and receives ``halo_size`` tokens from
    ``prev_peer`` = (rank-1)%world, send-before-recv, in one ``batch_isend_irecv``.
    The caller slices the last ``recv_count`` of the received block as its real
    straddle (rank 0's wraparound block and any over-send are ignored).
    Differentiable: backward is the reverse exchange (send grad to prev, recv from
    next).
    """

    @staticmethod
    def forward(ctx, x_local, group, halo_size, prev_peer, next_peer):
        """Exchange the chunk suffix; ``x_local`` is ``[B, C, L, D]``, returns ``[B, C, halo_size, D]``."""
        ctx.group = group
        ctx.halo_size = halo_size
        ctx.prev_peer = prev_peer
        ctx.next_peer = next_peer
        ctx.shape = tuple(x_local.shape)
        B, C, L, D = x_local.shape
        send_buf = x_local[:, :, L - halo_size :, :].contiguous()  # suffix sent to next
        recv_buf = torch.empty(B, C, halo_size, D, dtype=x_local.dtype, device=x_local.device)
        ops = [
            torch.distributed.P2POp(torch.distributed.isend, send_buf, next_peer, group=group),
            torch.distributed.P2POp(torch.distributed.irecv, recv_buf, prev_peer, group=group),
        ]
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()
        return recv_buf  # [B, C, halo_size, D] from prev rank's chunk suffix

    @staticmethod
    def backward(ctx, grad_recv):
        """Reverse exchange: route ``grad_recv`` ``[B, C, halo_size, D]`` back to the sending suffix."""
        B, C, L, D = ctx.shape
        hs = ctx.halo_size
        grad_send = torch.empty(B, C, hs, D, dtype=grad_recv.dtype, device=grad_recv.device)
        ops = [
            torch.distributed.P2POp(torch.distributed.isend, grad_recv.contiguous(), ctx.prev_peer, group=ctx.group),
            torch.distributed.P2POp(torch.distributed.irecv, grad_send, ctx.next_peer, group=ctx.group),
        ]
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()
        grad_x = grad_recv.new_zeros(B, C, L, D)
        grad_x[:, :, L - hs :, :] = grad_send  # grad for the suffix we sent to next
        return grad_x, None, None, None, None


def _blockdiag_halo_attention(query, key, value, doc_ids, group, plan, gm, row_offset, scale, backend, dropout_p):
    """Needed-only block-diagonal CP attention via the left-halo exchange.

    Each rank attends its local queries against ``[boundary-doc halo from rank-1] +
    [local chunk]`` instead of the full all-gathered sequence. The caller must fail
    fast when this returns ``None``: the halo collective has already executed, so a
    rank-local all-gather fallback would fork the CP group's collective order.

    Args:
        query: Local query shard ``[B, Hq, L, D]``.
        key: Local key shard ``[B, Hkv, L, D]``.
        value: Local value shard ``[B, Hkv, L, D]``.
        doc_ids: Replicated per-position document ids ``[B, S_full]`` (0 == pad).
        group: The CP process group.
        plan: Replicated halo plan from :func:`_compute_blockdiag_kv_plan`
            (augmented with ``rank``/``world``).
        gm: This rank's per-step varlen metadata (global ``s_first``/``real_end``).
        row_offset: Global position of this rank's first local query row.
        scale: Softmax scale (``None`` -> kernel default).
        backend: Varlen kernel backend, ``"flash"`` or ``"te"``.
        dropout_p: Attention dropout probability.

    Returns:
        Attention output ``[B, Hq, L, D]``, or ``None`` when the kernel is
        unavailable.
    """
    rank = plan["rank"]
    world = plan["world"]
    recv_count = plan["recv"][rank]
    # Uniform halo size across the whole ring (max straddle) so every rank posts
    # the same tensor shape.
    max_halo = max(plan["recv"]) if plan["recv"] else 0

    Hkv = key.shape[1]
    kv_local = torch.cat((key, value), dim=1)  # [B, 2*Hkv, L, D]
    if max_halo > 0:
        prev_peer = torch.distributed.get_global_rank(group, (rank - 1) % world)
        next_peer = torch.distributed.get_global_rank(group, (rank + 1) % world)
        recv_block = _LeftHaloExchange.apply(kv_local, group, max_halo, prev_peer, next_peer)  # [B,2Hkv,max_halo,D]
        halo = recv_block[:, :, max_halo - recv_count :, :]  # last recv_count == this rank's real straddle
    else:
        halo = kv_local[:, :, :0, :]  # no straddle anywhere -> empty
    kv_needed = torch.cat((halo, kv_local), dim=2)  # [B,2Hkv, recv_count+L, D] == global [s_first, (r+1)*L)
    key_needed = kv_needed[:, :Hkv]
    value_needed = kv_needed[:, Hkv:]

    n_real = gm["n_real"]
    if n_real == 0:
        # All-pad rank: no real queries. Keep the halo subgraph live (0-weighted touch)
        # so its backward fires symmetrically with the neighbor (counts are 0 here, so
        # both sides post nothing -- no hang). Output stays zeros.
        out = torch.zeros_like(query)
        return out + 0.0 * (key_needed.sum() + value_needed.sum()).to(out.dtype)

    # cu_seqlens are relative to s_first already; only the SLICE base changes: key_needed
    # IS the [s_first, real_end) slice, so index it at [0, recv+n_real).
    local_meta = dict(gm)
    local_meta["s_first"] = 0
    local_meta["real_end"] = recv_count + n_real
    gm.setdefault(
        "_needed_only_diagnostic",
        {
            "path": "halo",
            "rank": rank,
            "world": world,
            "row_offset": row_offset,
            "recv_count": recv_count,
            "max_halo": max_halo,
            "key_needed_len": int(key_needed.shape[2]),
            "kernel_meta": gm.get("_validation_snapshot"),
        },
    )
    return _cp_blockdiag_varlen(
        query,
        key_needed,
        value_needed,
        doc_ids,
        row_offset,
        scale,
        backend=backend,
        dropout_p=dropout_p,
        meta=local_meta,
    )


def _needed_kv_a2a_plan(plan: dict, rank: int, world: int, local_len: int, dev):
    """Per-(src,dst) split sizes + local gather index for the general needed-only
    all-to-all-v exchange (the >2-rank / single-doc case the halo can't cover).

    Each rank r needs the contiguous global range ``[s_first_r, real_end_r)``. As a
    SENDER, this rank s sends to each dst r the intersection of its owned chunk
    ``[s*L,(s+1)*L)`` with r's needed range (a token may go to several dsts, so the
    send buffer duplicates it once per destination). As a receiver, the pieces arrive
    in source order and concatenate into the contiguous needed range. All counts derive
    from the replicated plan, so every rank agrees.

    Args:
        plan: Replicated plan from :func:`_compute_blockdiag_kv_plan`.
        rank: This rank's index within the CP group.
        world: CP world size.
        local_len: Per-rank local sequence length ``L``.
        dev: Device for the produced ``send_index`` tensor.

    Returns:
        ``(in_splits, out_splits, send_index)``: per-destination send counts,
        per-source receive counts, and the 1D int64 local token indices (with
        duplication) to gather into the send buffer.
    """
    s_first = plan["s_first"]
    real_end = plan["real_end"]
    base = rank * local_len
    in_splits, out_splits = [], []
    send_idx = []
    for r in range(world):
        lo = max(base, s_first[r])
        hi = min(base + local_len, real_end[r])
        n = max(0, hi - lo)
        in_splits.append(n)
        if n > 0:
            send_idx.append(torch.arange(lo - base, hi - base, device=dev, dtype=torch.long))
    for s in range(world):
        lo = max(s * local_len, s_first[rank])
        hi = min((s + 1) * local_len, real_end[rank])
        out_splits.append(max(0, hi - lo))
    send_index = torch.cat(send_idx) if send_idx else torch.empty(0, dtype=torch.long, device=dev)
    return in_splits, out_splits, send_index


class _NeededKVExchange(torch.autograd.Function):
    """Differentiable multi-cast all-to-all-v K/V delivery.

    Delivers each rank exactly the K/V range it attends to, zero-redundantly (a
    source token is sent once per rank that needs it). Forward maps the local
    chunk ``[B, C, L, D]`` to the received needed range ``[B, C, R_recv, D]``.
    Backward scatter-ADDs the per-destination grads back to the source token
    (index_add), matching what the all-gather's reduce_scatter would sum.
    """

    @staticmethod
    def forward(ctx, x_local, group, in_splits, out_splits, send_index):
        """All-to-all-v ``x_local`` ``[B, C, L, D]`` into the needed range ``[B, C, R_recv, D]``."""
        ctx.group = group
        ctx.in_splits = in_splits
        ctx.out_splits = out_splits
        ctx.save_for_backward(send_index)
        ctx.local_len = x_local.shape[2]
        B, C, L, D = x_local.shape
        # gather (with duplication) the tokens to send, tokens-first for all_to_all_single
        xp = x_local.permute(2, 0, 1, 3).contiguous()  # [L, B, C, D]
        send_buf = xp.index_select(0, send_index).contiguous()  # [Tsend, B, C, D]
        recv_buf = send_buf.new_empty((sum(out_splits), B, C, D))
        torch.distributed.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=group,
        )
        return recv_buf.permute(1, 2, 0, 3).contiguous()  # [B, C, Rrecv, D]

    @staticmethod
    def backward(ctx, grad_out):
        """Reverse all-to-all-v of ``grad_out`` ``[B, C, R_recv, D]``; multi-cast grads accumulate."""
        (send_index,) = ctx.saved_tensors
        B, C, R, D = grad_out.shape
        gp = grad_out.permute(2, 0, 1, 3).contiguous()  # [Rrecv, B, C, D]
        grad_send = gp.new_empty((sum(ctx.in_splits), B, C, D))
        # reverse direction: swap in/out splits
        torch.distributed.all_to_all_single(
            grad_send,
            gp,
            output_split_sizes=ctx.in_splits,
            input_split_sizes=ctx.out_splits,
            group=ctx.group,
        )
        grad_x = gp.new_zeros((ctx.local_len, B, C, D))
        grad_x.index_add_(0, send_index, grad_send)  # multi-cast tokens accumulate
        return grad_x.permute(1, 2, 0, 3).contiguous(), None, None, None, None


def _blockdiag_a2a_attention(query, key, value, doc_ids, group, plan, gm, row_offset, scale, backend, dropout_p):
    """Needed-only block-diagonal CP attention via the general all-to-all-v exchange.

    Handles documents spanning >2 ranks (single long doc / unpacked long context)
    that the left-halo can't, still zero-redundantly (vs the full all-gather
    fallback). Same argument contract as :func:`_blockdiag_halo_attention`.

    Args:
        query: Local query shard ``[B, Hq, L, D]``.
        key: Local key shard ``[B, Hkv, L, D]``.
        value: Local value shard ``[B, Hkv, L, D]``.
        doc_ids: Replicated per-position document ids ``[B, S_full]`` (0 == pad).
        group: The CP process group.
        plan: Replicated plan from :func:`_compute_blockdiag_kv_plan`
            (augmented with ``rank``/``world``).
        gm: This rank's per-step varlen metadata (global ``s_first``/``real_end``).
        row_offset: Global position of this rank's first local query row.
        scale: Softmax scale (``None`` -> kernel default).
        backend: Varlen kernel backend, ``"flash"`` or ``"te"``.
        dropout_p: Attention dropout probability.

    Returns:
        Attention output ``[B, Hq, L, D]``, or ``None`` when the kernel is
        unavailable.
    """
    rank = plan["rank"]
    world = plan["world"]
    local_len = query.shape[2]
    in_splits, out_splits, send_index = _needed_kv_a2a_plan(plan, rank, world, local_len, query.device)

    Hkv = key.shape[1]
    kv_local = torch.cat((key, value), dim=1)  # [B, 2*Hkv, L, D]
    kv_needed = _NeededKVExchange.apply(kv_local, group, in_splits, out_splits, send_index)  # [B,2Hkv,need,D]
    key_needed = kv_needed[:, :Hkv]
    value_needed = kv_needed[:, Hkv:]

    n_real = gm["n_real"]
    if n_real == 0:
        out = torch.zeros_like(query)
        return out + 0.0 * (key_needed.sum() + value_needed.sum()).to(out.dtype)
    needed_len = plan["real_end"][rank] - plan["s_first"][rank]
    local_meta = dict(gm)
    local_meta["s_first"] = 0
    local_meta["real_end"] = needed_len
    gm.setdefault(
        "_needed_only_diagnostic",
        {
            "path": "a2a",
            "rank": rank,
            "world": world,
            "row_offset": row_offset,
            "needed_len": needed_len,
            "in_splits": list(in_splits),
            "out_splits": list(out_splits),
            "key_needed_len": int(key_needed.shape[2]),
            "kernel_meta": gm.get("_validation_snapshot"),
        },
    )
    return _cp_blockdiag_varlen(
        query,
        key_needed,
        value_needed,
        doc_ids,
        row_offset,
        scale,
        backend=backend,
        dropout_p=dropout_p,
        meta=local_meta,
    )
