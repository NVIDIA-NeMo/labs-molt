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

"""Local FlashAttention-2 compute for Gemma4 SLIDING-window CP layers.

Compute only: it takes the KV chunks the ring already collected
(:func:`cp_attention._collect_ring_kv_chunks`) and runs one FlashAttention forward+backward
over the causal window neighborhood. Ring communication and gradient routing stay in
``cp_attention``'s :class:`_Gemma4LocalKernelRingAttention`, so this reuses the flex/FFPA
rings' exact comm -- no new collectives.

The causally-valid chunks (owners ``<= cp_rank``) concat into one neighborhood; the local
query shard is its suffix, so a single ``flash_attn`` call with bottom-right causal + left
window places the shard correctly (no online-softmax merge). Packed multi-doc uses THD
``cu_seqlens`` from :func:`_build_packed_ring_segments` (cached per step). Backward is
no-recompute (see ``_fa_ll_*`` below).

Sliding layers only: global (all-chunk) layers keep FFPA; vision-bidirectional masks keep flex.
"""

from __future__ import annotations

from typing import Any

import torch

from nemo_automodel.components.models.gemma4_moe.cp_attention import (
    _RING_SEGMENT_CACHE,
    _build_packed_ring_segments,
    _gather_thd,
    _ring_segment_set_generation,
    _scatter_thd,
)


def _scale_of(ctx: Any) -> float:
    return float(ctx.scale) if ctx.scale is not None else ctx.query.shape[-1] ** -0.5


def _fold_pad(ids: torch.Tensor, pm: torch.Tensor | None) -> torch.Tensor:
    """Fold padding into the doc map (pad -> id 0) so ``_build_packed_ring_segments``
    excludes it; replaces the flex path's separate padding_mask handling. Returns ``ids``
    unchanged (same object, cache-stable) when there is no padding_mask."""
    return ids.masked_fill(pm, 0) if pm is not None else ids


def _concat_neighborhood(ctx: Any, chunks: list):
    """Concatenate the causally-valid collected chunks (owners ``<= cp_rank``) ascending.

    ``chunks`` is :func:`cp_attention._collect_ring_kv_chunks` output
    ``(owner, key, value, metadata)``. Returns ``(k_nb, v_nb, nb_ids, valid_owners)`` -- the
    neighborhood K/V (BHSD), the concatenated padding-folded ``_packed_seq_ids`` (or ``None``
    if unpacked), and the ascending list of kept owner ranks (matching the chunk order in
    ``k_nb``). Owners ``> cp_rank`` (future / wrapped) are causally dropped here; their grads
    are zero-routed by the ring Function."""
    valid = sorted((c for c in chunks if c[0] <= ctx.cp_rank), key=lambda c: c[0])
    k_nb = torch.cat([c[1] for c in valid], dim=2)
    v_nb = torch.cat([c[2] for c in valid], dim=2)
    ids = [c[3].get("_packed_seq_ids") for c in valid]
    nb_ids = None
    if all(x is not None for x in ids):
        nb_ids = torch.cat([_fold_pad(c[3]["_packed_seq_ids"], c[3].get("padding_mask")) for c in valid], dim=1)
    return k_nb, v_nb, nb_ids, [c[0] for c in valid]


# ------------- FlashAttention-2 low-level (no-recompute backward) -------------
# Wired one level below flash_attn_{func,varlen_func}'s internal autograd.Function: the forward
# saves the kernel's backward context (out, softmax LSE, rng); the backward calls
# ``_wrapped_flash_attn_*_backward`` with it, so the ring Function does NOT re-run the forward.
# The dense/varlen fwd return ``(out, state)``; the matching bwd consumes that ``state``.
def _fa_ll_dense_fwd(q, k, v, scale, win):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward

    qb, kb, vb = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # BHSD -> BSHD
    out, lse, _s, rng = _wrapped_flash_attn_forward(
        qb,
        kb,
        vb,
        0.0,
        scale,
        causal=True,
        window_size_left=win - 1,
        window_size_right=0,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    return out.transpose(1, 2), {"qb": qb, "kb": kb, "vb": vb, "out": out, "lse": lse, "rng": rng}


def _fa_ll_dense_bwd(grad_out, st, scale, win):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward

    doutb = grad_out.transpose(1, 2).contiguous()  # BHSD -> BSHD
    dq, dk, dv = torch.empty_like(st["qb"]), torch.empty_like(st["kb"]), torch.empty_like(st["vb"])
    _wrapped_flash_attn_backward(
        doutb,
        st["qb"],
        st["kb"],
        st["vb"],
        st["out"],
        st["lse"],
        dq,
        dk,
        dv,
        0.0,
        scale,
        True,
        win - 1,
        0,
        0.0,
        None,
        False,
        rng_state=st["rng"],
    )
    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)  # -> BHSD


def _fa_ll_varlen_fwd(qp, kp, vp, seg, scale, win):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_forward

    out_p, lse, _s, rng = _wrapped_flash_attn_varlen_forward(
        qp,
        kp,
        vp,
        seg["cu_q"],
        seg["cu_k"],
        int(seg["max_q"]),
        int(seg["max_k"]),
        0.0,
        scale,
        causal=True,
        window_size_left=win - 1,
        window_size_right=0,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        block_table=None,
    )
    return out_p, {"qp": qp, "kp": kp, "vp": vp, "out_p": out_p, "lse": lse, "rng": rng}


def _fa_ll_varlen_bwd(dout_p, st, seg, scale, win):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_backward

    dqp, dkp, dvp = torch.empty_like(st["qp"]), torch.empty_like(st["kp"]), torch.empty_like(st["vp"])
    _wrapped_flash_attn_varlen_backward(
        dout_p,
        st["qp"],
        st["kp"],
        st["vp"],
        st["out_p"],
        st["lse"],
        dqp,
        dkp,
        dvp,
        seg["cu_q"],
        seg["cu_k"],
        int(seg["max_q"]),
        int(seg["max_k"]),
        0.0,
        scale,
        True,
        win - 1,
        0,
        0.0,
        None,
        False,
        rng_state=st["rng"],
    )
    return dqp, dkp, dvp


def _owner_grad_dict(per_owner: dict, *, cp_rank, cp_size, n_prior, zeros_like):
    """Full ``owner -> dK/dV`` map the ring routing expects: every owner ``(cp_rank-d)%cp_size``
    for ``d in 0..n_prior``, with zeros for owners this shard didn't causally attend."""
    zeros = None
    full = {}
    for d in range(0, n_prior + 1):
        owner = (cp_rank - d) % cp_size
        g = per_owner.get(owner)
        if g is None:
            if zeros is None:
                zeros = torch.zeros_like(zeros_like)
            g = zeros
        full[owner] = g.contiguous()
    return full


def sliding_ring_compute_fa_fwd(ctx: Any, chunks: list):
    """No-recompute FlashAttention sliding-window forward over the collected ring chunks, saving
    the kernel's backward context so the ring Function needs no forward recompute. Returns
    ``(out[B,Hq,sl,D], saved)``; pass ``saved`` to :func:`sliding_ring_compute_fa_bwd`."""
    q = ctx.query
    B, Hq, sl, D = q.shape
    scale = _scale_of(ctx)
    win = int(getattr(ctx.module, "sliding_window"))
    k_nb, v_nb, nb_ids, valid_owners = _concat_neighborhood(ctx, chunks)
    Hkv = k_nb.shape[1]
    saved = {
        "cp_rank": ctx.cp_rank,
        "cp_size": ctx.cp_size,
        "n_prior": len(chunks) - 1,
        "valid_owners": valid_owners,
        "B": B,
        "Hq": Hq,
        "Hkv": Hkv,
        "sl": sl,
        "D": D,
        "scale": scale,
        "win": win,
        "dtype": q.dtype,
    }

    if nb_ids is None:  # single document per shard -> dense
        out, state = _fa_ll_dense_fwd(q, k_nb, v_nb, scale, win)
        saved.update(kind="dense", state=state)
        return out, saved

    pm = ctx.metadata.get("padding_mask")
    q_ids = _fold_pad(ctx.metadata["_packed_seq_ids"], pm)
    seg = _cached_sliding_segments(ctx, q_ids, nb_ids, valid_owners[0])
    if seg is None:  # no shared document -> all-pad shard: zero output, zero grads
        saved.update(kind="empty")
        return torch.zeros_like(q), saved
    qp = _gather_thd(q, seg["q_index"])
    kp = _gather_thd(k_nb, seg["k_index"])
    vp = _gather_thd(v_nb, seg["k_index"])
    out_p, state = _fa_ll_varlen_fwd(qp, kp, vp, seg, scale, win)
    out = _scatter_thd(out_p.to(q.dtype), seg["q_index"], B, Hq, sl)
    pad = seg.get("pad_rows")
    if pad is not None:
        out = out.masked_fill(pad[:, None, :, None], 0)
    saved.update(kind="varlen", state=state, seg=seg)
    return out, saved


def sliding_ring_compute_fa_bwd(saved: dict, grad_out: torch.Tensor):
    """Backward of :func:`sliding_ring_compute_fa_fwd` using FlashAttention's own backward (no
    recompute). Returns ``(grad_q[B,Hq,sl,D], grad_key_by_owner, grad_value_by_owner)`` --
    per-owner dK/dV maps ready for :func:`cp_attention._route_kv_grads_to_owners`."""
    kind = saved["kind"]
    B, Hq, Hkv, sl, D = saved["B"], saved["Hq"], saved["Hkv"], saved["sl"], saved["D"]
    scale, win, dtype = saved["scale"], saved["win"], saved["dtype"]
    zeros_like_chunk = grad_out.new_zeros(B, Hkv, sl, D)
    owner_kw = dict(cp_rank=saved["cp_rank"], cp_size=saved["cp_size"], n_prior=saved["n_prior"])

    if kind == "empty":  # all-pad shard: zero everything (routing still runs, rank-uniform)
        grad_q = grad_out.new_zeros(B, Hq, sl, D)
        gk = _owner_grad_dict({}, zeros_like=zeros_like_chunk, **owner_kw)
        return grad_q, gk, _owner_grad_dict({}, zeros_like=zeros_like_chunk, **owner_kw)

    if kind == "dense":
        dq, dk_nb, dv_nb = _fa_ll_dense_bwd(grad_out, saved["state"], scale, win)
    else:  # varlen
        seg = saved["seg"]
        dout_p = _gather_thd(grad_out, seg["q_index"])
        dqp, dkp, dvp = _fa_ll_varlen_bwd(dout_p, saved["state"], seg, scale, win)
        nb_len = len(saved["valid_owners"]) * sl
        dq = _scatter_thd(dqp, seg["q_index"], B, Hq, sl)
        dk_nb = _scatter_thd(dkp, seg["k_index"], B, Hkv, nb_len)
        dv_nb = _scatter_thd(dvp, seg["k_index"], B, Hkv, nb_len)

    per_owner_k, per_owner_v = {}, {}
    for i, owner in enumerate(saved["valid_owners"]):
        per_owner_k[owner] = dk_nb[:, :, i * sl : (i + 1) * sl, :]
        per_owner_v[owner] = dv_nb[:, :, i * sl : (i + 1) * sl, :]
    grad_k = _owner_grad_dict(per_owner_k, zeros_like=zeros_like_chunk, **owner_kw)
    grad_v = _owner_grad_dict(per_owner_v, zeros_like=zeros_like_chunk, **owner_kw)
    return dq.to(dtype), grad_k, grad_v


def _cached_sliding_segments(ctx: Any, folded_q_ids: torch.Tensor, folded_nb_ids: torch.Tensor, lo: int):
    """Per-step cache around the neighborhood segment build.

    The segment depends only on the doc maps + ring geometry, IDENTICAL across every sliding
    layer in a step (same ``_packed_seq_ids``, same ``sliding_window`` => same ``lo``) and
    forward/backward -- so the ``.tolist()`` D->H sync + Python pairing in
    :func:`_build_packed_ring_segments` runs ONCE per step, not once per sliding layer per
    pass (incl. activation-checkpoint recompute). Anchored on the *persistent*
    ``_packed_seq_ids`` tensor, shared with the FFPA ring's cache. ``None`` is cached too."""
    _ring_segment_set_generation(ctx.metadata.get("_packed_seq_ids"))
    key = (
        "gemma4_sliding_nb",
        ctx.cp_rank,
        lo,
        ctx.cp_size,
        folded_q_ids.shape[0],
        folded_q_ids.shape[1],
        folded_nb_ids.shape[1],
    )
    if key in _RING_SEGMENT_CACHE:
        return _RING_SEGMENT_CACHE[key]
    seg = _build_packed_ring_segments(folded_q_ids, folded_nb_ids)
    if len(_RING_SEGMENT_CACHE) >= 256:
        _RING_SEGMENT_CACHE.pop(next(iter(_RING_SEGMENT_CACHE)))
    _RING_SEGMENT_CACHE[key] = seg
    return seg
