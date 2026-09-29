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

"""Batch padding, sequential sharding, and block-diagonal CP context setup."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, ContextManager

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.blockdiag_cp import kernels
from nemo_automodel.components.distributed.blockdiag_cp import state as state_module
from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout


def _cp_blockdiag_doc_ids(batch: dict, seq_len: int, device, batch_size: int) -> torch.Tensor:
    """Resolve per-position document ids ``[B, S]`` (0 == padding) for the mask.

    Prefers the collator's ``_packed_seq_ids`` (1-based document index per token,
    present when a pack holds >1 document). Otherwise falls back to the 4-D
    block-causal ``attention_mask`` diagonal (valid positions) or, lacking both,
    treats the whole sequence as a single document.

    Args:
        batch: The training batch; may contain ``_packed_seq_ids`` ``[B, S]``
            (int document index per token) or ``attention_mask`` (``[B, 1, S, S]``
            block-causal bool, or ``[B, S]`` validity/indexed mask).
        seq_len: ``S``, the (unpadded) sequence length.
        device: Device for the returned tensor.
        batch_size: ``B``, used for the all-ones fallback.

    Returns:
        Per-position document ids ``[B, S]`` (int64, 0 == padding).
    """
    seq_ids = batch.get("_packed_seq_ids", None)
    if seq_ids is not None:
        return seq_ids.to(device=device, dtype=torch.long)
    attn = batch.get("attention_mask", None)
    if attn is not None and attn.dim() == 4:
        # [B, 1, S, S] block-causal bool -> diagonal gives per-position validity.
        valid = attn[:, 0].diagonal(dim1=-2, dim2=-1)  # [B, S]
        return valid.to(device=device, dtype=torch.long)
    if attn is not None and attn.dim() == 2:
        # [B, S] standard validity mask (0/1 or bool) or indexed packing ids.
        return attn.to(device=device, dtype=torch.long)
    return torch.ones(batch_size, seq_len, device=device, dtype=torch.long)


def make_cp_blockdiag_batch_and_ctx(
    cp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh | None,
    batch: dict[str, Any],
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
    shard_primary: bool = True,
) -> tuple[Callable[[], ContextManager], dict[str, Any], ShardLayout | None]:
    """Sequentially shard a batch for block-diagonal CP.

    Pads the sequence to a multiple of twice the CP world size, slices each
    selected sequence-aligned tensor to this rank's contiguous chunk, and
    returns a context whose lifetime activates per-document CP SDPA state.
    ``shard_primary=False`` leaves token ids or embeddings untouched for models
    that embed multimodal inputs inside ``forward``.

    Softmax attention must route through
    :func:`~nemo_automodel.components.distributed.blockdiag_cp.runtime.cp_blockdiag_sdpa`
    while this context is active. A model opts in by returning a
    :class:`~nemo_automodel.components.distributed.context_parallel.sharder.ContextParallelSharder`
    whose batch verb is this callable.

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        tp_mesh: Accepted for the shared sharder signature; unused
            (block-diagonal CP shards only the sequence dimension).
        batch: The training batch. Contains exactly one primary stream:
            ``inputs_embeds`` of shape ``[batch, sequence, hidden]`` or
            ``input_ids`` of shape ``[batch, sequence]``. The batch is mutated in
            place: ``attention_mask`` is dropped, ``padding_mask`` of shape
            ``[batch, sequence]`` (bool, True == pad) is added, and auxiliary
            sequence-aligned tensors are padded then sliced to this rank's
            ``[row_offset, row_offset + sequence/cp)`` chunk.
        loss_mask: Optional per-token loss mask ``[B, S]``; padded with 0 and
            sharded like the other sequence-aligned tensors (stored back into
            ``batch["loss_mask"]``).
        padding_token_id: Fill value for ``input_ids`` padding when the primary
            stream is sharded here.
        shard_primary: Whether to pad and shard the primary stream. Leave False
            for models that embed multimodal inputs and shard the resulting
            embeddings inside ``forward`` so FSDP hooks own the vision/embedding
            parameter lifecycle.

    Returns:
        ``(train_ctx, batch, layout)``: a zero-arg callable returning the
        per-step context manager, the sharded batch, and its contiguous
        :class:`~nemo_automodel.components.distributed.context_parallel.sharder.ShardLayout`.
    """
    from contextlib import nullcontext

    from torch.nn.attention import SDPBackend, sdpa_kernel

    world = cp_mesh.size()
    if world <= 1:
        primary = batch.get("inputs_embeds", batch.get("input_ids"))
        layout = None
        if primary is not None:
            layout = ShardLayout(original_seq_len=primary.shape[1], padded_seq_len=primary.shape[1])
        return nullcontext, batch, layout

    rank = cp_mesh.get_local_rank()
    group = cp_mesh.get_group()

    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    if has_inputs_embeds == has_input_ids:
        raise ValueError("block-diagonal CP requires exactly one of 'inputs_embeds' or 'input_ids' in the batch")
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary = batch[primary_key]
    B, S = primary.shape[0], primary.shape[1]
    device = primary.device

    # Resolve document ids BEFORE dropping attention_mask: for single-document /
    # text-only packs ``_packed_seq_ids`` is absent and the per-position validity
    # (real vs padding) comes from the 4D block-causal ``attention_mask`` diagonal.
    doc_ids = _cp_blockdiag_doc_ids(batch, S, device, B)  # [B, S] on full (unpadded) seq

    # The per-rank block-diagonal mask is rebuilt inside the CP SDPA from
    # ``doc_ids``; drop the (now stale, full-length) 4D mask so the model's own
    # mask machinery does not fire on the sharded local sequence.
    batch.pop("attention_mask", None)

    # Preserve the padding signal for MoE routing / load-balance statistics after
    # the full attention_mask is dropped (0 == padding in doc_ids). Built at the
    # unpadded length S so it shards through ``_shard`` like every other
    # sequence-aligned tensor -- the CP-pad tail is filled with
    # ``PAD_FILL["padding_mask"]=True`` there, matching ``doc_ids.eq(0)`` on the pad.
    batch["padding_mask"] = doc_ids.eq(0)

    if loss_mask is not None:
        batch["loss_mask"] = loss_mask

    # Match the in-forward contiguous primary sharder, which pads to
    # ``cp_size * 2`` so all model-owned CP layouts share one aux/primary shape
    # contract after the #2937 sharder refactor.
    pad_len = (-S) % (2 * world)
    if pad_len:
        doc_ids = torch.cat(
            [doc_ids, torch.zeros(B, pad_len, device=device, dtype=doc_ids.dtype)],
            dim=1,
        )
    S_full = S + pad_len
    local_len = S_full // world
    row_offset = rank * local_len

    # GatedDeltaNet's FLA CP context consumes GLOBAL packed boundaries. Compute
    # them once here, outside layer forwards and activation-checkpoint recompute,
    # so every recurrent layer observes the same document-reset contract without
    # introducing dynamic host synchronization between collectives.
    flat_doc_ids = doc_ids.reshape(-1)
    boundaries = torch.where(flat_doc_ids[1:] != flat_doc_ids[:-1])[0].to(torch.long) + 1
    packed_cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            boundaries,
            torch.tensor([flat_doc_ids.numel()], dtype=torch.long, device=device),
        )
    )
    packed_cu_seqlens_cpu = packed_cu_seqlens.detach().cpu()

    # Defense-in-depth (do not remove): the per-document SDPA all-gathers K/V over
    # ``group`` (``exchange._AllGatherSeqDiff``) to rebuild the full sequence, while
    # the sequence is sharded into ``world == cp_mesh.size()`` chunks above. The
    # all-gather group MUST therefore have exactly ``world`` ranks. On torch 2.8,
    # ``device_mesh["cp"].get_group()`` was mis-resolved on a mesh that carries a
    # flattened ``dp_shard_cp`` dim: with dp>1 it returned a dp*cp-sized group, so
    # the all-gather over-gathered -> ``key_full`` was dp* too long -> a cryptic
    # shape crash deep in the backward AC-recompute. Later torch versions resolve
    # it correctly. Fail loud and early here instead of crashing in backward.
    _group_world = torch.distributed.get_world_size(group)
    if _group_world != world:
        raise RuntimeError(
            f"block-diagonal CP: K/V all-gather group world ({_group_world}) != "
            f"cp_mesh.size() ({world}). The cp sub-group is mis-resolved -- a DeviceMesh "
            f"flatten/slice bug seen on torch 2.8 when dp>1; upgrade torch."
        )

    # Block-diagonal CP shards ONE packed sequence per rank and assumes local batch
    # B==1 (packing collapses many samples into a single sequence). B>1 is
    # unsupported: the deepstack visual-embed sharding below indexes batch row 0
    # (``vpm[0, ...]``), so B>1 would die with a cryptic size mismatch deep in the
    # model. Fail loud here.
    _cp_bsz = primary.shape[0]
    if _cp_bsz != 1:
        raise ValueError(
            f"block-diagonal context parallelism requires local_batch_size=1 (one packed "
            f"sequence per rank), got batch size {_cp_bsz}. Enable packing and set "
            f"the local batch size to 1."
        )

    # Per-tensor padding fill: each tensor's "ignore" value is semantic, not
    # dtype-derived (mirrors make_cp_batch_and_ctx's PAD_FILL).
    PAD_FILL = {
        "input_ids": padding_token_id,
        "labels": -100,
        "_packed_seq_ids": 0,
        "loss_mask": 0,
        "padding_mask": True,
        "visual_pos_masks": False,  # deepstack VLMs: pad positions are non-visual
    }

    def _shard(key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Pad ``tensor`` on its sequence dim then slice this rank's contiguous chunk.

        Args:
            key: Batch key, selects the pad fill sentinel and the sequence dim
                (``position_ids`` may be ``[3, B, S]`` -> seq dim 2; else dim 1).
            tensor: A sequence-aligned batch tensor (``[B, S, ...]`` or
                ``[3, B, S]`` for mRoPE position ids).

        Returns:
            The local shard ``[..., local_len, ...]`` along the sequence dim.
        """
        seq_dim = 2 if (key == "position_ids" and tensor.dim() == 3) else 1
        if pad_len:
            pad_shape = list(tensor.shape)
            pad_shape[seq_dim] = pad_len
            if tensor.dtype.is_floating_point:
                fill = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            else:
                fill = torch.full(
                    pad_shape,
                    PAD_FILL.get(key, 0),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            tensor = torch.cat([tensor, fill], dim=seq_dim)
        sl = [slice(None)] * tensor.dim()
        sl[seq_dim] = slice(row_offset, row_offset + local_len)
        return tensor[tuple(sl)]

    # Deepstack VLMs (e.g. Qwen3-VL): ``_deepstack_visual_embeds`` is a LIST of
    # [n_visual, H] tensors indexed by visual token (sequence order), NOT
    # sequence-aligned. Slice each to this rank's contiguous visual-token range so
    # the per-shard deepstack merge (hidden_states[visual_pos_masks] += embeds)
    # sees matching counts. Computed from the FULL (padded) visual_pos_masks
    # before it is sharded.
    if "_deepstack_visual_embeds" in batch and "visual_pos_masks" in batch:
        vpm = batch["visual_pos_masks"]  # [B, S] bool (packing -> B == 1)
        if pad_len:
            vpm = torch.cat(
                [vpm, torch.zeros(vpm.shape[0], pad_len, dtype=vpm.dtype, device=vpm.device)],
                dim=1,
            )
        v_start = int(vpm[0, :row_offset].sum())
        n_local = int(vpm[0, row_offset : row_offset + local_len].sum())
        batch["_deepstack_visual_embeds"] = [d[v_start : v_start + n_local] for d in batch["_deepstack_visual_embeds"]]

    seq_aligned = (
        "inputs_embeds",
        "input_ids",
        "labels",
        "position_ids",
        "_packed_seq_ids",
        "loss_mask",
        "padding_mask",
        "visual_pos_masks",
    )
    for key in seq_aligned:
        if not shard_primary and key == primary_key:
            continue
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = _shard(key, batch[key])

    runtime_config = state_module.cp_varlen_runtime_config()
    step_state = {
        "group": group,
        "doc_ids": doc_ids,
        "packed_cu_seqlens": packed_cu_seqlens,
        "packed_cu_seqlens_cpu": packed_cu_seqlens_cpu,
        "row_offset": row_offset,
        "seq_dim": 2,
        # Per-step snapshot of the runtime config, so cp_blockdiag_sdpa selects its
        # path from STATE (a function of (qkv, state)) rather than reading module
        # globals on the hot path. configure_cp_varlen() remains the config holder;
        # this captures it once per step (see _resolve_cp_varlen_config).
        "attn_backend": runtime_config["attn_backend"],
        "kv_exchange": runtime_config["kv_exchange"],
        # Precompute the varlen cu_seqlens once per step (step-constant in
        # doc_ids/row_offset/local_len) so the per-layer softmax SDPA does zero
        # GPU->CPU .item() syncs. The flash/te varlen path reads this; the dense
        # fallback ignores it.
        "varlen_meta": kernels.precompute_blockdiag_varlen_meta(doc_ids, row_offset, local_len, device),
        "model_state": state_module.BlockdiagCpModelState(
            group=group,
            packed_cu_seqlens=packed_cu_seqlens,
            packed_cu_seqlens_cpu=packed_cu_seqlens_cpu,
        ),
    }

    @contextlib.contextmanager
    def _ctx():
        token = state_module._CP_BLOCKDIAG_STATE.set(step_state)
        # EFFICIENT/MATH support arbitrary masks (flash does not); both are correct
        # on the local (non-DTensor) tensors produced by the K/V exchange.
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            try:
                yield
            finally:
                state_module._CP_BLOCKDIAG_STATE.reset(token)

    return _ctx, batch, ShardLayout(original_seq_len=S, padded_seq_len=S_full)
