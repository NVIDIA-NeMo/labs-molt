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

"""Context-parallel helpers for the DeepSeek V4 custom model.

This implements the Miles-style training path: each CP rank owns a contiguous
query shard, while K/V and compressed K/V are all-gathered with autograd-aware
collectives before DSV4 sparse attention consumes them.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

_SEQ_LENS_PADDING_VALUE = -1000


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else max(a, b)


def dsv4_cp_local_seq_multiple(model_or_config) -> int:
    """Required per-CP-rank sequence-length multiple for DSV4 Miles-style CP.

    Compress-ratio layers constrain how the sequence may be split across CP ranks:
    a ratio-R layer needs each local shard divisible by R, and ratio-4 layers use
    cross-window overlap so they need ``2*R``. The returned value is the LCM across
    all configured ``compress_ratios`` (1 when none are configured).
    """
    config = getattr(model_or_config, "config", model_or_config)
    ratios = [int(r) for r in (getattr(config, "compress_ratios", None) or []) if int(r) > 0]
    multiple = 1
    for ratio in ratios:
        required = 2 * ratio if ratio == 4 else ratio
        multiple = _lcm(multiple, required)
    return max(multiple, 1)


def dsv4_cp_enabled(cp_group) -> bool:
    """Return whether a real CP process group is active."""
    return (
        cp_group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=cp_group) > 1
    )


def dsv4_cp_rank(cp_group) -> int:
    """Return this rank's index in the DSV4 CP group, or 0 without CP."""
    return dist.get_rank(group=cp_group) if dsv4_cp_enabled(cp_group) else 0


def dsv4_cp_size(cp_group) -> int:
    """Return the DSV4 CP group size, or 1 without CP."""
    return dist.get_world_size(group=cp_group) if dsv4_cp_enabled(cp_group) else 1


def dsv4_cp_all_gather(tensor: torch.Tensor, *, dim: int, cp_group) -> torch.Tensor:
    """All-gather activation tensors across CP ranks and concatenate on ``dim``.

    The distributed.nn functional collective preserves autograd, so backward
    routes gradients for gathered remote slices back to their owning ranks.
    """
    if not dsv4_cp_enabled(cp_group):
        return tensor

    try:
        from torch.distributed.nn.functional import all_gather
    except (ImportError, AttributeError) as exc:  # pragma: no cover - version guard
        raise RuntimeError(
            "DeepSeek V4 context parallelism requires torch.distributed.nn.functional.all_gather "
            "for differentiable activation gathers."
        ) from exc

    parts = all_gather(tensor.contiguous(), group=cp_group)
    return torch.cat(tuple(parts), dim=dim)


def dsv4_cp_all_gather_metadata(tensor: torch.Tensor | None, *, dim: int, cp_group) -> torch.Tensor | None:
    """All-gather non-differentiable metadata such as padding masks."""
    if tensor is None or not dsv4_cp_enabled(cp_group):
        return tensor
    local = tensor.contiguous()
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size(group=cp_group))]
    dist.all_gather(parts, local, group=cp_group)
    return torch.cat(parts, dim=dim)


def build_packed_seq_ids(
    seq_lens_padded: torch.Tensor,
    *,
    seq_len: int,
    device: torch.device,
    padding_value: int = _SEQ_LENS_PADDING_VALUE,
) -> torch.Tensor:
    """Build per-token packed sequence IDs from padded packed lengths.

    IDs are 1-based within each batch row; 0 marks trailing pack padding that
    belongs to no sequence. ``seq_lens_padded`` may be right-padded with
    ``padding_value`` to make rows rectangular.
    """
    if seq_lens_padded.dim() == 1:
        seq_lens_padded = seq_lens_padded.unsqueeze(0)
    lengths = seq_lens_padded.to(device=device, dtype=torch.long)
    seq_ids = torch.zeros((lengths.shape[0], seq_len), dtype=torch.long, device=device)
    for batch_idx in range(lengths.shape[0]):
        offset = 0
        doc_id = 1
        for length in lengths[batch_idx].tolist():
            if int(length) == padding_value:
                continue
            length = max(int(length), 0)
            if length == 0:
                doc_id += 1
                continue
            end = min(offset + length, seq_len)
            if end > offset:
                seq_ids[batch_idx, offset:end] = doc_id
            offset += length
            doc_id += 1
            if offset >= seq_len:
                break
    return seq_ids


def build_dsv4_cp_packed_causal_padding_mask(
    *,
    position_ids: torch.Tensor,
    packed_seq_ids: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    cp_group,
    padding_mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build local-query/global-key additive mask for packed DSV4 CP.

    ``packed_seq_ids`` is local to the CP rank. The function all-gathers sequence
    IDs and document-local positions for the key side, then applies same-document,
    causal, padding, and optional sliding-window constraints.
    """
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if packed_seq_ids.dim() == 1:
        packed_seq_ids = packed_seq_ids.unsqueeze(0)

    q_pos = position_ids.to(device=device, dtype=torch.long)
    q_seq = packed_seq_ids.to(device=device, dtype=torch.long)
    k_pos = dsv4_cp_all_gather_metadata(q_pos, dim=1, cp_group=cp_group)
    k_seq = dsv4_cp_all_gather_metadata(q_seq, dim=1, cp_group=cp_group)

    allowed = (q_seq.unsqueeze(-1) > 0) & (q_seq.unsqueeze(-1) == k_seq.unsqueeze(1))
    allowed = allowed & (k_pos.unsqueeze(1) <= q_pos.unsqueeze(-1))
    if sliding_window is not None:
        allowed = allowed & ((q_pos.unsqueeze(-1) - k_pos.unsqueeze(1)) < sliding_window)

    padding_mask_full = dsv4_cp_all_gather_metadata(padding_mask, dim=1, cp_group=cp_group)
    if padding_mask_full is not None:
        allowed = allowed & ~padding_mask_full.to(device=device, dtype=torch.bool).unsqueeze(1)

    min_value = torch.finfo(dtype).min
    return torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )


def build_dsv4_cp_causal_padding_mask(
    *,
    position_ids: torch.Tensor,
    key_len: int,
    dtype: torch.dtype,
    device: torch.device,
    cp_group,
    padding_mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build local-query/global-key additive mask for Miles-style DSV4 CP.

    ``position_ids`` are the local query positions after contiguous CP slicing.
    Keys are in global sequence order because DSV4 gathers K/V along sequence.
    ``padding_mask`` follows the internal convention True=padding.
    """
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    q_pos = position_ids.to(device=device, dtype=torch.long)
    batch, seq_len = q_pos.shape
    k_pos = torch.arange(key_len, device=device, dtype=torch.long)

    allowed = k_pos.view(1, 1, key_len) <= q_pos.view(batch, seq_len, 1)
    if sliding_window is not None:
        allowed = allowed & ((q_pos.view(batch, seq_len, 1) - k_pos.view(1, 1, key_len)) < sliding_window)

    padding_mask_full = dsv4_cp_all_gather_metadata(padding_mask, dim=1, cp_group=cp_group)
    if padding_mask_full is not None:
        allowed = allowed & ~padding_mask_full.to(device=device, dtype=torch.bool).view(batch, 1, key_len)

    min_value = torch.finfo(dtype).min
    return torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )


# ---------------------------------------------------------------------------
# Model-owned CP batch sharding (Miles-style contiguous query shard).
#
# ``ContextParallelSharder`` construction delegates manual all-gather CP to the model
# via the ``ContextParallelSharder`` returned by ``prepare_model_inputs_for_cp``. DSV4's
# sharder pads + contiguously shards the sequence per CP rank (via the shared
# contiguous implementation in ``components/distributed/context_parallel/sharder.py``) and
# hands the CP process group to the forward (``_dsv4_cp_group``) so DSV4
# attention can all-gather K/V.
# ---------------------------------------------------------------------------


def _valid_packed_lengths(row: torch.Tensor, padding_value: int = _SEQ_LENS_PADDING_VALUE) -> list[int]:
    return [max(int(x), 0) for x in row.tolist() if int(x) != padding_value]


def _pad_length(length: int, multiple: int) -> int:
    multiple = max(int(multiple), 1)
    return length + ((-length) % multiple)


def _pad_1d(values: list[int], width: int, padding_value: int = _SEQ_LENS_PADDING_VALUE) -> torch.Tensor:
    if len(values) < width:
        values = [*values, *([padding_value] * (width - len(values)))]
    return torch.tensor(values, dtype=torch.long)


def _repad_dsv4_packed_batch(
    batch: dict,
    *,
    cp_size: int,
    pad_multiple: int,
    padding_token_id: int,
    sync_packed_length: bool = False,
    loss_mask: torch.Tensor | None = None,
) -> tuple[dict, torch.Tensor | None, torch.Tensor]:
    """Insert DSV4 compression-safe padding into packed BSHD rows before CP slicing.

    The generic packed dataset may pad each packed sequence only for TE CP. DSV4
    compression additionally needs document boundaries to align to compressor
    windows; CSA then uses ``packed_seq_ids`` to reset the previous-window overlap.
    This routine rebuilds each row from real sequence spans, pads every span to
    ``pad_multiple``, and appends row-level pack padding with sequence ID 0. When
    requested for HybridEP, the final physical length is max-reduced across ranks
    before CP slicing so every rank in a flattened DP x CP expert group is uniform.
    """
    if "seq_lens" not in batch:
        raise KeyError("DSV4 packed context parallelism requires `seq_lens` in the batch.")

    seq_lens = batch["seq_lens"]
    seq_lens_padded = batch.get("seq_lens_padded", seq_lens)
    if seq_lens.dim() == 1:
        seq_lens = seq_lens.unsqueeze(0)
    if seq_lens_padded.dim() == 1:
        seq_lens_padded = seq_lens_padded.unsqueeze(0)

    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "make_dsv4_contiguous_shard_cp_batch_and_ctx requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary = batch[primary_key]
    labels = batch.get("labels")
    if labels is None:
        raise KeyError("DSV4 context parallelism requires `labels` in the batch.")

    batch_size = primary.shape[0]
    # Per-row map from input position to rebuilt-row column (-1 = input pad
    # slot whose token was dropped); the ContextParallelSharder token verbs restore the
    # caller's coordinates through it.
    input_positions = torch.full((batch_size, primary.shape[1]), -1, dtype=torch.long, device=primary.device)
    rebuilt_primary = []
    rebuilt_labels = []
    rebuilt_loss_mask = [] if loss_mask is not None else None
    rebuilt_positions = []
    rebuilt_padding = []
    rebuilt_seq_ids = []
    new_seq_lens: list[list[int]] = []
    new_seq_lens_padded: list[list[int]] = []
    row_lengths = []

    for batch_idx in range(batch_size):
        real_lengths = _valid_packed_lengths(seq_lens[batch_idx])
        padded_lengths = _valid_packed_lengths(seq_lens_padded[batch_idx])
        if len(padded_lengths) < len(real_lengths):
            padded_lengths.extend(real_lengths[len(padded_lengths) :])

        primary_parts = []
        label_parts = []
        loss_parts = []
        pos_parts = []
        padding_parts = []
        seq_id_parts = []
        row_seq_lens = []
        row_padded_lens = []
        old_offset = 0
        new_offset = 0

        for doc_idx, real_len in enumerate(real_lengths):
            old_padded_len = max(padded_lengths[doc_idx], real_len)
            if old_offset + real_len > primary.shape[1]:
                raise ValueError(
                    "Packed DSV4 batch metadata exceeds token row length: "
                    f"{old_offset + real_len=} > {primary.shape[1]=}"
                )
            new_padded_len = _pad_length(real_len, pad_multiple)
            pad_len = new_padded_len - real_len

            primary_real = primary[batch_idx, old_offset : old_offset + real_len]
            labels_real = labels[batch_idx, old_offset : old_offset + real_len]
            primary_parts.append(primary_real)
            label_parts.append(labels_real)
            if loss_mask is not None:
                loss_parts.append(loss_mask[batch_idx, old_offset : old_offset + real_len])

            if pad_len:
                if has_inputs_embeds:
                    primary_pad = primary.new_zeros((pad_len, *primary.shape[2:]))
                else:
                    primary_pad = torch.full((pad_len,), padding_token_id, dtype=primary.dtype, device=primary.device)
                primary_parts.append(primary_pad)
                label_parts.append(torch.full((pad_len,), -100, dtype=labels.dtype, device=labels.device))
                if loss_mask is not None:
                    loss_parts.append(torch.zeros((pad_len,), dtype=loss_mask.dtype, device=loss_mask.device))

            pos_parts.append(torch.arange(new_padded_len, dtype=torch.long, device=primary.device))
            padding_parts.append(
                torch.cat(
                    (
                        torch.zeros(real_len, dtype=torch.bool, device=primary.device),
                        torch.ones(pad_len, dtype=torch.bool, device=primary.device),
                    )
                )
            )
            seq_id_parts.append(torch.full((new_padded_len,), doc_idx + 1, dtype=torch.long, device=primary.device))
            # Input->output position map for this document's real tokens; the
            # dropped input pad slots keep -1 (see input_positions init).
            input_positions[batch_idx, old_offset : old_offset + real_len] = torch.arange(
                new_offset, new_offset + real_len, device=primary.device
            )
            row_seq_lens.append(real_len)
            row_padded_lens.append(new_padded_len)
            old_offset += old_padded_len
            new_offset += new_padded_len

        rebuilt_primary.append(torch.cat(primary_parts, dim=0))
        rebuilt_labels.append(torch.cat(label_parts, dim=0))
        if loss_mask is not None and rebuilt_loss_mask is not None:
            rebuilt_loss_mask.append(torch.cat(loss_parts, dim=0))
        rebuilt_positions.append(torch.cat(pos_parts, dim=0))
        rebuilt_padding.append(torch.cat(padding_parts, dim=0))
        rebuilt_seq_ids.append(torch.cat(seq_id_parts, dim=0))
        new_seq_lens.append(row_seq_lens)
        new_seq_lens_padded.append(row_padded_lens)
        row_lengths.append(rebuilt_primary[-1].shape[0])

    total_seq_len = _pad_length(max(row_lengths), cp_size * pad_multiple)
    if sync_packed_length and dist.is_available() and dist.is_initialized():
        # HybridEP flattens DP x CP into one EP group and requires every rank to
        # contribute the same number of tokens. Different DP packs can acquire
        # different amounts of per-document compression padding.
        length = torch.tensor(total_seq_len, dtype=torch.int64, device=primary.device)
        dist.all_reduce(length, op=dist.ReduceOp.MAX)
        total_seq_len = int(length.item())

    def _right_pad_rows(rows: list[torch.Tensor], fill_value, *, dtype=None) -> torch.Tensor:
        padded = []
        for row in rows:
            pad_len = total_seq_len - row.shape[0]
            if pad_len:
                pad_shape = (pad_len, *row.shape[1:])
                pad = torch.full(pad_shape, fill_value, dtype=dtype or row.dtype, device=row.device)
                row = torch.cat((row, pad), dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)

    if has_inputs_embeds:
        batch["inputs_embeds"] = _right_pad_rows(rebuilt_primary, 0)
    else:
        batch["input_ids"] = _right_pad_rows(rebuilt_primary, padding_token_id)
    batch["labels"] = _right_pad_rows(rebuilt_labels, -100)
    batch["position_ids"] = _right_pad_rows(rebuilt_positions, 0, dtype=torch.long)
    batch["padding_mask"] = _right_pad_rows(rebuilt_padding, True, dtype=torch.bool)
    batch["packed_seq_ids"] = _right_pad_rows(rebuilt_seq_ids, 0, dtype=torch.long)

    width = max(len(row) for row in new_seq_lens)
    batch["seq_lens"] = torch.stack(
        [_pad_1d(row, width).to(device=primary.device) for row in new_seq_lens],
        dim=0,
    )
    batch["seq_lens_padded"] = torch.stack(
        [_pad_1d(row, width).to(device=primary.device) for row in new_seq_lens_padded],
        dim=0,
    )
    batch["qkv_format"] = "thd"

    if loss_mask is not None and rebuilt_loss_mask is not None:
        loss_mask = _right_pad_rows(rebuilt_loss_mask, 0)
    return batch, loss_mask, input_positions


def make_dsv4_contiguous_shard_cp_batch_and_ctx(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    pad_multiple: int | None = None,
    sync_packed_length: bool = False,
):
    """Contiguously shard a batch for DeepSeek V4 Miles-style context parallelism.

    Exposed as ``ContextParallelSharder.shard_batch`` (via ``functools.partial`` to bind
    ``pad_multiple``) and invoked by the CP dispatch. HybridEP can
    first max-reduce packed lengths so every rank contributes a uniform token count.
    Each CP rank then keeps one ``seq_start:seq_end`` slice; DSV4 attention all-gathers
    K/V across CP ranks during forward. Returns ``(nullcontext, batch)``.

    ``pad_multiple`` is the required *per-CP-rank* shard multiple (from
    ``dsv4_cp_local_seq_multiple``); the global sequence is padded so it is divisible
    by ``cp_size`` and each local shard is divisible by ``pad_multiple`` (>= 2).
    At CP size one, the native THD route only marks packed input as THD and leaves
    its tensors and packing metadata unchanged.
    """
    import contextlib

    from nemo_automodel.components.distributed.context_parallel.sharder import (  # noqa: PLC0415
        ShardLayout,
        convert_attention_mask_to_padding_mask,
        shard_batch_contiguous,
    )

    cp_size = cp_mesh.size()
    packed = batch.get("qkv_format") == "thd" or "seq_lens" in batch or "cu_seqlens" in batch
    if cp_size <= 1:
        if packed:
            batch["qkv_format"] = "thd"
        if "labels" not in batch and loss_mask is not None:
            batch["labels"] = loss_mask
        elif loss_mask is not None:
            batch["loss_mask"] = loss_mask
        return contextlib.nullcontext, batch, None

    local_multiple = max(int(pad_multiple or 2), 2)

    if "cu_seqlens" in batch and "seq_lens" not in batch:
        raise NotImplementedError(
            "DeepSeek V4 model-owned packed CP expects BSHD packed metadata (`seq_lens`); "
            "pre-flattened `cu_seqlens` batches should use the TE CP path."
        )

    input_positions = None
    if packed:
        # Preserve the packed-document boundaries while the shared contiguous
        # sharder handles the common padding/slicing mechanics.
        convert_attention_mask_to_padding_mask(batch)
        batch, loss_mask, input_positions = _repad_dsv4_packed_batch(
            batch,
            cp_size=cp_size,
            pad_multiple=local_multiple,
            padding_token_id=padding_token_id,
            sync_packed_length=sync_packed_length,
            loss_mask=loss_mask,
        )

    ctx, batch, layout = shard_batch_contiguous(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        pad_multiple=local_multiple,
        extra_seq_keys={"packed_seq_ids": 1} if packed else None,
        extra_pad_values={"packed_seq_ids": 0} if packed else None,
    )
    if packed:
        # The repad rebuilt the rows, so no single original length exists; the
        # caller's coordinates are restored through the position map instead.
        layout = ShardLayout(
            padded_seq_len=layout.padded_seq_len,
            input_token_stream_positions=input_positions,
        )
    # Hand the CP process group to the model forward (read from attn_kwargs) so
    # DSV4 attention all-gathers K/V across CP ranks. Not a tensor -> not sharded.
    batch["_dsv4_cp_group"] = cp_mesh.get_group()
    return ctx, batch, layout
