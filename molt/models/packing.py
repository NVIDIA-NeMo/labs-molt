# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense RL batches to and from model-native packed token layouts."""

import torch
import torch.nn.functional as F


def pack_padded_batch(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layout: str,
    sequence_alignment: int = 1,
    padding_token_id: int = 0,
):
    """Pack padded next-token examples into one physical token row.

    Args:
        sequences: Token IDs with shape ``[batch, sequence]``.
        attention_mask: Contiguous valid-token spans matching ``sequences``.
        layout: ``"thd"`` or ``"indexed_mask"``.
        sequence_alignment: Physical alignment for each THD document.
        padding_token_id: Fill for synthetic input tokens.

    Returns:
        Packed input IDs, positions, next-token targets, a physical-token to
        dense-source index map, and model attention metadata. Each real sample
        contributes ``valid_tokens - 1`` prediction positions; synthetic map
        entries are ``-1``.
    """
    if sequences.ndim != 2 or attention_mask.shape != sequences.shape:
        raise ValueError("packing requires matching [batch, sequence] token and attention tensors")
    if layout not in {"thd", "indexed_mask"}:
        raise ValueError(f"unknown packing layout: {layout!r}")
    if sequence_alignment < 1:
        raise ValueError(f"sequence_alignment must be positive, got {sequence_alignment}")
    if layout == "indexed_mask" and sequence_alignment != 1:
        raise ValueError("indexed-mask packing does not support per-document alignment")

    batch, seqlen = sequences.shape
    packed_ids_parts = []
    target_parts = []
    position_parts = []
    physical_to_dense_parts = []
    document_id_parts = []
    seq_lens = []
    padded_lens = []
    for row in range(batch):
        valid = attention_mask[row].bool().nonzero(as_tuple=False).flatten()
        if valid.numel() < 2:
            raise ValueError("packed samples require at least two contiguous valid tokens")
        start, stop = int(valid[0]), int(valid[-1]) + 1
        if valid.numel() != stop - start:
            raise ValueError("packed samples require one contiguous attention span")

        real_length = stop - start - 1
        padded_length = ((real_length + sequence_alignment - 1) // sequence_alignment) * sequence_alignment
        pad = padded_length - real_length
        packed_ids_parts.append(F.pad(sequences[row, start : stop - 1], (0, pad), value=padding_token_id))
        target_parts.append(F.pad(sequences[row, start + 1 : stop], (0, pad)))
        position_parts.append(torch.arange(padded_length, device=sequences.device, dtype=torch.long))
        dense_positions = torch.arange(start, stop - 1, device=sequences.device) + row * seqlen
        physical_to_dense_parts.append(F.pad(dense_positions, (0, pad), value=-1))
        document_id_parts.append(torch.full((padded_length,), row + 1, device=sequences.device, dtype=torch.long))
        seq_lens.append(real_length)
        padded_lens.append(padded_length)

    packed_ids = torch.cat(packed_ids_parts).unsqueeze(0)
    targets = torch.cat(target_parts).unsqueeze(0)
    position_ids = torch.cat(position_parts).unsqueeze(0)
    physical_to_dense = torch.cat(physical_to_dense_parts)

    if layout == "indexed_mask":
        indexed_mask = torch.cat(document_id_parts).unsqueeze(0)
        return packed_ids, position_ids, targets, physical_to_dense, {"attention_mask": indexed_mask}

    seq_lens_tensor = torch.tensor(seq_lens, device=sequences.device, dtype=torch.int32)
    padded_lens_tensor = torch.tensor(padded_lens, device=sequences.device, dtype=torch.int32)
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=sequences.device),
            seq_lens_tensor.cumsum(dim=0, dtype=torch.int32),
        )
    )
    cu_seqlens_padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=sequences.device),
            padded_lens_tensor.cumsum(dim=0, dtype=torch.int32),
        )
    )

    attention = {
        "qkv_format": "thd",
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens_padded,
        "max_seqlen": int(padded_lens_tensor.max()),
        "seq_lens": seq_lens_tensor.unsqueeze(0),
        "seq_lens_padded": padded_lens_tensor.unsqueeze(0),
        "padding_mask": physical_to_dense.lt(0).unsqueeze(0),
    }
    return packed_ids, position_ids, targets, physical_to_dense, attention


def unpack_to_padded(packed: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Scatter physical ``[tokens, ...]`` outputs to dense source positions."""
    values = packed.squeeze(0) if packed.ndim > 1 and packed.shape[0] == 1 else packed
    if values.shape[0] < indices.numel():
        raise ValueError(f"packed output has {values.shape[0]} tokens for a {indices.numel()}-entry restore map")
    values = values[: indices.numel()]
    valid = indices >= 0
    output = values.new_zeros((batch * seqlen, *values.shape[1:]))
    output.index_copy_(0, indices[valid], values[valid])
    return output.view(batch, seqlen, *values.shape[1:])
