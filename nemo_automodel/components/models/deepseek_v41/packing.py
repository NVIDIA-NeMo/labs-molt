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

"""One packed text layout for direct forwards and contiguous context parallelism."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedSequenceLayout:
    """Compression-aligned packed rows and reversible token coordinates.

    Attributes:
        source_indices: Original token columns [batch, padded_sequence]; -1 marks padding.
        input_positions: Aligned columns [batch, original_sequence]; -1 marks discarded padding.
        position_ids: Document-local RoPE positions [batch, padded_sequence].
        sequence_ids: Document IDs [batch, padded_sequence], starting at one; zero marks padding.
    """

    source_indices: torch.Tensor
    input_positions: torch.Tensor
    position_ids: torch.Tensor
    sequence_ids: torch.Tensor

    def pack(self, values: torch.Tensor, *, fill: int | float = 0) -> torch.Tensor:
        """Insert alignment padding without changing real token order.

        Args:
            values: Tensor of shape [batch, original_sequence, ...], with arbitrary trailing axes.
            fill: Value for padding slots.

        Returns:
            Independent tensor [batch, padded_sequence, ...], preserving gradients of real tokens.
        """
        rows = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
        result = values[rows, self.source_indices.clamp_min(0)]
        valid = self.source_indices >= 0
        valid = valid.reshape(*valid.shape, *([1] * (values.ndim - 2)))
        return result.masked_fill(~valid, fill)

    def restore(self, values: torch.Tensor) -> torch.Tensor:
        """Restore caller coordinates, returning zeros at original padding slots.

        Args:
            values: Tensor of shape [batch, padded_sequence, ...], with arbitrary trailing axes.

        Returns:
            Independent tensor [batch, original_sequence, ...] retaining real-token gradients.
        """
        rows = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
        result = values[rows, self.input_positions.clamp_min(0)]
        valid = self.input_positions >= 0
        valid = valid.reshape(*valid.shape, *([1] * (values.ndim - 2)))
        return result.masked_fill(~valid, 0)


def packed_layout(
    seq_lens: torch.Tensor,
    *,
    seq_lens_padded: torch.Tensor | None,
    input_shape: tuple[int, int],
    alignment: int,
    minimum_length: int = 0,
) -> PackedSequenceLayout:
    """Validate packed spans and align document starts for compression groups.

    Args:
        seq_lens: Integer real document lengths [batch, documents] or [documents] for batch one.
            Zero and -1000 entries denote absent documents.
        seq_lens_padded: Optional integer physical span lengths with the same shape. These include
            existing per-document padding; omitted means documents are directly concatenated.
        input_shape: Original [batch, sequence] dimensions.
        alignment: Positive multiple for each document's physical span.
        minimum_length: Minimum aligned row length, used to retain a fixed pack budget.

    Returns:
        Layout whose tensor fields are documented by PackedSequenceLayout, on seq_lens.device.
        Metadata and caller tensors are never mutated.
    """
    if alignment < 1:
        raise ValueError("Packed alignment must be positive")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens must contain integer document lengths")
    lengths = seq_lens.unsqueeze(0) if seq_lens.ndim == 1 else seq_lens
    spans = lengths if seq_lens_padded is None else seq_lens_padded
    spans = spans.unsqueeze(0) if spans.ndim == 1 else spans
    if lengths.ndim != 2 or lengths.shape[0] != input_shape[0] or spans.shape != lengths.shape:
        raise ValueError("seq_lens and seq_lens_padded must have matching [batch, documents] shapes")
    if spans.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens_padded must contain integer document lengths")
    documents = []
    max_length = max(minimum_length, 1)
    for real_row, span_row in zip(lengths.tolist(), spans.tolist(), strict=True):
        row = []
        old_start = new_start = 0
        for document, (length, span) in enumerate(zip(real_row, span_row, strict=True), start=1):
            if length in (0, -1000):
                if span not in (0, -1000):
                    raise ValueError("An absent packed document cannot have a physical span")
                continue
            if length < 0 or span < length or old_start + span > input_shape[1]:
                raise ValueError("Packed document lengths exceed their physical spans or input row")
            row.append((old_start, new_start, length, document))
            old_start += span
            new_start += (length + alignment - 1) // alignment * alignment
        documents.append(row)
        max_length = max(max_length, new_start)
    max_length = (max_length + alignment - 1) // alignment * alignment
    shape = (input_shape[0], max_length)
    sources = torch.full(shape, -1, device=seq_lens.device, dtype=torch.long)
    positions = torch.zeros(shape, device=seq_lens.device, dtype=torch.long)
    sequence_ids = torch.zeros_like(positions)
    input_positions = torch.full(input_shape, -1, device=seq_lens.device, dtype=torch.long)
    for batch_idx, row in enumerate(documents):
        for old_start, new_start, length, document in row:
            offsets = torch.arange(length, device=seq_lens.device)
            sources[batch_idx, new_start : new_start + length] = old_start + offsets
            input_positions[batch_idx, old_start : old_start + length] = new_start + offsets
            positions[batch_idx, new_start : new_start + length] = offsets
            sequence_ids[batch_idx, new_start : new_start + length] = document
    return PackedSequenceLayout(sources, input_positions, positions, sequence_ids)
