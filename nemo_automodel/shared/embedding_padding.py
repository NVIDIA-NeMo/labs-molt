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
"""DTensor-safe zeroing of one embedding row (the ``padding_idx`` step of weight init)."""

from __future__ import annotations

import logging

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

try:  # private torch helper; if it moves, degrade to "leave the row as initialized" below
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
except ImportError:  # pragma: no cover - depends on the torch build
    compute_local_shape_and_global_offset = None  # type: ignore[assignment]

try:  # FSDP2 x TP shards parameters with _StridedShard on the FSDP mesh dim
    from torch.distributed.tensor.placement_types import _StridedShard
except ImportError:  # pragma: no cover - older torch
    _StridedShard = ()  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)
_SUPPORTED_PLACEMENTS = (Replicate, Shard) + ((_StridedShard,) if _StridedShard else ())


@torch.no_grad()
def zero_embedding_row_(weight: torch.Tensor, row: int) -> bool:
    """Zero ``weight[row]`` in place without integer-indexing a DTensor.

    ``weight[row].zero_()`` on a DTensor whose vocabulary dim is sharded triggers a
    redistribute (an all-gather of the whole embedding) and fails outright for TP
    shards. This touches only the rank-local shard, and only when that shard owns
    the row.

    Args:
        weight: Embedding matrix of shape [vocab, hidden]; a plain tensor or a DTensor whose
            placements are ``Replicate`` / ``Shard`` / ``_StridedShard`` (FSDP2 x TP) on either matrix axis.
        row: Global row index to zero (negative indices count from the end).

    Returns:
        True when this rank held part of the row and zeroed it; False when the row lives
        entirely on other ranks, or when the layout is not understood (then the row is
        left as initialized and a warning is logged rather than aborting model
        construction).
    """
    vocab = weight.shape[0]
    if row < 0:
        row += vocab
    if not 0 <= row < vocab:
        raise IndexError(f"row {row} out of range for embedding with {vocab} rows")
    if not isinstance(weight, DTensor):
        weight[row].zero_()
        return True
    placements = tuple(weight.placements)
    unsupported = [p for p in placements if not isinstance(p, _SUPPORTED_PLACEMENTS)]
    if unsupported:
        # Never fail model construction over the padding row: it only needs to be zero
        # for HF parity, and a from_pretrained load overwrites it anyway.
        logger.warning(
            "zero_embedding_row_: leaving padding row %d as initialized; unsupported DTensor placements %s",
            row,
            unsupported,
        )
        return False
    try:
        if compute_local_shape_and_global_offset is None:
            raise ImportError("torch.distributed.tensor._utils.compute_local_shape_and_global_offset is unavailable")
        local_shape, global_offset = compute_local_shape_and_global_offset(
            tuple(weight.shape), weight.device_mesh, placements
        )
    except Exception as exc:  # noqa: BLE001 - offset math for exotic layouts must not abort init
        logger.warning(
            "zero_embedding_row_: leaving padding row %d as initialized; could not resolve the local shard of "
            "placements %s: %s",
            row,
            placements,
            exc,
        )
        return False
    start = global_offset[0]
    if not start <= row < start + local_shape[0]:
        return False
    local = weight.to_local()
    if local.numel() == 0:
        return False
    local[row - start].zero_()
    return True
