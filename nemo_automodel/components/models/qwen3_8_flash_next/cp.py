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

"""Contiguous context parallelism for the language-only Qwen3.8-Flash-Next model."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ShardLayout,
    shard_batch_contiguous,
)


@dataclass(frozen=True)
class Qwen3_8_FlashNextCPContext:
    """Per-forward metadata for Qwen3.8-Flash-Next's contiguous CP sequence shard.

    Attributes:
        group: Process group whose rank-ordered shards form one sequence, or
            ``None`` for the size-one identity path.
        rank: This process's zero-based rank within ``group``.
        size: Number of contiguous sequence shards.
        global_input_ids: Padded raw tokenizer IDs of shape ``[batch,
            global_sequence]``. The tensor is replicated across the CP group
            and is used only for PLE hashing.
        global_padding_mask: Replicated boolean tensor of shape ``[batch,
            global_sequence]`` where ``True`` marks right-tail padding.
        local_sequence_start: Global position of local sequence position zero.
        local_sequence_length: Number of physical sequence positions on this
            rank, including CP padding.
        global_cu_seqlens: Optional global packed-document boundaries
            ``[num_docs + 1]`` for a THD batch of one row. The final boundary
            may be smaller than the padded global length; the remaining tail
            is CP padding that belongs to no document.
    """

    group: dist.ProcessGroup | None
    rank: int
    size: int
    global_input_ids: torch.Tensor
    global_padding_mask: torch.Tensor
    local_sequence_start: int
    local_sequence_length: int
    global_cu_seqlens: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate the replicated metadata and contiguous rank mapping."""
        if self.size <= 0 or self.rank < 0 or self.rank >= self.size:
            raise ValueError(f"Invalid Qwen3.8-Flash-Next CP rank/size: rank={self.rank}, size={self.size}")
        if self.global_input_ids.ndim != 2 or self.global_input_ids.dtype not in (
            torch.int32,
            torch.int64,
            torch.long,
        ):
            raise ValueError(
                "Qwen3.8-Flash-Next CP global_input_ids must be int32/int64 [batch, global_sequence]; "
                f"got shape={tuple(self.global_input_ids.shape)}, dtype={self.global_input_ids.dtype}"
            )
        if self.global_padding_mask.shape != self.global_input_ids.shape:
            raise ValueError(
                "Qwen3.8-Flash-Next CP global padding/ID axes differ: "
                f"mask={tuple(self.global_padding_mask.shape)}, ids={tuple(self.global_input_ids.shape)}"
            )
        if self.global_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"Qwen3.8-Flash-Next CP global_padding_mask must be bool, got {self.global_padding_mask.dtype}"
            )
        if self.global_padding_mask.device != self.global_input_ids.device:
            raise ValueError("Qwen3.8-Flash-Next CP global padding mask and raw IDs must be on the same device")
        if self.local_sequence_length <= 0:
            raise ValueError(
                f"Qwen3.8-Flash-Next CP local sequence length must be positive, got {self.local_sequence_length}"
            )
        expected_global_length = self.local_sequence_length * self.size
        if self.global_input_ids.shape[1] != expected_global_length:
            raise ValueError(
                "Qwen3.8-Flash-Next CP global sequence must equal local_sequence_length * size; "
                f"got global={self.global_input_ids.shape[1]}, local={self.local_sequence_length}, size={self.size}"
            )
        expected_start = self.rank * self.local_sequence_length
        if self.local_sequence_start != expected_start:
            raise ValueError(
                "Qwen3.8-Flash-Next CP local start must use contiguous rank order; "
                f"got start={self.local_sequence_start}, expected={expected_start}"
            )
        if self.global_cu_seqlens is not None:
            boundaries = self.global_cu_seqlens
            if boundaries.ndim != 1 or boundaries.numel() < 2:
                raise ValueError(
                    f"Qwen3.8-Flash-Next CP global_cu_seqlens must be a rank-1 boundary tensor, got {tuple(boundaries.shape)}"
                )
            if self.global_input_ids.shape[0] != 1:
                raise ValueError("Qwen3.8-Flash-Next packed CP requires a THD batch of one row")
            if (
                int(boundaries[0]) != 0
                or int(boundaries[-1]) > expected_global_length
                or bool((boundaries.diff() <= 0).any())
            ):
                raise ValueError(
                    "Qwen3.8-Flash-Next CP global_cu_seqlens must start at 0, be strictly increasing, and end at or "
                    f"before the padded global length {expected_global_length}; got last={int(boundaries[-1])}"
                )

    @property
    def global_sequence_length(self) -> int:
        """Return the padded global physical sequence length."""
        return self.global_input_ids.shape[1]

    @property
    def local_sequence_end(self) -> int:
        """Return the exclusive global end of this rank's sequence shard."""
        return self.local_sequence_start + self.local_sequence_length

    @property
    def global_sequence_lengths(self) -> torch.Tensor:
        """Return logical right-padded lengths as int64 ``[batch]``."""
        return self.global_padding_mask.logical_not().sum(dim=-1, dtype=torch.long)


def _validate_right_tail_mask(mask: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Validate and normalize a full-sequence token-validity mask.

    Args:
        mask: Binary validity tensor of shape ``[batch, global_sequence]``.
        input_ids: Raw IDs of shape ``[batch, global_sequence]`` whose axes
            establish the expected mask shape and device.

    Returns:
        Boolean validity tensor of shape ``[batch, global_sequence]`` on the
        same device as ``input_ids``.
    """
    if mask.ndim != 2 or mask.shape != input_ids.shape:
        raise NotImplementedError(
            "Qwen3.8-Flash-Next CP requires a non-packed [batch, sequence] attention/padding mask; "
            f"got mask={tuple(mask.shape)}, input_ids={tuple(input_ids.shape)}"
        )
    mask = mask.to(device=input_ids.device)
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Qwen3.8-Flash-Next CP masks must contain only binary 0/1 values")
    valid = mask.bool()
    lengths = valid.sum(dim=-1, dtype=torch.long)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    if not bool(torch.equal(valid, positions < lengths.unsqueeze(1))):
        raise NotImplementedError(
            "Qwen3.8-Flash-Next CP supports only right-tail padding; left padding, interior padding, and packing are unsupported"
        )
    return valid


def _pad_right(tensor: torch.Tensor, length: int, value: int | bool) -> torch.Tensor:
    """Right-pad a rank-two token tensor to ``length``.

    Args:
        tensor: Tensor of shape ``[batch, sequence]``.
        length: Requested output sequence length, no shorter than ``sequence``.
        value: Scalar fill value for appended positions.

    Returns:
        Tensor of shape ``[batch, length]``. The input is returned unchanged
        when no padding is required.
    """
    pad_length = length - tensor.shape[1]
    if pad_length < 0:
        raise ValueError(f"Cannot pad sequence length {tensor.shape[1]} down to {length}")
    if pad_length == 0:
        return tensor
    padding = torch.full(
        (tensor.shape[0], pad_length),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=1)


def shard_batch_for_qwen3_8_flash_next_cp(
    cp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh | None,
    batch: dict[str, Any],
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
    pad_multiple: int = 4,
) -> tuple[Callable[[], contextlib.AbstractContextManager[Any]], dict[str, Any], ShardLayout]:
    """Validate, pad, and contiguously shard a Qwen3.8-Flash-Next text batch.

    Args:
        cp_mesh: One-dimensional CP device mesh. Rank ``r`` owns the contiguous
            global interval ``[r * local_sequence, (r + 1) * local_sequence)``.
        tp_mesh: Optional TP device mesh. Qwen3.8-Flash-Next CP requires this mesh to be
            absent or size one.
        batch: Mutable full-sequence batch. ``input_ids``, ``labels``, and
            optional ``attention_mask``/``padding_mask`` have shape ``[batch,
            global_sequence]``; ``position_ids`` has shape ``[batch,
            global_sequence]``. A packed THD row uses batch size one plus
            ``cu_seqlens`` or loader ``seq_lens`` document boundaries; packed
            attention and padding masks remain unsupported.
        loss_mask: Optional tensor of shape ``[batch, global_sequence]`` used
            by the shared sharder when labels are absent.
        padding_token_id: Raw token ID appended for CP divisibility.
        pad_multiple: Required multiple of every rank's local sequence length.
            QSA requires its compression ratio, four for the released model.

    Returns:
        A null context factory, the mutated batch containing local token
        tensors plus a replicated :class:`Qwen3_8_FlashNextCPContext`, and the global
        pre/post-padding :class:`ShardLayout`.
    """
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise NotImplementedError("Qwen3.8-Flash-Next context parallelism cannot be composed with tensor parallelism")
    packed_keys = (
        "cu_seqlens",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_padded",
        "seq_lens",
        "seq_lens_padded",
        "packed_seq_ids",
        "_packed_seq_ids",
        "indices",
        "max_seqlen",
        "max_seqlen_q",
        "max_seqlen_kv",
    )
    packed_cu_seqlens = batch.pop("cu_seqlens", None)
    loader_seq_lens = batch.pop("seq_lens", None)
    batch.pop("seq_lens_padded", None)
    if packed_cu_seqlens is None and loader_seq_lens is not None:
        packed_cu_seqlens = packed_boundaries_from_seq_lens(
            loader_seq_lens,
            total_tokens=batch["input_ids"].shape[1],
        )
    supported_packed_keys = {"cu_seqlens", "seq_lens", "seq_lens_padded"}
    unsupported_packed_keys = [
        key for key in packed_keys if key not in supported_packed_keys and batch.get(key) is not None
    ]
    if unsupported_packed_keys or (batch.get("qkv_format") == "thd" and packed_cu_seqlens is None):
        raise NotImplementedError(
            "Qwen3.8-Flash-Next packed context parallelism supports only cu_seqlens document boundaries; "
            f"got unsupported keys {unsupported_packed_keys}"
        )
    batch.pop("qkv_format", None)
    if "input_ids" not in batch or "inputs_embeds" in batch:
        raise NotImplementedError(
            "Qwen3.8-Flash-Next context parallelism requires raw input_ids as the sole primary stream for PLE hashing"
        )
    full_input_ids = batch["input_ids"]
    if full_input_ids.ndim != 2 or full_input_ids.dtype not in (torch.int32, torch.int64, torch.long):
        raise ValueError(
            "Qwen3.8-Flash-Next CP input_ids must be an int32/int64 [batch, sequence] tensor; "
            f"got shape={tuple(full_input_ids.shape)}, dtype={full_input_ids.dtype}"
        )
    if full_input_ids.shape[1] == 0:
        raise ValueError("Qwen3.8-Flash-Next context parallelism requires a non-empty sequence")
    if pad_multiple <= 1:
        raise ValueError(f"Qwen3.8-Flash-Next CP pad_multiple must exceed one, got {pad_multiple}")

    if packed_cu_seqlens is not None:
        if full_input_ids.shape[0] != 1:
            raise ValueError(
                f"Qwen3.8-Flash-Next packed CP requires a THD batch of one row, got batch={full_input_ids.shape[0]}"
            )
        boundaries = packed_cu_seqlens.reshape(-1).to(dtype=torch.long)
        if (
            boundaries.numel() < 2
            or int(boundaries[0]) != 0
            or int(boundaries[-1]) != full_input_ids.shape[1]
            or bool((boundaries.diff() <= 0).any())
        ):
            raise ValueError(
                "Qwen3.8-Flash-Next packed CP cu_seqlens must start at 0, be strictly increasing, and end at the "
                f"packed token count {full_input_ids.shape[1]}"
            )
        if batch.get("attention_mask") is not None or batch.get("padding_mask") is not None:
            raise ValueError(
                "Qwen3.8-Flash-Next packed CP takes document boundaries from cu_seqlens; masks are unsupported"
            )
        packed_cu_seqlens = boundaries

    attention_mask = batch.get("attention_mask")
    padding_mask = batch.get("padding_mask")
    if attention_mask is not None:
        global_valid_mask = _validate_right_tail_mask(attention_mask, full_input_ids)
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool and not bool(((padding_mask == 0) | (padding_mask == 1)).all()):
                raise ValueError("Qwen3.8-Flash-Next CP padding_mask must contain only binary 0/1 values")
            padding_valid_mask = _validate_right_tail_mask(padding_mask.logical_not(), full_input_ids)
            if not bool(torch.equal(global_valid_mask, padding_valid_mask)):
                raise ValueError("Qwen3.8-Flash-Next CP attention_mask and padding_mask disagree")
    elif padding_mask is not None:
        if padding_mask.dtype != torch.bool and not bool(((padding_mask == 0) | (padding_mask == 1)).all()):
            raise ValueError("Qwen3.8-Flash-Next CP padding_mask must contain only binary 0/1 values")
        global_valid_mask = _validate_right_tail_mask(padding_mask.logical_not(), full_input_ids)
    else:
        global_valid_mask = torch.ones_like(full_input_ids, dtype=torch.bool)

    # Normalize even a user-provided integer padding mask to bool. The MoE
    # dispatcher applies bitwise ``~padding_mask``; retaining int 0/1 would
    # produce -1/-2 and incorrectly mark every padding position as active. This
    # also seeds a mask for the no-mask case so CP-added tail tokens are padded
    # with True by the shared sharder.
    batch["padding_mask"] = global_valid_mask.logical_not()

    original_sequence_length = full_input_ids.shape[1]
    ctx_factory, sharded_batch, layout = shard_batch_contiguous(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        pad_multiple=pad_multiple,
    )
    if layout is None or layout.padded_seq_len is None:
        raise RuntimeError("Qwen3.8-Flash-Next CP sharding did not report its padded global sequence length")

    cp_size = cp_mesh.size()
    cp_group = cp_mesh.get_group() if cp_size > 1 else None
    if cp_size > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Qwen3.8-Flash-Next CP size greater than one requires torch.distributed initialization")
        cp_rank = dist.get_rank(cp_group)
    else:
        cp_rank = 0
    local_sequence_length = layout.padded_seq_len // cp_size
    global_input_ids = _pad_right(full_input_ids, layout.padded_seq_len, padding_token_id)
    global_padding_mask = _pad_right(global_valid_mask.logical_not(), layout.padded_seq_len, True)
    local_sequence_start = cp_rank * local_sequence_length
    expected_local_ids = global_input_ids[:, local_sequence_start : local_sequence_start + local_sequence_length]
    if not bool(torch.equal(sharded_batch["input_ids"], expected_local_ids)):
        raise RuntimeError(
            "Qwen3.8-Flash-Next CP sharder produced local raw IDs inconsistent with its global PLE context"
        )
    if packed_cu_seqlens is not None:
        packed_cu_seqlens = packed_cu_seqlens.to(device=global_input_ids.device)
    sharded_batch["_qwen3_8_flash_next_cp_context"] = Qwen3_8_FlashNextCPContext(
        group=cp_group,
        rank=cp_rank,
        size=cp_size,
        global_input_ids=global_input_ids,
        global_padding_mask=global_padding_mask,
        local_sequence_start=local_sequence_start,
        local_sequence_length=local_sequence_length,
        global_cu_seqlens=packed_cu_seqlens,
    )
    return (
        ctx_factory,
        sharded_batch,
        ShardLayout(
            original_seq_len=original_sequence_length,
            padded_seq_len=layout.padded_seq_len,
        ),
    )


def qwen3_8_flash_next_cp_all_gather(
    tensor: torch.Tensor,
    context: Qwen3_8_FlashNextCPContext,
    *,
    sequence_dim: int = 1,
    differentiable: bool = True,
) -> torch.Tensor:
    """Gather equal contiguous sequence shards in global rank order.

    Args:
        tensor: Local tensor whose ``sequence_dim`` axis has length
            ``context.local_sequence_length`` (or a fixed compressed fraction
            of it shared by every rank). All non-sequence axes are replicated
            in shape across the CP group.
        context: Qwen3.8-Flash-Next contiguous CP metadata.
        sequence_dim: Axis on which rank-ordered parts are concatenated.
        differentiable: Use PyTorch's autograd-aware collective. Set ``False``
            only for frozen routing values or integer metadata.

    Returns:
        Tensor with the same axis order as ``tensor`` and a ``sequence_dim``
        length multiplied by ``context.size``.
    """
    if context.size <= 1:
        return tensor
    if context.group is None:
        raise RuntimeError("Qwen3.8-Flash-Next CP context is missing its process group")
    if differentiable:
        parts = differentiable_all_gather(tensor.contiguous(), group=context.group)
    else:
        parts = [torch.empty_like(tensor) for _ in range(context.size)]
        dist.all_gather(parts, tensor.contiguous(), group=context.group)
    return torch.cat(tuple(parts), dim=sequence_dim)


def qwen3_8_flash_next_cp_left_halo(
    tensor: torch.Tensor,
    context: Qwen3_8_FlashNextCPContext,
    *,
    history: int,
) -> torch.Tensor:
    """Collect only the preceding causal boundary needed by a local operator.

    Every rank contributes at most ``history`` trailing tokens. Autograd-aware
    All-Gather routes gradients from a later rank's halo use back to the rank
    that owns those tokens. Rank zero and globally short prefixes are zero
    padded on the left.

    Args:
        tensor: Local sequence tensor of shape ``[batch, local_sequence,
            channels]`` using contiguous rank order.
        context: Qwen3.8-Flash-Next contiguous CP metadata.
        history: Number of immediately preceding global tokens required.

    Returns:
        Left context of shape ``[batch, history, channels]``. The result does
        not include any token from the current rank.
    """
    if tensor.ndim != 3 or tensor.shape[1] != context.local_sequence_length:
        raise ValueError(
            "Qwen3.8-Flash-Next CP halo input must be [batch, local_sequence, channels]; "
            f"got {tuple(tensor.shape)} for local_sequence={context.local_sequence_length}"
        )
    if history < 0:
        raise ValueError(f"Qwen3.8-Flash-Next CP halo history must be non-negative, got {history}")
    if history == 0:
        return tensor[:, :0]
    if context.size <= 1:
        return tensor.new_zeros((tensor.shape[0], history, tensor.shape[2]))

    tail_length = min(history, context.local_sequence_length)
    gathered_tails = qwen3_8_flash_next_cp_all_gather(
        tensor[:, -tail_length:],
        context,
        sequence_dim=1,
        differentiable=True,
    ).unflatten(1, (context.size, tail_length))
    preceding = gathered_tails[:, : context.rank].flatten(1, 2)
    preceding = preceding[:, -history:]
    missing = history - preceding.shape[1]
    if missing:
        preceding = torch.cat(
            (tensor.new_zeros((tensor.shape[0], missing, tensor.shape[2])), preceding),
            dim=1,
        )

    # Rank zero has no real predecessor and would otherwise drop the collective
    # from its autograd graph, making backward collective participation asymmetric.
    collective_anchor = gathered_tails[:, :, :0].sum()
    return preceding + collective_anchor


def packed_boundaries_from_seq_lens(
    seq_lens: torch.Tensor,
    *,
    total_tokens: int | None = None,
    sentinel: int = -1000,
) -> torch.Tensor:
    """Convert loader ``seq_lens`` metadata to physical document boundaries.

    The THD packer concatenates documents contiguously and pads only the pack
    tail, so physical boundaries follow the REAL lengths (``seq_lens``); the
    loader's ``seq_lens_padded`` is TE-specific virtual-layout metadata and
    must not be used for physical offsets. When ``total_tokens`` exceeds the
    packed length, the trailing pack padding becomes its own segment so pad
    tokens never join a real document.

    Args:
        seq_lens: Real per-document lengths ``[num_docs]`` or ``[1, num_docs]``.
        total_tokens: Physical row length including trailing pack padding.
        sentinel: Filler value marking unused length slots.

    Returns:
        int64 boundaries ``[num_docs (+1 pad segment) + 1]`` starting at zero.
    """
    widths = seq_lens.reshape(-1).to(dtype=torch.long)
    widths = widths[widths != sentinel]
    if widths.numel() == 0 or bool((widths <= 0).any()):
        raise ValueError(f"packed seq_lens must contain positive document widths, got {widths.tolist()}")
    boundaries = torch.zeros(widths.numel() + 1, dtype=torch.long, device=widths.device)
    boundaries[1:] = torch.cumsum(widths, dim=0)
    if total_tokens is not None:
        packed_length = int(boundaries[-1])
        if packed_length > total_tokens:
            raise ValueError(f"packed seq_lens sum {packed_length} exceeds the physical row length {total_tokens}")
        if packed_length < total_tokens:
            boundaries = torch.cat([boundaries, boundaries.new_tensor([total_tokens])])
    return boundaries


__all__ = [
    "Qwen3_8_FlashNextCPContext",
    "packed_boundaries_from_seq_lens",
    "qwen3_8_flash_next_cp_all_gather",
    "qwen3_8_flash_next_cp_left_halo",
    "shard_batch_for_qwen3_8_flash_next_cp",
]
