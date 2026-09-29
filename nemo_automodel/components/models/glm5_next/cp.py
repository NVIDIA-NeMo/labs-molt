# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contiguous packed context parallelism for GLM-5.3-Flash.

Kimi Delta Attention carries recurrent and short-convolution state from left to
right, so the generic load-balanced CP permutation is not valid.  This module
keeps one contiguous token interval per CP rank and one global document-id map
that both KDA and KPool-DSA consume.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout
from nemo_automodel.shared.import_utils import safe_import_from

_PAD_DOC_ID = 0
_FLA_CP_AVAILABLE, _build_cp_context = safe_import_from(
    "fla.ops.cp",
    "build_cp_context",
    msg="GLM-5.3 context parallelism requires the `fla` optional dependency.",
)


@dataclass
class Glm5NextPackedContext:
    """Global packed-document layout for one model step.

    Attributes:
        doc_ids: Integer document ids ``[batch, global_sequence]``. Zero marks
            padding and positive values identify independent packed documents.
        seq_start: Global offset of this rank's contiguous local token interval.
        cp_size: Number of context-parallel ranks.
        original_seq_len: Sequence length before CP divisibility padding.
    """

    doc_ids: torch.Tensor
    seq_start: int = 0
    cp_size: int = 1
    original_seq_len: int | None = None
    # FSDP mixed-precision hooks recursively rebuild dataclass kwargs with
    # ``dataclasses.replace``.  Keep the cache constructor-visible so that
    # transform remains valid after packed metadata enters an FSDP root.
    _cu_seqlens: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @property
    def cp_enabled(self) -> bool:
        """Return whether the sequence is split across more than one rank."""
        return self.cp_size > 1

    @property
    def local_seq_len(self) -> int:
        """Return the padded local sequence length."""
        return self.doc_ids.shape[1] // self.cp_size

    @property
    def local_doc_ids(self) -> torch.Tensor:
        """Return document ids ``[batch, local_sequence]`` for this rank."""
        return self.doc_ids[:, self.seq_start : self.seq_start + self.local_seq_len]

    def row_cu_seqlens(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return device/CPU segment boundaries for one packed batch row."""
        cached = self._cu_seqlens.get(row)
        if cached is None:
            device_boundaries = segment_cu_seqlens(self.doc_ids[row]).to(torch.long)
            cached = (device_boundaries, device_boundaries.cpu())
            self._cu_seqlens[row] = cached
        return cached


def doc_ids_from_seq_lens(seq_lens: torch.Tensor, seq_len: int, *, padding_value: int = -1000) -> torch.Tensor:
    """Convert per-document lengths ``[batch, documents]`` to ids ``[batch, sequence]``."""
    if seq_lens.ndim == 1:
        seq_lens = seq_lens.unsqueeze(0)
    doc_ids = torch.zeros((seq_lens.shape[0], seq_len), dtype=torch.int32, device=seq_lens.device)
    for row in range(seq_lens.shape[0]):
        offset = 0
        lengths = seq_lens[row]
        lengths = lengths[lengths != padding_value]
        for doc_index, length in enumerate(lengths.tolist()):
            if length <= 0 or offset >= seq_len:
                continue
            end = min(offset + int(length), seq_len)
            doc_ids[row, offset:end] = doc_index + 1
            offset = end
    return doc_ids


def doc_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Convert cumulative document boundaries to ids ``[1, sequence]``."""
    boundaries = cu_seqlens.flatten().to(torch.long)
    boundaries = boundaries[boundaries >= 0]
    if boundaries.numel() < 2:
        return torch.ones((1, seq_len), dtype=torch.int32, device=cu_seqlens.device)
    positions = torch.arange(seq_len, device=cu_seqlens.device)
    doc_ids = torch.bucketize(positions, boundaries[1:], right=True) + 1
    doc_ids = torch.where(positions < boundaries[-1], doc_ids, torch.zeros_like(doc_ids))
    return doc_ids.to(torch.int32).unsqueeze(0)


def segment_cu_seqlens(doc_ids_row: torch.Tensor) -> torch.Tensor:
    """Return boundaries for consecutive document-id runs covering the full row."""
    if doc_ids_row.numel() == 0:
        return torch.zeros(1, dtype=torch.int32, device=doc_ids_row.device)
    starts = torch.nonzero(doc_ids_row[1:] != doc_ids_row[:-1], as_tuple=False).flatten() + 1
    return torch.cat(
        (
            torch.zeros(1, dtype=starts.dtype, device=starts.device),
            starts,
            torch.full((1,), doc_ids_row.numel(), dtype=starts.dtype, device=starts.device),
        )
    ).to(torch.int32)


def build_fla_cp_context(
    packed_context: Glm5NextPackedContext,
    row: int,
    cp_group: Any,
    conv_kernel_size: int,
):
    """Build FLA's KDA context for one batch row.

    Args:
        packed_context: Global document layout.
        row: Batch row being executed.
        cp_group: Context-parallel process group.
        conv_kernel_size: Short-convolution width used for the left halo.

    Returns:
        FLA ``FLACPContext`` carrying segment and process-group metadata.
    """
    if not _FLA_CP_AVAILABLE:
        raise RuntimeError("GLM-5.3 context parallelism requires the `fla` optional dependency")

    cu_seqlens, cu_seqlens_cpu = packed_context.row_cu_seqlens(row)
    return _build_cp_context(
        cu_seqlens=cu_seqlens,
        group=cp_group,
        conv1d_kernel_size=conv_kernel_size,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )


class _AllGatherSequence(torch.autograd.Function):
    """Autograd-aware all-gather of equal contiguous sequence shards."""

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group: Any, dim: int) -> torch.Tensor:
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered, local_tensor.contiguous(), group=group)
        ctx.group = group
        ctx.dim = dim
        ctx.rank = dist.get_rank(group)
        ctx.local_size = local_tensor.shape[dim]
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.rank * ctx.local_size
        return grad_output.narrow(ctx.dim, start, ctx.local_size).contiguous(), None, None


class _AllGatherBackwardAnchor(torch.autograd.Function):
    """Return zero while retaining a backward edge to a gathered tensor."""

    @staticmethod
    def forward(ctx, gathered: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = gathered.shape
        return gathered.new_zeros((gathered.shape[0], 1, 1))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.new_zeros(ctx.input_shape)


def all_gather_sequence(tensor: torch.Tensor, cp_group: Any, *, dim: int = 1) -> torch.Tensor:
    """Gather a sequence-sharded tensor while preserving K/V gradient flow."""
    return _AllGatherSequence.apply(tensor, cp_group, dim)


def all_gather_backward_anchor(gathered: torch.Tensor) -> torch.Tensor:
    """Create a zero-valued dependency that keeps gather backward collective-safe."""
    return _AllGatherBackwardAnchor.apply(gathered)


def _pad_sequence_dim(tensor: torch.Tensor, pad_len: int, value: float | int | bool) -> torch.Tensor:
    if pad_len <= 0:
        return tensor
    pad = torch.full(
        (tensor.shape[0], pad_len, *tensor.shape[2:]),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, pad), dim=1)


def _normalize_batch_axis(batch: dict[str, Any]) -> None:
    """Restore the placeholder batch axis used by THD VLM collaters."""
    input_ids = batch["input_ids"]
    if input_ids.ndim != 1:
        return
    sequence_length = input_ids.shape[0]
    # Media tensors are flattened over vision patches, not text tokens.  Restrict
    # this repair to fields whose leading dimension is contractually the text
    # sequence so an image with the same number of patches is never reshaped.
    for key in ("input_ids", "labels", "position_ids", "attention_mask", "padding_mask", "_packed_seq_ids"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == sequence_length:
            batch[key] = value.unsqueeze(0)


def _global_doc_ids_from_batch(batch: dict[str, Any], seq_len: int) -> torch.Tensor:
    """Resolve the global document map before removing packing metadata."""
    packed_ids = batch.get("_packed_seq_ids")
    if isinstance(packed_ids, torch.Tensor):
        return packed_ids.to(torch.int32) if packed_ids.ndim == 2 else packed_ids.unsqueeze(0).to(torch.int32)

    attention_mask = batch.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        # Binary masks naturally describe one document; indexed masks already
        # carry one-based packed document ids.
        return attention_mask.to(torch.int32)

    seq_lens = batch.get("seq_lens_padded", batch.get("seq_lens"))
    if isinstance(seq_lens, torch.Tensor):
        return doc_ids_from_seq_lens(seq_lens, seq_len)

    cu_seqlens = batch.get("cu_seqlens")
    if isinstance(cu_seqlens, torch.Tensor):
        return doc_ids_from_cu_seqlens(cu_seqlens, seq_len)

    doc_ids = torch.ones(
        (batch["input_ids"].shape[0], seq_len),
        dtype=torch.int32,
        device=batch["input_ids"].device,
    )
    padding_mask = batch.get("padding_mask")
    if isinstance(padding_mask, torch.Tensor):
        doc_ids.masked_fill_(padding_mask.bool(), _PAD_DOC_ID)
    return doc_ids


def shard_batch_for_glm5_next_cp(
    cp_mesh,
    tp_mesh,
    batch: dict[str, Any],
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    shard_primary: bool = False,
):
    """Contiguously shard packed GLM-5.3 token streams.

    The top-level VLM uses ``shard_primary=False``: image features must be
    spliced into the full embedding sequence inside ``forward`` before that
    differentiable primary stream is sliced. Labels and other no-grad token
    streams are still sharded here.

    Args:
        cp_mesh: One-dimensional CP mesh or ``None``.
        tp_mesh: Unused tensor-parallel mesh, accepted by the sharder protocol.
        batch: Batch containing token tensors with shape ``[batch, sequence]``.
        loss_mask: Optional loss mask ``[batch, sequence]``.
        padding_token_id: Fill value for padded token ids.
        shard_primary: Whether to shard ``input_ids`` in this function.

    Returns:
        Context factory, mutated local batch, and the global shard layout.
    """
    del tp_mesh
    _normalize_batch_axis(batch)
    input_ids = batch["input_ids"]
    seq_len = input_ids.shape[1]
    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    doc_ids = _global_doc_ids_from_batch(batch, seq_len)

    for key in (
        "attention_mask",
        "_packed_seq_ids",
        "seq_lens",
        "seq_lens_padded",
        "cu_seqlens",
        "cu_seqlens_padded",
        "max_seqlen",
        "qkv_format",
    ):
        batch.pop(key, None)

    if "position_ids" not in batch:
        batch["position_ids"] = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

    pad_len = (-seq_len) % cp_size
    padded_seq_len = seq_len + pad_len
    if pad_len:
        for key, pad_value in (("labels", -100), ("position_ids", 0), ("padding_mask", True)):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == seq_len:
                batch[key] = _pad_sequence_dim(value, pad_len, pad_value)
        if shard_primary:
            batch["input_ids"] = _pad_sequence_dim(batch["input_ids"], pad_len, padding_token_id)
        doc_ids = _pad_sequence_dim(doc_ids, pad_len, _PAD_DOC_ID)
        if isinstance(loss_mask, torch.Tensor):
            loss_mask = _pad_sequence_dim(loss_mask, pad_len, 0)

    batch.setdefault("padding_mask", doc_ids <= _PAD_DOC_ID)
    seq_start = 0 if cp_mesh is None else cp_mesh.get_local_rank() * (padded_seq_len // cp_size)
    batch["glm5_next_packed_context"] = Glm5NextPackedContext(
        doc_ids=doc_ids,
        seq_start=seq_start,
        cp_size=cp_size,
        original_seq_len=seq_len,
    )

    local_seq_len = padded_seq_len // cp_size
    seq_end = seq_start + local_seq_len
    shard_keys = ["labels", "position_ids", "padding_mask"]
    if shard_primary:
        shard_keys.append("input_ids")
    for key in shard_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == padded_seq_len:
            batch[key] = value[:, seq_start:seq_end].contiguous()
    if isinstance(loss_mask, torch.Tensor):
        batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()

    return contextlib.nullcontext, batch, ShardLayout(original_seq_len=seq_len, padded_seq_len=padded_seq_len)
