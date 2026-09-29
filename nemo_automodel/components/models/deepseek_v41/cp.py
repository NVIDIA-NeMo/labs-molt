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

"""Contiguous context parallelism for the V4.1 text backbone."""

import contextlib
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_gather

from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout, shard_batch_contiguous
from nemo_automodel.components.models.deepseek_v41.packing import packed_layout


def gather_sequence(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Gather rank-ordered sequence shards, summing remote uses in backward.

    Args:
        tensor: Tensor of shape [batch, local_sequence, ...], with arbitrary
            trailing dimensions. Equal sequence lengths are required on every rank.
        group: CP process group, or None for an identity operation.

    Returns:
        Tensor of shape [batch, global_sequence, ...]. Floating activations retain
        autograd history; integer and boolean metadata are gathered without gradients.
        At CP1 the result aliases the input. No input is mutated.
    """
    if group is None or dist.get_world_size(group) == 1:
        return tensor
    if tensor.requires_grad:
        parts = all_gather(tensor.contiguous(), group=group)
    else:
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=1)


def shard_cp_batch(
    cp_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None,
    batch: dict[str, Any],
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int | None = 0,
    pad_multiple: int = 1,
    packed_alignment: int = 1,
    sync_packed_length: bool = False,
) -> tuple[Callable, dict[str, Any], ShardLayout]:
    """Prepare packed boundaries once, then keep one contiguous query shard.

    Args:
        cp_mesh: Context-parallel mesh, or None for a local packed forward.
        tp_mesh: Optional tensor-parallel mesh.
        batch: Text tensors input_ids, labels, position_ids and optional binary attention_mask
            [batch, global_sequence]. Packed input adds seq_lens and optional seq_lens_padded
            [batch, documents]. Labels must already be shifted independently within each document.
            Replaced in place with local tensors and document IDs [batch, local_sequence].
        loss_mask: Optional tensor of shape [batch, global_sequence].
        padding_token_id: Input token used for padding; None uses zero. Validity comes from metadata.
        pad_multiple: LCM of this model's active compression ratios.
        packed_alignment: Document alignment for compression groups, independent of CP size.
        sync_packed_length: Synchronize the physical packed length across WORLD for HybridEP's uniform input.

    Returns:
        Null context factory, the local batch with runtime cp_group, and its ShardLayout.
        Packed layouts retain original-to-aligned token coordinates [batch, original_sequence].
        Token tensors have independent storage; input tensor storage is never modified.
    """
    if batch.get("pixel_values") is not None or batch.get("inputs_embeds") is not None:
        raise ValueError("DeepSeek V4.1 context parallelism currently supports text input_ids only")
    padding_token_id = 0 if padding_token_id is None else padding_token_id
    packed = batch.get("seq_lens") is not None
    if not packed and (batch.get("cu_seqlens") is not None or batch.get("qkv_format") == "thd"):
        raise ValueError("DeepSeek V4.1 packed input requires [batch, sequence] tokens and seq_lens")
    input_positions = None
    if packed:
        lengths = batch["seq_lens"]
        real_lengths = lengths.clamp_min(0)
        row_lengths = ((real_lengths + packed_alignment - 1) // packed_alignment * packed_alignment).sum(-1)
        length = torch.maximum(row_lengths.max(), lengths.new_tensor(batch["input_ids"].shape[1]))
        if sync_packed_length and dist.is_initialized():
            dist.all_reduce(length, op=dist.ReduceOp.MAX)
        layout = packed_layout(
            lengths,
            seq_lens_padded=batch.get("seq_lens_padded"),
            input_shape=tuple(batch["input_ids"].shape),
            alignment=packed_alignment,
            minimum_length=int(length),
        )
        input_positions = layout.input_positions
        for key, fill in (("input_ids", padding_token_id), ("labels", -100), ("loss_mask", 0)):
            if key in batch:
                batch[key] = layout.pack(batch[key], fill=fill)
        if loss_mask is not None:
            loss_mask = layout.pack(loss_mask)
        batch["position_ids"] = layout.position_ids
        batch["packed_seq_ids"] = layout.sequence_ids
        batch["padding_mask"] = layout.sequence_ids == 0
        for key in ("seq_lens", "seq_lens_padded", "qkv_format", "attention_mask"):
            batch.pop(key, None)
    elif "padding_mask" not in batch and "attention_mask" not in batch:
        batch["padding_mask"] = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
    if cp_mesh is None:
        if loss_mask is not None:
            batch["loss_mask"] = loss_mask
        ctx = contextlib.nullcontext
        layout = ShardLayout(
            padded_seq_len=batch["input_ids"].shape[1],
            local_token_global_indices=torch.arange(batch["input_ids"].shape[1], device=batch["input_ids"].device),
        )
    else:
        ctx, batch, layout = shard_batch_contiguous(
            cp_mesh,
            tp_mesh,
            batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
            pad_multiple=pad_multiple,
            extra_seq_keys={"packed_seq_ids": 1} if packed else None,
            extra_pad_values={"packed_seq_ids": 0} if packed else None,
        )
    if input_positions is not None:
        layout = ShardLayout(
            padded_seq_len=layout.padded_seq_len,
            local_token_global_indices=layout.local_token_global_indices,
            input_token_stream_positions=input_positions,
        )
    batch["attention_mask"] = ~batch.pop("padding_mask")
    batch["cp_group"] = None if cp_mesh is None else cp_mesh.get_group()
    return ctx, batch, layout
