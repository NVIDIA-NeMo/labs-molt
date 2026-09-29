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

"""Synchronization helpers for parameters replicated across tensor parallel ranks."""

from collections.abc import Iterator
from itertools import chain
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

_TP_REPLICA_GRAD_REDUCTION_ATTR = "_nemo_tp_replica_grad_reduction"
_MODEL_OWNED_GRAD_DIVISOR_ATTR = "_nemo_model_owned_grad_divisor"
_MAX_FLAT_BUFFER_BYTES = 256 * 1024 * 1024


def mark_tp_replica_gradient_reduction(
    module: torch.nn.Module,
    reduction: Literal["mean", "sum"],
) -> None:
    """Mark direct parameters of a module with their TP-replica reduction semantic.

    Args:
        module: Module whose direct parameters are replicated across TP ranks.
        reduction: ``"mean"`` for redundant full computation or ``"sum"`` for
            disjoint partial contributions.
    """
    setattr(module, _TP_REPLICA_GRAD_REDUCTION_ATTR, reduction)


def exclude_from_tp_replica_sync(module: torch.nn.Module) -> None:
    """Exclude an owner-sharded module subtree from TP replica synchronization.

    Some parameter owners use a mesh that folds physical TP ranks into another
    logical axis. Their local tensors are different shards, even though they are
    not DTensors carrying an explicit ``tp`` placement. Marking the complete
    subtree keeps both parameter and buffer synchronization from treating those
    owner-local tensors as TP replicas.

    Args:
        module: Owner-sharded module whose current subtree must remain local.
    """
    for child in module.modules():
        setattr(child, _TP_REPLICA_GRAD_REDUCTION_ATTR, "skip")


def _get_tp_mesh(device_mesh: DeviceMesh | None, tp_axis_name: str) -> DeviceMesh | None:
    """Return a non-trivial TP submesh when one is present."""
    if not isinstance(device_mesh, DeviceMesh) or tp_axis_name not in (device_mesh.mesh_dim_names or ()):
        return None
    tp_mesh = device_mesh[tp_axis_name]
    return tp_mesh if tp_mesh.size() > 1 else None


def _is_tp_replicated(
    tensor: torch.Tensor,
    tp_group_ranks: tuple[int, ...],
    current_rank: int,
    tp_axis_name: str,
) -> bool:
    """Return whether rank-local tensor storage is replicated across TP peers.

    A DTensor may live on a DP-only submesh even though physical copies exist for
    every TP coordinate. Conversely, an MoE mesh may fold the TP ranks into axes
    named ``ep`` or ``ep_shard``. Comparing mesh coordinates and placements keeps
    both cases explicit instead of assuming every unnamed TP dimension is a copy.

    Args:
        tensor: Tensor or DTensor of arbitrary global and rank-local shape.
        tp_group_ranks: Ordered global ranks in the current TP group.
        current_rank: Current global rank.
        tp_axis_name: Name of the tensor-parallel mesh dimension.

    Returns:
        Whether every TP peer represented in the tensor mesh differs only along
        dimensions carrying ``Replicate`` placements.
    """
    if not isinstance(tensor, DTensor):
        return True
    mesh_names = tensor.device_mesh.mesh_dim_names or ()
    if tp_axis_name in mesh_names:
        tp_dim = list(mesh_names).index(tp_axis_name)
        return isinstance(tensor.placements[tp_dim], Replicate)

    mesh_ranks = tensor.device_mesh.mesh
    current_coordinate = (mesh_ranks == current_rank).nonzero(as_tuple=False)
    if current_coordinate.numel() == 0:
        raise RuntimeError(f"Current rank {current_rank} is absent from a local DTensor mesh")
    current_coordinate = current_coordinate[0].tolist()
    for peer_rank in tp_group_ranks:
        peer_coordinate = (mesh_ranks == peer_rank).nonzero(as_tuple=False)
        if peer_coordinate.numel() == 0:
            continue
        for mesh_dim, (current_index, peer_index) in enumerate(zip(current_coordinate, peer_coordinate[0].tolist())):
            if current_index != peer_index and not isinstance(tensor.placements[mesh_dim], Replicate):
                return False
    return True


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return rank-local storage for a Tensor or DTensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _iter_unique_parameters(
    model_parts: list[torch.nn.Module],
) -> Iterator[tuple[torch.nn.Parameter, Literal["mean", "sum"]]]:
    """Yield unmarked parameters once with their module reduction semantic."""
    seen: dict[int, Literal["mean", "sum"]] = {}
    for model_part in model_parts:
        for module in model_part.modules():
            reduction = getattr(module, _TP_REPLICA_GRAD_REDUCTION_ATTR, "mean")
            if reduction == "skip":
                continue
            if reduction not in ("mean", "sum"):
                raise ValueError(f"Unsupported TP replica gradient reduction: {reduction!r}")
            for parameter in module.parameters(recurse=False):
                if getattr(parameter, _MODEL_OWNED_GRAD_DIVISOR_ATTR, None) is not None:
                    continue
                parameter_id = id(parameter)
                prior_reduction = seen.get(parameter_id)
                if prior_reduction is not None:
                    if prior_reduction != reduction:
                        raise RuntimeError(
                            "A shared parameter has conflicting TP replica gradient reductions: "
                            f"{prior_reduction!r} and {reduction!r}"
                        )
                    continue
                seen[parameter_id] = reduction
                yield parameter, reduction


def _iter_unique_parameters_by_name(
    model_parts: list[torch.nn.Module],
) -> Iterator[tuple[torch.nn.Parameter, Literal["mean", "sum"]]]:
    """Yield parameters once in rank-stable fully qualified name order."""
    parameter_names: dict[int, tuple[int, str]] = {}
    for part_index, model_part in enumerate(model_parts):
        for name, parameter in model_part.named_parameters(remove_duplicate=False):
            parameter_id = id(parameter)
            parameter_name = (part_index, name)
            parameter_names[parameter_id] = min(parameter_name, parameter_names.get(parameter_id, parameter_name))

    yield from sorted(_iter_unique_parameters(model_parts), key=lambda entry: parameter_names[id(entry[0])])


def _iter_unique_buffers(model_parts: list[torch.nn.Module]) -> Iterator[torch.Tensor]:
    """Yield model buffers once in rank-stable fully qualified name order."""
    excluded_buffer_ids = {
        id(buffer)
        for model_part in model_parts
        for module in model_part.modules()
        if getattr(module, _TP_REPLICA_GRAD_REDUCTION_ATTR, "mean") == "skip"
        for buffer in module.buffers(recurse=False)
    }
    named_buffers = [
        (part_index, name, buffer)
        for part_index, model_part in enumerate(model_parts)
        for name, buffer in model_part.named_buffers(remove_duplicate=False)
    ]
    seen: set[int] = set()
    for _, _, buffer in sorted(named_buffers, key=lambda entry: (entry[0], entry[1])):
        buffer_id = id(buffer)
        if buffer_id in seen or buffer_id in excluded_buffer_ids:
            continue
        seen.add(buffer_id)
        yield buffer


@torch.no_grad()
def broadcast_tp_replicas(
    model_parts: list[torch.nn.Module],
    device_mesh: DeviceMesh | None,
    tp_axis_name: str = "tp",
) -> int:
    """Broadcast replicated parameters and buffers from the first TP rank.

    Intended TP shards are excluded by their DTensor placement, and model-owned
    shards are excluded by their explicit parameter or module marker. The
    collective operates on each tensor's rank-local storage, whose shape is
    identical within a TP group even when another mesh axis shards the global
    tensor.

    Args:
        model_parts: Local pipeline-stage modules containing tensors of arbitrary
            shape. Shared tensors are visited once.
        device_mesh: Root mesh containing the tensor-parallel axis.
        tp_axis_name: Name of the tensor-parallel mesh dimension.

    Returns:
        Number of rank-local tensors broadcast on this rank.
    """
    tp_mesh = _get_tp_mesh(device_mesh, tp_axis_name)
    if tp_mesh is None:
        return 0

    group = tp_mesh.get_group()
    source_rank = dist.get_global_rank(group, 0)
    tp_group_ranks = tuple(dist.get_process_group_ranks(group))
    current_rank = dist.get_rank()
    placement_cache: dict[tuple[int, tuple[object, ...]], bool] = {}

    def is_replicated(tensor: torch.Tensor) -> bool:
        if not isinstance(tensor, DTensor):
            return True
        cache_key = (id(tensor.device_mesh), tuple(tensor.placements))
        if cache_key not in placement_cache:
            placement_cache[cache_key] = _is_tp_replicated(
                tensor,
                tp_group_ranks,
                current_rank,
                tp_axis_name,
            )
        return placement_cache[cache_key]

    synchronized = 0
    parameters = (parameter for parameter, _ in _iter_unique_parameters_by_name(model_parts))
    for tensor in chain(parameters, _iter_unique_buffers(model_parts)):
        if not is_replicated(tensor):
            continue
        local = _local_tensor(tensor)
        if local.is_meta or local.numel() == 0:
            continue
        if local.device.type == tp_mesh.device_type:
            dist.broadcast(local, src=source_rank, group=group)
        else:
            communication_tensor = local.to(device=tp_mesh.device_type)
            dist.broadcast(communication_tensor, src=source_rank, group=group)
            local.copy_(communication_tensor.to(device=local.device))
        synchronized += 1
    return synchronized


