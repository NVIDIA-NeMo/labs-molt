# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from typing import Optional

import torch
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh


def is_dtensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a DTensor."""
    return isinstance(tensor, DTensor)


def get_submesh(device_mesh: DeviceMesh, dims: tuple[str, ...]) -> DeviceMesh:
    """Access a submesh by dim names from the given mesh."""
    return device_mesh[dims]


def _get_expert_mesh_dim_index(dtensor: DTensor) -> int | None:
    """Find the device-mesh dimension that partitions a grouped expert tensor.

    Args:
        dtensor: DTensor with global shape [experts, ...] and arbitrary trailing dimensions. A named ``ep`` mesh
            dimension owns the expert partition when present; otherwise a ``Shard(0)`` placement identifies an
            EP-free FSDP partition of the experts axis.

    Returns:
        Index of the expert-partitioning mesh dimension, or ``None`` when every rank retains all experts.
    """
    mesh_dim_names = tuple(dtensor.device_mesh.mesh_dim_names)
    if "ep" in mesh_dim_names:
        return mesh_dim_names.index("ep")

    return next(
        (
            dim_idx
            for dim_idx, placement in enumerate(dtensor.placements)
            if isinstance(placement, Shard) and placement.dim == 0
        ),
        None,
    )


def get_expert_slice_for_rank(experts_tensor: torch.Tensor, n_experts: int) -> tuple[torch.Tensor, int, int]:
    """
    Get the slice of experts present on the current rank for a DTensor.

    For non-DTensors, returns the full tensor with start_expert=0, end_expert=n_experts.
    For DTensors sharded along the expert dimension (dim=0), returns only the local experts. The mesh dimension
    may be the explicit ``ep`` dimension or an EP-free FSDP dimension such as ``dp_shard_cp``.

    Args:
        experts_tensor: Tensor with global shape [experts, ...] and arbitrary trailing dimensions. For a DTensor,
            the per-rank local shape is [local_experts, ...] when a mesh dimension uses ``Shard(0)``; other
            placements retain ``experts`` on the first local axis.
        n_experts: Total number of experts across all ranks.

    Returns:
        Tuple containing the local tensor of shape [local_experts, ...], the inclusive global ID of its first
        expert, and the exclusive global ID after its last expert. The local tensor aliases the input's local
        storage.
    """
    if not is_dtensor(experts_tensor):
        return experts_tensor, 0, n_experts

    dtensor = experts_tensor
    local_tensor = dtensor.to_local()

    device_mesh = dtensor.device_mesh
    expert_mesh_dim_idx = _get_expert_mesh_dim_index(dtensor)
    if expert_mesh_dim_idx is None:
        return local_tensor, 0, n_experts

    expert_mesh_dim_name = device_mesh.mesh_dim_names[expert_mesh_dim_idx]
    expert_mesh = get_submesh(device_mesh, (expert_mesh_dim_name,))
    placement = dtensor.placements[expert_mesh_dim_idx]
    if isinstance(placement, Shard) and placement.dim == 0:
        # Tensor is sharded along expert dimension
        current_rank = expert_mesh.get_local_rank()
        world_size = expert_mesh.size()

        # Calculate expert range for this rank
        experts_per_rank = n_experts // world_size
        remainder = n_experts % world_size

        if current_rank < remainder:
            # First `remainder` ranks get one extra expert
            experts_on_rank = experts_per_rank + 1
            start_expert = current_rank * experts_on_rank
        else:
            # Remaining ranks get standard number of experts
            experts_on_rank = experts_per_rank
            start_expert = remainder * (experts_per_rank + 1) + (current_rank - remainder) * experts_per_rank

        end_expert = start_expert + experts_on_rank
        return local_tensor, start_expert, end_expert
    elif isinstance(placement, Replicate):
        # Tensor is replicated - all ranks have full data
        return local_tensor, 0, n_experts
    # Other sharding patterns retain every expert on the rank.
    return local_tensor, 0, n_experts


def split_experts_weights_dtensor_aware(weight: torch.Tensor, n_experts: int) -> tuple[list[torch.Tensor], list[int]]:
    """
    Split expert weights, handling both regular tensors and DTensors.

    For DTensors sharded on the expert axis, only splits the experts present on the current rank. Other DTensor
    placements retain all experts and are adjusted after removing the expert tensor dimension.

    Args:
        weight: Tensor with global shape [experts, ...] and arbitrary trailing dimensions. A ``Shard(0)`` DTensor
            has local shape [local_experts, ...]; inner-axis shards retain shape [experts, ...] locally.
        n_experts: Total number of experts across all ranks.

    Returns:
        Tuple containing per-expert tensors of shape [...] and their global expert IDs. Each output aliases the
        corresponding local input slice. Any retained DTensor ``Shard(d)`` placement becomes ``Shard(d - 1)``
        after the experts axis is removed.
    """
    local_tensor, start_expert, end_expert = get_expert_slice_for_rank(weight, n_experts)
    local_n_experts = end_expert - start_expert
    if local_tensor.shape[0] != local_n_experts:
        raise ValueError(
            f"Expected local tensor first dimension to be {local_n_experts} "
            f"(experts {start_expert}:{end_expert}), got {local_tensor.shape[0]}"
        )

    split_weights = []
    expert_ids = []

    # Check if weight is a DTensor to preserve placements
    is_weight_dtensor = is_dtensor(weight)
    if is_weight_dtensor:
        device_mesh = weight.device_mesh
        original_placements = weight.placements
        mesh_dim_names = list(weight.device_mesh.mesh_dim_names)

        # Remove the mesh dimension that partitions experts. With EP disabled, this may be an FSDP dimension.
        expert_mesh_dim_idx = _get_expert_mesh_dim_index(weight)
        if expert_mesh_dim_idx is None:
            remaining_mesh_dims = mesh_dim_names
            new_placements_template = original_placements
        else:
            remaining_mesh_dims = mesh_dim_names[:expert_mesh_dim_idx] + mesh_dim_names[expert_mesh_dim_idx + 1 :]
            new_placements_template = (
                original_placements[:expert_mesh_dim_idx] + original_placements[expert_mesh_dim_idx + 1 :]
            )

        # Build a device mesh without the dimension that partitioned experts.
        if remaining_mesh_dims and any(map(lambda x: get_submesh(device_mesh, (x,)).size() > 1, remaining_mesh_dims)):
            new_device_mesh = get_submesh(device_mesh, tuple(remaining_mesh_dims))
        else:
            new_device_mesh = None
            is_weight_dtensor = False

    for i in range(local_n_experts):
        expert_weight = local_tensor[i]  # Shape: [...] (expert dimension removed)
        global_expert_id = start_expert + i

        # If original weight was DTensor, wrap the sliced expert weight in DTensor with adjusted placements
        if is_weight_dtensor:
            new_placements = []

            for placement in new_placements_template:
                if isinstance(placement, Shard) and placement.dim > 0:
                    # Adjust shard dimensions since we removed dimension 0
                    new_placements.append(Shard(placement.dim - 1))
                elif isinstance(placement, Shard) and placement.dim == 0:
                    # Can't shard on dim 0 anymore since we removed it
                    new_placements.append(Replicate())
                else:
                    # Keep other placements as-is (e.g., Replicate)
                    new_placements.append(placement)

            # Create DTensor with new device mesh and placements
            expert_weight = DTensor.from_local(expert_weight, new_device_mesh, new_placements)

        split_weights.append(expert_weight)
        expert_ids.append(global_expert_id)

    return split_weights, expert_ids


def validate_dtensor_expert_sharding(tensor: torch.Tensor, expected_experts: int, tensor_name: str = "tensor") -> bool:
    """
    Validate that a DTensor is properly sharded for expert parallelism.

    Args:
        tensor: Tensor to validate
        expected_experts: Expected total number of experts
        tensor_name: Name for error messages

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not is_dtensor(tensor):
        if tensor.shape[0] != expected_experts:
            raise ValueError(f"{tensor_name} has shape {tensor.shape[0]} experts, expected {expected_experts}")
        return True

    dtensor = tensor

    if dtensor.shape[0] != expected_experts:
        raise ValueError(f"{tensor_name} global shape has {dtensor.shape[0]} experts, expected {expected_experts}")

    placement = dtensor.placements[-1] if dtensor.placements else None

    if isinstance(placement, Shard) and placement.dim == 0:
        return True
    elif isinstance(placement, Replicate):
        return True
    else:
        raise ValueError(
            f"{tensor_name} has unsupported DTensor placement: {placement}. "
            f"Expected Shard(dim=0) or Replicate for expert parallelism."
        )


def create_dtensor_from_local(
    local_tensor: torch.Tensor, device_mesh: Optional["DeviceMesh"], rank: int | None = None
) -> torch.Tensor:
    """
    Create a DTensor from a local tensor for expert parallelism.

    Args:
        local_tensor: Local portion of the tensor on this rank
        device_mesh: Device mesh for DTensor creation
        rank: Current rank (for device placement)

    Returns:
        DTensor if device_mesh is provided and DTensor is available, otherwise local_tensor
    """
    if device_mesh is None:
        return local_tensor

    if rank is not None and torch.cuda.is_available():
        local_tensor = local_tensor.to(f"cuda:{torch.cuda.current_device()}")

    # Create placements based on device mesh dimensions
    placements = []
    ep_sharded = any(
        map(
            lambda x: x in device_mesh.mesh_dim_names and get_submesh(device_mesh, (x,)).size() > 1,
            ["ep_shard", "ep_replicate"],
        )
    )
    dim_names_for_placements = []
    mesh_dim_names = device_mesh.mesh_dim_names
    assert all(map(lambda x: x in ["ep", "ep_shard", "ep_replicate"], mesh_dim_names)), (
        f"Expected mesh dimension names to contain only 'ep', 'ep_shard', 'ep_replicate', got {mesh_dim_names}"
    )
    for _, dim_name in enumerate(mesh_dim_names):
        if dim_name == "ep":
            # Expert parallelism: shard across experts (dim 0)
            placements.append(Shard(0))
            dim_names_for_placements.append(dim_name)
        elif dim_name == "ep_shard":
            if ep_sharded:
                placements.append(Shard(1))
                dim_names_for_placements.append(dim_name)
        elif dim_name == "ep_replicate":
            if ep_sharded:
                # Expert replication: replicate across this dimension
                placements.append(Replicate())
                dim_names_for_placements.append(dim_name)
        else:
            raise ValueError(f"Unexpected mesh dimension name: {dim_name}")

    dtensor = DTensor.from_local(local_tensor, get_submesh(device_mesh, tuple(dim_names_for_placements)), placements)
    return dtensor


def get_expert_range_for_rank_from_mesh(device_mesh: Optional["DeviceMesh"], n_experts: int) -> tuple[int, int]:
    """
    Get the range of experts that should be loaded for the current rank.

    Args:
        device_mesh: Device mesh for expert parallelism
        n_experts: Total number of experts

    Returns:
        Tuple of (start_expert_id, end_expert_id) for this rank
    """
    if device_mesh is None:
        return 0, n_experts

    ep_mesh = get_submesh(device_mesh, ("ep",)) if "ep" in device_mesh.mesh_dim_names else device_mesh
    world_size = ep_mesh.size()
    rank = ep_mesh.get_local_rank()

    experts_per_rank = n_experts // world_size
    remainder = n_experts % world_size

    if rank < remainder:
        experts_per_rank += 1
        start_expert = rank * experts_per_rank
    else:
        start_expert = rank * experts_per_rank + remainder

    end_expert = start_expert + experts_per_rank
    return start_expert, end_expert


def should_load_expert_for_rank(expert_id: int, device_mesh: Optional["DeviceMesh"], n_experts: int) -> bool:
    """
    Check if a specific expert should be loaded on the current rank.

    Args:
        expert_id: The expert ID to check
        device_mesh: Device mesh for expert parallelism
        n_experts: Total number of experts

    Returns:
        True if this expert should be loaded on the current rank
    """
    start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
    return start_expert <= expert_id < end_expert
