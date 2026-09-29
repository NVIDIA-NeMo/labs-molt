# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Device mesh construction and access utilities for distributed training."""

import datetime
import itertools
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from nemo_automodel.components.distributed.config import (
    DistributedStrategyConfig,
    FSDP2Config,
)
from nemo_automodel.components.distributed.mesh import MeshAxisName, ParallelismSizes

__all__ = [
    "_create_device_meshes",
    "_create_fsdp2_device_mesh",
    "_unflatten_compat",
    "get_flat_mesh",
    "get_submesh",
    "get_fsdp_dp_mesh",
]


def _degree(value: int | None) -> int:
    return value if isinstance(value, int) and value > 0 else 1


@dataclass(frozen=True)
class _MeshSpec:
    """Named mesh shape plus derived flattened axes."""

    shape: tuple[int, ...]
    axes: tuple[MeshAxisName, ...]
    flattened_axes: dict[MeshAxisName, tuple[MeshAxisName, ...]] = field(default_factory=dict)


def _create_device_meshes(
    strategy_config: DistributedStrategyConfig,
    parallelism: ParallelismSizes,
    *,
    world_size: int,
    timeout_minutes: int | None = None,
    ranks: list[int] | tuple[int, ...] | None = None,
) -> tuple[DeviceMesh | None, DeviceMesh | None]:
    """Create raw device meshes based on distributed config type."""
    if (
        parallelism.dp_replicate_size is not None
        and parallelism.dp_replicate_size > 1
        and not isinstance(strategy_config, FSDP2Config)
    ):
        raise ValueError("dp_replicate_size is only supported with FSDP2Config")

    if isinstance(strategy_config, FSDP2Config):
        return _create_fsdp2_device_mesh(
            parallelism,
            world_size=world_size,
            timeout_minutes=timeout_minutes,
            ranks=ranks,
        )
    else:
        raise ValueError(f"Unknown distributed strategy config type: {type(strategy_config)}")


def _infer_dp_size(
    dp_size: int | None,
    *,
    world_size: int,
    non_dp_size: int,
    expression: str,
    factors: tuple[int, ...],
) -> int:
    if dp_size is not None and dp_size > 0:
        return dp_size

    if world_size % non_dp_size != 0:
        factors_str = " * ".join(str(factor) for factor in factors)
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ({expression}) ({factors_str} = {non_dp_size})"
        )
    return world_size // non_dp_size


def _mesh_device_type() -> str:
    if dist.is_available() and dist.is_initialized():
        backend = str(dist.get_backend()).lower()
        return "cuda" if "nccl" in backend and torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _init_named_mesh(
    spec: _MeshSpec,
    *,
    timeout_minutes: int | None = None,
    ranks: list[int] | tuple[int, ...] | None = None,
) -> DeviceMesh:
    _validate_mesh_spec(spec)
    device_type = _mesh_device_type()
    backend_override = _nccl_backend_override(spec.axes, device_type=device_type, timeout_minutes=timeout_minutes)
    if ranks is None:
        device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=spec.shape,
            mesh_dim_names=spec.axes,
            backend_override=backend_override,
        )
    else:
        expected_size = 1
        for size in spec.shape:
            expected_size *= size
        if len(ranks) != expected_size:
            raise ValueError(f"mesh ranks ({len(ranks)}) must match mesh size ({expected_size})")
        if len(set(ranks)) != len(ranks):
            raise ValueError("mesh ranks must be unique")
        device_mesh = DeviceMesh(
            device_type=device_type,
            mesh=torch.tensor(ranks, dtype=torch.int64).reshape(spec.shape),
            mesh_dim_names=spec.axes,
            backend_override=backend_override,
        )
    if ranks is None:
        _register_flattened_axes(device_mesh, spec.flattened_axes, timeout_minutes=timeout_minutes)
    else:
        _register_flattened_axes_for_rank_subset(
            device_mesh,
            spec,
            ranks=ranks,
            timeout_minutes=timeout_minutes,
        )
    return device_mesh


def _nccl_backend_override(
    axes: tuple[str, ...],
    *,
    device_type: str,
    timeout_minutes: int | None,
):
    """Create per-axis NCCL options for DeviceMesh subgroups.

    ``init_process_group(timeout=...)`` configures the default process group, but
    ``init_device_mesh`` creates additional per-axis process groups. Without a
    backend override those groups keep PyTorch's default NCCL timeout.
    """
    if timeout_minutes is None or device_type != "cuda":
        return None

    timeout = datetime.timedelta(minutes=timeout_minutes)
    override = {}
    for axis in axes:
        options = dist.ProcessGroupNCCL.Options()
        options._timeout = timeout
        override[axis] = ("nccl", options)
    return override


def _can_reuse_default_group(group_size: int) -> bool:
    """Return whether a derived mesh dimension spans the default process group."""
    return dist.is_initialized() and group_size == dist.get_world_size()


def _validate_mesh_spec(spec: _MeshSpec) -> None:
    for shape, axis in zip(spec.shape, spec.axes):
        assert isinstance(shape, int), f"Expected {axis} to be an int, but got {type(shape)}"
        assert shape > 0, f"Expected {axis} > 0, got {shape}"


def _register_flattened_axes(
    device_mesh: DeviceMesh,
    flattened_axes: dict[MeshAxisName, tuple[MeshAxisName, ...]],
    *,
    timeout_minutes: int | None = None,
) -> None:
    if not flattened_axes:
        return
    if not hasattr(device_mesh, "_flatten_mapping"):
        device_mesh._flatten_mapping = {}
    backend_overrides = _nccl_backend_override(
        tuple(flattened_axes),
        device_type=device_mesh.device_type,
        timeout_minutes=timeout_minutes,
    )
    for flattened_axis, source_axes in flattened_axes.items():
        source_mesh = device_mesh[source_axes]
        backend_override = backend_overrides.get(flattened_axis) if backend_overrides else None
        # The default process group already has the configured timeout. Supplying
        # an override for the same world-sized ranks would allocate another NCCL communicator.
        if backend_override is not None and _can_reuse_default_group(source_mesh.size()):
            backend_override = None
        flattened_mesh = source_mesh._flatten(
            mesh_dim_name=flattened_axis,
            backend_override=backend_override,
        )
        device_mesh._flatten_mapping.setdefault(flattened_axis, flattened_mesh)


def _register_flattened_axes_for_rank_subset(
    device_mesh: DeviceMesh,
    spec: _MeshSpec,
    *,
    ranks: list[int] | tuple[int, ...],
    timeout_minutes: int | None,
) -> None:
    """Register flattened axes without slicing a mesh on non-member ranks.

    PyTorch cannot slice an explicit ``DeviceMesh`` on ranks outside that mesh.
    Separate-model jobs still need every global rank to create process groups in
    the same order, so construct each flattened subgroup explicitly and retain
    the subgroup local to the current rank.
    """
    if not spec.flattened_axes:
        return
    if not hasattr(device_mesh, "_flatten_mapping"):
        device_mesh._flatten_mapping = {}

    rank_tensor = torch.tensor(ranks, dtype=torch.int64).reshape(spec.shape)
    current_rank = dist.get_rank()
    for flattened_axis, source_axes in spec.flattened_axes.items():
        source_indices = tuple(spec.axes.index(axis) for axis in source_axes)
        other_indices = tuple(index for index in range(len(spec.axes)) if index not in source_indices)
        other_shapes = tuple(spec.shape[index] for index in other_indices)
        local_ranks = None
        local_group = None
        first_ranks = None
        for other_coordinate in itertools.product(*(range(size) for size in other_shapes)):
            index = [slice(None)] * len(spec.axes)
            for axis_index, coordinate in zip(other_indices, other_coordinate):
                index[axis_index] = coordinate
            subgroup_ranks = rank_tensor[tuple(index)].reshape(-1).tolist()
            first_ranks = first_ranks or subgroup_ranks
            group = dist.new_group(ranks=subgroup_ranks)
            if current_rank in subgroup_ranks:
                local_ranks = subgroup_ranks
                local_group = group

        flat_ranks = local_ranks or first_ranks
        flattened_mesh = DeviceMesh(
            device_mesh.device_type,
            mesh=torch.tensor(flat_ranks, dtype=torch.int64),
            mesh_dim_names=(flattened_axis,),
            _init_backend=False,
        )
        flattened_mesh._dim_group_names = [local_group.group_name] if local_group is not None else []
        device_mesh._flatten_mapping.setdefault(flattened_axis, flattened_mesh)


def _create_fsdp2_device_mesh(
    parallelism: ParallelismSizes,
    *,
    world_size: int,
    timeout_minutes: int | None = None,
    ranks: list[int] | tuple[int, ...] | None = None,
) -> tuple[DeviceMesh, DeviceMesh | None]:
    """Create the FSDP2 root mesh and optional MoE mesh."""
    tp_size = _degree(parallelism.tp_size)
    cp_size = _degree(parallelism.cp_size)
    pp_size = _degree(parallelism.pp_size)
    ep_size = _degree(parallelism.ep_size)
    dp_replicate_size = _degree(parallelism.dp_replicate_size)
    dp_size = _infer_dp_size(
        parallelism.dp_size,
        world_size=world_size,
        non_dp_size=tp_size * cp_size * pp_size,
        expression="tp_size * cp_size * pp_size",
        factors=(tp_size, cp_size, pp_size),
    )

    if dp_size % dp_replicate_size != 0:
        raise ValueError("dp_size must be a multiple of dp_replicate_size")
    if dp_replicate_size >= dp_size and dp_replicate_size != 1:
        raise ValueError(
            f"dp_replicate_size={dp_replicate_size} must be less than dp_size={dp_size} "
            "since DDP usecase is not supported by FSDP2"
        )

    non_pp_size = dp_size * cp_size * tp_size
    if non_pp_size % ep_size != 0:
        raise ValueError(f"{non_pp_size=} must be a multiple of {ep_size=}")
    ep_shard_size = non_pp_size // ep_size if ep_size < non_pp_size else 1
    dp_shard_size = dp_size // dp_replicate_size

    device_mesh = _init_named_mesh(
        _MeshSpec(
            shape=(pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size),
            axes=(
                MeshAxisName.PP,
                MeshAxisName.DP_REPLICATE,
                MeshAxisName.DP_SHARD,
                MeshAxisName.CP,
                MeshAxisName.TP,
            ),
            flattened_axes={
                MeshAxisName.DP: (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD),
                MeshAxisName.DP_SHARD_CP: (MeshAxisName.DP_SHARD, MeshAxisName.CP),
                MeshAxisName.DP_CP: (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD, MeshAxisName.CP),
            },
        ),
        timeout_minutes=timeout_minutes,
        ranks=ranks,
    )

    moe_mesh = None
    if ep_size > 1:
        moe_mesh = _create_moe_mesh(
            device_mesh,
            ep_shard_size=ep_shard_size,
            ep_size=ep_size,
            pp_size=pp_size,
            timeout_minutes=timeout_minutes,
            ranks=ranks,
        )

    return device_mesh, moe_mesh


def _create_moe_mesh(
    device_mesh: DeviceMesh,
    *,
    ep_shard_size: int,
    ep_size: int,
    pp_size: int = 1,
    timeout_minutes: int | None = None,
    ranks: list[int] | tuple[int, ...] | None = None,
) -> DeviceMesh:
    non_pp_axes = (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD, MeshAxisName.CP, MeshAxisName.TP)
    if ranks is not None:
        current_rank = dist.get_rank()
        ranks_per_stage = ep_shard_size * ep_size
        backend_override = _nccl_backend_override(
            (MeshAxisName.EP_SHARD, MeshAxisName.EP),
            device_type=device_mesh.device_type,
            timeout_minutes=timeout_minutes,
        )
        local_mesh = None
        first_mesh = None
        for stage in range(pp_size):
            stage_ranks = ranks[stage * ranks_per_stage : (stage + 1) * ranks_per_stage]
            stage_mesh = DeviceMesh(
                device_mesh.device_type,
                mesh=torch.tensor(stage_ranks, dtype=torch.int64).reshape(ep_shard_size, ep_size),
                mesh_dim_names=(MeshAxisName.EP_SHARD, MeshAxisName.EP),
                backend_override=backend_override,
            )
            if first_mesh is None:
                first_mesh = stage_mesh
            if current_rank in stage_ranks:
                local_mesh = stage_mesh
        if first_mesh is None:
            raise RuntimeError("Failed to construct an expert-parallel mesh")
        return local_mesh if local_mesh is not None else first_mesh
    return _unflatten_compat(
        device_mesh[non_pp_axes]._flatten(),
        axis=0,
        sizes=(ep_shard_size, ep_size),
        names=(MeshAxisName.EP_SHARD, MeshAxisName.EP),
        timeout_minutes=timeout_minutes,
    )


def _unflatten_compat(
    flat_mesh: DeviceMesh,
    axis: int,
    sizes: tuple,
    names: tuple,
    *,
    timeout_minutes: int | None = None,
) -> DeviceMesh:
    """Unflatten a mesh with its NCCL timeout, including the PyTorch 2.9 fallback.

    PyTorch reuses the default process group for a world-sized dimension only
    when no backend override is supplied. The default group already has the
    configured distributed timeout, so omitting that redundant override avoids
    allocating another NCCL communicator.
    """
    if hasattr(flat_mesh, "_unflatten"):
        if timeout_minutes is not None and flat_mesh.device_type == "cuda":
            backend_override = _nccl_backend_override(
                names,
                device_type=flat_mesh.device_type,
                timeout_minutes=timeout_minutes,
            )
            backend_override = {
                name: backend_override[name] for name, size in zip(names, sizes) if not _can_reuse_default_group(size)
            }
            if backend_override:
                return flat_mesh._unflatten(
                    axis,
                    sizes,
                    names,
                    backend_override=backend_override,
                )
        return flat_mesh._unflatten(axis, sizes, names)
    new_mesh_tensor = flat_mesh.mesh.reshape(sizes)
    from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh

    return _DeviceMesh(flat_mesh.device_type, new_mesh_tensor, mesh_dim_names=names)


def get_flat_mesh(device_mesh: "DeviceMesh", name: str) -> "DeviceMesh":
    """Access a 1D submesh by parallelism name (e.g. ``"dp"``, ``"tp"``, ``"dp_cp"``).

    PyTorch 2.11 deprecates ``root_mesh["name"]`` for dimensions created via
    ``_flatten()``.  This reads the ``_flatten()`` result directly.

    Args:
        device_mesh: Any DeviceMesh (root or submesh).
        name: Parallelism dimension name.
    """
    if name in device_mesh.mesh_dim_names:
        return device_mesh[name]
    # _get_root_mesh() was added in PyTorch 2.10; fall back for 2.9.x.
    if hasattr(device_mesh, "_get_root_mesh"):
        root = device_mesh._get_root_mesh()
    else:
        root = device_mesh
    if hasattr(root, "_flatten_mapping") and name in root._flatten_mapping:
        return root._flatten_mapping[name]
    raise KeyError(
        f"Mesh dim {name!r} not found in mesh_dim_names {device_mesh.mesh_dim_names} "
        f"or root _flatten_mapping {set(getattr(root, '_flatten_mapping', {}))}"
    )


def get_submesh(device_mesh: "DeviceMesh", names: tuple) -> "DeviceMesh":
    """Access a submesh by parallelism dim names.

    Handles all cases: single dims, multi-dim slices, and combinations that
    include ``_flatten()``-created dims (e.g. ``("dp_replicate", "dp_shard_cp")``).
    For the latter, finds the parent ``_flatten()`` result and calls ``_unflatten()``
    to decompose it into the requested shape.

    Args:
        device_mesh: Any DeviceMesh (root or submesh).
        names: Tuple of dimension names.
    """
    if len(names) == 1:
        return get_flat_mesh(device_mesh, names[0])
    if all(n in device_mesh.mesh_dim_names for n in names):
        return device_mesh[names]

    # Some dims were created via _flatten(); resolve sizes and unflatten from parent.
    # Strategy: find a parent flattened mesh whose size equals the product of
    # requested dims, unflatten it, then validate that process groups for any
    # dim that also exists on the root mesh are identical (guards against
    # ambiguous size collisions between different flattened meshes).
    from math import prod

    import torch.distributed as dist

    sizes = tuple(get_flat_mesh(device_mesh, n).size() for n in names)
    target = prod(sizes)
    if hasattr(device_mesh, "_get_root_mesh"):
        root = device_mesh._get_root_mesh()
    else:
        root = device_mesh

    for fm in getattr(root, "_flatten_mapping", {}).values():
        if fm.size() != target:
            continue
        try:
            result = _unflatten_compat(fm, 0, sizes, names)
        except (ValueError, RuntimeError):
            continue
        # Validate: for each requested dim, verify its process group matches
        # what get_flat_mesh returns (works for both mesh dims and flattened dims).
        valid = True
        for name in names:
            expected = set(dist.get_process_group_ranks(get_flat_mesh(device_mesh, name).get_group()))
            actual = set(dist.get_process_group_ranks(result[name].get_group()))
            if expected != actual:
                valid = False
                break
        if valid:
            return result
    raise KeyError(
        f"No parent flattened mesh found for dims {names} with target size {target}. "
        f"Available: {set(root._flatten_mapping)}"
    )


def get_fsdp_dp_mesh(
    device_mesh: "DeviceMesh",
    dp_replicate_name: str = MeshAxisName.DP_REPLICATE,
    dp_shard_cp_name: str = MeshAxisName.DP_SHARD_CP,
) -> "DeviceMesh":
    """Return the DP mesh for FSDP2 without losing the original root mesh.

    ``get_submesh()`` may rebuild a fresh DeviceMesh when asked to compose native
    and flattened dims like ``("dp_replicate", "dp_shard_cp")``. That is fine
    for many local operations, but FSDP2 expects its DP mesh to share the same
    root mesh as TP/EP meshes. On multi-node TP runs this can break group
    construction in non-obvious ways.

    Prefer native dimensions whenever possible:
    - cp=1, dp_replicate=1  -> ``device_mesh["dp_shard"]``
    - cp=1, dp_replicate>1  -> ``device_mesh[("dp_replicate", "dp_shard")]``
    - cp>1, dp_replicate=1  -> ``device_mesh["dp_shard_cp"]``

    When both CP and replicated DP are active we fall back to ``get_submesh()``
    because the composed mesh is genuinely multi-level.
    """

    dp_shard_name = MeshAxisName.DP_SHARD
    cp_name = MeshAxisName.CP

    native_dims_available = (
        dp_replicate_name in device_mesh.mesh_dim_names
        and dp_shard_name in device_mesh.mesh_dim_names
        and cp_name in device_mesh.mesh_dim_names
    )
    if native_dims_available:
        cp_size = device_mesh[cp_name].size()
        dp_replicate_size = device_mesh[dp_replicate_name].size()

        if dp_replicate_size > 1 and cp_size > 1:
            pass
        elif dp_replicate_size > 1:
            return device_mesh[(dp_replicate_name, dp_shard_name)]
        elif cp_size > 1:
            return get_flat_mesh(device_mesh, dp_shard_cp_name)
        else:
            return device_mesh[dp_shard_name]

    return get_submesh(device_mesh, (dp_replicate_name, dp_shard_cp_name))


