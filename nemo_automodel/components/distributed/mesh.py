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

"""MeshContext dataclass, construction, and validation.

``MeshContext`` is the single source of truth for distributed topology:
device meshes, parallelism sizes, and axis names.

Parallelism sizes (``tp_size``, ``pp_size``, etc.) are derived at runtime
from the attached ``DeviceMesh`` objects via ``@property``.  When no mesh
is present the properties return safe defaults (1 for sizes, ``None`` for
dp / hsdp).

All inputs and outputs are typed Python objects (dataclasses, enums, etc.).
YAML / dict parsing belongs in the recipe layer — see
``nemo_automodel.recipes._dist_utils``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

from nemo_automodel.components.distributed.config import DistributedStrategyConfig
from nemo_automodel.components.distributed.init_utils import get_world_size_safe

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh


class MeshAxisName(str, Enum):
    """Canonical mesh axis names used by ``DeviceMesh`` and helpers.

    Inherits from ``str`` so each member compares equal to (and can be
    used wherever) a plain string — e.g. ``MeshAxisName.TP == "tp"``.
    """

    PP = "pp"
    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_SHARD_CP = "dp_shard_cp"
    DP_CP = "dp_cp"
    CP = "cp"
    TP = "tp"
    EP = "ep"
    EP_SHARD = "ep_shard"


#: All values accepted as ``DeviceMesh`` axis names.
_VALID_AXIS_NAMES: frozenset = frozenset(MeshAxisName)


@dataclass(frozen=True, kw_only=True)
class ParallelismSizes:
    """Build-time requested parallelism sizes.

    This is durable user intent, not runtime topology. ``MeshContext`` derives
    its size properties from live ``DeviceMesh`` objects after build.
    """

    dp_size: int | None = None
    dp_replicate_size: int | None = None
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1


@dataclass
class MeshContext:
    """Runtime distributed topology context.

    Parallelism sizes (``tp_size``, ``pp_size``, etc.) are **not** stored as
    fields; they are ``@property`` accessors that read directly from the
    attached ``DeviceMesh`` / ``moe_mesh``.  When no mesh is present the
    properties return safe defaults (``1`` for sizes, ``None`` for dp / hsdp).

    All ``DeviceMesh`` objects passed in must use axis names from
    :class:`MeshAxisName`; a ``ValueError`` is raised on construction if
    any unknown name is encountered.

    Lifecycle
    ---------
    1. Recipes parse YAML to obtain sizes and strategy configs.
    2. Sizes are passed to :meth:`build` to build ``DeviceMesh``
       objects.
    3. ``MeshContext`` is created with those meshes; axis names are
       validated automatically in ``__post_init__``.

    Alternatively, :meth:`from_meshes` constructs an instance directly from
    ``DeviceMesh`` objects (used by ``NeMoAutoModel.from_pretrained``).

    Attributes:
        device_mesh: Device mesh for distributed training.
        moe_mesh: MoE-specific device mesh.
        process_group: Optional model-local group for recipe-level collectives
            that must not involve ranks outside this mesh.
    """

    # runtime mesh references
    device_mesh: Optional["DeviceMesh"] = field(default=None, repr=False)
    moe_mesh: Optional["DeviceMesh"] = field(default=None, repr=False)
    process_group: "ProcessGroup | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_mesh_axis_names(self)

    # Parallelism sizes — derived from the attached meshes
    @property
    def pp_size(self) -> int:
        """Pipeline-parallel degree (from ``device_mesh``, default ``1``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.PP)

    @property
    def pp_enabled(self) -> bool:
        """``True`` when ``pp_size > 1``."""
        return self.pp_size > 1

    @property
    def tp_size(self) -> int:
        """Tensor-parallel degree (from ``device_mesh``, default ``1``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.TP)

    @property
    def cp_size(self) -> int:
        """Context-parallel degree (from ``device_mesh``, default ``1``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.CP)

    @property
    def ep_size(self) -> int:
        """Expert-parallel degree (from ``moe_mesh``, default ``1``)."""
        return _get_axis_size(self.moe_mesh, MeshAxisName.EP)

    @property
    def dp_size(self) -> int | None:
        """Data-parallel degree (from ``device_mesh``, default ``None``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.DP, default=None)

    @property
    def dp_replicate_size(self) -> int | None:
        """HSDP replication degree (from ``device_mesh``, default ``None``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.DP_REPLICATE, default=None)

    @property
    def dp_shard_size(self) -> int:
        """DP shard degree (from ``device_mesh``, default ``1``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.DP_SHARD, default=1)

    # Axis-name helpers (used by AutoPipeline and parallelize_model)
    def _dp_axis_names(self) -> Tuple[str, ...]:
        """DP axis names for FSDP mesh slicing."""
        if self.device_mesh is not None:
            names = self.device_mesh.mesh_dim_names
            if (
                MeshAxisName.DP_REPLICATE in names
                and _get_axis_size(self.device_mesh, MeshAxisName.DP_SHARD_CP, default=None) is not None
            ):
                return (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD_CP)
        return (MeshAxisName.DP_SHARD_CP,)

    # @akoumpa: we will deprecate `pipeline_axis_kwargs` in 26.04.
    def pipeline_axis_kwargs(self) -> Dict[str, object]:
        """Axis-name kwargs for ``AutoPipeline``."""
        return {
            "pp_axis_name": MeshAxisName.PP,
        } | self.parallelize_axis_kwargs()

    # @akoumpa: we will deprecate `parallelize_axis_kwargs` in 26.04.
    def parallelize_axis_kwargs(self) -> Dict[str, object]:
        """Axis-name kwargs for ``parallelize_fn`` (EP/FSDP, no ``pp_axis_name``)."""
        return {
            "dp_axis_names": self._dp_axis_names(),
            "cp_axis_name": _optional_axis(self.device_mesh, MeshAxisName.CP),
            "tp_axis_name": _optional_axis(self.device_mesh, MeshAxisName.TP),
            "ep_axis_name": _optional_axis(self.moe_mesh, MeshAxisName.EP),
            "ep_shard_axis_names": (MeshAxisName.EP_SHARD,)
            if _optional_axis(self.moe_mesh, MeshAxisName.EP_SHARD)
            else None,
        }

    @classmethod
    def build(
        cls,
        strategy_config: DistributedStrategyConfig,
        parallelism_sizes: ParallelismSizes | None = None,
        *,
        world_size: int | None = None,
        timeout_minutes: int | None = None,
        ranks: Sequence[int] | None = None,
    ) -> "MeshContext":
        """Build a topology-only :class:`MeshContext` from parallelism sizes.

        Args:
            strategy_config: Already-instantiated distributed strategy config.
            parallelism_sizes: Requested data, tensor, pipeline, context, and expert
                parallelism sizes. If ``None``, defaults to no parallelism with
                DP inferred from ``world_size``.
            world_size: Total process count. If ``None``, inferred from the
                distributed environment.
            timeout_minutes: Optional timeout for process groups created by
                ``DeviceMesh`` axes.
            ranks: Optional ordered global ranks to use for the mesh. When
                omitted, the mesh uses every rank in the default process group.
        """
        if world_size is None:
            world_size = get_world_size_safe()
        if parallelism_sizes is None:
            parallelism_sizes = ParallelismSizes()

        from nemo_automodel.components.distributed.mesh_utils import _create_device_meshes

        device_mesh, moe_mesh = _create_device_meshes(
            strategy_config,
            parallelism_sizes,
            world_size=world_size,
            timeout_minutes=timeout_minutes,
            ranks=ranks,
        )
        return cls.from_meshes(device_mesh, moe_mesh)

    # Convenience constructor
    @classmethod
    def from_meshes(
        cls,
        device_mesh: Optional["DeviceMesh"],
        moe_mesh: Optional["DeviceMesh"] = None,
    ) -> "MeshContext":
        """Build a :class:`MeshContext` from ``DeviceMesh`` objects.

        This is the entry-point used by ``NeMoAutoModel.from_pretrained`` /
        ``from_config`` where the caller has raw meshes rather than a parsed
        YAML config.
        """
        return cls(
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
        )


# misc utils
def _get_axis_size(mesh: Optional["DeviceMesh"], axis: MeshAxisName, default=1) -> int | None:
    """Return the size of *axis* if present in *mesh*, else *default*."""
    if mesh is None:
        return default
    # Check mesh axes and _flatten() results on root mesh.
    if axis in mesh.mesh_dim_names:
        return mesh[axis].size()
    if hasattr(mesh, "_get_root_mesh"):
        root = mesh._get_root_mesh()
    else:
        root = mesh
    if hasattr(root, "_flatten_mapping") and axis in root._flatten_mapping:
        return root._flatten_mapping[axis].size()
    return default


def _optional_axis(mesh: Optional["DeviceMesh"], axis: MeshAxisName) -> str | None:
    """Return *axis* if present in *mesh*, else ``None``."""
    if mesh is not None and axis in mesh.mesh_dim_names:
        return axis
    return None


# Validation utils
def _validate_mesh_axis_names(mesh_context: "MeshContext") -> None:
    """Ensure every axis name in the attached meshes is a :class:`MeshAxisName`."""
    for label in ("device_mesh", "moe_mesh"):
        mesh = getattr(mesh_context, label)
        if mesh is None:
            continue
        bad = {n for n in mesh.mesh_dim_names if n not in _VALID_AXIS_NAMES}
        if bad:
            raise ValueError(
                f"{label} contains unknown axis names {bad}; allowed names are {sorted(_VALID_AXIS_NAMES)}"
            )


__all__ = [
    "MeshAxisName",
    "MeshContext",
    "ParallelismSizes",
]
