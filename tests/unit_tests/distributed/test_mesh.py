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

"""Tests for the **component-layer** mesh module (MeshContext and validation).

Dict-parsing tests live in ``tests/unit_tests/recipes/test_dist_utils.py``.
"""

from unittest.mock import Mock


from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.mesh import (
    MeshAxisName,
    MeshContext,
    _get_axis_size,
)

# ---------------------------------------------------------------------------
# MeshContext – defaults (no mesh attached)
# ---------------------------------------------------------------------------


class TestMeshContextDefaults:
    def test_sizes_default_to_1_or_none(self):
        ctx = MeshContext()

        assert ctx.tp_size == 1
        assert ctx.pp_size == 1
        assert ctx.cp_size == 1
        assert ctx.ep_size == 1
        assert ctx.dp_size is None
        assert ctx.dp_replicate_size is None

    def test_pp_enabled_false_by_default(self):
        ctx = MeshContext()
        assert ctx.pp_enabled is False


# ---------------------------------------------------------------------------
# MeshContext.from_meshes (no real mesh — smoke test)
# ---------------------------------------------------------------------------


class TestFromMeshes:
    def test_from_none_meshes(self):
        ctx = MeshContext.from_meshes(None)
        assert ctx.device_mesh is None
        assert ctx.moe_mesh is None
        assert ctx.tp_size == 1


# ---------------------------------------------------------------------------
# MeshContext – helper methods
# ---------------------------------------------------------------------------


class TestMeshAxisNameEnum:
    def test_enum_is_str(self):
        """MeshAxisName members compare equal to plain strings."""
        assert MeshAxisName.TP == "tp"
        assert MeshAxisName.PP == "pp"
        assert MeshAxisName.DP_SHARD_CP == "dp_shard_cp"
        assert isinstance(MeshAxisName.TP, str)

    def test_all_expected_members(self):
        names = {m.value for m in MeshAxisName}
        assert names == {
            "pp",
            "dp",
            "dp_replicate",
            "dp_shard",
            "dp_shard_cp",
            "dp_cp",
            "cp",
            "tp",
            "ep",
            "ep_shard",
        }


# ---------------------------------------------------------------------------
# _get_axis_size – supports _flatten() created dims
# ---------------------------------------------------------------------------


class TestGetAxisSize:
    def _make_mock_mesh(self, dim_names, flatten_mapping=None):
        mesh = Mock()
        mesh.mesh_dim_names = dim_names
        mesh._get_root_mesh = Mock(return_value=mesh)
        mesh._flatten_mapping = flatten_mapping or {}

        def getitem(name):
            submesh = Mock()
            submesh.size = Mock(return_value=4)
            return submesh

        mesh.__getitem__ = Mock(side_effect=getitem)
        return mesh

    def test_none_mesh_returns_default(self):
        assert _get_axis_size(None, MeshAxisName.TP) == 1

    def test_none_mesh_returns_custom_default(self):
        assert _get_axis_size(None, MeshAxisName.DP, default=None) is None

    def test_physical_dim_returns_size(self):
        mesh = self._make_mock_mesh(("dp", "tp"))
        result = _get_axis_size(mesh, MeshAxisName.TP)
        assert result == 4
        mesh.__getitem__.assert_called_once_with(MeshAxisName.TP)

    def test_flattened_dim_returns_size(self):
        dp_flat = Mock()
        dp_flat.size = Mock(return_value=8)
        mesh = self._make_mock_mesh(
            ("dp_replicate", "dp_shard", "cp", "tp"),
            flatten_mapping={"dp": dp_flat},
        )
        result = _get_axis_size(mesh, MeshAxisName.DP)
        assert result == 8
        # Should NOT go through __getitem__
        mesh.__getitem__.assert_not_called()

    def test_missing_dim_returns_default(self):
        mesh = self._make_mock_mesh(("dp", "tp"), flatten_mapping={})
        result = _get_axis_size(mesh, MeshAxisName.PP)
        assert result == 1


class TestHelperMethods:
    def test_pipeline_axis_kwargs(self):
        ctx = MeshContext()
        kwargs = ctx.pipeline_axis_kwargs()
        assert "pp_axis_name" in kwargs
        assert kwargs["pp_axis_name"] == MeshAxisName.PP
        assert kwargs["dp_axis_names"] == (MeshAxisName.DP_SHARD_CP,)

    def test_parallelize_axis_kwargs(self):
        ctx = MeshContext()
        kwargs = ctx.parallelize_axis_kwargs()
        assert "pp_axis_name" not in kwargs
        assert kwargs["dp_axis_names"] == (MeshAxisName.DP_SHARD_CP,)

    def test_dp_axis_names_include_replicate_axis_for_flattened_shard_cp(self):
        flattened_dp_mesh = Mock()
        flattened_dp_mesh.size.return_value = 8
        device_mesh = Mock()
        device_mesh.mesh_dim_names = (
            MeshAxisName.PP,
            MeshAxisName.DP_REPLICATE,
            MeshAxisName.DP_SHARD,
            MeshAxisName.CP,
            MeshAxisName.TP,
        )
        device_mesh._get_root_mesh.return_value = device_mesh
        device_mesh._flatten_mapping = {MeshAxisName.DP_SHARD_CP: flattened_dp_mesh}

        ctx = MeshContext(device_mesh=device_mesh)

        assert ctx._dp_axis_names() == (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD_CP)


# ---------------------------------------------------------------------------
# DistributedSetup – simple runtime bundle
# ---------------------------------------------------------------------------


class TestDistributedSetup:
    def test_minimal_setup_holds_mesh_and_policy(self):
        setup = DistributedSetup(mesh_context=MeshContext(), strategy_config=FSDP2Config(activation_checkpointing=True))

        assert isinstance(setup.mesh_context, MeshContext)
        assert setup.strategy_config.activation_checkpointing is True


# ---------------------------------------------------------------------------
# Integration: validate with full configs
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_fsdp2_validates_on_distributed_setup(self):
        """DistributedSetup validates strategy policy against mesh topology."""
        setup = DistributedSetup(
            mesh_context=MeshContext(),
            strategy_config=FSDP2Config(
                sequence_parallel=True,
                activation_checkpointing=True,
                defer_fsdp_grad_sync=False,
            ),
        )
        # No meshes → sizes default to 1 / None, which is valid for FSDP2.
        assert setup.mesh_context.tp_size == 1

