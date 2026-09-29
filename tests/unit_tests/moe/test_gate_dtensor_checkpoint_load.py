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

"""Distributed checkpoint-load coverage for the adaptive MoE routing bias."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


def _make_gate() -> Gate:
    """Construct a tiny gate with a persistent adaptive routing-bias buffer."""
    return Gate(
        MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.1,
            aux_loss_coeff=0.0,
            score_func="softmax_with_bias",
            route_scale=1.0,
            dim=8,
            inter_dim=16,
            moe_inter_dim=4,
            norm_topk_prob=True,
            dtype=torch.bfloat16,
        )
    )


def _run_replicated_bias_load(rank: int, world_size: int, init_file: str) -> None:
    """Load plain and distributed routing-bias tensors into a replicated DTensor buffer."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = DeviceMesh("cpu", torch.arange(world_size), mesh_dim_names=("dp",))
        gate = _make_gate()
        gate.e_score_correction_bias = distribute_tensor(
            gate.e_score_correction_bias,
            device_mesh=mesh,
            placements=[Replicate()],
        )

        plain_bias = torch.tensor([0.25, -0.5, 0.75, -1.0], dtype=torch.float32)
        set_model_state_dict(
            gate,
            {"e_score_correction_bias": plain_bias},
            options=StateDictOptions(strict=False),
        )

        assert isinstance(gate.e_score_correction_bias, DTensor)
        assert gate.e_score_correction_bias.placements == (Replicate(),)
        assert gate.e_score_correction_bias.dtype == torch.float32
        torch.testing.assert_close(gate.e_score_correction_bias.to_local(), plain_bias)

        distributed_bias = distribute_tensor(
            torch.tensor([-1.25, 1.5, -1.75, 2.0], dtype=torch.float32),
            device_mesh=mesh,
            placements=[Replicate()],
        )
        set_model_state_dict(
            gate,
            {"e_score_correction_bias": distributed_bias},
            options=StateDictOptions(strict=False),
        )
        torch.testing.assert_close(gate.e_score_correction_bias.to_local(), distributed_bias.to_local())

        sharded_gate = _make_gate()
        sharded_gate.e_score_correction_bias = distribute_tensor(
            sharded_gate.e_score_correction_bias,
            device_mesh=mesh,
            placements=[Shard(0)],
        )
        try:
            set_model_state_dict(
                sharded_gate,
                {"e_score_correction_bias": plain_bias},
                options=StateDictOptions(strict=False),
            )
        except RuntimeError as error:
            assert "must be replicated on every mesh dimension" in str(error)
        else:
            raise AssertionError("A full routing-bias tensor must not load into a sharded DTensor")
    finally:
        dist.destroy_process_group()


def test_gate_loads_plain_bias_into_replicated_dtensor(tmp_path) -> None:
    """A two-rank load accepts HF/adapter tensors and preserves native DCP tensors."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with Gloo is unavailable")

    mp.spawn(
        _run_replicated_bias_load,
        args=(2, str(tmp_path / "gate_bias_pg")),
        nprocs=2,
        join=True,
    )
