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

"""Distributed gradient scaling and clipping tests for owner-sharded Engram rows."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


class _EngramOnlyModel(nn.Module):
    """Minimal parameter owner used to exercise common training utilities."""

    def __init__(self) -> None:
        super().__init__()
        self.table = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=6,
            embedding_dim=2,
            initializer_range=0.0,
        ).build(process_group=dist.group.WORLD, dtype=torch.float32)
        self.table.parallelize_weight(
            DeviceMesh.from_group(
                dist.group.WORLD,
                device_type="cpu",
                mesh_dim_names=("dp_shard_cp",),
            )
        )


def _engram_scale_and_norm_worker(rank: int, world_size: int, store_path: str) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )
        model = _EngramOnlyModel()
        assert isinstance(model.table.weight, DTensor)
        assert tuple(model.table.weight.shape) == (6, 2)
        assert tuple(model.table.weight.placements) == (Shard(0),)
        assert model.table.weight._nemo_model_owned_grad_divisor == float(world_size)
        model.table.weight.grad = torch.full_like(model.table.weight, float(rank + 1))

        total_norm = scale_grads_and_clip_grad_norm(
            1.0e9,
            [model],
            foreach=True,
        )

        expected_grad = torch.full_like(model.table.weight.to_local(), float(rank + 1) / world_size)
        assert isinstance(model.table.weight.grad, DTensor)
        torch.testing.assert_close(model.table.weight.grad.to_local(), expected_grad, rtol=0, atol=0)
        # Six local values across two ranks: rank 0 contributes 6*(1/2)^2,
        # rank 1 contributes 6*(2/2)^2.
        expected_norm = torch.tensor(7.5, dtype=torch.float64).sqrt()
        torch.testing.assert_close(total_norm.cpu(), expected_norm, rtol=1e-7, atol=1e-7)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_engram_grad_scaling_and_global_norm_use_owner_group(tmp_path: Path) -> None:
    mp.spawn(
        _engram_scale_and_norm_worker,
        args=(2, str(tmp_path / "engram-training-utils-pg")),
        nprocs=2,
        join=True,
    )
