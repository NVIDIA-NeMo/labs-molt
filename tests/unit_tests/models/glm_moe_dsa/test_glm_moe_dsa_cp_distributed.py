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

"""Two-rank NCCL regression for GLM DSA's differentiable CP gather."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.models.glm_moe_dsa.cp import glm_dsa_cp_all_gather

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = [
    pytest.mark.timeout(60),
    pytest.mark.run_only_on("GPU"),
]

_WORLD_SIZE = 2


def _free_port() -> int:
    """Return an unused localhost port for the process-group rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gather_worker(rank: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=_WORLD_SIZE)

        device = torch.device("cuda", rank)
        local_k = (torch.arange(4, device=device, dtype=torch.float32).reshape(2, 2) + 10 * rank).requires_grad_()
        local_v = (torch.arange(2, device=device, dtype=torch.float32).reshape(2, 1) + 100 * rank).requires_grad_()

        gathered_k = glm_dsa_cp_all_gather(local_k, dim=0, cp_group=dist.group.WORLD)
        gathered_v = glm_dsa_cp_all_gather(local_v, dim=0, cp_group=dist.group.WORLD)

        expected_k = torch.cat(
            [torch.arange(4, device=device, dtype=torch.float32).reshape(2, 2) + 10 * source for source in range(2)]
        )
        expected_v = torch.cat(
            [torch.arange(2, device=device, dtype=torch.float32).reshape(2, 1) + 100 * source for source in range(2)]
        )
        torch.testing.assert_close(gathered_k, expected_k)
        torch.testing.assert_close(gathered_v, expected_v)

        k_weight = rank + 1
        v_weight = 2 * rank + 1
        (k_weight * gathered_k.sum() + v_weight * gathered_v.sum()).backward()

        # Differentiable all-gather sums every consumer's contribution before
        # returning the gradient shard owned by this rank.
        torch.testing.assert_close(local_k.grad, torch.full_like(local_k, 3.0))
        torch.testing.assert_close(local_v.grad, torch.full_like(local_v, 4.0))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason="requires NCCL and at least two CUDA devices",
)
def test_glm_dsa_cp_all_gather_preserves_rank_order_and_sums_kv_gradients() -> None:
    mp.spawn(_gather_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
