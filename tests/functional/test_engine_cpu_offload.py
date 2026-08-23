# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from nemo_automodel.components.datasets.datum import Datum, LossInputLayout
from nemo_automodel.engine import Engine
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

from molt.trainer.fsdp.refit import gather_full_param


class _TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        del attention_mask, position_ids
        return self.projection(input_ids.float().unsqueeze(-1)).squeeze(-1)


def _cpu_offload_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(1234)
        device = torch.device("cuda", rank)
        model = _TinyRegressor().to(device)
        offload_policy = CPUOffloadPolicy(pin_memory=False)
        fully_shard(model.projection, offload_policy=offload_policy)
        fully_shard(model, offload_policy=offload_policy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        datums = [
            Datum(
                model_inputs={"input_ids": torch.tensor([rank + 1, rank + 2])},
                loss_fn_inputs={"weights": torch.ones(2)},
                loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
            )
        ]

        engine = Engine(model, device=device, optimizers=optimizer, max_grad_norm=1.0)

        def squared_loss(output, _loss_inputs):
            return output.square().sum()

        result = engine.forward_backward([datums], squared_loss)
        optim_result = engine.step()

        assert torch.isfinite(result.loss)
        assert torch.isfinite(optim_result.grad_norm)
        parameter = next(model.parameters())
        assert isinstance(parameter, DTensor)
        assert parameter.to_local().device.type == "cpu"

        full, full_shape = gather_full_param(parameter)
        assert full.device.type == "cuda"
        assert full.shape == full_shape == parameter.shape
        assert torch.isfinite(full).all()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_engine_full_cpu_offload_and_refit_gather(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _cpu_offload_worker,
        args=(2, str(tmp_path / "cpu-offload-init")),
        nprocs=2,
    )
