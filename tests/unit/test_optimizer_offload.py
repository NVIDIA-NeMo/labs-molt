# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from molt.trainer.fsdp.optimizer_offload import CpuOptimizerOffloader, offload_moments_to_cpu


def test_cpu_step_matches_plain_adamw_and_keeps_moments_on_cpu():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4, 3))
    reference = torch.nn.Parameter(param.detach().clone())
    grad = torch.randn(4, 3)

    offloaded = CpuOptimizerOffloader(torch.optim.AdamW([param], lr=0.1, foreach=False, fused=False))
    plain = torch.optim.AdamW([reference], lr=0.1, foreach=False, fused=False)
    for _ in range(3):
        param.grad = grad.clone()
        reference.grad = grad.clone()
        offloaded.step()
        plain.step()
        offloaded.zero_grad()
        plain.zero_grad()

    torch.testing.assert_close(param, reference)
    for state in offloaded.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"


def test_offload_moments_to_cpu_pages_restored_state():
    param = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=0.1, foreach=False, fused=False)
    param.grad = torch.randn(2, 2)
    optimizer.step()

    offload_moments_to_cpu(optimizer)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
