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

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common.utils import Float32RMSNorm


@pytest.mark.parametrize("shape", [(7,), (2, 3, 7), (0, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("trainable", ["both", "input", "weight"])
def test_float32_rms_norm_forward_and_gradients(shape, dtype, trainable):
    torch.manual_seed(17)
    x = torch.randn(shape, dtype=dtype, requires_grad=trainable != "weight")
    module = Float32RMSNorm(7, dtype=dtype)
    with torch.no_grad():
        module.weight.normal_()
    module.weight.requires_grad_(trainable != "input")
    ref_x = x.detach().clone().requires_grad_(x.requires_grad)
    ref_weight = module.weight.detach().clone().requires_grad_(module.weight.requires_grad)

    with torch.compiler.set_stance("force_eager"):
        actual = module(x)
    expected = torch.nn.functional.rms_norm(ref_x.float(), (7,), ref_weight.float(), module.eps).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)

    # Allow rounding from different fp32 reduction orders and bf16 casts.
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (0.016, 1e-5)
    for result, reference in [(x, ref_x), (module.weight, ref_weight)]:
        if result.requires_grad:
            torch.testing.assert_close(result.grad, reference.grad, rtol=rtol, atol=atol)
        else:
            assert result.grad is None


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "strided_hidden"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA layout validation requires a GPU"),
        ),
    ],
)
# opcheck includes AOTAutograd compilation, which exceeds the default 5 seconds.
@pytest.mark.timeout(60)
def test_float32_rms_norm_custom_op_contract(layout, device):
    x = torch.randn(3, 2, 14 if layout == "strided_hidden" else 7, device=device)
    if layout == "transposed":
        x = x.transpose(0, 1)
    elif layout == "strided_hidden":
        x = x[..., ::2]
    x.requires_grad_()
    weight = torch.randn(7, device=device, requires_grad=True)
    torch.library.opcheck(torch.ops.nemo_automodel.float32_rms_norm.default, (x, weight, 1e-5))


# The child process isolates registrations and reproduces the install import checker.
@pytest.mark.timeout(60)
def test_float32_rms_norm_reimport():
    script = """
import importlib
import sys
import torch
name = "nemo_automodel.components.models.common.utils"
original = importlib.import_module(name)
for _ in range(2):
    del sys.modules[name]
    current = importlib.import_module(name)
    for module in (original, current):
        x = torch.randn(7, requires_grad=True)
        norm = module.Float32RMSNorm(7, dtype=torch.float32)
        with torch.compiler.set_stance("force_eager"):
            norm(x).sum().backward()
        assert norm.weight.grad.shape == (7,)
        assert torch.isfinite(x.grad).all()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=50,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def _dtensor_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        torch.manual_seed(17)
        values = torch.randn(3, 5, 8)
        weights = torch.randn(8)
        upstream = torch.randn_like(values)
        for placement in [Replicate(), Shard(0), Shard(1), Shard(2)]:
            ref_x = values.clone().requires_grad_()
            ref_weight = weights.clone().requires_grad_()
            expected = torch.nn.functional.rms_norm(ref_x, (8,), ref_weight, 1e-5)
            expected.backward(upstream)

            x = distribute_tensor(values.clone(), mesh, [placement]).requires_grad_()
            weight = distribute_tensor(weights.clone(), mesh, [Replicate()]).requires_grad_()
            actual = torch.ops.nemo_automodel.float32_rms_norm(x, weight, 1e-5)
            expected_placement = Replicate() if placement == Shard(2) else placement
            assert actual.placements == (expected_placement,)
            torch.testing.assert_close(actual.full_tensor(), expected, rtol=1e-5, atol=1e-6)
            actual.backward(distribute_tensor(upstream, mesh, list(actual.placements)))
            torch.testing.assert_close(x.grad.full_tensor(), ref_x.grad, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(weight.grad.full_tensor(), ref_weight.grad, rtol=1e-5, atol=1e-6)

            # A sequence-sharded input produces a Partial weight gradient. Sum it
            # before the optimizer, as the training gradient-sync boundary does.
            weight.grad = weight.grad.redistribute(placements=[Replicate()])
            norm = torch.nn.utils.clip_grad_norm_([weight], 0.3)
            ref_norm = torch.nn.utils.clip_grad_norm_([ref_weight], 0.3)
            torch.testing.assert_close(norm.full_tensor(), ref_norm, rtol=1e-5, atol=1e-6)
            torch.optim.SGD([weight], lr=0.1).step()
            torch.optim.SGD([ref_weight], lr=0.1).step()
            torch.testing.assert_close(weight.full_tensor(), ref_weight, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
# Spawning PyTorch workers and initializing Gloo exceeds the default 5 seconds.
@pytest.mark.timeout(60)
def test_float32_rms_norm_dtensor(world_size, tmp_path):
    mp.spawn(_dtensor_worker, args=(world_size, str(tmp_path / "init")), nprocs=world_size, join=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required to reproduce the compiled grad/no_grad regression"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
# This checks actual Inductor compilation, including activation recomputation.
@pytest.mark.timeout(60)
def test_float32_rms_norm_compiled_determinism(dtype, monkeypatch):
    from torch._dynamo.testing import CompileCounterWithBackend

    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    torch.manual_seed(17)
    module = Float32RMSNorm(257, device="cuda", dtype=dtype)
    with torch.no_grad():
        module.weight.normal_()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(module, backend=counter, fullgraph=True, dynamic=True)
    for shape, recompute in [((2, 7, 257), False), ((3, 5, 257), True)]:
        x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        ref_weight = module.weight.detach().clone().requires_grad_()
        actual = checkpoint(compiled, x, use_reentrant=False) if recompute else compiled(x)
        with torch.no_grad():
            no_grad_output = compiled(x)
        torch.testing.assert_close(actual, no_grad_output, rtol=0, atol=0)
        expected = torch.nn.functional.rms_norm(ref_x.float(), (257,), ref_weight.float(), module.eps).to(dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (0.016, 1e-5)
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(module.weight.grad, ref_weight.grad, rtol=rtol, atol=atol)
        module.zero_grad()
    assert counter.frame_count > 0
