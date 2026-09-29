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

import importlib.util
import math
from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import nemo_automodel.components.training.utils as training_utils
from nemo_automodel.components.training.utils import _all_reduce_scalar, clip_grad_norm, count_tail_padding, get_expert_tp_replication_factor, scale_grads_and_clip_grad_norm

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


def test_docstring_example():
    labels = torch.tensor(
        [
            [-100, 1, 1, -100, -100],  # 2 tail -100s
            [-100, -100, 2, 3, 4],  # 0 tail -100s
            [5, 6, -100, -100, -100],  # 3 tail -100s
        ]
    )
    assert count_tail_padding(labels) == 5


@pytest.mark.parametrize(
    "labels, expected",
    [
        # No padding at all
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), 0),
        # Entire sequence is padding
        (torch.full((2, 4), -100), 8),
        # Different ignore label
        (torch.tensor([[9, 0, 0], [0, 0, 0]]), 5),
    ],
)
def test_various_cases(labels, expected):
    """
    Covers:
    1. no ignore_label present
    2. every position is ignore_label
    3. custom ignore_label value (0)
    """
    ignore_label = 0 if (labels == 0).any() else -100
    assert count_tail_padding(labels, ignore_label=ignore_label) == expected


def test_random_shapes():
    """
    Generate random examples and compare with a simple-but-slow reference
    implementation to guard against shape / broadcasting regressions.
    """
    torch.manual_seed(0)
    for _ in range(10):
        batch = torch.randint(
            1,
            8,
            size=(
                torch.randint(1, 5, ()).item(),  # batch size
                torch.randint(1, 10, ()).item(),
            ),
        )  # seq len
        # randomly sprinkle ignore tokens
        mask = torch.rand_like(batch.float()) < 0.3
        batch[mask] = -100

        # brute-force reference
        ref = 0
        for row in batch:
            idx = (row != -100).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                ref += row.numel()
            else:
                ref += (row[idx[-1] + 1 :] == -100).sum().item()

        assert count_tail_padding(batch) == ref


def test_clip_grad_norm_with_pp_and_tp():
    """Test that clip_grad_norm works with PP and TP enabled (no longer skips)."""
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)

    device_mesh = Mock()
    device_mesh.mesh_dim_names = ["pp", "tp"]
    # device_type must be a real string: a DeviceMesh always exposes one, and the norm
    # reduction compares it against the accumulator's device to pick the comm device.
    device_mesh.__getitem__ = Mock(side_effect=lambda key: Mock(size=Mock(return_value=2), device_type="cpu"))

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=True,
        device_mesh=device_mesh,
        pp_axis_name="pp",
    )

    # Should now clip (not skip) with the new sharding-aware implementation
    assert grad_norm > 0


def test_clip_grad_norm_works_without_pp():
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=False,
    )

    assert isinstance(grad_norm, torch.Tensor)
    assert grad_norm > 0


@pytest.mark.parametrize(
    ("gradient", "expected"),
    [
        (0.0, 0.0),
        (float("inf"), float("inf")),
        (float("nan"), float("nan")),
    ],
)
def test_clip_grad_norm_preserves_zero_and_nonfinite_results(gradient: float, expected: float):
    """Branch-free scaling preserves the prior zero and non-finite norm behavior."""
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, gradient)

    grad_norm = clip_grad_norm(max_grad_norm=1.0, model_parts=[model])

    assert isinstance(grad_norm, torch.Tensor)
    if math.isnan(expected):
        assert torch.isnan(grad_norm)
    else:
        torch.testing.assert_close(grad_norm, torch.tensor(expected, dtype=grad_norm.dtype))


def test_clip_grad_norm_uses_torch_fast_path_when_requested(monkeypatch):
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)

    clip_grad_norm_mock = Mock(return_value=torch.tensor(3.0))
    clip_grads_with_norm_mock = Mock()
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_grad_norm_mock)
    monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_", clip_grads_with_norm_mock)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=False,
        foreach=True,
        use_torch_clip_grad_norm=True,
    )

    assert grad_norm == 3.0
    clip_grad_norm_mock.assert_called_once()
    assert clip_grad_norm_mock.call_args.kwargs["norm_type"] == 2.0
    assert clip_grad_norm_mock.call_args.kwargs["error_if_nonfinite"] is False
    assert clip_grad_norm_mock.call_args.kwargs["foreach"] is True
    clip_grads_with_norm_mock.assert_not_called()


@pytest.mark.parametrize("backend", [None, "triton", "te"])
def test_clip_grad_norm_selects_requested_backend(monkeypatch, backend):
    model = torch.nn.Linear(2, 1, bias=False)
    gradient = torch.tensor([[3.0, 4.0]])
    model.weight.grad = gradient.clone()

    triton_norm = Mock(return_value=torch.tensor(25.0, dtype=torch.float64))
    te_norm = Mock(return_value=torch.tensor(5.0, dtype=torch.float64))
    monkeypatch.setattr(training_utils, "_use_fused_grad_norm", lambda *_: True)
    monkeypatch.setattr(training_utils, "multi_tensor_sumsq", triton_norm)
    monkeypatch.setattr(training_utils, "_local_te_l2_norm", te_norm)

    options = {} if backend is None else {"grad_norm_backend": backend}
    observed = clip_grad_norm(max_grad_norm=1.0, model_parts=[model], **options)

    torch.testing.assert_close(observed, torch.tensor(5.0, dtype=torch.float64))
    torch.testing.assert_close(model.weight.grad, gradient / (5.0 + 1e-6))
    assert triton_norm.call_count == (backend != "te")
    assert te_norm.call_count == (backend == "te")


def test_clip_grad_norm_rejects_invalid_backend():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)

    with pytest.raises(ValueError, match="grad_norm_backend must be 'triton' or 'te'"):
        clip_grad_norm(max_grad_norm=1.0, model_parts=[model], grad_norm_backend="invalid")


def test_clip_grad_norm_disables_torch_fast_path_for_owner_shard(monkeypatch):
    """A model-owned local shard requires the contract's global norm group."""
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    model.weight._nemo_model_owned_grad_divisor = 1.0

    torch_clip_mock = Mock(return_value=torch.tensor(-1.0))
    sharding_aware_clip_mock = Mock()
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", torch_clip_mock)
    monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_", sharding_aware_clip_mock)

    grad_norm = clip_grad_norm(
        max_grad_norm=10.0,
        model_parts=[model],
        use_torch_clip_grad_norm=True,
    )

    torch.testing.assert_close(grad_norm, torch.tensor(5.0, dtype=torch.float64))
    torch_clip_mock.assert_not_called()
    sharding_aware_clip_mock.assert_called_once()


def test_clip_grad_norm_returns_zero_when_max_grad_norm_is_none():
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)

    grad_norm = clip_grad_norm(
        max_grad_norm=None,
        model_parts=[model],
        pp_enabled=False,
    )

    assert grad_norm == 0


def test_clip_grad_norm_with_multiple_models():
    """Test that clip_grad_norm works with multiple model parts."""
    model1 = torch.nn.Linear(10, 10)
    model2 = torch.nn.Linear(20, 20)

    model1.weight.grad = torch.randn_like(model1.weight)
    model1.bias.grad = torch.randn_like(model1.bias)
    model2.weight.grad = torch.randn_like(model2.weight)
    model2.bias.grad = torch.randn_like(model2.bias)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model1, model2],
        pp_enabled=False,
    )

    assert grad_norm > 0
    # Verify gradients were actually clipped
    assert model1.weight.grad.norm().item() <= 1.0 + 1e-5
    assert model2.weight.grad.norm().item() <= 1.0 + 1e-5


def test_clip_grad_norm_actually_clips():
    """Test that gradients are actually clipped to max_norm."""
    model = torch.nn.Linear(10, 10)
    # Set large gradients
    model.weight.grad = torch.ones_like(model.weight) * 10.0
    model.bias.grad = torch.ones_like(model.bias) * 10.0

    initial_norm = torch.nn.utils.clip_grad_norm_([model.weight, model.bias], float("inf")).item()

    # Reset gradients
    model.weight.grad = torch.ones_like(model.weight) * 10.0
    model.bias.grad = torch.ones_like(model.bias) * 10.0

    max_norm = 1.0
    grad_norm = clip_grad_norm(
        max_grad_norm=max_norm,
        model_parts=[model],
        pp_enabled=False,
    )

    assert grad_norm > 0
    # The reported norm should be the original (unclipped) norm
    assert abs(grad_norm - initial_norm) < 1e-3

    # Verify the actual gradients are clipped
    clipped_norm = torch.sqrt(model.weight.grad.pow(2).sum() + model.bias.grad.pow(2).sum()).item()
    assert abs(clipped_norm - max_norm) < 1e-3


def test_clip_grad_norm_large_finite_gradients_do_not_overflow():
    """Finite rare-token embedding gradients can exceed fp32 squared-norm range."""
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[6.0e31, 4.0]], dtype=model.weight.dtype)

    max_norm = 0.3
    grad_norm = clip_grad_norm(
        max_grad_norm=max_norm,
        model_parts=[model],
        pp_enabled=False,
    )

    assert math.isfinite(grad_norm)
    assert grad_norm > 1.0e31
    assert torch.isfinite(model.weight.grad).all()
    assert torch.linalg.vector_norm(model.weight.grad.double(), ord=2).item() <= max_norm + 1.0e-6


def _run_empty_local_shard_clip_worker(rank: int, world_size: int, init_file: str) -> None:
    """Exercise clipping when one rank owns an empty DTensor gradient shard."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        local_param = torch.ones(1, 2) if rank == 0 else torch.empty(0, 2)
        local_grad = torch.tensor([[3.0, 4.0]]) if rank == 0 else torch.empty(0, 2)
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(
            DTensor.from_local(
                local_param,
                mesh,
                [Shard(0)],
                run_check=False,
                shape=(1, 2),
                stride=(2, 1),
            )
        )
        model.weight.grad = DTensor.from_local(
            local_grad,
            mesh,
            [Shard(0)],
            run_check=False,
            shape=(1, 2),
            stride=(2, 1),
        )

        grad_norm = clip_grad_norm(
            max_grad_norm=1.0,
            model_parts=[model],
            pp_enabled=False,
            foreach=False,
        )

        assert grad_norm == pytest.approx(5.0)
        if rank == 0:
            torch.testing.assert_close(model.weight.grad.to_local(), torch.tensor([[0.6, 0.8]]))
        else:
            assert model.weight.grad.to_local().numel() == 0
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def test_clip_grad_norm_handles_empty_local_dtensor_shards(tmp_path):
    """Empty rank-local DTensor shards contribute zero without breaking collectives."""
    torch.multiprocessing.spawn(
        _run_empty_local_shard_clip_worker,
        args=(2, str(tmp_path / "empty_local_shard_pg")),
        nprocs=2,
        join=True,
    )


class _FakeMesh:
    """Minimal DeviceMesh stand-in; ``_all_reduce_scalar`` needs only these two members."""

    def __init__(self, device_type: str):
        self.device_type = device_type

    def get_group(self, mesh_dim=None):
        return f"pg:{mesh_dim}"


def test_all_reduce_scalar_reduces_in_place_when_devices_match(monkeypatch):
    """Without CPU offload the scalar already sits on the mesh device: reduce in place.

    This is the guard that the fix is inert for every non-offload recipe -- the same
    tensor object goes into the same in-place all_reduce and comes back out.
    """
    seen = {}

    def fake_all_reduce(tensor, op=None, group=None):
        seen["tensor"] = tensor
        seen["group"] = group
        tensor.add_(1.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    mesh = _FakeMesh("cpu")
    scalar = torch.zeros((), dtype=torch.float64)
    out = _all_reduce_scalar(scalar, torch.distributed.ReduceOp.MAX, mesh, 0)

    assert out is scalar
    assert seen["tensor"] is scalar
    assert seen["group"] == "pg:0"
    assert out.item() == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_all_reduce_scalar_moves_offloaded_scalar_to_mesh_device(monkeypatch):
    """CPU-offloaded gradients must not be all-reduced on the NCCL group as CPU tensors.

    Regression test for the CPUOffloadPolicy crash: FSDP2 leaves the local gradient
    shard -- and therefore the norm accumulator derived from it -- on CPU, while the
    mesh's process group is NCCL and has no CPU backend.
    """
    seen = {}

    def fake_all_reduce(tensor, op=None, group=None):
        seen["device"] = tensor.device
        tensor.add_(1.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    mesh = _FakeMesh("cuda")
    scalar = torch.zeros((), dtype=torch.float64)  # on CPU, as under CPUOffloadPolicy
    out = _all_reduce_scalar(scalar, torch.distributed.ReduceOp.SUM, mesh, 1)

    # Pre-fix this was "cpu", which NCCL rejects with
    # "No backend type associated with device type cpu".
    assert seen["device"].type == "cuda"
    # The result returns to the gradients' device so the clip math stays consistent.
    assert out.device.type == "cpu"
    assert out.item() == 1.0


def test_clip_grad_norm_with_inf_norm():
    """Test clip_grad_norm with infinity norm."""
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)
    model.bias.grad = torch.randn_like(model.bias)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        norm_type=float("inf"),
        pp_enabled=False,
    )

    assert grad_norm > 0


def test_clip_grad_norm_with_empty_gradients():
    """Test that clip_grad_norm handles parameters without gradients."""
    model = torch.nn.Linear(10, 10)
    # Only set gradient for weight, not bias
    model.weight.grad = torch.randn_like(model.weight)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=False,
    )

    # Should work even with some None gradients
    assert grad_norm > 0


def test_clip_grad_norm_with_all_none_gradients():
    """Test that clip_grad_norm handles all None gradients gracefully."""
    model = torch.nn.Linear(10, 10)
    # Don't set any gradients

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=False,
    )

    # Should return 0 when no gradients exist
    assert grad_norm == 0.0


def test_clip_grad_norm_different_norm_types():
    """Test clip_grad_norm with different norm types (L1, L2, Linf)."""
    model = torch.nn.Linear(10, 10)

    for norm_type in [1.0, 2.0, float("inf")]:
        # Reset gradients
        model.weight.grad = torch.randn_like(model.weight)
        model.bias.grad = torch.randn_like(model.bias)

        grad_norm = clip_grad_norm(
            max_grad_norm=1.0,
            model_parts=[model],
            norm_type=norm_type,
            pp_enabled=False,
        )

        assert grad_norm >= 0, f"Norm type {norm_type} failed"


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        self.register_buffer("scale", torch.ones(1))


def _all_tensors_on_device(module: nn.Module, device_type: str) -> bool:
    for p in module.parameters():
        if p.device.type != device_type:
            return False
    for b in module.buffers():
        if b.device.type != device_type:
            return False
    return True


class _TEGroupedLinearMock(nn.Module):
    """Mock TE GroupedLinear with weight0 parameter naming."""

    def __init__(self):
        super().__init__()
        # Parameter name matches TE GroupedLinear pattern (weight0, weight1, etc.)
        self.weight0 = nn.Parameter(torch.randn(4, 2))


class _ExpertsModule(nn.Module):
    """Mock experts module with parameters matching GroupedExpertsTE pattern."""

    def __init__(self):
        super().__init__()
        # Submodule names match GroupedExpertsTE: gate_up_linear, down_linear
        self.gate_up_linear = _TEGroupedLinearMock()


class _MoEModule(nn.Module):
    """Mock MoE module with expert parameters for testing EP scaling."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        # FQN will be mlp.experts.gate_up_linear.weight0, matching _EXPERT_PARAM_PATTERN
        self.mlp = nn.ModuleDict({"experts": _ExpertsModule()})


class TestScaleGradsAndClipGradNorm:
    """Tests for scale_grads_and_clip_grad_norm with EP scaling."""

    def test_owner_shard_uses_explicit_gradient_divisor(self):
        """Owner scaling is declared by the model and independent of DP arguments."""
        model = nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.full_like(model.weight, 8.0)
        model.weight._nemo_model_owned_grad_divisor = 4.0

        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,
            model_parts=[model],
            dp_group_size=999,
        )

        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 2.0))

    def test_ep_scaling_for_expert_params_by_name(self):
        """Test that expert params are scaled by EP ratio based on param name."""
        model = _MoEModule()
        expert_param = model.mlp["experts"].gate_up_linear.weight0
        model.gate.weight.grad = torch.ones_like(model.gate.weight) * 2.0
        expert_param.grad = torch.ones_like(expert_param) * 2.0

        # Mock moe_mesh to trigger EP scaling
        moe_mesh = Mock()
        moe_mesh.mesh_dim_names = ["ep_shard"]
        moe_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))

        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,  # Disable clipping to test scaling only
            model_parts=[model],
            pp_enabled=False,
            moe_mesh=moe_mesh,
            dp_group_size=4,
        )

        # ep_ratio = dp_group_size / ep_shard_size = 4 / 2 = 2
        # Expert params (mlp.experts.gate_up_linear.weight0) should be scaled by 1/2
        # Non-expert params (gate) should NOT be scaled
        assert torch.allclose(model.gate.weight.grad, torch.ones_like(model.gate.weight) * 2.0)
        assert torch.allclose(expert_param.grad, torch.ones_like(expert_param) * 1.0)

    def test_ep_scaling_removes_replicated_tp_token_factor_from_experts_only(self):
        """TP-replicated MoE inputs add a TP factor only to expert gradients."""
        model = _MoEModule()
        expert_param = model.mlp["experts"].gate_up_linear.weight0
        model.gate.weight.grad = torch.ones_like(model.gate.weight) * 8.0
        expert_param.grad = torch.ones_like(expert_param) * 8.0

        moe_mesh = Mock()
        moe_mesh.mesh_dim_names = ["ep_shard"]
        moe_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))

        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,
            model_parts=[model],
            moe_mesh=moe_mesh,
            dp_group_size=4,
            expert_tp_replication_factor=2,
        )

        # Base EP divisor = 4/2 = 2; replicated TP tokens add another 2.
        assert torch.allclose(expert_param.grad, torch.ones_like(expert_param) * 2.0)
        # Router/dense replicas must never receive the expert-only divisor;
        # their TP synchronization is owned separately at the optimizer boundary.
        assert torch.allclose(model.gate.weight.grad, torch.ones_like(model.gate.weight) * 8.0)

    @pytest.mark.parametrize(
        "factor,exc",
        [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
    )
    def test_ep_tp_replication_factor_is_strictly_validated(self, factor, exc):
        with pytest.raises(exc, match="expert_tp_replication_factor"):
            scale_grads_and_clip_grad_norm(
                max_grad_norm=None,
                model_parts=[_MoEModule()],
                expert_tp_replication_factor=factor,
            )

    def test_get_expert_tp_replication_factor_requires_moe_tp_marker_and_tp_axis(self):
        """The factor is tp_size only when the custom-MoE TP path marked the model."""
        marked = _MoEModule()
        marked._nemo_moe_tp_requires_replica_sync = True
        unmarked = _MoEModule()

        tp_mesh = Mock()
        tp_mesh.mesh_dim_names = ("dp", "tp")
        tp_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))
        no_tp_mesh = Mock()
        no_tp_mesh.mesh_dim_names = ("dp",)

        assert get_expert_tp_replication_factor([marked], tp_mesh) == 2
        assert get_expert_tp_replication_factor([unmarked], tp_mesh) == 1
        assert get_expert_tp_replication_factor([marked], no_tp_mesh) == 1
        assert get_expert_tp_replication_factor([marked], None) == 1

        tp1_mesh = Mock()
        tp1_mesh.mesh_dim_names = ("dp", "tp")
        tp1_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=1)))
        assert get_expert_tp_replication_factor([marked], tp1_mesh) == 1

    def test_no_ep_scaling_without_moe_mesh(self):
        """Test that no EP scaling occurs when moe_mesh is None."""
        model = _MoEModule()
        expert_param = model.mlp["experts"].gate_up_linear.weight0
        model.gate.weight.grad = torch.ones_like(model.gate.weight) * 2.0
        expert_param.grad = torch.ones_like(expert_param) * 2.0

        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,
            model_parts=[model],
            pp_enabled=False,
            moe_mesh=None,
            dp_group_size=4,
        )

        # No scaling should occur
        assert torch.allclose(model.gate.weight.grad, torch.ones_like(model.gate.weight) * 2.0)
        assert torch.allclose(expert_param.grad, torch.ones_like(expert_param) * 2.0)

    def test_ep_scaling_combined_with_pp_scaling(self):
        """Test that PP and EP scaling work together correctly."""
        model = _MoEModule()
        expert_param = model.mlp["experts"].gate_up_linear.weight0
        model.gate.weight.grad = torch.ones_like(model.gate.weight) * 8.0
        expert_param.grad = torch.ones_like(expert_param) * 8.0

        # Mock device_mesh for PP
        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["pp"]
        device_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))

        # Mock moe_mesh for EP
        moe_mesh = Mock()
        moe_mesh.mesh_dim_names = ["ep_shard"]
        moe_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))

        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,
            model_parts=[model],
            pp_enabled=True,
            device_mesh=device_mesh,
            pp_axis_name="pp",
            moe_mesh=moe_mesh,
            dp_group_size=4,
            num_label_tokens=8,  # pp_divisor = num_label_tokens / dp_group_size = 8 / 4 = 2
        )

        # PP scaling: all grads divided by (num_label_tokens / dp_group_size) = 8/4 = 2 -> 8 / 2 = 4
        # EP scaling: expert grads divided by ep_ratio (4/2=2) -> 4 / 2 = 2
        # Non-expert params: only PP scaling -> 4
        assert torch.allclose(model.gate.weight.grad, torch.ones_like(model.gate.weight) * 4.0)
        assert torch.allclose(expert_param.grad, torch.ones_like(expert_param) * 2.0)


@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
@pytest.mark.parametrize("transposed", [False, True])
def test_clip_large_gradient_matches_dense_float64_reference(norm_type, transposed):
    """Check large contiguous/strided gradients and clipping against a dense FP64 reference."""
    torch.manual_seed(4128)
    gradient = torch.randn(1025, 2049)
    if transposed:
        gradient = gradient.t()
    parameter = torch.nn.Parameter(torch.empty_like(gradient))
    parameter.grad = gradient.clone(memory_format=torch.preserve_format)
    reference_norm = torch.linalg.vector_norm(gradient.double(), ord=norm_type)
    reference = gradient * (0.3 / (reference_norm + 1e-6)).clamp(max=1.0)
    from nemo_automodel.components.training.utils import _clip_grad_norm_impl

    actual_norm = _clip_grad_norm_impl([parameter], 0.3, norm_type=norm_type)
    torch.testing.assert_close(actual_norm, reference_norm, atol=0, rtol=2e-7)
    torch.testing.assert_close(parameter.grad, reference, atol=0, rtol=3e-7)


@pytest.mark.parametrize("backend", ["triton", "te"])
def test_optional_te_is_not_loaded_for_import_or_cpu_clipping(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """A broken optional TE binary must not prevent import or CPU gradient clipping."""
    from nemo_automodel.components.training import utils
    from nemo_automodel.shared import import_utils

    original_import = import_utils.safe_import
    te_imports = []

    def import_with_broken_te(module, **kwargs):
        if module.startswith("transformer_engine"):
            te_imports.append(module)
            raise OSError("undefined symbol: cublasLtGroupedMatrixLayoutInit_internal")
        return original_import(module, **kwargs)

    monkeypatch.setattr(import_utils, "safe_import", import_with_broken_te)
    spec = importlib.util.spec_from_file_location("_training_utils_import_test", utils.__file__)
    isolated_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated_utils)

    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    norm = isolated_utils._clip_grad_norm_impl([parameter], 1.0, grad_norm_backend=backend)
    torch.testing.assert_close(norm, torch.tensor(5.0, dtype=torch.float64))
    expected_gradient = torch.tensor([3.0, 4.0]) / (5.0 + 1e-6)
    torch.testing.assert_close(parameter.grad, expected_gradient)
    torch.optim.SGD([parameter], lr=0.1).step()
    torch.testing.assert_close(parameter, torch.tensor([1.0, 2.0]) - 0.1 * expected_gradient)
    assert not te_imports
