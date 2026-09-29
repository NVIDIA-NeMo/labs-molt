# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for async-TP linear graph shaping (_tp_linear helpers and TPLinear)."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard, distribute_tensor

from nemo_automodel.components.distributed.parallel_styles import TPLinear
from nemo_automodel.shared.tp_linear import _async_tp_linear, _is_async_tp_linear_enabled


@pytest.fixture
def micro_pipeline_tp_enabled():
    """Enable torch._inductor.config._micro_pipeline_tp and restore it afterwards."""
    original = torch._inductor.config._micro_pipeline_tp
    torch._inductor.config._micro_pipeline_tp = True
    yield
    torch._inductor.config._micro_pipeline_tp = original


@pytest.fixture
def micro_pipeline_tp_disabled():
    """Force torch._inductor.config._micro_pipeline_tp off and restore it afterwards."""
    original = torch._inductor.config._micro_pipeline_tp
    torch._inductor.config._micro_pipeline_tp = False
    yield
    torch._inductor.config._micro_pipeline_tp = original


@pytest.fixture
def single_rank_pg():
    """Provide a single-rank gloo process group for DTensor placement tests."""
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    already = dist.is_initialized()
    if not already:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield
    finally:
        if not already:
            dist.destroy_process_group()


def _capture_compiled_graphs(module: nn.Module, x: torch.Tensor) -> tuple[list[torch.fx.GraphModule], torch.Tensor]:
    """Compile ``module`` with a graph-capturing backend and run it on ``x``.

    Args:
        module: Module to compile; its forward must accept a single tensor.
        x: Input activations of shape ``[..., in_features]``.

    Returns:
        A tuple of (captured Dynamo FX graphs, output tensor of shape
        ``[..., out_features]``).
    """
    graphs: list[torch.fx.GraphModule] = []

    def _backend(gm: torch.fx.GraphModule, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(module, backend=_backend, fullgraph=True)
    out = compiled(x)
    return graphs, out


def _call_function_targets(graphs: list[torch.fx.GraphModule]) -> set:
    """Collect call_function targets from a list of Dynamo FX graphs."""
    return {node.target for gm in graphs for node in gm.graph.nodes if node.op == "call_function"}


class TestIsAsyncTpLinearEnabled:
    """Tests for the _is_async_tp_linear_enabled gate."""

    def test_false_in_eager_even_with_flag_set(self, micro_pipeline_tp_enabled):
        """The gate must stay closed outside torch.compile tracing."""
        assert not torch.compiler.is_compiling()
        assert _is_async_tp_linear_enabled() is False

    def test_false_when_compiling_without_flag(self, micro_pipeline_tp_disabled):
        """The gate must stay closed when _micro_pipeline_tp is not set."""
        with patch("torch.compiler.is_compiling", return_value=True):
            assert _is_async_tp_linear_enabled() is False

    def test_true_when_compiling_with_flag(self, micro_pipeline_tp_enabled):
        """The gate opens only under compile with _micro_pipeline_tp set."""
        with patch("torch.compiler.is_compiling", return_value=True):
            assert _is_async_tp_linear_enabled() is True


class TestAsyncTpLinearNumerics:
    """Numerical equivalence of _async_tp_linear against F.linear."""

    @pytest.mark.parametrize("shape", [(8, 16), (2, 5, 16)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_matches_f_linear_forward(self, shape, use_bias):
        """_async_tp_linear must match F.linear for 2-D and 3-D inputs."""
        torch.manual_seed(0)
        x = torch.randn(*shape)
        weight = torch.randn(12, 16)
        bias = torch.randn(12) if use_bias else None

        out = _async_tp_linear(x, weight, bias)
        ref = F.linear(x, weight, bias)

        assert out.shape == ref.shape
        assert torch.allclose(out, ref)

    def test_matches_f_linear_backward(self):
        """Gradients through _async_tp_linear must match F.linear."""
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16, requires_grad=True)
        weight = torch.randn(12, 16, requires_grad=True)
        bias = torch.randn(12, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)

        grad = torch.randn(2, 5, 12)
        _async_tp_linear(x, weight, bias).backward(grad)
        F.linear(x_ref, weight_ref, bias_ref).backward(grad)

        assert torch.allclose(x.grad, x_ref.grad)
        assert torch.allclose(weight.grad, weight_ref.grad)
        assert torch.allclose(bias.grad, bias_ref.grad)


class TestTPLinearGraphShaping:
    """TPLinear.forward must emit the graph async-TP fusion can pattern-match."""

    def _make_tp_linear(self) -> nn.Linear:
        torch.manual_seed(0)
        linear = nn.Linear(16, 12)
        linear.__class__ = TPLinear
        return linear

    def test_async_tp_mode_emits_native_linear(self, micro_pipeline_tp_enabled):
        """With _micro_pipeline_tp set, compile must trace F.linear, not bmm."""
        linear = self._make_tp_linear()
        x = torch.randn(2, 5, 16)

        graphs, out = _capture_compiled_graphs(linear, x)
        targets = _call_function_targets(graphs)

        assert torch.bmm not in targets
        assert F.linear in targets
        assert torch.allclose(out, F.linear(x, linear.weight, linear.bias))

    def test_default_compile_path_keeps_bmm(self, micro_pipeline_tp_disabled):
        """Without the flag, the DTensor-safe bmm path must be preserved."""
        linear = self._make_tp_linear()
        x = torch.randn(2, 5, 16)

        graphs, out = _capture_compiled_graphs(linear, x)
        targets = _call_function_targets(graphs)

        assert torch.bmm in targets
        assert F.linear not in targets
        assert torch.allclose(out, F.linear(x, linear.weight, linear.bias))

    def test_default_compile_path_2d_uses_mm(self, micro_pipeline_tp_disabled):
        """Without the flag, 2-D input under compile must keep TPLinear's torch.mm numerics."""
        linear = self._make_tp_linear()
        x = torch.randn(8, 16)

        graphs, out = _capture_compiled_graphs(linear, x)
        targets = _call_function_targets(graphs)

        assert torch.mm in targets
        assert torch.bmm not in targets
        assert F.linear not in targets
        assert torch.allclose(out, F.linear(x, linear.weight, linear.bias), atol=1e-6)

    def test_eager_path_unaffected_by_flag(self, micro_pipeline_tp_enabled):
        """Eager TPLinear.forward must not change when only the flag is set."""
        linear = self._make_tp_linear()
        x = torch.randn(2, 5, 16)

        out = linear(x)

        assert torch.allclose(out, F.linear(x, linear.weight, linear.bias))


class TestTPLinearShardedInputUnderAsyncTp:
    """DTensor placement must select the safe, fusable async-TP graph."""

    def test_dim1_sharded_dtensor_takes_bmm_not_async_shaping(self, single_rank_pg):
        """A sequence-sharded DTensor input must bypass async-TP shaping and not crash.

        Builds a TPLinear with replicated DTensor weight/bias on a world-size-1
        mesh and feeds a ``[B, S, in_features]`` DTensor input sharded on dim 1
        (the sequence-parallel layout async-TP mandates), with the async-TP gate
        mocked open.  The forward must route through ``torch.bmm``, never through
        ``_async_tp_linear`` (whose ``F.linear`` view cannot flatten the sharded
        dim), and match the local F.linear reference.
        """
        torch.manual_seed(0)
        linear = nn.Linear(16, 12)
        linear.__class__ = TPLinear
        weight_local = linear.weight.detach().clone()
        bias_local = linear.bias.detach().clone()

        mesh = DeviceMesh("cpu", torch.arange(1))
        linear.weight = nn.Parameter(distribute_tensor(weight_local, mesh, [Replicate()]))
        linear.bias = nn.Parameter(distribute_tensor(bias_local, mesh, [Replicate()]))
        x_local = torch.randn(2, 5, 16)
        x = DTensor.from_local(x_local, mesh, [Shard(1)], run_check=False)

        with (
            patch("nemo_automodel.shared.tp_linear._is_async_tp_linear_enabled", return_value=True),
            patch("nemo_automodel.shared.tp_linear._async_tp_linear") as async_spy,
            patch("torch.bmm", wraps=torch.bmm) as bmm_spy,
        ):
            out = linear(x)

        async_spy.assert_not_called()
        bmm_spy.assert_called_once()
        assert isinstance(out, DTensor)
        assert torch.allclose(out.full_tensor(), F.linear(x_local, weight_local, bias_local), atol=1e-6)

    def test_negative_last_dim_shard_takes_async_shaping(self, single_rank_pg):
        """A row-parallel ``Shard(-1)`` input must take the fusable linear graph.

        Builds a bias-free TPLinear with a row-sharded ``[out_features,
        in_features]`` weight and feeds a ``[B, S, in_features]`` DTensor input
        sharded on its last/feature dimension. ``Shard(-1)`` is equivalent to
        ``Shard(2)`` for this 3-D input and must not be mistaken for batch or
        sequence sharding.
        """
        torch.manual_seed(0)
        linear = nn.Linear(16, 12, bias=False)
        linear.__class__ = TPLinear
        weight_local = linear.weight.detach().clone()

        mesh = DeviceMesh("cpu", torch.arange(1))
        linear.weight = nn.Parameter(distribute_tensor(weight_local, mesh, [Shard(1)]))
        x_local = torch.randn(2, 5, 16)
        x = DTensor.from_local(x_local, mesh, [Shard(-1)], run_check=False)

        with (
            patch("nemo_automodel.shared.tp_linear._is_async_tp_linear_enabled", return_value=True),
            patch("nemo_automodel.shared.tp_linear._async_tp_linear", wraps=_async_tp_linear) as async_spy,
            patch("torch.bmm", wraps=torch.bmm) as bmm_spy,
        ):
            out = linear(x)

        async_spy.assert_called_once()
        bmm_spy.assert_not_called()
        assert isinstance(out, DTensor)
        assert torch.allclose(out.full_tensor(), F.linear(x_local, weight_local), atol=1e-6)
