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

import pytest
import torch

import nemo_automodel.components.moe.experts as _experts_unused  # noqa: F401
import nemo_automodel.components.moe.optimized_ops as experts_mod
from nemo_automodel.components.moe.optimized_ops import (
    _RW_CHUNK_THRESHOLD,
    _apply_router_weight_fp32,
    _compile_router_weight_cores,
    _RouterWeightMulFunction,
    _rw_bwd_probs_core,
    _rw_bwd_x_core,
    _rw_fwd_core,
)


def _eager_reference(x: torch.Tensor, probs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Plain-autograd fp32 router-weight multiply, the trusted reference.

    Args:
        x: Expert outputs of shape [tokens, hidden].
        probs: Routing probabilities of shape [tokens, 1].
        out_dtype: Output dtype.

    Returns:
        Tensor of shape [tokens, hidden] and dtype ``out_dtype``.
    """
    return (x.float() * probs.float()).to(out_dtype)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
def test_forward_is_bitwise_identical_to_eager(x_dtype, out_dtype):
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=x_dtype)
    probs = torch.rand(64, 1, dtype=torch.float32)

    actual = _RouterWeightMulFunction.apply(x, probs, out_dtype, False)
    expected = _eager_reference(x, probs, out_dtype)

    assert actual.dtype == out_dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_backward_matches_autograd_reference(out_dtype):
    torch.manual_seed(1)
    x = torch.randn(64, 32, dtype=torch.float32)
    probs = torch.rand(64, 1, dtype=torch.float32)
    grad_out = torch.randn(64, 32, dtype=out_dtype)

    x_c = x.clone().requires_grad_()
    p_c = probs.clone().requires_grad_()
    _RouterWeightMulFunction.apply(x_c, p_c, out_dtype, True).backward(grad_out)

    x_e = x.clone().requires_grad_()
    p_e = probs.clone().requires_grad_()
    _eager_reference(x_e, p_e, out_dtype).backward(grad_out)

    torch.testing.assert_close(x_c.grad, x_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-6, atol=5e-6)


def test_multi_chunk_forward_and_backward_match_eager(monkeypatch):
    """Force several chunks per call so chunk-boundary handling is exercised."""
    monkeypatch.setattr(experts_mod, "_RW_CHUNK_ROWS", 7)
    torch.manual_seed(2)
    x = torch.randn(30, 16, dtype=torch.float32)
    probs = torch.rand(30, 1, dtype=torch.float32)
    grad_out = torch.randn(30, 16, dtype=torch.float32)

    x_c = x.clone().requires_grad_()
    p_c = probs.clone().requires_grad_()
    out = _RouterWeightMulFunction.apply(x_c, p_c, torch.float32, True)

    x_e = x.clone().requires_grad_()
    p_e = probs.clone().requires_grad_()
    expected = _eager_reference(x_e, p_e, torch.float32)

    assert torch.equal(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(x_c.grad, x_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-6, atol=5e-6)


def test_apply_router_weight_fp32_routes_large_inputs_through_function():
    torch.manual_seed(3)
    rows = _RW_CHUNK_THRESHOLD + 1
    x = torch.randn(rows, 8, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(rows, 1, dtype=torch.float32)

    out = _apply_router_weight_fp32(x, probs, torch.bfloat16)

    assert type(out.grad_fn).__name__.startswith("_RouterWeightMulFunction")
    assert torch.equal(out, _eager_reference(x.detach(), probs, torch.bfloat16))

    out.backward(torch.randn_like(out))
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_apply_router_weight_fp32_small_inputs_keep_eager_path():
    torch.manual_seed(4)
    x = torch.randn(8, 8, dtype=torch.float32, requires_grad=True)
    probs = torch.rand(8, 1, dtype=torch.float32)

    out = _apply_router_weight_fp32(x, probs, torch.float32)

    assert not type(out.grad_fn).__name__.startswith("_RouterWeightMulFunction")
    assert torch.equal(out, _eager_reference(x.detach(), probs, torch.float32))


def test_apply_router_weight_fp32_broadcast_fallback():
    """Non [tokens, 1] probs shapes must keep the eager broadcast semantics."""
    torch.manual_seed(5)
    x = torch.randn(20000, 4, dtype=torch.float32)
    probs = torch.rand(1, 4, dtype=torch.float32)

    out = _apply_router_weight_fp32(x, probs, torch.float32)

    assert torch.equal(out, _eager_reference(x, probs, torch.float32))


def test_zero_row_input():
    x = torch.zeros(0, 8, dtype=torch.float32, requires_grad=True)
    probs = torch.zeros(0, 1, dtype=torch.float32)

    out = _RouterWeightMulFunction.apply(x, probs, torch.float32, False)

    assert out.shape == (0, 8)
    out.backward(torch.zeros_like(out))
    assert x.grad.shape == x.shape


def test_saves_x_only_when_probs_needs_grad():
    torch.manual_seed(6)
    x = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)

    for probs_requires_grad, expected_saved in ((False, 1), (True, 2)):
        probs = torch.rand(16, 1, dtype=torch.float32, requires_grad=probs_requires_grad)
        saved: list[torch.Tensor] = []

        def _pack(tensor: torch.Tensor) -> torch.Tensor:
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t):
            out = _RouterWeightMulFunction.apply(x, probs, torch.bfloat16, probs.requires_grad)

        # With grad-free probs (e.g. FakeBalancedGate) x is not saved, so no
        # full-size [tokens, hidden] tensor is pinned for backward.
        assert len(saved) == expected_saved, f"probs_requires_grad={probs_requires_grad}"
        out.backward(torch.randn_like(out))
        if probs_requires_grad:
            assert probs.grad is not None
        assert x.grad is not None
        x.grad = None


def test_grad_only_for_probs():
    torch.manual_seed(7)
    x = torch.randn(16, 8, dtype=torch.float32)
    probs = torch.rand(16, 1, dtype=torch.float32)
    grad_out = torch.randn(16, 8, dtype=torch.float32)

    p_e = probs.clone().requires_grad_()
    _eager_reference(x, p_e, torch.float32).backward(grad_out)

    p_c = probs.clone().requires_grad_()
    out = _RouterWeightMulFunction.apply(x, p_c, torch.float32, True)
    out.backward(grad_out)

    assert p_c.grad is not None
    torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-6, atol=5e-6)


# --------------------------------------------------------------------------------------
# Compiled (fused whole-tensor) path: BackendConfig.compile_router_weight
# --------------------------------------------------------------------------------------


@pytest.fixture
def restore_rw_cores():
    """Snapshot and restore the module-level router-weight dispatch state around a test."""
    snapshot = (
        experts_mod._rw_fwd_dispatch,
        experts_mod._rw_bwd_x_dispatch,
        experts_mod._rw_bwd_probs_dispatch,
        experts_mod._RW_CORES_COMPILED,
    )
    yield
    (
        experts_mod._rw_fwd_dispatch,
        experts_mod._rw_bwd_x_dispatch,
        experts_mod._rw_bwd_probs_dispatch,
        experts_mod._RW_CORES_COMPILED,
    ) = snapshot


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
def test_fused_cores_match_eager_reference(x_dtype, out_dtype):
    """The whole-tensor cores compute the same fp32 math as the chunked eager Function."""
    torch.manual_seed(10)
    x = torch.randn(64, 32, dtype=x_dtype)
    probs = torch.rand(64, 1, dtype=torch.float32)
    grad_out = torch.randn(64, 32, dtype=out_dtype)
    assert torch.equal(_rw_fwd_core(x, probs, out_dtype), _eager_reference(x, probs, out_dtype))
    x_e = x.clone().float().requires_grad_()
    p_e = probs.clone().requires_grad_()
    _eager_reference(x_e, p_e, out_dtype).backward(grad_out)
    torch.testing.assert_close(_rw_bwd_x_core(grad_out, probs, torch.float32), x_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(_rw_bwd_probs_core(grad_out, x, probs.dtype), p_e.grad, rtol=1e-6, atol=5e-6)


def test_compiled_path_skips_chunk_loop_and_matches_eager(restore_rw_cores, monkeypatch):
    """With the cores marked compiled, each pass is one dispatch call (no per-chunk work)."""
    calls: dict[str, int] = {"fwd": 0, "bwd_x": 0, "bwd_p": 0}

    def _count(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(experts_mod, "_RW_CHUNK_ROWS", 7)  # eager path would take 5 chunks
    experts_mod._rw_fwd_dispatch = _count("fwd", _rw_fwd_core)
    experts_mod._rw_bwd_x_dispatch = _count("bwd_x", _rw_bwd_x_core)
    experts_mod._rw_bwd_probs_dispatch = _count("bwd_p", _rw_bwd_probs_core)
    experts_mod._RW_CORES_COMPILED = True

    torch.manual_seed(11)
    x = torch.randn(30, 16, dtype=torch.float32)
    probs = torch.rand(30, 1, dtype=torch.float32)
    grad_out = torch.randn(30, 16, dtype=torch.float32)
    x_c = x.clone().requires_grad_()
    p_c = probs.clone().requires_grad_()
    out = _RouterWeightMulFunction.apply(x_c, p_c, torch.float32, True)
    x_e = x.clone().requires_grad_()
    p_e = probs.clone().requires_grad_()
    expected = _eager_reference(x_e, p_e, torch.float32)
    assert torch.equal(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(x_c.grad, x_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-6, atol=5e-6)
    assert calls == {"fwd": 1, "bwd_x": 1, "bwd_p": 1}


def test_compiled_path_zero_rows_fall_back_to_eager(restore_rw_cores):
    def _boom(*_args, **_kwargs):
        raise AssertionError("fused dispatch must not run on zero-row inputs")

    experts_mod._rw_fwd_dispatch = _boom
    experts_mod._rw_bwd_x_dispatch = _boom
    experts_mod._rw_bwd_probs_dispatch = _boom
    experts_mod._RW_CORES_COMPILED = True
    x = torch.zeros(0, 8, dtype=torch.float32, requires_grad=True)
    probs = torch.zeros(0, 1, dtype=torch.float32)
    out = _RouterWeightMulFunction.apply(x, probs, torch.float32, False)
    out.backward(torch.zeros_like(out))
    assert x.grad.shape == x.shape


def test_compile_router_weight_cores_wraps_once(restore_rw_cores):
    experts_mod._RW_CORES_COMPILED = False
    experts_mod._rw_fwd_dispatch = _rw_fwd_core
    _compile_router_weight_cores()
    assert experts_mod._RW_CORES_COMPILED is True
    assert experts_mod._rw_fwd_dispatch is not _rw_fwd_core
    assert experts_mod._rw_bwd_x_dispatch is not _rw_bwd_x_core
    assert experts_mod._rw_bwd_probs_dispatch is not _rw_bwd_probs_core
    compiled = (experts_mod._rw_fwd_dispatch, experts_mod._rw_bwd_x_dispatch, experts_mod._rw_bwd_probs_dispatch)
    _compile_router_weight_cores()
    assert (
        experts_mod._rw_fwd_dispatch,
        experts_mod._rw_bwd_x_dispatch,
        experts_mod._rw_bwd_probs_dispatch,
    ) == compiled


def _moe_config():
    from nemo_automodel.components.moe.config import MoEConfig

    return MoEConfig(
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.1,
        aux_loss_coeff=0.01,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.float32,
    )


def test_backend_flag_defaults_false_and_wires_expert_modules(restore_rw_cores, monkeypatch):
    import nemo_automodel.components.moe.experts as experts
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.moe.experts import GroupedExperts

    assert BackendConfig().compile_router_weight is False
    called: list[int] = []
    monkeypatch.setattr(experts, "_compile_router_weight_cores", lambda: called.append(1))
    common = dict(attn="eager", linear="torch", experts="torch", dispatcher="torch", enable_hf_state_dict_adapter=False)
    GroupedExperts(_moe_config(), backend=BackendConfig(**common))
    assert called == []
    GroupedExperts(_moe_config(), backend=BackendConfig(**common, compile_router_weight=True))
    assert called == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled execution needs a GPU and inductor")
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="torch.compile of the router-weight cores on first use.",
)
def test_compiled_execution_matches_eager_on_gpu(restore_rw_cores):
    experts_mod._RW_CORES_COMPILED = False
    _compile_router_weight_cores()
    torch.manual_seed(12)
    x = torch.randn(4096, 64, dtype=torch.bfloat16, device="cuda")
    probs = torch.rand(4096, 1, dtype=torch.float32, device="cuda")
    grad_out = torch.randn(4096, 64, dtype=torch.bfloat16, device="cuda")
    x_c = x.clone().requires_grad_()
    p_c = probs.clone().requires_grad_()
    out = _RouterWeightMulFunction.apply(x_c, p_c, torch.bfloat16, True)
    x_e = x.clone().requires_grad_()
    p_e = probs.clone().requires_grad_()
    expected = _eager_reference(x_e, p_e, torch.bfloat16)
    torch.testing.assert_close(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(x_c.grad, x_e.grad)
    torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-4, atol=1e-4)
