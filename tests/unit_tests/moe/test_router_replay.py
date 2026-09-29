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

"""Unit tests for Rollout Routing Replay (R3)."""

import pytest
import torch

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate
from nemo_automodel.components.moe.router_replay import (
    RouterReplay,
    RouterReplayMode,
    replay_selection,
)

# Every score_func branch in Gate.forward selects experts with its own top-k call;
# routing replay must override the selection identically in all of them.
SCORE_FUNCS = [
    "softmax",
    "softmax_with_bias",
    "sqrtsoftplus",
    "sigmoid_with_bias",
    "sigmoid",
]


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global RouterReplay registry across tests."""
    RouterReplay.clear_registry()
    yield
    RouterReplay.clear_registry()


def make_config(score_func="softmax", enable_routing_replay=True, **overrides):
    cfg = dict(
        n_routed_experts=8,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func=score_func,
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        dtype=torch.float32,
        enable_routing_replay=enable_routing_replay,
    )
    cfg.update(overrides)
    return MoEConfig(**cfg)


def make_gate(score_func="softmax", enable_routing_replay=True, **overrides):
    gate = Gate(make_config(score_func, enable_routing_replay, **overrides))
    # Spread the router weights so two different inputs route differently.
    with torch.no_grad():
        gate.weight.normal_(0.0, 1.0)
    return gate


def run(gate, x):
    token_mask = torch.ones(x.shape[0], dtype=torch.bool)
    return gate(x, token_mask, None)


# --------------------------------------------------------------------------- #
# Config + helper
# --------------------------------------------------------------------------- #


def test_config_default_off():
    assert MoEConfig.__dataclass_fields__["enable_routing_replay"].default is False
    gate = Gate(make_config(enable_routing_replay=False))
    assert gate.router_replay is None
    # A disabled gate registers nothing.
    assert RouterReplay.instances() == []


def test_enabled_gate_registers_once():
    gate = make_gate()
    assert isinstance(gate.router_replay, RouterReplay)
    assert RouterReplay.instances() == [gate.router_replay]


def test_replay_selection_passthrough_when_disabled():
    idx = torch.tensor([[0, 1], [2, 3]])
    assert replay_selection(None, idx) is idx


# --------------------------------------------------------------------------- #
# No-op guarantees: an enabled-but-inactive gate matches a disabled gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("score_func", SCORE_FUNCS)
def test_disabled_path_is_noop(score_func):
    torch.manual_seed(0)
    x = torch.randn(6, 16)

    g_off = make_gate(score_func, enable_routing_replay=False)
    g_on = make_gate(score_func, enable_routing_replay=True)
    # Same router weights so the only difference would be the replay plumbing.
    g_on.load_state_dict(g_off.state_dict())

    w_off, i_off, _ = run(g_off, x)
    w_on, i_on, _ = run(g_on, x)  # enabled but no mode set -> inactive

    torch.testing.assert_close(w_off, w_on)
    assert torch.equal(i_off, i_on)


def test_record_mode_does_not_change_output():
    torch.manual_seed(0)
    x = torch.randn(6, 16)
    gate = make_gate()

    w_plain, i_plain, _ = run(gate, x)
    with RouterReplay.record():
        w_rec, i_rec, _ = run(gate, x)

    torch.testing.assert_close(w_plain, w_rec)
    assert torch.equal(i_plain, i_rec)


# --------------------------------------------------------------------------- #
# Record -> replay roundtrip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("score_func", SCORE_FUNCS)
def test_record_replay_roundtrip(score_func):
    torch.manual_seed(0)
    gate = make_gate(score_func)

    x_rollout = torch.randn(6, 16)
    x_train = torch.randn(6, 16)  # different tokens -> different natural routing

    # Sanity: the two inputs route differently without replay.
    _, i_rollout, _ = run(gate, x_rollout)
    _, i_train_free, _ = run(gate, x_train)
    assert not torch.equal(i_rollout, i_train_free)

    # Record on the rollout forward, replay on the training forward.
    with RouterReplay.record():
        _, i_recorded, _ = run(gate, x_rollout)
    captured = RouterReplay.collect()
    assert len(captured) == 1
    assert torch.equal(captured[0], i_recorded)

    with RouterReplay.replay(captured):
        w_replay, i_replay, _ = run(gate, x_train)

    # Replayed selection matches the rollout, not the training input's natural pick.
    assert torch.equal(i_replay, i_rollout)
    assert not torch.equal(i_replay, i_train_free)


def test_replay_recomputes_weights_from_live_scores():
    """Weights under replay come from the *current* scores at the replayed indices."""
    torch.manual_seed(1)
    gate = make_gate("sigmoid")  # weights = sigmoid(scores).gather(indices), simple to mirror

    x_rollout = torch.randn(5, 16)
    x_train = torch.randn(5, 16)

    with RouterReplay.record():
        run(gate, x_rollout)
    captured = RouterReplay.collect()

    with RouterReplay.replay(captured):
        w_replay, i_replay, _ = run(gate, x_train)

    # Reference: sigmoid of the live (train) logits gathered at the replayed indices.
    live_scores = torch.sigmoid(torch.nn.functional.linear(x_train, gate.weight))
    expected = live_scores.gather(1, i_replay)
    torch.testing.assert_close(w_replay, expected)


def test_replay_keeps_gradient_to_router():
    torch.manual_seed(2)
    gate = make_gate("softmax")

    x_rollout = torch.randn(4, 16)
    x_train = torch.randn(4, 16)
    with RouterReplay.record():
        run(gate, x_rollout)
    captured = RouterReplay.collect()

    with RouterReplay.replay(captured):
        w_replay, _, _ = run(gate, x_train)
    w_replay.sum().backward()

    assert gate.weight.grad is not None
    assert torch.any(gate.weight.grad != 0)


# --------------------------------------------------------------------------- #
# Multi-layer global control
# --------------------------------------------------------------------------- #


def test_multilayer_distribute_and_collect():
    torch.manual_seed(3)
    gates = [make_gate("softmax") for _ in range(3)]
    assert len(RouterReplay.instances()) == 3

    xs = [torch.randn(4, 16) for _ in gates]
    with RouterReplay.record():
        recorded = [run(g, x)[1] for g, x in zip(gates, xs)]
    captured = RouterReplay.collect()
    assert len(captured) == 3
    for cap, rec in zip(captured, recorded):
        assert torch.equal(cap, rec)

    # Replay each layer with its own captured selection on fresh inputs.
    new_xs = [torch.randn(4, 16) for _ in gates]
    with RouterReplay.replay(captured):
        replayed = [run(g, x)[1] for g, x in zip(gates, new_xs)]
    for rep, rec in zip(replayed, recorded):
        assert torch.equal(rep, rec)


# --------------------------------------------------------------------------- #
# Error handling + context-manager cleanup
# --------------------------------------------------------------------------- #


def test_collect_before_record_raises():
    make_gate()
    with pytest.raises(RuntimeError, match="no recorded selection"):
        RouterReplay.collect()


def test_set_replay_indices_length_mismatch_raises():
    make_gate()
    make_gate()
    with pytest.raises(ValueError, match="replay tensors"):
        RouterReplay.set_replay_indices([torch.zeros(4, 2, dtype=torch.long)])


def test_replay_without_target_raises():
    gate = make_gate()
    RouterReplay.set_mode(RouterReplayMode.REPLAY)
    try:
        with pytest.raises(RuntimeError, match="no target indices"):
            run(gate, torch.randn(4, 16))
    finally:
        RouterReplay.set_mode(None)


def test_replay_shape_mismatch_raises():
    gate = make_gate()
    with RouterReplay.record():
        run(gate, torch.randn(6, 16))
    captured = RouterReplay.collect()
    # Replay onto a different token count -> gather/shape guard fires.
    with pytest.raises(ValueError, match="does not match"):
        with RouterReplay.replay(captured):
            run(gate, torch.randn(4, 16))


def test_context_managers_clear_state_on_exit():
    gate = make_gate()
    with RouterReplay.record():
        run(gate, torch.randn(4, 16))
    captured = RouterReplay.collect()

    with RouterReplay.replay(captured):
        assert gate.router_replay.mode == RouterReplayMode.REPLAY
        assert gate.router_replay.target_indices is not None
    # Mode reset and target cleared so a later forward routes normally.
    assert gate.router_replay.mode is None
    assert gate.router_replay.target_indices is None

    with RouterReplay.record():
        assert gate.router_replay.mode == RouterReplayMode.RECORD
    assert gate.router_replay.mode is None


def test_clear_registry_unregisters():
    make_gate()
    make_gate()
    assert len(RouterReplay.instances()) == 2
    RouterReplay.clear_registry()
    assert RouterReplay.instances() == []


def test_clear_indices_drops_state_but_keeps_registration():
    gate = make_gate()
    with RouterReplay.record():
        run(gate, torch.randn(4, 16))
    gate.router_replay.set_target(torch.zeros(4, 2, dtype=torch.long))
    assert gate.router_replay.recorded_indices is not None
    assert gate.router_replay.target_indices is not None

    RouterReplay.clear_indices()
    assert gate.router_replay.recorded_indices is None
    assert gate.router_replay.target_indices is None
    # Instance is still registered after clearing its indices.
    assert RouterReplay.instances() == [gate.router_replay]


# --------------------------------------------------------------------------- #
# Gemma4Gate (custom router) honours routing replay too
# --------------------------------------------------------------------------- #


def _gemma4_text_config():
    from types import SimpleNamespace

    return SimpleNamespace(
        hidden_size=16,
        num_experts=8,
        top_k_experts=2,
        rms_norm_eps=1e-6,
        torch_dtype=torch.float32,
    )


def _make_gemma4_gate(enable_routing_replay=True):
    from nemo_automodel.components.models.gemma4_moe.model import Gemma4Gate

    gate = Gemma4Gate(_gemma4_text_config(), enable_routing_replay=enable_routing_replay)
    with torch.no_grad():
        gate.proj.weight.normal_(0.0, 1.0)
    return gate


def test_gemma4_gate_disabled_has_no_handle():
    gate = _make_gemma4_gate(enable_routing_replay=False)
    assert gate.router_replay is None
    assert RouterReplay.instances() == []


def test_gemma4_gate_record_replay_roundtrip():
    torch.manual_seed(4)
    gate = _make_gemma4_gate()
    assert RouterReplay.instances() == [gate.router_replay]

    x_rollout = torch.randn(6, 16)
    x_train = torch.randn(6, 16)

    _, i_rollout, _ = gate(x_rollout)
    _, i_train_free, _ = gate(x_train)
    assert not torch.equal(i_rollout, i_train_free)

    with RouterReplay.record():
        gate(x_rollout)
    captured = RouterReplay.collect()

    with RouterReplay.replay(captured):
        w_replay, i_replay, _ = gate(x_train)

    assert torch.equal(i_replay, i_rollout)
    # Weights are renormalized probabilities gathered from the live train logits.
    assert torch.allclose(w_replay.sum(dim=-1), torch.ones(6), atol=1e-5)


def test_gemma4_gate_replay_keeps_gradient():
    torch.manual_seed(5)
    gate = _make_gemma4_gate()
    with RouterReplay.record():
        gate(torch.randn(4, 16))
    captured = RouterReplay.collect()
    with RouterReplay.replay(captured):
        w_replay, _, _ = gate(torch.randn(4, 16))
    w_replay.sum().backward()
    assert gate.proj.weight.grad is not None
    assert torch.any(gate.proj.weight.grad != 0)
