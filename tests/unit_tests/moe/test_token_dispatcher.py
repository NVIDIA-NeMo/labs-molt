# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    _DeepepManager,
    _HybridEPManager,
    _HybridEPMetadataProcessor,
)


@pytest.fixture
def hybrid_ep_manager():
    """Create a _HybridEPManager with mocked hybrid_ep_dispatch import."""
    with patch(
        "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
        new=lambda *a, **kw: None,
    ):
        manager = _HybridEPManager(
            group=None,
            num_local_experts=2,
            num_experts=8,
            router_topk=2,
        )
    return manager


class TestIndicesToMultihot:
    """Tests for _HybridEPManager._indices_to_multihot."""

    def test_basic(self, hybrid_ep_manager):
        """Basic topk=2 case with valid indices."""
        indices = torch.tensor([[0, 3], [1, 5]])
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.3]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.shape == (2, 8)
        assert routing_map[0, 0] and routing_map[0, 3]
        assert routing_map[1, 1] and routing_map[1, 5]
        assert routing_map.sum() == 4

        assert multihot_probs[0, 0] == pytest.approx(0.6)
        assert multihot_probs[0, 3] == pytest.approx(0.4)
        assert multihot_probs[1, 1] == pytest.approx(0.7)
        assert multihot_probs[1, 5] == pytest.approx(0.3)

    def test_scoped_processor_matches_existing_conversion(self, hybrid_ep_manager):
        indices = torch.tensor([[0, 3], [1, -1]])
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.0]])
        processor = _HybridEPMetadataProcessor(num_experts=8, permute_fusion=False)

        expected = hybrid_ep_manager._indices_to_multihot(indices, probs)
        actual = processor(indices, probs)

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_topk_1(self, hybrid_ep_manager):
        """Each token routed to exactly one expert."""
        indices = torch.tensor([[2], [7]])
        probs = torch.tensor([[1.0], [1.0]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.sum() == 2
        assert routing_map[0, 2] and routing_map[1, 7]

    def test_all_minus_one(self, hybrid_ep_manager):
        """All indices are -1 (no valid routing)."""
        indices = torch.tensor([[-1, -1], [-1, -1]])
        probs = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.sum() == 0
        assert multihot_probs.sum() == 0

    def test_partial_minus_one(self, hybrid_ep_manager):
        """Some indices are -1 (partial routing)."""
        indices = torch.tensor([[3, -1], [-1, 6]])
        probs = torch.tensor([[0.8, 0.0], [0.0, 0.5]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.sum() == 2
        assert routing_map[0, 3] and routing_map[1, 6]
        assert multihot_probs[0, 3] == pytest.approx(0.8)
        assert multihot_probs[1, 6] == pytest.approx(0.5)

    def test_single_token(self, hybrid_ep_manager):
        """Single token with multiple expert assignments."""
        indices = torch.tensor([[0, 7]])
        probs = torch.tensor([[0.5, 0.5]])

        routing_map, multihot_probs = hybrid_ep_manager._indices_to_multihot(indices, probs)

        assert routing_map.shape == (1, 8)
        assert routing_map.sum() == 2
        assert routing_map[0, 0] and routing_map[0, 7]


@pytest.mark.parametrize("enabled", [False, True])
def test_token_unpermutation_applies_async_setting_to_deepep_combine(enabled):
    dispatcher = object.__new__(MoEFlexTokenDispatcher)
    manager = object.__new__(_DeepepManager)
    manager.get_restored_hidden_states_by_experts = Mock(side_effect=lambda tensor: tensor)
    manager.combine = Mock(side_effect=lambda tensor, async_finish, allocate_on_comm_stream: tensor)
    dispatcher._comm_manager = manager
    dispatcher.config = SimpleNamespace(moe_deepep_async_dispatch=enabled)
    dispatcher.hidden_shape = (2, 4)
    hidden_states = torch.randn(2, 4)

    actual = dispatcher.token_unpermutation(hidden_states)

    torch.testing.assert_close(actual, hidden_states)
    manager.combine.assert_called_once_with(hidden_states, enabled, enabled)


class TestHybridEPTokenCountEqualization:
    """dispatch() must pad unequal per-rank token counts up to the EP-group max."""

    def _run(self, hybrid_ep_manager, monkeypatch, num_tokens, group_max):
        import nemo_automodel.components.moe.megatron.token_dispatcher as td

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

        def fake_all_reduce(tensor, op=None, group=None):
            tensor.fill_(group_max)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        dispatched = {}

        def fake_dispatch(x, routing_map, probs, **kwargs):
            dispatched.update(x=x, routing_map=routing_map, probs=probs)
            return x, probs, None, routing_map.sum(dim=0), "handle"

        monkeypatch.setattr(td, "hybrid_ep_dispatch", fake_dispatch)
        monkeypatch.setattr(td, "hybrid_ep_combine", lambda x, **kwargs: dispatched["x"])

        hidden = torch.randn(num_tokens, 4)
        hybrid_ep_manager.routing_map = torch.ones(num_tokens, 8, dtype=torch.bool)
        hybrid_ep_manager.token_probs = torch.full((num_tokens, 8), 0.125)
        out = hybrid_ep_manager.dispatch(hidden)
        combined = hybrid_ep_manager.combine(out)
        return hidden, dispatched, combined

    def test_shorter_rank_pads_to_aligned_group_max_and_slices_back(self, hybrid_ep_manager, monkeypatch):
        # group max 5 rounds up to the 4-token kernel alignment -> 8.
        hidden, dispatched, combined = self._run(hybrid_ep_manager, monkeypatch, num_tokens=3, group_max=5)

        assert dispatched["x"].shape[0] == 8
        assert dispatched["routing_map"].shape[0] == 8
        assert dispatched["probs"].shape[0] == 8
        # Padded rows carry zero hidden state and route to no expert.
        assert torch.equal(dispatched["x"][3:], torch.zeros(5, 4))
        assert not dispatched["routing_map"][3:].any()
        assert not dispatched["probs"][3:].any()
        # combine() returns only this rank's real tokens.
        assert combined.shape[0] == 3
        assert torch.equal(combined, hidden)
        assert hybrid_ep_manager.num_unpadded_tokens is None

    def test_equal_unaligned_counts_pad_to_alignment(self, hybrid_ep_manager, monkeypatch):
        hidden, dispatched, combined = self._run(hybrid_ep_manager, monkeypatch, num_tokens=6, group_max=6)

        assert dispatched["x"].shape[0] == 8
        assert combined.shape[0] == 6
        assert torch.equal(combined, hidden)

    def test_equal_aligned_counts_do_not_pad(self, hybrid_ep_manager, monkeypatch):
        hidden, dispatched, combined = self._run(hybrid_ep_manager, monkeypatch, num_tokens=4, group_max=4)

        assert dispatched["x"].shape[0] == 4
        assert combined.shape[0] == 4
        assert torch.equal(combined, hidden)


class TestHybridEPStaticRoutingPadPin:
    """Under benchmark_static_routing the padded EP-group token count is computed once (one MAX all-reduce +
    host sync) and reused by every later dispatch; NEMO_STATIC_ROUTING_PAD_PIN=0 keeps the per-dispatch path."""

    def _manager(self, static_routing):
        with patch(
            "nemo_automodel.components.moe.megatron.token_dispatcher.hybrid_ep_dispatch",
            new=lambda *a, **kw: None,
        ):
            return _HybridEPManager(
                group=None, num_local_experts=2, num_experts=8, router_topk=2, benchmark_static_routing=static_routing
            )

    def _dispatch_n(self, manager, monkeypatch, num_tokens_seq, group_max, pin):
        import nemo_automodel.components.moe.megatron.token_dispatcher as td

        monkeypatch.setattr(td, "_STATIC_ROUTING_PAD_PIN", pin)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
        calls = []

        def fake_all_reduce(tensor, op=None, group=None):
            calls.append(int(tensor))
            tensor.fill_(group_max)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        sizes = []

        def fake_dispatch(x, routing_map, probs, **kwargs):
            sizes.append(x.shape[0])
            return x, probs, None, routing_map.sum(dim=0), "handle"

        monkeypatch.setattr(td, "hybrid_ep_dispatch", fake_dispatch)
        for n in num_tokens_seq:
            manager.routing_map = torch.ones(n, 8, dtype=torch.bool)
            manager.token_probs = torch.full((n, 8), 0.125)
            manager.dispatch(torch.randn(n, 4))
        return calls, sizes

    def test_static_routing_all_reduces_once_and_reuses_the_padded_size(self, monkeypatch):
        m = self._manager(static_routing=True)
        calls, sizes = self._dispatch_n(m, monkeypatch, [6, 6, 6], group_max=6, pin=True)
        assert calls == [6], "only the first dispatch pays the EP-group max all-reduce"
        assert sizes == [8, 8, 8] and m._static_target_tokens == 8

    def test_pin_env_off_keeps_the_per_dispatch_all_reduce(self, monkeypatch):
        m = self._manager(static_routing=True)
        calls, sizes = self._dispatch_n(m, monkeypatch, [6, 6, 6], group_max=6, pin=False)
        assert calls == [6, 6, 6] and sizes == [8, 8, 8] and m._static_target_tokens is None

    def test_dynamic_routing_never_pins(self, monkeypatch):
        m = self._manager(static_routing=False)
        calls, _ = self._dispatch_n(m, monkeypatch, [6, 6], group_max=6, pin=True)
        assert calls == [6, 6] and m._static_target_tokens is None

    def test_pinned_size_smaller_than_a_later_batch_falls_back_to_the_all_reduce(self, monkeypatch):
        m = self._manager(static_routing=True)
        calls, sizes = self._dispatch_n(m, monkeypatch, [4, 12], group_max=12, pin=True)
        # first call: group max 12 -> pin 12; second call fits -> no all-reduce. Then a larger batch re-derives.
        assert calls == [4] and sizes == [12, 12]
        calls2, sizes2 = self._dispatch_n(m, monkeypatch, [16], group_max=16, pin=True)
        assert calls2 == [16] and sizes2 == [16] and m._static_target_tokens == 16
