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

"""Unit tests for the pipeline-parallel PEFT-adapter gather.

Background
----------
Under ``pp_size > 1`` the local PEFT save path in
:class:`ModelState` only collects adapter tensors from the local PP stages,
so the on-disk adapter is missing every layer owned by another stage (a
grouped-experts MoE could save only ~1/pp of its layers). The fix all-gathers
the per-rank PEFT dicts across the PP group and unions ``target_modules`` the
same way.

These tests fake ``torch.distributed`` so the merge/union logic runs on CPU
with no real process group — the distributed primitive is trivial; the
correctness we care about is the merge, the dedup, the world==1 short-circuit,
and the degenerate-gather warning.
"""

import logging

import pytest
import torch
from torch import nn

from nemo_automodel.components.checkpoint.addons import _extract_target_modules, _extract_target_parameters
from nemo_automodel.components.checkpoint.stateful_wrappers import (
    _gather_peft_state_dict_across_pp,
)
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin


class _FakePPGroup:
    """Marker object standing in for a real ProcessGroup in tests."""

    def __init__(self, ranks_state_dicts):
        # ranks_state_dicts: list of per-rank payloads the fake all_gather returns.
        self.ranks_state_dicts = ranks_state_dicts


class _V5MoEAdapter(MoESplitExpertsStateDictMixin):
    """Minimal adapter that opts into fused PEFT v5 expert parameters."""

    @property
    def _v5_peft_target_parameters(self):
        return ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def _install_fake_distributed(monkeypatch, per_rank_payloads):
    """Patch torch.distributed so collectives operate on ``per_rank_payloads``.

    ``get_world_size`` returns len(per_rank_payloads); ``all_gather_object``
    fills the output list with a copy of ``per_rank_payloads`` (i.e. what every
    rank would receive after the gather).
    """
    world = len(per_rank_payloads)

    def fake_get_world_size(group=None):
        return world

    def fake_all_gather_object(out_list, obj, group=None):
        # Simulate the collective: every rank ends up with all ranks' objects.
        for i, payload in enumerate(per_rank_payloads):
            out_list[i] = payload

    monkeypatch.setattr(torch.distributed, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)


def _t(val):
    """Tiny tensor factory so dicts hold real tensors (CPU)."""
    return torch.tensor([float(val)])


# ---------------------------------------------------------------------------
# _gather_peft_state_dict_across_pp
# ---------------------------------------------------------------------------
class TestGatherPeftStateDictAcrossPP:
    def test_merges_disjoint_layers_from_all_ranks(self, monkeypatch):
        """The canonical bug case: each rank holds a disjoint layer subset;
        the gather must produce the union of all of them."""
        # pp=4, 2 layers/rank, keys disjoint by layer index (mirrors real PP).
        rank0 = {
            "base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0),
            "base_model.model.model.layers.1.self_attn.wq_a.lora_A.weight": _t(1),
        }
        rank1 = {
            "base_model.model.model.layers.2.self_attn.wq_a.lora_A.weight": _t(2),
            "base_model.model.model.layers.3.self_attn.wq_a.lora_A.weight": _t(3),
        }
        rank2 = {
            "base_model.model.model.layers.4.self_attn.wq_a.lora_A.weight": _t(4),
            "base_model.model.model.layers.5.self_attn.wq_a.lora_A.weight": _t(5),
        }
        rank3 = {
            "base_model.model.model.layers.6.self_attn.wq_a.lora_A.weight": _t(6),
            "base_model.model.model.layers.7.self_attn.wq_a.lora_A.weight": _t(7),
        }
        payloads = [rank0, rank1, rank2, rank3]
        _install_fake_distributed(monkeypatch, payloads)

        # Call from rank 0's perspective (local_state_dict == rank0).
        merged = _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert len(merged) == 8
        for layer in range(8):
            key = f"base_model.model.model.layers.{layer}.self_attn.wq_a.lora_A.weight"
            assert key in merged, f"missing layer {layer}"

    def test_world_size_one_returns_local_unchanged(self, monkeypatch):
        """pp=1 (single rank): no gather, return the local dict object as-is."""
        local = {"base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0)}
        _install_fake_distributed(monkeypatch, [local])

        merged = _gather_peft_state_dict_across_pp(local, _FakePPGroup([local]))

        assert merged is local  # short-circuit returns the same object

    def test_duplicate_key_resolves_to_lowest_rank(self, monkeypatch):
        """If the same FQN appears on two ranks (e.g. a replicated param),
        the lowest-rank value must win deterministically."""
        dup_key = "base_model.model.model.embed_tokens.lora_A.weight"
        rank0 = {dup_key: _t(100)}
        rank1 = {dup_key: _t(200)}
        payloads = [rank0, rank1]
        _install_fake_distributed(monkeypatch, payloads)

        merged = _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert len(merged) == 1
        assert merged[dup_key].item() == 100.0  # rank 0 wins

    def test_handles_empty_rank_payload(self, monkeypatch):
        """A rank that trained no PEFT params (empty dict) must not break the
        merge (defensive: e.g. a stage that holds only the embed/head)."""
        rank0 = {"base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0)}
        rank1 = {}  # empty
        rank2 = {"base_model.model.model.layers.4.self_attn.wq_a.lora_A.weight": _t(4)}
        payloads = [rank0, rank1, rank2]
        _install_fake_distributed(monkeypatch, payloads)

        merged = _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert len(merged) == 2
        assert "base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight" in merged
        assert "base_model.model.model.layers.4.self_attn.wq_a.lora_A.weight" in merged

    def test_degenerate_gather_emits_warning(self, monkeypatch, caplog):
        """If the gather adds nothing beyond the largest single rank (e.g. a
        wrong/global group was passed), warn loudly — this is the failure this
        fix exists to catch."""
        # All ranks report the SAME single key → merged size == max per-rank size.
        same = {"base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0)}
        payloads = [same, same]
        _install_fake_distributed(monkeypatch, payloads)

        with caplog.at_level(logging.WARNING):
            merged = _gather_peft_state_dict_across_pp(same, _FakePPGroup(payloads))

        assert len(merged) == 1
        assert any("may be INCOMPLETE" in rec.message for rec in caplog.records)

    def test_single_nonempty_rank_does_not_warn(self, monkeypatch, caplog):
        """A valid PP layout can place all trainable adapters on one stage, with
        every other stage contributing an empty dict (per_rank=[N, 0, ...]).
        Then merged_n == max(per_rank) is correct, not a collapse, so the
        INCOMPLETE warning must NOT fire."""
        rank0 = {
            "base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0),
            "base_model.model.model.layers.1.self_attn.wq_a.lora_A.weight": _t(1),
        }
        payloads = [rank0, {}, {}]  # only rank 0 holds adapters
        _install_fake_distributed(monkeypatch, payloads)

        with caplog.at_level(logging.WARNING):
            merged = _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert len(merged) == 2
        assert not any("may be INCOMPLETE" in rec.message for rec in caplog.records)

    def test_healthy_gather_does_not_warn(self, monkeypatch, caplog):
        rank0 = {"base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0)}
        rank1 = {"base_model.model.model.layers.1.self_attn.wq_a.lora_A.weight": _t(1)}
        payloads = [rank0, rank1]
        _install_fake_distributed(monkeypatch, payloads)

        with caplog.at_level(logging.WARNING):
            _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert not any("may be INCOMPLETE" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _extract_target_modules PP union
# ---------------------------------------------------------------------------
def _make_model_with_named_modules(module_names):
    """Build a dummy model whose ``named_modules`` yields the given names
    (each target gets a ``.lora_A`` leaf so _extract_target_modules picks it up)."""
    root = nn.Module()
    for name in module_names:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Identity())
    return root


class TestExtractTargetModulesPPUnion:
    def test_unions_target_modules_across_pp_ranks(self, monkeypatch):
        """adapter_config.json target_modules must list every stage's layers,
        not just the local stage's (the second half of the truncation bug)."""
        # Local (rank 0) model only has layer 0's modules.
        local_model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.wq_a.lora_A",
                "model.layers.0.self_attn.wkv.lora_A",
            ]
        )
        # Other ranks contribute layers 1 and 2 (what the gather would supply).
        rank1_targets = ["model.layers.1.self_attn.wq_a", "model.layers.1.self_attn.wkv"]
        rank2_targets = ["model.layers.2.self_attn.wq_a", "model.layers.2.self_attn.wkv"]
        local_targets = ["model.layers.0.self_attn.wkv", "model.layers.0.self_attn.wq_a"]
        payloads = [local_targets, rank1_targets, rank2_targets]
        _install_fake_distributed(monkeypatch, payloads)

        result = _extract_target_modules(local_model, pp_group=_FakePPGroup(payloads))

        # Union covers all three layers.
        for layer in (0, 1, 2):
            assert f"model.layers.{layer}.self_attn.wq_a" in result
            assert f"model.layers.{layer}.self_attn.wkv" in result

    def test_no_pp_group_returns_local_only(self):
        """Without a PP group (pp_size==1 or non-PP), behaviour is unchanged:
        only the local model's modules are returned."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.wq_a.lora_A",
            ]
        )
        result = _extract_target_modules(model, pp_group=None)
        assert result == ["model.layers.0.self_attn.wq_a"]


class TestExtractTargetParametersPPUnion:
    def test_remote_expert_lora_enables_v5_metadata_on_every_stage(self, monkeypatch):
        """A PP rank without experts must emit metadata when another rank owns them."""
        local_model = nn.Module()
        local_model.state_dict_adapter = _V5MoEAdapter()
        payloads = [False, True]
        _install_fake_distributed(monkeypatch, payloads)

        result = _extract_target_parameters(local_model, pp_group=_FakePPGroup(payloads))

        assert result == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
