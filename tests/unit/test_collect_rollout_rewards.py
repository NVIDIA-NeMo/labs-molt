# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout reward metrics must not weight a trajectory by how many turns it took."""
import types

import pytest

torch = pytest.importorskip("torch")

from molt.trainer.rl_trainer import _collect_rollout_rewards
from molt.trainer.rollout.experience_maker import rollout_and_group_ids


def _sample(rollout_ids, group_ids, rewards):
    return types.SimpleNamespace(
        info={"reward": torch.tensor(rewards, dtype=torch.float)},
        rollout_ids=rollout_ids,
        group_ids=group_ids,
        index=list(range(len(rewards))),
    )


def test_long_failures_do_not_outweigh_short_successes():
    # Two groups, each a 2-turn success and a 6-turn failure. Flattened: 4/16 = 0.25.
    # Honest pass rate: 2/4 = 0.5. That 2x is the bug, at the magnitude seen on OSWorld.
    samples = [
        _sample(["A"] * 2 + ["B"] * 6, ["g0"] * 8, [1.0] * 2 + [0.0] * 6),
        _sample(["C"] * 2 + ["D"] * 6, ["g1"] * 8, [1.0] * 2 + [0.0] * 6),
    ]
    assert torch.cat([s.info["reward"] for s in samples]).mean().item() == pytest.approx(0.25)

    per_rollout, per_group = _collect_rollout_rewards(samples)
    assert per_rollout == [1.0, 0.0, 1.0, 0.0]
    assert per_group == [0.5, 0.5]


def test_a_rollout_split_across_samples_is_counted_once():
    samples = [_sample(["A"], ["g0"], [1.0]), _sample(["A", "B"], ["g0", "g0"], [1.0, 0.0])]
    per_rollout, per_group = _collect_rollout_rewards(samples)
    assert per_rollout == [1.0, 0.0]
    assert per_group == [0.5]


def test_single_turn_legacy_path_is_an_identity():
    # No ids stamped: every row is its own rollout and its own group.
    sample = types.SimpleNamespace(
        info={"reward": torch.tensor([1.0, 0.0, 1.0, 0.0])},
        rollout_ids=None,
        group_ids=None,
        index=[0, 1, 2, 3],
    )
    per_rollout, per_group = _collect_rollout_rewards([sample])
    assert per_rollout == [1.0, 0.0, 1.0, 0.0]
    assert per_group == [1.0, 0.0, 1.0, 0.0]


def test_samples_without_a_reward_are_skipped():
    ok = _sample(["A"], ["g0"], [1.0])
    missing = types.SimpleNamespace(info={}, rollout_ids=["B"], group_ids=["g1"], index=[0])
    assert _collect_rollout_rewards([ok, missing]) == ([1.0], [1.0])


def test_empty_input():
    assert _collect_rollout_rewards([]) == ([], [])


def test_id_length_mismatch_is_not_silently_truncated():
    bad = _sample(["A", "B"], ["g0", "g0"], [1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        _collect_rollout_rewards([bad])


def test_shared_id_fallbacks():
    stamped = _sample(["A"], ["g0"], [1.0])
    assert rollout_and_group_ids(stamped) == (["A"], ["g0"])

    no_group = types.SimpleNamespace(rollout_ids=["A"], group_ids=None, index=[0])
    assert rollout_and_group_ids(no_group) == (["A"], ["A"])

    bare = types.SimpleNamespace(rollout_ids=None, group_ids=None, index=[7])
    assert rollout_and_group_ids(bare) == ([7], [7])
