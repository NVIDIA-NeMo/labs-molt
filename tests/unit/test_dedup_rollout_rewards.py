# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the per-rollout reward aggregation in rl_trainer."""
import types

import pytest

torch = pytest.importorskip("torch")

from molt.trainer.rl_trainer import _dedup_rollout_rewards


def _sample(rollout_ids, group_ids, rewards):
    s = types.SimpleNamespace()
    s.info = {"reward": torch.tensor(rewards, dtype=torch.float)}
    s.rollout_ids = rollout_ids
    s.group_ids = group_ids
    s.index = list(range(len(rewards)))
    return s


def test_long_failures_do_not_outweigh_short_successes():
    # Two prompt groups, each with one 2-turn success and one 6-turn failure. The flattened mean
    # is 4/16 = 0.25; the honest pass rate is 2/4 = 0.5. This 2x gap is the bug, at the same
    # magnitude seen on OSWorld (reported 0.28 against a measured 0.4487).
    samples = [
        _sample(["A"] * 2 + ["B"] * 6, ["g0"] * 8, [1.0] * 2 + [0.0] * 6),
        _sample(["C"] * 2 + ["D"] * 6, ["g1"] * 8, [1.0] * 2 + [0.0] * 6),
    ]
    flat = torch.cat([s.info["reward"] for s in samples])
    assert flat.mean().item() == pytest.approx(0.25)

    per_rollout, per_group = _dedup_rollout_rewards(samples)
    assert per_rollout.numel() == 4
    assert per_group.numel() == 2
    assert per_rollout.mean().item() == pytest.approx(0.5)
    assert per_group.mean().item() == pytest.approx(0.5)


def test_group_pass_rate_averages_within_group_first():
    # One group solves 1 of 2 rollouts, the other solves 2 of 2. Per-rollout mean is 0.75;
    # per-group mean is (0.5 + 1.0) / 2 = 0.75 here, but the group view is what eval reports,
    # so keep both and let them differ when group sizes differ.
    samples = [
        _sample(["A", "B", "B"], ["g0"] * 3, [1.0, 0.0, 0.0]),
        _sample(["C", "D"], ["g1"] * 2, [1.0, 1.0]),
    ]
    per_rollout, per_group = _dedup_rollout_rewards(samples)
    assert per_rollout.numel() == 4
    assert per_rollout.mean().item() == pytest.approx(0.75)
    assert per_group.numel() == 2
    assert per_group.mean().item() == pytest.approx(0.75)


def test_single_turn_legacy_path_is_identity():
    # No ids: every row is its own rollout, so dedup must not collapse anything.
    s = types.SimpleNamespace()
    s.info = {"reward": torch.tensor([1.0, 0.0, 1.0, 0.0])}
    s.rollout_ids = None
    s.group_ids = None
    s.index = [0, 1, 2, 3]
    per_rollout, _ = _dedup_rollout_rewards([s])
    assert per_rollout.numel() == 4
    assert per_rollout.mean().item() == pytest.approx(0.5)


def test_samples_without_reward_are_skipped():
    ok = _sample(["A"], ["g0"], [1.0])
    missing = types.SimpleNamespace(info={}, rollout_ids=["B"], group_ids=["g1"], index=[0])
    per_rollout, per_group = _dedup_rollout_rewards([ok, missing])
    assert per_rollout.numel() == 1
    assert per_group.numel() == 1


def test_empty_input():
    per_rollout, per_group = _dedup_rollout_rewards([])
    assert per_rollout.numel() == 0
    assert per_group.numel() == 0
