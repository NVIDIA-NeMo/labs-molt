# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical tests for the group advantage estimators (grpo / dr_grpo /
reinforce_baseline / rloo).

Each estimator turns per-rollout scalar rewards into per-token advantages using a
prompt-group baseline. We assert on the un-whitened ``returns`` (the second output),
which for these estimators is the raw group math broadcast onto the action tokens
(``scalar_advantage * action_mask``). Masks are fully active (one action token per
rollout) so ``* mask`` is the identity and the numbers are exact. Expected values are
derived directly from each estimator's definition (not captured from the code).

A multi-group case pins the defining property — each prompt group uses its own
baseline — which a single global baseline would silently violate.
"""

import torch

from molt.trainer.algorithm.advantage import AdvantageContext
from molt.trainer.algorithm.advantage import get_advantage_estimator as get_estimator


def _ctx(n: int) -> "AdvantageContext":
    """One experience of ``n`` single-token, fully-active rollouts (no KL, no values)."""
    return AdvantageContext(
        sample_to_rollout=torch.arange(n),
        exp_len=[n],
        action_masks=[torch.ones(n, 1)],
        kl_coef=0.0,
        gamma=1.0,
        lam=1.0,
        kls=[torch.zeros(n, 1)],
        values=None,
    )


def _run(name: str, rewards, groups):
    adv, ret = get_estimator(name)(torch.tensor(rewards), groups, _ctx(len(rewards)))
    return adv[0].flatten(), ret[0].flatten()


# One prompt group of four rollouts, one success — the canonical asymmetric case.
_G = [[0, 1, 2, 3]]
_R = [1.0, 0.0, 0.0, 0.0]


def test_reinforce_baseline_subtracts_group_mean():
    """returns = r - mean(group); mean = 0.25."""
    _adv, ret = _run("reinforce_baseline", _R, _G)
    assert torch.allclose(ret, torch.tensor([0.75, -0.25, -0.25, -0.25]))


def test_dr_grpo_subtracts_group_mean():
    """dr_grpo = group-mean baseline, same returns as reinforce_baseline."""
    _adv, ret = _run("dr_grpo", _R, _G)
    assert torch.allclose(ret, torch.tensor([0.75, -0.25, -0.25, -0.25]))


def test_grpo_normalizes_by_group_std():
    """returns = (r - mean) / (std + eps). torch .std() is sample std (n-1): for
    [1,0,0,0] mean=0.25, sum sq dev=0.75, var=0.75/3=0.25, std=0.5 -> 0.75/0.5 = 1.5."""
    _adv, ret = _run("grpo", _R, _G)
    assert torch.allclose(ret, torch.tensor([1.5, -0.5, -0.5, -0.5]), atol=1e-5)


def test_rloo_leaves_one_out():
    """Each baseline is the mean of the OTHER rollouts: (sum - r) / (n - 1). The lone
    success keeps a full advantage of 1.0 (its baseline excludes itself), unlike the
    0.75 a plain group-mean baseline gives it."""
    _adv, ret = _run("rloo", _R, _G)
    assert torch.allclose(ret, torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]), atol=1e-6)


def test_groups_use_independent_baselines():
    """Two groups with different reward scales must each be centered on their OWN mean
    — a single global baseline would give different numbers. Group [1,0] -> mean 0.5;
    group [5,3] -> mean 4. rloo (n=2): each baseline is the other member."""
    rewards, groups = [1.0, 0.0, 5.0, 3.0], [[0, 1], [2, 3]]
    _adv, ret = _run("dr_grpo", rewards, groups)
    assert torch.allclose(ret, torch.tensor([0.5, -0.5, 1.0, -1.0]))
    _adv_r, ret_r = _run("rloo", rewards, groups)
    assert torch.allclose(ret_r, torch.tensor([1.0, -1.0, 2.0, -2.0]))


def test_only_reinforce_baseline_whitens():
    """reinforce_baseline whitens across the batch (advantages != returns; whitened
    advantages have ~zero mean, unit population std). grpo / dr_grpo / rloo skip
    whitening by contract, so their advantages equal their returns."""
    adv, ret = _run("reinforce_baseline", _R, _G)
    assert not torch.allclose(adv, ret)
    assert adv.mean().abs() < 1e-5
    assert abs(adv.std(unbiased=False).item() - 1.0) < 1e-5
    for name in ("grpo", "dr_grpo", "rloo"):
        adv, ret = _run(name, _R, _G)
        assert torch.allclose(adv, ret), f"{name} should not whiten"


def test_grpo_constant_reward_group_is_zero():
    """All rewards equal -> numerator is 0 everywhere -> advantage 0 (no learning
    signal from a group the model scored uniformly). Documents current behavior."""
    _adv, ret = _run("grpo", [0.5, 0.5, 0.5, 0.5], _G)
    assert torch.allclose(ret, torch.zeros(4))


def test_singleton_group_behavior_differs_between_grpo_and_rloo():
    """A group of one has no within-group baseline, and the estimators diverge: grpo
    returns 0 (std=0 -> (r-mean)/1e-9 = 0), while rloo keeps the raw reward. Documents
    current behavior."""
    _adv_g, ret_g = _run("grpo", [2.0], [[0]])
    _adv_r, ret_r = _run("rloo", [2.0], [[0]])
    assert torch.allclose(ret_g, torch.zeros(1))
    assert torch.allclose(ret_r, torch.tensor([2.0]))
