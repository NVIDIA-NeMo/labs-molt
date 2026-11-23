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

"""Numerical tests for the `on_policy_distill` advantage estimator.

On-policy distillation has no reward: the advantage is the per-token reverse KL to
the teacher, ``-kl_coef * (log pi_student - log pi_teacher)``, read from
``ctx.kls``. We assert the sign and scale, that masked positions stay zero, and
that rewards and prompt groups are ignored. The k1 estimator that feeds the KL
must not be clamped -- the most divergent tokens are the signal.
"""

import importlib.util
import pathlib

import torch

from molt.trainer.algorithm.advantage import AdvantageContext
from molt.trainer.algorithm.advantage import get_advantage_estimator as get_estimator

# molt.models.__init__ imports the AutoModel-backed Actor, which the host test env
# does not need for a pure-tensor KL helper; load utils.py by path instead.
_spec = importlib.util.spec_from_file_location(
    "molt_models_utils", pathlib.Path(__file__).resolve().parents[2] / "molt/models/utils.py"
)
models_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(models_utils)


def _ctx(kl_coef: float, kls: torch.Tensor, action_masks: torch.Tensor) -> AdvantageContext:
    n = kls.shape[0]
    return AdvantageContext(
        sample_to_rollout=torch.arange(n),
        exp_len=[n],
        action_masks=[action_masks],
        kl_coef=kl_coef,
        gamma=1.0,
        lam=1.0,
        kls=[kls],
        values=None,
    )


def _run(kl_coef, kls, mask, rewards=None, groups=None):
    adv, ret = get_estimator("on_policy_distill")(
        torch.tensor(rewards if rewards is not None else [0.0] * kls.shape[0]),
        groups if groups is not None else [[0, 1]],
        _ctx(kl_coef, kls, mask),
    )
    return adv[0], ret[0]


def test_advantage_is_negative_kl_coef_times_kl():
    """ret = adv = -kl_coef * kl on action tokens, with no whitening."""
    kls = torch.tensor([[0.1, -0.2], [0.05, 0.0]])
    adv, ret = _run(2.0, kls, torch.ones(2, 2))
    torch.testing.assert_close(ret, torch.tensor([[-0.2, 0.4], [-0.1, 0.0]]))
    torch.testing.assert_close(adv, ret)


def test_masked_positions_stay_zero():
    """Padding / observation tokens carry no distillation signal."""
    kls = torch.ones(2, 2)
    adv, ret = _run(1.0, kls, torch.tensor([[1, 0], [0, 1]]))
    torch.testing.assert_close(ret, torch.tensor([[-1.0, 0.0], [0.0, -1.0]]))
    torch.testing.assert_close(adv, ret)


def test_rewards_and_groups_are_ignored():
    """The signal is the KL alone: task rewards and the prompt grouping do not enter."""
    kls = torch.tensor([[0.3, -0.1], [0.2, 0.0]])
    mask = torch.ones(2, 2)
    baseline = _run(1.0, kls, mask)
    shifted = _run(1.0, kls, mask, rewards=[100.0, -100.0], groups=[[0], [1]])
    torch.testing.assert_close(shifted[0], baseline[0])
    torch.testing.assert_close(shifted[1], baseline[1])


def test_k1_log_ratio_is_not_clipped():
    """k1 keeps the raw signed log-ratio, so the reverse-KL reward is not capped on
    the most divergent tokens; the non-negative k2/k3 estimators stay in +-10."""
    log_probs = torch.tensor([100.0])
    base = torch.tensor([0.0])
    assert models_utils.compute_approx_kl(log_probs, base, kl_estimator="k1").item() == 100.0
    assert models_utils.compute_approx_kl(log_probs, base, kl_estimator="k2").item() == 10.0
    assert models_utils.compute_approx_kl(log_probs, base, kl_estimator="k3").item() == 10.0