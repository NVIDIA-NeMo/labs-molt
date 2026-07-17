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

import math

import pytest
import torch

from molt.models import PolicyLoss
from molt.models.utils import compute_approx_kl


def test_seq_mask_tis_uses_raw_token_importance_weights_for_kept_sequences():
    loss_fn = PolicyLoss(
        is_correction_threshold=[0.5, 2.0],
        is_correction_level="geo",
        is_correction_mode="mask",
    )
    rollout_log_probs = torch.tensor([[-math.log(3.0), math.log(3.0)]])

    loss, *_ = loss_fn(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        rollout_log_probs=rollout_log_probs,
    )

    torch.testing.assert_close(loss, torch.tensor(-(3.0 + 1.0 / 3.0) / 2.0))


def test_force_on_policy_old_equals_action_gives_reinforce_gradient():
    # force_on_policy skips the old-logprob forward and sets old = action.detach()
    # (policy_actor). The PPO ratio is then exactly 1, so the surrogate's gradient is
    # REINFORCE: proportional to -advantage. Verify the grad has that structure.
    logp = torch.randn(3, 4, requires_grad=True)
    adv = torch.tensor([[1.0, -2.0, 0.5, -1.5]]).repeat(3, 1)
    mask = torch.ones(3, 4, dtype=torch.bool)
    loss_fn = PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.2)  # no IS, no dual-clip

    loss, *_ = loss_fn(logp, logp.detach(), adv, action_mask=mask)
    loss.backward()

    # grad_i = c * (-adv_i) for one shared token-mean weight c > 0 -> ratio is constant.
    ratio = logp.grad / (-adv)
    torch.testing.assert_close(ratio, ratio.flatten()[:1].expand_as(ratio).contiguous())
    assert (ratio > 0).all()


def test_force_on_policy_reinforce_still_applies_is_correction():
    # With old = action.detach(), the IS correction still runs vs the rollout log-probs
    # (behavior policy = vLLM). vllm_kl = mean(rollout - old).
    logp = torch.zeros(1, 2)
    rollout = torch.tensor([[-0.1, -0.1]])  # old(0) - rollout(-0.1) = 0.1
    loss_fn = PolicyLoss(
        is_correction_threshold=[0.0, 100.0],
        is_correction_level="token",
        is_correction_mode="trunc",
    )

    _, _, _, _, vllm_kl, _ = loss_fn(
        logp,
        logp.detach(),
        torch.ones(1, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        rollout_log_probs=rollout,
    )

    torch.testing.assert_close(vllm_kl, torch.tensor(-0.1))


def test_tis_caps_large_importance_weights_without_flooring_small_weights():
    loss_fn = PolicyLoss(
        is_correction_threshold=[0.5, 2.0],
        is_correction_level="token",
        is_correction_mode="trunc",
    )

    loss, *_ = loss_fn(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        rollout_log_probs=torch.tensor([[math.log(10.0), -math.log(10.0)]]),
    )

    torch.testing.assert_close(loss, torch.tensor(-(0.1 + 2.0) / 2.0))


def test_icepop_importance_weights_do_not_nan_on_overflow():
    loss_fn = PolicyLoss(
        is_correction_threshold=[0.5, 2.0],
        is_correction_level="token",
        is_correction_mode="mask",
    )

    loss, *_ = loss_fn(
        torch.zeros(1, 1),
        torch.full((1, 1), 1000.0),
        torch.ones(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        rollout_log_probs=torch.zeros(1, 1),
    )

    assert torch.isfinite(loss)


def test_policy_loss_zero_advantage_handles_extreme_policy_ratio():
    loss_fn = PolicyLoss()

    loss, *_ = loss_fn(
        torch.full((1, 1), 1000.0),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_policy_loss_zero_advantage_handles_equal_negative_infinity_logprobs():
    loss_fn = PolicyLoss()

    loss, _reported, clip_ratio, policy_kl, *_ = loss_fn(
        torch.full((1, 1), -torch.inf),
        torch.full((1, 1), -torch.inf),
        torch.zeros(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert torch.isfinite(clip_ratio)
    assert torch.isfinite(policy_kl)


def test_policy_loss_empty_action_mask_returns_zero():
    loss_fn = PolicyLoss()

    loss, _reported, clip_ratio, policy_kl, *_ = loss_fn(
        torch.full((1, 2), torch.nan),
        torch.full((1, 2), torch.nan),
        torch.ones(1, 2),
        action_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(clip_ratio, torch.tensor(0.0))
    torch.testing.assert_close(policy_kl, torch.tensor(0.0))


def test_tis_caps_overflowed_importance_weights_at_high_threshold():
    loss_fn = PolicyLoss(
        is_correction_threshold=[0.5, 2.0],
        is_correction_level="token",
        is_correction_mode="trunc",
    )

    loss, *_ = loss_fn(
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        rollout_log_probs=torch.full((1, 1), -1000.0),
    )

    torch.testing.assert_close(loss, torch.tensor(-2.0))


def test_seq_mask_tis_drops_overflowed_importance_weight_without_nan():
    loss_fn = PolicyLoss(
        is_correction_threshold=[0.5, 2.0],
        is_correction_level="geo",
        is_correction_mode="mask",
    )

    loss, *_, is_filter_ratio = loss_fn(
        torch.full((1, 1), 1000.0),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        rollout_log_probs=torch.full((1, 1), -1000.0),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(is_filter_ratio, torch.tensor(1.0))


def test_policy_loss_sanitizes_nonfinite_vllm_kl_metric():
    loss_fn = PolicyLoss(
        is_correction_threshold=[0.5, 2.0],
        is_correction_level="token",
        is_correction_mode="trunc",
    )

    loss, *_, vllm_kl, is_filter_ratio = loss_fn(
        torch.zeros(1, 1),
        torch.full((1, 1), -torch.inf),
        torch.zeros(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        rollout_log_probs=torch.full((1, 1), -torch.inf),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert torch.isfinite(vllm_kl)
    assert torch.isfinite(is_filter_ratio)


def test_token_clip_clamps_each_token_weight_without_dropping():
    # token x clip: keep every token, clamp its IS weight into [low, high]
    # (contrast with token x mask, which zeros out-of-band tokens).
    loss_fn = PolicyLoss(
        is_correction_threshold=[0.5, 2.0],
        is_correction_level="token",
        is_correction_mode="clip",
    )
    # token ratios pi_train/pi_rollout = [3, 1/3]  (old=0, rollout=[-log3, +log3])
    loss, *_, is_filter_ratio = loss_fn(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        rollout_log_probs=torch.tensor([[-math.log(3.0), math.log(3.0)]]),
    )
    # base per-token loss -1; weights clamp to [2.0, 0.5] -> mean(-(2.0+0.5)/2)
    torch.testing.assert_close(loss, torch.tensor(-(2.0 + 0.5) / 2.0))
    torch.testing.assert_close(is_filter_ratio, torch.tensor(1.0))  # both clamped, none dropped


def test_level_off_disables_correction_even_with_rollout_logprobs():
    # level="off" (default) folds in the old enable=False: no IS weight is applied
    # even when rollout_log_probs are passed, and the vllm_kl / is_filter metrics
    # stay None. Loss is the plain policy loss (ratio 1 -> -advantage).
    loss_fn = PolicyLoss()  # default is_correction_level="off"
    loss, *_, vllm_kl, is_filter_ratio = loss_fn(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        rollout_log_probs=torch.tensor([[-math.log(3.0), math.log(3.0)]]),
    )
    torch.testing.assert_close(loss, torch.tensor(-1.0))  # no IS scaling applied
    assert vllm_kl is None
    assert is_filter_ratio is None


@pytest.mark.parametrize("level", ["seq", "geo"])
@pytest.mark.parametrize("mode", ["clip", "trunc"])
def test_sequence_geometric_levels_only_allow_mask(level, mode):
    # seq/geo are per-sequence rejection FILTERS (survivors keep their per-token IS
    # weight), so clip/trunc — which would replace that with one clamped sequence
    # weight — are rejected at construction.
    with pytest.raises(ValueError, match="only supports is_correction_mode=mask"):
        PolicyLoss(
            is_correction_threshold=[0.5, 2.0],
            is_correction_level=level,
            is_correction_mode=mode,
        )


def test_cispo_clamps_only_ratios_above_the_ceiling():
    # ratios = exp(log_probs - old_log_probs) = exp([0.5, -0.5]) ~= [1.65, 0.61].
    # Ceiling 1.2: the first token is clamped down to 1.2, the second (below the
    # ceiling, and below 1.0) passes through unclamped -- CISPO has no lower clip.
    loss_fn = PolicyLoss(loss_mode="cispo", clip_eps_high=1.2)
    log_probs = torch.tensor([[0.5, -0.5]])
    old_log_probs = torch.zeros(1, 2)
    advantages = torch.ones(1, 2)

    loss, *_ = loss_fn(
        log_probs,
        old_log_probs,
        advantages,
        action_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    ratio = log_probs.exp()
    expected = -(1.2 * 1.0 * log_probs[0, 0] + ratio[0, 1] * 1.0 * log_probs[0, 1]) / 2.0
    torch.testing.assert_close(loss, expected)


def test_cispo_clip_ratio_metric_reports_fraction_above_ceiling():
    loss_fn = PolicyLoss(loss_mode="cispo", clip_eps_high=1.2)
    # ratios ~= [1.65, 0.61, 1.65]: 2 of 3 tokens exceed the 1.2 ceiling.
    log_probs = torch.tensor([[0.5, -0.5, 0.5]])
    old_log_probs = torch.zeros(1, 3)

    _, _, clip_ratio, *_ = loss_fn(
        log_probs,
        old_log_probs,
        torch.ones(1, 3),
        action_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(clip_ratio, torch.tensor(2.0 / 3.0))


def test_cispo_gradient_flows_only_through_log_probs_not_the_clip_weight():
    # The defining CISPO property: the IS weight is stop-gradient'd (detach()), so the
    # gradient is a plain REINFORCE term through log_probs, not a PPO-style ratio*advantage
    # gradient through the (clipped) ratio itself.
    loss_fn = PolicyLoss(loss_mode="cispo", clip_eps_high=1.2)
    log_probs = torch.zeros(1, 2, requires_grad=True)  # ratio = 1, below the ceiling
    old_log_probs = torch.zeros(1, 2)
    advantages = torch.tensor([[2.0, -3.0]])

    loss, *_ = loss_fn(
        log_probs,
        old_log_probs,
        advantages,
        action_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    loss.backward()

    # d(-1 * A * log_probs)/d(log_probs) = -A, token-mean aggregated.
    torch.testing.assert_close(log_probs.grad, -advantages / 2.0)


def test_compute_approx_kl_sanitizes_equal_negative_infinity_logprobs():
    kl = compute_approx_kl(
        torch.full((1, 1), -torch.inf),
        torch.full((1, 1), -torch.inf),
        kl_estimator="k2",
    )

    torch.testing.assert_close(kl, torch.zeros(1, 1))


def test_policy_loss_registry_rejects_unknown_and_duplicate_names():
    from molt.models.loss import POLICY_LOSSES, get_policy_loss, register_policy_loss

    assert {"ppo", "cispo"} <= set(POLICY_LOSSES)  # built-in surrogates are registered
    with pytest.raises(ValueError, match="Unknown policy loss"):
        get_policy_loss("nope")
    with pytest.raises(ValueError, match="Unknown policy loss"):
        PolicyLoss(loss_mode="nope")  # surfaced at construction
    with pytest.raises(ValueError, match="already registered"):
        register_policy_loss("ppo")(lambda *a, **k: None)  # no silent shadowing
