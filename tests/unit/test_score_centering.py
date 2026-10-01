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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
"""Score centering (https://arxiv.org/abs/2609.20807) against the paper's equations.

Every test differentiates the surrogate w.r.t. the policy LOGITS and compares with the closed form
the paper derives, so a wrong sign, a wrong stop-gradient or a wrong tail factor cannot cancel out.
"""

import pytest
import torch
import torch.nn.functional as F

from molt.models import PolicyLoss
from molt.models.score_centering import head_mass, importance_weights, score_centering_correction

VOCAB, STEPS = 7, 5
BOUNDS = {"none": (None, None), "trunc": (None, 2.0), "clip": (0.5, 2.0), "mask": (0.5, 2.0)}


def _case(k=None, seed=0):
    """A policy p (trainable logits), a drifted sampler q, tokens sampled from q and the sampler's
    top-k head (the whole vocabulary when k is None)."""
    torch.manual_seed(seed)
    logits = torch.randn(1, STEPS, VOCAB, requires_grad=True)
    log_p = torch.log_softmax(logits, dim=-1)
    q = torch.softmax(torch.randn(1, STEPS, VOCAB) * 1.5, dim=-1)
    sampled = torch.multinomial(q.view(-1, VOCAB), 1).view(1, STEPS)
    ids = torch.arange(VOCAB).expand(1, STEPS, VOCAB) if k is None else q.topk(k, dim=-1).indices
    adv = torch.randn(1, STEPS)
    return logits, log_p, q, sampled, ids, adv


def _surrogate(log_p, q, sampled, ids, adv, mode="none"):
    """`-A f(p_y/q_y) log p_y + A * correction`, the per-token loss PolicyLoss assembles."""
    low, high = BOUNDS[mode]
    log_p_y = log_p.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    log_q_y = q.log().gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    w_y = importance_weights((log_p_y - log_q_y).exp(), mode, low, high).detach()
    correction, _, _ = score_centering_correction(
        log_p.gather(-1, ids), q.log().gather(-1, ids), mode=mode, low=low, high=high
    )
    return -adv * w_y * log_p_y + adv * correction


def _paper_eq14(log_p, q, sampled, ids, adv, mode):
    """Eq. 14 written out term by term: `-A (f(r_y) log p_y - sum_H sg[q_v f(r_v) - alpha p_v] log p_v)`
    with `rho = (1 - sum_H q) / (1 - sum_H p)` and `alpha = rho f(1/rho)` (`f(1) = 1` when H covers
    everything, i.e. the vanilla Eq. 12 weights `q_v - p_v`)."""
    low, high = BOUNDS[mode]
    f = lambda r: importance_weights(r, mode, low, high)  # noqa: E731
    p = log_p.detach().exp()
    p_y = p.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    q_y = q.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    p_h, q_h = p.gather(-1, ids), q.gather(-1, ids)
    rho = ((1 - q_h.sum(-1)).clamp_min(1e-6) / (1 - p_h.sum(-1)).clamp_min(1e-6)).unsqueeze(-1)
    alpha = rho * f(1 / rho)
    head = ((q_h * f(p_h / q_h) - alpha * p_h) * log_p.gather(-1, ids)).sum(-1)
    return -adv * (f(p_y / q_y) * log_p.gather(-1, sampled.unsqueeze(-1)).squeeze(-1) - head)


def test_full_vocabulary_head_gives_the_sampler_centered_score():
    # Paper Eq. 4/5: with the whole vocabulary as head the logit gradient is -A (onehot(y) - q): REINFORCE's
    # (e_y - p) with the trained p replaced by the sampler q, so E_q[grad] = 0 whatever the drift.
    logits, log_p, q, sampled, ids, adv = _case()
    _surrogate(log_p, q, sampled, ids, adv).sum().backward()
    expected = -adv.unsqueeze(-1) * (F.one_hot(sampled, VOCAB).float() - q)
    torch.testing.assert_close(logits.grad, expected, atol=1e-6, rtol=1e-5)


def test_top_k_head_reconstructs_the_tail_as_rescaled_policy():
    # Paper Eq. 9/10: on a top-k head H the centered distribution is q on H and rho * p off H.
    logits, log_p, q, sampled, ids, adv = _case(k=3)
    _surrogate(log_p, q, sampled, ids, adv).sum().backward()
    p, q_h = log_p.detach().exp(), q.gather(-1, ids)
    rho = (1 - q_h.sum(-1, keepdim=True)) / (1 - p.gather(-1, ids).sum(-1, keepdim=True))
    q_hat = (rho * p).scatter(-1, ids, q_h)
    expected = -adv.unsqueeze(-1) * (F.one_hot(sampled, VOCAB).float() - q_hat)
    torch.testing.assert_close(logits.grad, expected, atol=1e-6, rtol=1e-5)


def test_a_matched_sampler_reduces_to_plain_reinforce():
    # q == p on the head: rho == 1, every head weight is 0 and only -A log p_y is left.
    logits, log_p, _, sampled, ids, adv = _case(k=3)
    correction, q_mass, p_mass = score_centering_correction(log_p.gather(-1, ids), log_p.detach().gather(-1, ids))
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=1e-6, rtol=0)
    torch.testing.assert_close(q_mass, p_mass)
    (-adv * log_p.gather(-1, sampled.unsqueeze(-1)).squeeze(-1) + adv * correction).sum().backward()
    expected = -adv.unsqueeze(-1) * (F.one_hot(sampled, VOCAB).float() - log_p.detach().exp())
    torch.testing.assert_close(logits.grad, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["none", "trunc", "clip", "mask"])
def test_composed_with_an_importance_weight_matches_equation_14(mode):
    logits, log_p, q, sampled, ids, adv = _case(k=4, seed=1)
    ours = _surrogate(log_p, q, sampled, ids, adv, mode)
    paper = _paper_eq14(log_p, q, sampled, ids, adv, mode)
    torch.testing.assert_close(ours, paper, atol=1e-6, rtol=1e-5)
    grads = [torch.autograd.grad(t.sum(), logits, retain_graph=True)[0] for t in (ours, paper)]
    torch.testing.assert_close(*grads, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["none", "trunc", "clip", "mask"])
def test_the_expected_update_under_the_sampler_is_zero(mode):
    # The point of the construction: summing the logit gradient over y ~ q (A = 1, full head) is 0 for any f.
    logits, log_p, q, _, ids, _ = _case(seed=2)
    ones = torch.ones(1, STEPS)
    total = torch.zeros_like(logits)
    for y in range(VOCAB):
        sampled = torch.full((1, STEPS), y)
        loss = _surrogate(log_p, q, sampled, ids, ones, mode)
        grad = torch.autograd.grad(loss.sum(), logits, retain_graph=True)[0]
        total += q[..., y].unsqueeze(-1) * grad
    torch.testing.assert_close(total, torch.zeros_like(total), atol=1e-6, rtol=0)


def test_importance_weight_modes_follow_the_is_correction_vocabulary():
    r = torch.tensor([0.1, 0.5, 1.0, 3.0])
    torch.testing.assert_close(importance_weights(r, "none"), torch.ones(4))
    torch.testing.assert_close(importance_weights(r, "trunc", high=2.0), torch.tensor([0.1, 0.5, 1.0, 2.0]))
    torch.testing.assert_close(importance_weights(r, "clip", 0.5, 2.0), torch.tensor([0.5, 0.5, 1.0, 2.0]))
    torch.testing.assert_close(importance_weights(r, "mask", 0.5, 2.0), torch.tensor([0.0, 0.5, 1.0, 0.0]))
    with pytest.raises(ValueError, match="unknown importance weight mode"):
        importance_weights(r, "tis")


def test_head_mass_and_tail_floor():
    log_probs = torch.log(torch.tensor([[[0.5, 0.3], [0.9, 0.1]]]))
    torch.testing.assert_close(head_mass(log_probs), torch.tensor([[0.8, 1.0]]))
    # A head that (numerically) covers everything must not divide by zero: rho = eps / eps = 1.
    correction, *_ = score_centering_correction(log_probs, log_probs)
    assert torch.isfinite(correction).all()


# --- through PolicyLoss -------------------------------------------------------------------------


def _head_inputs(log_p, q, ids):
    return dict(top_log_probs=log_p.gather(-1, ids), rollout_top_log_probs=q.log().gather(-1, ids))


def test_policy_loss_reinforce_with_score_centering_is_the_token_mean_of_the_surrogate():
    logits, log_p, q, sampled, ids, adv = _case(k=3, seed=3)
    log_p_y = log_p.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    mask = torch.ones(1, STEPS, dtype=torch.bool)
    loss_fn = PolicyLoss(loss_mode="reinforce", score_centering=True)
    loss, reported, clip_ratio, *_ = loss_fn(
        log_p_y, log_p_y.detach(), adv, action_mask=mask, **_head_inputs(log_p, q, ids)
    )
    expected = _surrogate(log_p, q, sampled, ids, adv).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(reported, expected.detach())
    assert clip_ratio == 0  # nothing is clipped in REINFORCE
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=1e-6,
        rtol=1e-5,
    )


def test_policy_loss_weights_the_sampled_token_and_the_head_with_the_same_truncated_ratio():
    # TIS variant (paper App. A.2): is_correction token/trunc caps p_y/q_y at `high`, and the head term
    # must use the same cap -- score_centering_correction reads it off the loss's own IS config.
    logits, log_p, q, sampled, ids, adv = _case(k=3, seed=4)
    log_p_y = log_p.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    log_q_y = q.log().gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    mask = torch.ones(1, STEPS, dtype=torch.bool)
    loss_fn = PolicyLoss(
        loss_mode="reinforce",
        score_centering=True,
        is_correction_level="token",
        is_correction_mode="trunc",
        is_correction_threshold=[0.0, 2.0],
    )
    loss, *_ = loss_fn(
        log_p_y, log_p_y.detach(), adv, action_mask=mask, rollout_log_probs=log_q_y, **_head_inputs(log_p, q, ids)
    )
    expected = _surrogate(log_p, q, sampled, ids, adv, "trunc").mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=1e-6,
        rtol=1e-5,
    )


def test_policy_loss_rejects_score_centering_outside_its_assumptions():
    with pytest.raises(ValueError, match="REINFORCE"):
        PolicyLoss(loss_mode="ppo", score_centering=True)
    with pytest.raises(ValueError, match="per-token ratio weight"):
        PolicyLoss(
            loss_mode="reinforce", score_centering=True, is_correction_level="seq", is_correction_threshold=[0.5, 5]
        )
    with pytest.raises(ValueError, match="per-token ratio weight"):
        PolicyLoss(
            loss_mode="reinforce",
            score_centering=True,
            is_correction_level="token",
            is_correction_gating="binary_kl",
            is_correction_threshold=[0, 5e-3],
        )
    loss_fn = PolicyLoss(loss_mode="reinforce", score_centering=True)
    with pytest.raises(ValueError, match="top-k"):
        loss_fn(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2), action_mask=torch.ones(1, 2, dtype=torch.bool))


def test_reinforce_surrogate_is_minus_advantage_times_log_prob():
    logp = torch.tensor([[-0.5, -1.0]], requires_grad=True)
    adv = torch.tensor([[2.0, -1.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss, *_ = PolicyLoss(loss_mode="reinforce")(logp, logp.detach() - 0.3, adv, action_mask=mask)
    torch.testing.assert_close(loss, (-adv * logp.detach()).mean())  # the stale `old` log-prob plays no role
    loss.backward()
    torch.testing.assert_close(logp.grad, -adv / 2)
