# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable advantage estimators.

An estimator turns per-rollout rewards (already clipped, deduped, and grouped by
prompt) into per-token advantages and returns, one tensor per experience:

    estimator(rewards, groups, ctx) -> (advantages, returns)
        rewards     (R,) tensor          one clipped scalar reward per rollout
        groups      list[list[int]]      rollout rows sharing a prompt (the GRPO group)
        ctx         AdvantageContext     sample/mask/kl/gamma tensors + flags
        advantages  list[(B, L) tensor]  per experience, ready for the policy loss
        returns     list[(B, L) tensor]  per experience (advantages before whitening)

Estimators call the helpers `broadcast_advantages` (scalar advantage -> token-level
`advantage * action_mask`) and `normalize_advantages` (cross-batch whitening), both
operating on plain tensors — estimators never see `Experience`. The outcome-reward
estimators broadcast flat; `reinforce` is REINFORCE++ (per-token KL reward +
discounted cumulative returns). The trainer clips rewards, builds the context, and
assembles the results back onto the experiences.

Register a custom estimator without editing this file::

    @register_advantage_estimator("my_estimator")
    def my_estimator(rewards, groups, ctx):
        adv = rewards.clone()
        for group in groups:
            adv[group] = rewards[group] - rewards[group].median()
        returns = broadcast_advantages(adv, ctx)
        return normalize_advantages(returns, ctx), returns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch


@dataclass
class AdvantageContext:
    """Small tensor/scalar inputs an estimator needs (never the `Experience` objects)."""

    sample_to_rollout: torch.Tensor  # (S,) maps each concat-order sample to its rollout row
    exp_len: List[int]  # samples per experience (to re-split per-sample tensors)
    action_masks: List[torch.Tensor]  # per experience (B, L)
    no_std_norm: bool  # skip the std term when whitening
    kl_coef: float  # per-token KL reward coefficient (REINFORCE++ / GAE)
    gamma: float  # discount factor (REINFORCE++ / GAE)
    lam: float  # GAE lambda (PPO); 1.0 = Monte-Carlo return minus the value baseline
    kls: List[torch.Tensor]  # per experience (B, L) per-token KL
    values: List[torch.Tensor] | None = None  # per experience (B, L) critic V(s) at collection (PPO/gae)


Estimator = Callable[
    [torch.Tensor, List[List[int]], "AdvantageContext"],
    Tuple[List[torch.Tensor], List[torch.Tensor]],
]

ADVANTAGE_ESTIMATORS: Dict[str, Estimator] = {}


def register_advantage_estimator(name: str) -> Callable[[Estimator], Estimator]:
    """Register an estimator under `name`."""

    def decorator(fn: Estimator) -> Estimator:
        if name in ADVANTAGE_ESTIMATORS and ADVANTAGE_ESTIMATORS[name] is not fn:
            raise ValueError(f"Advantage estimator '{name}' is already registered")
        ADVANTAGE_ESTIMATORS[name] = fn
        return fn

    return decorator


def get_advantage_estimator(name: str) -> Estimator:
    if name not in ADVANTAGE_ESTIMATORS:
        raise ValueError(f"Unknown advantage estimator '{name}'. Registered: {sorted(ADVANTAGE_ESTIMATORS)}")
    return ADVANTAGE_ESTIMATORS[name]


# ──────────────── tensor helpers (no Experience; the trainer assembles those) ────────────────


def broadcast_advantages(rollout_advantages: torch.Tensor, ctx: AdvantageContext) -> List[torch.Tensor]:
    """Broadcast each rollout's scalar advantage onto its response tokens (`advantage * action_mask`)."""
    sample_advantages = rollout_advantages[ctx.sample_to_rollout].split(ctx.exp_len)
    return [adv.unsqueeze(-1) * mask for adv, mask in zip(sample_advantages, ctx.action_masks)]


def normalize_advantages(advantages: List[torch.Tensor], ctx: AdvantageContext) -> List[torch.Tensor]:
    """Whiten per-experience (B, L) advantages across the batch (action-mask-weighted statistics)."""
    flat_adv = torch.cat([a.flatten() for a in advantages], dim=0).float()
    flat_mask = torch.cat([m.flatten() for m in ctx.action_masks], dim=0)
    num_actions = flat_mask.sum()
    if num_actions == 0:
        # No action tokens anywhere in the batch -> nothing to whiten. Return zeros
        # rather than a 0/0 NaN mean (mirrors agg_loss's denom==0 guard).
        return [torch.zeros_like(a) for a in advantages]

    mean = (flat_adv * flat_mask).sum() / num_actions
    if not ctx.no_std_norm:
        rstd = (((flat_adv - mean).pow(2) * flat_mask).sum() / num_actions).clamp(min=1e-8).rsqrt()
    else:
        rstd = 1

    return [(a - mean) * rstd for a in advantages]


# Estimators that normalize rewards within a prompt group; only these need the
# rollout-reward merge (one reward per rollout, grouped by prompt). reinforce / gae /
# on_policy_distill score each sample independently, so in multi-turn rollouts that
# split one trajectory into several samples they must NOT merge (see compute_advantages).
GROUP_ADVANTAGE_ESTIMATORS = frozenset({"grpo", "dr_grpo", "reinforce_baseline", "rloo"})


# ──────────────────────────────── estimators ────────────────────────────────


@register_advantage_estimator("reinforce")
def reinforce(
    rewards: torch.Tensor, groups: List[List[int]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """REINFORCE++ (https://arxiv.org/abs/2501.03262): no group baseline.

    Each rollout's clipped scalar reward is placed on its last response token, a per-token KL
    penalty (-kl_coef * kl) is added, and discounted cumulative returns are accumulated. The
    resulting advantages are whitened across the batch.
    """
    sample_rewards = rewards[ctx.sample_to_rollout].split(ctx.exp_len)
    returns = []
    for reward, mask, kl in zip(sample_rewards, ctx.action_masks, ctx.kls):
        token_reward = (-ctx.kl_coef * kl).float()  # (B, L) per-token KL penalty
        has_action = mask.bool().any(dim=1)
        last = mask.size(1) - 1 - mask.long().fliplr().argmax(dim=1, keepdim=True)  # last action token
        if has_action.any():
            token_reward[has_action] = token_reward[has_action].scatter_add(
                1, last[has_action], reward[has_action, None].to(token_reward.dtype)
            )
        token_reward = token_reward * mask

        # discounted reverse-cumulative returns: G_t = r_t + gamma * G_{t+1}
        seq_returns = torch.zeros_like(token_reward)
        running = torch.zeros(token_reward.size(0), device=token_reward.device)
        for t in reversed(range(token_reward.size(1))):
            running = token_reward[:, t] + ctx.gamma * running
            seq_returns[:, t] = running
        returns.append(seq_returns)

    return normalize_advantages(returns, ctx), returns


@register_advantage_estimator("on_policy_distill")
def on_policy_distill(
    rewards: torch.Tensor, groups: List[List[int]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """On-policy distillation (slime / Thinking Machines): the dense per-token training
    signal is the *reverse* KL to a frozen teacher, which here is just the reference model.

    With ``--algo.kl.estimator k1`` each ``ctx.kls`` entry is the per-token ``log pi_student -
    log pi_teacher``, so the advantage is ``-kl_coef * kl = kl_coef * (log pi_teacher -
    log pi_student)`` — the negative per-token reverse KL. No scalar task reward, no group
    baseline, no whitening, discount 0 (immediate per-token term only). The policy loss then
    gives the standard policy-gradient estimator of the reverse-KL gradient, so the student is
    pulled onto the teacher on its *own* on-policy samples. Point the teacher at a (bigger)
    checkpoint with ``--ref.model_name_or_path`` and set the strength with ``--algo.kl.init_coef``;
    keep ``--algo.kl.use_loss`` off so the KL flows through the advantage, not a separate loss term.
    """
    returns = [(-ctx.kl_coef * kl) * mask for kl, mask in zip(ctx.kls, ctx.action_masks)]
    return [ret.clone() for ret in returns], returns


@register_advantage_estimator("reinforce_baseline")
def reinforce_baseline(
    rewards: torch.Tensor, groups: List[List[int]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Group-mean baseline; advantages whitened across the batch."""
    advantages = rewards.clone()
    for group in groups:
        advantages[group] = rewards[group] - rewards[group].mean()
    returns = broadcast_advantages(advantages, ctx)
    return normalize_advantages(returns, ctx), returns


@register_advantage_estimator("dr_grpo")
def dr_grpo(
    rewards: torch.Tensor, groups: List[List[int]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Dr.GRPO (https://arxiv.org/abs/2503.20783): group-mean baseline, no std, no whitening."""
    advantages = rewards.clone()
    for group in groups:
        advantages[group] = rewards[group] - rewards[group].mean()
    returns = broadcast_advantages(advantages, ctx)
    return [ret.clone() for ret in returns], returns


@register_advantage_estimator("grpo")
def grpo(
    rewards: torch.Tensor, groups: List[List[int]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """GRPO group-normalized advantage: (r - mean) / (std + eps)."""
    advantages = rewards.clone()
    for group in groups:
        group_rewards = rewards[group]
        std = group_rewards.std() if len(group) > 1 else 0.0
        advantages[group] = (group_rewards - group_rewards.mean()) / (std + 1e-9)
    returns = broadcast_advantages(advantages, ctx)
    return [ret.clone() for ret in returns], returns


@register_advantage_estimator("rloo")
def rloo(
    rewards: torch.Tensor, groups: List[List[int]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Leave-one-out baseline (RLOO, https://arxiv.org/abs/2402.14740).

    Each rollout's baseline is the mean of the *other* rollouts in its group: `(sum - r) / (n - 1)`.
    """
    advantages = rewards.clone()
    for group in groups:
        group_rewards = rewards[group]
        if len(group) > 1:
            advantages[group] = group_rewards - (group_rewards.sum() - group_rewards) / (len(group) - 1)
        # singleton group: no leave-one-out baseline exists, so the advantage stays the raw
        # reward (REINFORCE without baseline) — intentional, not zeroed.
    returns = broadcast_advantages(advantages, ctx)
    return [ret.clone() for ret in returns], returns


@register_advantage_estimator("gae")
def gae(
    rewards: torch.Tensor, groups: List[List[int]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """PPO advantage with a learned value baseline (GAE).

    The per-token reward mirrors ``reinforce`` — ``-kl_coef * kl`` on every step,
    plus the clipped outcome reward on the last action token — and the advantages
    and returns follow the standard GAE recursion::

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lam * A_{t+1}        (V(s_{T+1}) = 0)
        ret_t   = A_t + V(s_t)                            (value-regression target)

    ``ctx.values`` holds the critic's collection-time V(s), one (B, L) tensor per
    experience. Masked positions (right-padding and multi-turn observation/tool
    tokens) are made *transparent* to the recursion: the bootstrap value
    ``V(s_{t+1})`` and the running GAE are carried across them, so the last action
    token before an interior gap bootstraps off the next action token's value rather
    than a spurious terminal ``V = 0``, and a masked token contributes no TD error of
    its own. This masked-GAE treatment is required for correct multi-turn advantages
    whenever ``lam < 1``. At the
    default ``lam = 1`` it reduces to ``A_t = G_t - V(s_t)`` with ``G_t`` the
    discounted return (interior values cancel by telescoping, so carry-vs-terminal is
    invisible there).

    Advantages are then batch-whitened unconditionally (mean/std over action tokens,
    std gated by ``no_std_norm``). Whitening touches only the advantages fed to the policy loss;
    ``returns = A + V(s)`` is left un-whitened so it stays the correct value-regression
    target for the value loss.
    """
    if ctx.values is None:
        raise ValueError("gae requires AdvantageContext.values (advantage_estimator=gae needs a critic)")
    sample_rewards = rewards[ctx.sample_to_rollout].split(ctx.exp_len)
    advantages, returns = [], []
    for reward, mask, kl, values in zip(sample_rewards, ctx.action_masks, ctx.kls, ctx.values):
        # per-token reward: KL penalty + clipped scalar reward on the last action token
        token_reward = (-ctx.kl_coef * kl).float()  # (B, L)
        has_action = mask.bool().any(dim=1)
        last = mask.size(1) - 1 - mask.long().fliplr().argmax(dim=1, keepdim=True)  # last action token
        if has_action.any():
            token_reward[has_action] = token_reward[has_action].scatter_add(
                1, last[has_action], reward[has_action, None].to(token_reward.dtype)
            )
        token_reward = token_reward * mask
        # Zero V(s) off the action span so the value-regression target below
        # (returns = A + V) carries no critic output on masked positions; the
        # recursion itself never reads these (they are carried over, see below).
        values = values * mask

        # GAE reverse recursion: A_t = (r_t + gamma * V_{t+1} - V_t) + gamma * lam * A_{t+1}.
        # Masked (padding / multi-turn observation) tokens are transparent: carry the
        # bootstrap value and the running GAE across them so an action token before a
        # gap bootstraps from the next action token's value, not a spurious terminal
        # V = 0 (matters at lam < 1, a no-op at lam = 1).
        adv = torch.zeros_like(token_reward)
        running = torch.zeros(token_reward.size(0), device=token_reward.device)
        next_value = torch.zeros_like(running)  # V(s_{T+1}) = 0
        m = mask.float()
        for t in reversed(range(token_reward.size(1))):
            delta = token_reward[:, t] + ctx.gamma * next_value - values[:, t]
            running_t = delta + ctx.gamma * ctx.lam * running
            next_value = values[:, t] * m[:, t] + (1 - m[:, t]) * next_value
            running = running_t * m[:, t] + (1 - m[:, t]) * running
            adv[:, t] = running
        advantages.append(adv * mask)
        returns.append((adv + values) * mask)  # value-regression target: returns = A + V(s)
    # Advantages must be whitened (mean/std over action tokens, std gated by
    # no_std_norm), re-masked so off-action positions stay 0; returns are left
    # un-whitened (they remain the value-regression target A + V(s)).
    advantages = [a * m for a, m in zip(normalize_advantages(advantages, ctx), ctx.action_masks)]
    return advantages, returns
