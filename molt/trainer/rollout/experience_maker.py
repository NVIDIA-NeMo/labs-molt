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
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, List

import ray
import torch

from molt.models.utils import compute_approx_kl, masked_mean
from molt.trainer.algorithm.advantage import (
    GROUP_ADVANTAGE_ESTIMATORS,
    AdvantageContext,
    get_advantage_estimator,
)
from molt.trainer.algorithm.experience import Experience
from molt.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from molt.trainer.workers.actor_group import RayActorGroup

logger = init_logger(__name__)


class RemoteExperienceMaker:
    """Builds train-ready experiences from rollout samples and remote model forwards."""

    def __init__(
        self,
        actor_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy,
        tokenizer,
        critic_model_group: RayActorGroup = None,
        **kwargs,
    ):
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args
        self.advantage_estimator = strategy.args.algo.advantage.estimator

        self.actor_model_group = actor_model_group
        self.initial_model_group = initial_model_group
        self.critic_model_group = critic_model_group
        self.tokenizer = tokenizer
        self.kl_ctl = kl_controller

    def build_experiences(self, rollout_samples: List[Experience]) -> List[Experience]:
        """Turn balanced rollout samples into train-ready experiences: recompute the log-probs and
        values the loss needs (make_experience), then the advantages and returns."""
        experiences = self.make_experience(rollout_samples)
        return self.compute_advantages_and_returns(experiences)

    @torch.no_grad()
    def make_experience(self, experiences: List[Experience]) -> List[Experience]:
        """Recompute the log-probs and values the policy loss needs. Each model's forward runs on
        its DP ranks, which fetch their samples with Experience.reload() — so the heavy tensors stay
        in shared memory and never reach the controller. Attaches values (GAE critic),
        base_action_log_probs (KL recipes), old action_log_probs, and the per-token kl, ready for
        advantage estimation."""
        args = self.args
        if self.critic_model_group is not None:
            self._dispatch_forward(experiences, self.critic_model_group, "values")
        if self.initial_model_group is not None:
            self._dispatch_forward(experiences, self.initial_model_group, "base_action_log_probs")

        # Old actor log-probs. Force-on-policy with no KL reward (kl_coef==0): old == the training
        # forward, so the PPO ratio is 1 (REINFORCE) and policy_train recomputes old itself — skip
        # the redundant pass. Otherwise (off-policy, or a KL/distill reward that compares old vs the
        # ref) run the actor forward here.
        skip_actor_old = args.train.force_on_policy and args.algo.kl.init_coef == 0
        if not skip_actor_old:
            self._dispatch_forward(experiences, self.actor_model_group, "action_log_probs")

        for i, experience in enumerate(experiences):
            experience.index = [i]
            # KL-as-reward (on_policy_distill, or reinforce/gae with kl_coef>0 and KL kept off the
            # loss): the advantage's per-token signal is the student->teacher KL. With KL in the loss
            # (or no ref) the advantage sees no KL reward, so kl stays zero.
            if (
                self.initial_model_group is not None
                and not args.algo.kl.use_loss
                and experience.action_log_probs is not None
            ):
                experience.kl = compute_approx_kl(
                    experience.action_log_probs,
                    experience.base_action_log_probs,
                    kl_estimator=args.algo.kl.estimator,
                )
                logprobs_diff = experience.action_log_probs.float() - experience.base_action_log_probs.float()
            else:
                experience.kl = torch.zeros_like(experience.action_mask, dtype=torch.float32)
                logprobs_diff = torch.zeros_like(experience.action_mask, dtype=torch.float32)
            experience.info["kl"] = masked_mean(experience.kl, experience.action_mask, dim=-1)
            experience.info["logprobs_diff"] = masked_mean(logprobs_diff, experience.action_mask, dim=-1)
            # With KL as a reward (or no ref) the loss needs no separate KL term, so drop the ref
            # log-probs; the KL-in-loss path keeps them for policy_train's loss.
            if not args.algo.kl.use_loss:
                experience.base_action_log_probs = None
        return experiences

    def _dispatch_forward(self, experiences: List[Experience], group: "RayActorGroup", result_attr: str) -> None:
        """Run ``group``'s forward on every sample — distributed across its DP ranks, each reloading its
        own heavy tensors so they never reach the controller — and store the per-sample result on the
        Experience under ``result_attr`` ("values" / "base_action_log_probs" / "action_log_probs"). Every
        cp/tp rank in a DP group returned the same per-sample results, so keep one copy per group (drop
        the duplicates) and flatten to one result per sample, in ``experiences`` order. Frees the
        forward's cache before the colocated actor trains on the same GPUs."""
        refs = group.async_run_method_batch(method_name="forward", experience=experiences)
        outputs = list(itertools.chain.from_iterable(ray.get(refs)[:: group.duplicate_actors]))
        for experience, output in zip(experiences, outputs):
            setattr(experience, result_attr, output)
        ray.get(group.async_run_method(method_name="empty_cache"))

    # Advantage and return computation

    def _merge_rollout_rewards(self, experiences: List[Experience]) -> dict:
        """Preprocessing: merge a rollout's multi-turn step-samples into one reward per rollout.

        Multi-turn agents emit several step-samples per rollout that share a rollout_id and the
        same terminal reward. We keep one reward per rollout and record, for every sample, which
        rollout it belongs to — so an estimator's per-rollout advantage can be scattered back to
        all of its steps. Without ids (legacy path) each sample is its own rollout and prompt.

        Example — two experiences, rollout "A" has 2 steps, "B" has 1, "C" has 2:
            e0.rollout_ids = ["A", "A", "B"]   e1.rollout_ids = ["C", "C"]
            e0.group_ids   = ["g0", "g0", "g0"]  e1.group_ids   = ["g1", "g1"]
            e0.rewards     = [1.0, 1.0, 0.0]   e1.rewards     = [0.5, 0.5]
        After concat the 5 samples are [A, A, B, C, C]; merging by first-seen rollout_id gives
            rewards           = [1.0, 0.0, 0.5]   # (R=3) one per unique rollout A, B, C
            groups            = [[0, 1], [2]]     # rollout rows grouped by prompt (g0, g1)
            sample_to_rollout = [0, 0, 1, 2, 2]   # (S=5) sample i -> its rollout's row in `rewards`
            exp_len           = [3, 2]            # samples per experience, to re-split later
        """
        exp_len = [len(e.index) for e in experiences]
        rollout_ids = list(itertools.chain.from_iterable(e.rollout_ids or list(e.index) for e in experiences))
        group_ids = list(
            itertools.chain.from_iterable(e.group_ids or e.rollout_ids or list(e.index) for e in experiences)
        )
        rewards = torch.cat([e.rewards for e in experiences], dim=0)
        if not (len(rollout_ids) == len(group_ids) == rewards.numel()):
            raise ValueError(
                f"id/reward length mismatch: {len(rollout_ids)} rollout_ids, "
                f"{len(group_ids)} group_ids, {rewards.numel()} rewards"
            )

        rollout_rewards: list = []
        sample_to_rollout: list = []
        prompt_groups: dict = {}  # prompt id -> rollout rows (preserves first-seen order)
        first_seen: dict = {}
        for rid, gid, reward in zip(rollout_ids, group_ids, rewards):
            if rid not in first_seen:
                first_seen[rid] = len(rollout_rewards)
                rollout_rewards.append(reward)
                prompt_groups.setdefault(gid, []).append(first_seen[rid])
            sample_to_rollout.append(first_seen[rid])

        return {
            "rewards": torch.stack(rollout_rewards),
            "groups": list(prompt_groups.values()),
            "sample_to_rollout": torch.tensor(sample_to_rollout),
            "exp_len": exp_len,
        }

    @staticmethod
    def _per_sample_rewards(experiences: List[Experience]) -> dict:
        """No-merge path: every sample is its own rollout (one-element groups).

        reinforce / gae / on_policy_distill score each sample independently, so a
        multi-turn rollout split into several samples must NOT collapse to one reward
        (only the group baselines need that). Each sample keeps its own reward; the
        identity sample->row map leaves the per-sample broadcast unchanged.
        """
        rewards = torch.cat([e.rewards for e in experiences], dim=0)
        n = rewards.numel()
        return {
            "rewards": rewards,
            "groups": [[i] for i in range(n)],
            "sample_to_rollout": torch.arange(n),
            "exp_len": [len(e.index) for e in experiences],
        }

    @torch.no_grad()
    def compute_advantages_and_returns(self, experiences: List[Experience]) -> List[Experience]:
        """Clip rewards, run the estimator (which returns per-token advantages/returns), assemble onto exps.

        Estimators live in `advantage.py` and never see `Experience`: this method extracts the small
        tensor inputs (rewards, action masks, per-token KL), builds the `AdvantageContext`, and
        writes the returned advantages/returns/info back onto each experience. Only the group
        baselines merge multi-turn step-samples to one reward per rollout; reinforce/gae score
        each sample independently (see `_per_sample_rewards`).
        """
        args = self.args
        if self.advantage_estimator in GROUP_ADVANTAGE_ESTIMATORS:
            rollouts = self._merge_rollout_rewards(experiences)
        else:
            rollouts = self._per_sample_rewards(experiences)

        # Clip the raw per-rollout reward before the baseline.
        clip = args.reward.clip_range
        rewards = rollouts["rewards"].clamp(min=clip[0], max=clip[1]) if clip else rollouts["rewards"]

        # PPO/gae is the only estimator that consumes a learned value baseline; the
        # critic filled exp.values during make_experience. Other estimators ignore it.
        needs_values = self.advantage_estimator == "gae"
        ctx = AdvantageContext(
            sample_to_rollout=rollouts["sample_to_rollout"],
            exp_len=rollouts["exp_len"],
            action_masks=[exp.action_mask for exp in experiences],
            kl_coef=self.kl_ctl.value,
            gamma=args.algo.advantage.gamma,
            kls=[exp.kl for exp in experiences],
            lam=args.algo.advantage.lam,
            values=[exp.values for exp in experiences] if needs_values else None,
            no_whiten=args.algo.advantage.no_whiten,
        )
        advantages, returns = get_advantage_estimator(self.advantage_estimator)(rewards, rollouts["groups"], ctx)

        # Per-group reward std (on clipped rewards), broadcast to samples, for logging only.
        rollout_stds = torch.zeros_like(rewards)
        for group in rollouts["groups"]:
            rollout_stds[group] = rewards[group].std() if len(group) > 1 else 0.0
        sample_stds = rollout_stds[rollouts["sample_to_rollout"]].split(rollouts["exp_len"])

        # Assemble the experiences from the computed tensors.
        for exp, adv, ret, std in zip(experiences, advantages, returns, sample_stds):
            exp.advantages = adv
            exp.returns = ret
            exp.info["return"] = masked_mean(ret, exp.action_mask, dim=-1)
            if args.rollout.n_samples_per_prompt > 1:
                exp.info["group_reward_std"] = std
            exp.kl = None

        return experiences
