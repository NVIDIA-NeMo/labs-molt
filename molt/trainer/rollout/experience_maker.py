# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from __future__ import annotations

import itertools
import time
from datetime import timedelta
from typing import TYPE_CHECKING, List

import ray
import torch

from molt.models.utils import compute_approx_kl, masked_mean
from molt.trainer.algorithm.advantage import (
    GROUP_ADVANTAGE_ESTIMATORS,
    AdvantageContext,
    get_advantage_estimator,
)
from molt.trainer.algorithm.experience import Experience, get_model_parallel_size
from molt.utils.logging_utils import init_logger
from molt.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions

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

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.train.dynamic_batch_enable:
            total_lengths = [int(s.total_length.item()) for s in rollout_samples]
            actor_world_size = self.args.actor.num_nodes * self.args.actor.num_gpus_per_node
            effective_actor_num = actor_world_size // get_model_parallel_size(self.args)
            if effective_actor_num <= 0:
                raise ValueError(f"Invalid effective actor count: {effective_actor_num}")
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout.max_tokens_per_gpu,
                self.args.fsdp.cp_size,
                self.args.fsdp.tp_size,
            )
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.rollout.micro_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def build_experiences(self, rollout_samples) -> List[Experience]:
        """Turn already-generated rollout samples into train-ready experiences.

        Splits the rollout batch across DP ranks, runs the remote log-prob / KL
        forwards, then computes advantages and returns. (Sequences and rewards
        are produced earlier by the SamplesGenerator, not here.)
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs and KL divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    # Remote model dispatch helpers

    def _dispatch_forward(self, group, sync_condition, **kwargs):
        """Dispatch a batched forward call and optionally sync + empty cache."""
        ref = group.async_run_method_batch(method_name="forward", **kwargs)
        if sync_condition:
            ray.get(ref)
            ray.get(group.async_run_method(method_name="empty_cache"))
        return ref

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """Turn samples into experience by calculating logprobs and KL divergence."""
        start_time = time.time()
        logger.info(f"Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.args
        cp_tp_copies = get_model_parallel_size(args)
        n_samples = len(samples_list)

        # Extract tensors for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]
        forward_kwargs = dict(
            sequences=sequences_list, action_mask=action_mask_list, attention_mask=attention_mask_list
        )

        # VLM: pre-processed multimodal inputs needed by actor and reference models
        vlm_forward_kwargs = dict(forward_kwargs)
        if any(s.mm_train_inputs for s in samples_list):
            vlm_forward_kwargs["mm_train_inputs_list"] = [s.mm_train_inputs for s in samples_list]

        if any(samples.rewards is None for samples in samples_list):
            raise ValueError(
                "Rollout samples must include rewards. Use --train.agent_path and return "
                "`rewards` or `reward` from the agent."
            )

        colocated_policy_workers = getattr(args.train, "colocate_fsdp_models", False) and (
            self.initial_model_group is not None or self.critic_model_group is not None
        )
        # On-policy with no KL (kl_coef == 0): old == the training forward, so the PPO
        # ratio is 1 and the loss is REINFORCE — the old-logprob recompute is redundant.
        # Skip it; policy_train sets old = action.detach(), sharing the exact R3 routing.
        # (A KL/distill reward, kl_coef > 0, still needs old to compare against the ref.)
        skip_actor_old = args.train.force_on_policy and args.algo.kl.init_coef == 0
        if not skip_actor_old:
            actor_forward_kwargs = dict(vlm_forward_kwargs)
            if any(s.routed_experts is not None for s in samples_list):
                # R3: replay the rollout routing so old picks the same experts as training.
                actor_forward_kwargs["routed_experts"] = [s.routed_experts for s in samples_list]
            action_log_probs_ref = self._dispatch_forward(
                self.actor_model_group,
                colocated_policy_workers,
                **actor_forward_kwargs,
            )

        # Reference model (also receives mm_train_inputs_list for VLM). If there is no
        # reference, base log-probs stay None (filled below).
        base_action_log_probs_ref = None
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self._dispatch_forward(
                self.initial_model_group,
                colocated_policy_workers,
                **vlm_forward_kwargs,
            )

        # Critic value model (PPO/gae only). Its own Ray group; CriticModelActor.forward
        # returns the per-token V(s) on the action span — same dispatch shape as the ref.
        use_critic = self.advantage_estimator == "gae"
        if use_critic and self.critic_model_group is None:
            raise ValueError("advantage_estimator=gae requires a critic_model_group.")
        values_ref = (
            self._dispatch_forward(self.critic_model_group, colocated_policy_workers, **vlm_forward_kwargs)
            if use_critic
            else None
        )

        # Gather each forward's per-sample results, dropping the duplicate CP/TP rank
        # copies. A forward we didn't run (skipped actor / no reference / no critic)
        # yields a per-sample None list.
        if skip_actor_old:
            action_log_probs_list = [None] * n_samples
        else:
            action_log_probs_list = list(itertools.chain.from_iterable(ray.get(action_log_probs_ref)[::cp_tp_copies]))

        if base_action_log_probs_ref is not None:
            base_action_log_probs_list = list(
                itertools.chain.from_iterable(ray.get(base_action_log_probs_ref)[::cp_tp_copies])
            )
        else:
            base_action_log_probs_list = [None] * n_samples

        if use_critic:
            values_list = list(itertools.chain.from_iterable(ray.get(values_ref)[::cp_tp_copies]))
        else:
            values_list = None

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}"
        if use_critic:
            assert len(values_list) == len(samples_list), f"values {len(values_list)} != samples {len(samples_list)}"

        # Compute KL and attach results to experiences

        for i, (samples, action_log_probs, base_action_log_probs) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list)
        ):
            if (self.initial_model_group is not None) and (not args.algo.kl.use_loss) and action_log_probs is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=args.algo.kl.estimator,
                )
                logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
            else:
                # action_log_probs may be None (force_on_policy skipped the recompute),
                # so size the zero KL / logprobs_diff from the always-present action mask.
                kl = torch.zeros_like(samples.action_mask, dtype=torch.float32, device="cpu")
                logprobs_diff = torch.zeros_like(samples.action_mask, dtype=torch.float32, device="cpu")
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            logprobs_diff_mean = masked_mean(logprobs_diff, samples.action_mask, dim=-1)

            if not args.algo.kl.use_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.kl = kl
            samples.info["kl"] = kl_mean
            samples.info["logprobs_diff"] = logprobs_diff_mean
            if use_critic:
                # Collection-time V(s) on the action span; the PPO old_values that
                # gae subtracts and the value loss clips around.
                samples.values = values_list[i]

        time_str = str(timedelta(seconds=time.time() - start_time)).split(".")[0]
        logger.info(f"Experience making completed in {time_str}")
        return samples_list

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
            no_std_norm=args.algo.advantage.no_std_norm,
            kl_coef=self.kl_ctl.value,
            gamma=args.algo.advantage.gamma,
            kls=[exp.kl for exp in experiences],
            lam=args.algo.advantage.lam,
            values=[exp.values for exp in experiences] if needs_values else None,
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
