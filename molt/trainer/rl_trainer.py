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

import asyncio
import os
import time
from typing import Dict, Tuple

import ray
import torch
from ray.util.queue import Queue
from tqdm import tqdm

from molt.agents.base import _first_scalar
from molt.datasets import PromptDataset
from molt.datasets.utils import blending_datasets
from molt.trainer.algorithm.experience import balance_experiences
from molt.trainer.algorithm.kl_controller import AdaptiveKLController, FixedKLController
from molt.trainer.fsdp import FsdpStrategy
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker
from molt.trainer.rollout.samples_generator import SamplesGenerator
from molt.trainer.vllm.vllm_engine import batch_vllm_engine_call
from molt.trainer.workers.actor_group import RayActorGroup
from molt.utils.distributed_sampler import DistributedSampler
from molt.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from molt.utils.utils import get_tokenizer

logger = init_logger(__name__)


def prepare_datasets(strategy, tokenizer):
    args = strategy.args

    # BOTH runner types consume the SAME chat-format dataset (--data.apply_chat_template);
    # Runner.PRERENDER_PROMPT only decides WHERE the template is applied. The step runner needs the
    # dataset to pre-render; the chat runner hands the raw messages to the chat server, which renders
    # them exactly once with the model's own template (a dataset pre-render would double-template and
    # drop the image on structured-content VLMs) — so for chat agents the flag is required, never
    # applied dataset-side.
    from molt.agents.base import load_agent_runner

    prerender = True
    if getattr(args.train, "agent_path", None):
        prerender = getattr(load_agent_runner(args.train.agent_path), "PRERENDER_PROMPT", True)
    if not prerender and not args.data.apply_chat_template:
        raise ValueError(
            "Chat agents consume the same chat-format dataset as step agents: pass "
            "--data.apply_chat_template. The dataset hands the messages through raw; the chat "
            "server renders them once with the model's own template."
        )

    # prepare datasets
    train_data = blending_datasets(
        args.data.prompt_dataset,
        args.data.prompt_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.prompt_split,
    )

    # Create train dataset
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, prerender=prerender)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=prompts_dataset.collate_fn,
        num_workers=args.data.dataloader_num_workers,
    )

    # Create eval dataset if eval data exists
    if getattr(args.eval, "dataset", None):
        eval_data = blending_datasets(
            args.eval.dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.eval.split,
        )
        eval_data = eval_data.select(range(min(args.data.max_samples, len(eval_data))))
        eval_dataset = PromptDataset(eval_data, tokenizer, strategy, prerender=prerender)
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
            num_workers=args.data.dataloader_num_workers,
        )
    else:
        eval_dataloader = None

    if args.train.force_on_policy:
        # On-policy: one optimizer step per rollout batch (per epoch), regardless
        # of how many samples multi-turn flatten produces. The generator consumes
        # rollout.batch_size prompt-groups per round, so the LR scheduler decays
        # over len(prompts) // rollout.batch_size, not samples // train.batch_size.
        max_steps = len(prompts_dataset) // args.rollout.batch_size * args.train.num_episodes * args.train.max_epochs
    else:
        max_steps = (
            len(prompts_dataset)
            * args.rollout.n_samples_per_prompt
            // args.train.batch_size
            * args.train.num_episodes
            * args.train.max_epochs
        )
    return prompts_dataloader, eval_dataloader, max_steps


def compute_eval_metrics(eval_dataloader, samples_list, n_samples_per_prompt):
    """Compute pass@k eval metrics from generated samples.

    Robust to dropped rollouts: samples are grouped by their rollout ``group_id``
    (falling back to the prompt string when group ids are absent) rather than a
    rigid ``reshape(-1, n_samples_per_prompt)``. A rollout dropped during
    generation (empty / zero-action / VLM-truncated — see
    ``SamplesGenerator._process_response_into_experience``) only shrinks that
    prompt's group instead of crashing the reshape or misaligning samples across
    prompt boundaries.
    """
    if not samples_list:
        return {}

    prompt_to_datasource = {}
    for datasources, prompts, labels, _images, _tools in eval_dataloader:
        for prompt, datasource in zip(prompts, datasources):
            if isinstance(prompt, list):
                # Chat rows pass through as messages; key on the last user turn's text —
                # the same scalar ChatAgentRunner stores as Trajectory.prompt.
                texts = [
                    m.get("content") for m in prompt if m.get("role") == "user" and isinstance(m.get("content"), str)
                ]
                prompt = texts[-1] if texts else str(prompt)
            prompt_to_datasource[prompt] = datasource

    # Each Experience here is a single rollout sample (B=1). Group the per-sample
    # scalars by the rollout group_id (one uuid per prompt group, i.e.
    # per eval prompt instance), NOT by the prompt STRING: two distinct eval rows
    # that render to the same string (same question across blended eval sets, dup
    # rows, or short templated prompts) would otherwise merge into one pass@k
    # group and distort pass1/count/eval_num_samples. Fall back to the prompt
    # string only when group ids are absent. Datasource is still resolved via the
    # prompt string (a benign attribution choice for genuine duplicates).
    grouped: Dict[str, Dict[str, list]] = {}
    group_order = []
    group_prompt: Dict[str, str] = {}
    for s in samples_list:
        prompt = s.prompts[0]
        key = s.group_ids[0] if getattr(s, "group_ids", None) else prompt
        if key not in grouped:
            grouped[key] = {"rewards": [], "lengths": [], "truncated": []}
            group_order.append(key)
            group_prompt[key] = prompt
        grouped[key]["rewards"].append(_first_scalar(s.rewards))
        grouped[key]["lengths"].append(_first_scalar(s.response_length))
        grouped[key]["truncated"].append(_first_scalar(s.truncated))

    metrics = {}
    for key in group_order:
        g = grouped[key]
        prompt = group_prompt[key]
        rewards = [r for r in g["rewards"] if r is not None]
        if not rewards:
            continue
        ds = prompt_to_datasource.get(prompt, "unknown")
        if ds not in metrics:
            metrics[ds] = {
                f"pass{n_samples_per_prompt}": 0.0,
                "pass1": 0.0,
                "count": 0,
                "lengths": [],
                "truncated": [],
            }
        if n_samples_per_prompt > 1:
            metrics[ds][f"pass{n_samples_per_prompt}"] += max(rewards)
        metrics[ds]["pass1"] += sum(rewards) / len(rewards)
        metrics[ds]["count"] += 1
        metrics[ds]["lengths"].extend(x for x in g["lengths"] if x is not None)
        metrics[ds]["truncated"].extend(t for t in g["truncated"] if t is not None)

    logs = {}
    total_lengths = []
    total_truncated = []
    for ds, m in metrics.items():
        logs[f"eval_{ds}_pass{n_samples_per_prompt}"] = m[f"pass{n_samples_per_prompt}"] / m["count"]
        logs[f"eval_{ds}_pass1"] = m["pass1"] / m["count"]
        if m["lengths"]:
            logs[f"eval_{ds}_response_length_mean"] = sum(m["lengths"]) / len(m["lengths"])
            total_lengths.extend(m["lengths"])
        if m["truncated"]:
            logs[f"eval_{ds}_truncated_rate"] = sum(m["truncated"]) / len(m["truncated"])
            total_truncated.extend(m["truncated"])

    if total_lengths:
        logs["eval_response_length_mean"] = sum(total_lengths) / len(total_lengths)
    if total_truncated:
        logs["eval_truncated_rate"] = sum(total_truncated) / len(total_truncated)
    logs["eval_num_samples"] = float(len(samples_list))

    return logs


class BaseRLTrainer:
    """Training-side base class for non-critic policy RL."""

    def __init__(
        self,
        strategy: FsdpStrategy,
        actor_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines,
        tokenizer,
        critic_model_group: RayActorGroup = None,
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args

        self.actor_model_group = actor_model_group
        self.reference_model_group = reference_model_group
        self.critic_model_group = critic_model_group
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer

        # Critic warmup: freeze the actor's policy update for the first N optimizer
        # steps so the value model can fit the initial rollouts before its early,
        # high-variance advantages start moving the policy. The critic still trains
        # (and the actor buffer is still drained) while frozen. Only with a critic;
        # 0 disables.
        self.freezing_actor_steps = self.args.actor.freezing_steps if critic_model_group is not None else 0

        if self.args.algo.kl.target:
            self.kl_ctl = AdaptiveKLController(
                self.args.algo.kl.init_coef, self.args.algo.kl.target, self.args.algo.kl.horizon
            )
        else:
            self.kl_ctl = FixedKLController(self.args.algo.kl.init_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            tokenizer,
            critic_model_group=self.critic_model_group,
        )

        # Tracking backends
        self.wandb_logger = WandbLogger(self.args) if self.args.logger.wandb.key else None
        self.tensorboard_logger = TensorboardLogger(self.args) if self.args.logger.tensorboard_dir else None

        # Best eval metric tracking
        self.best_eval_metric_value = float("-inf")
        self.best_eval_metric_key = getattr(self.args.ckpt, "best_metric_key", "") or ""
        self._latest_eval_metric_value = None

    def restore_best_metric_tracker(self, checkpoint_states) -> None:
        if not checkpoint_states:
            return

        checkpoint_metric_key = checkpoint_states.get("best_eval_metric_key")
        checkpoint_metric_value = checkpoint_states.get("best_eval_metric_value")

        if checkpoint_metric_key:
            self.best_eval_metric_key = checkpoint_metric_key
        if checkpoint_metric_value is not None:
            self.best_eval_metric_value = checkpoint_metric_value
            self._latest_eval_metric_value = checkpoint_metric_value

    def fit(self, global_step: int = 0) -> None:
        raise NotImplementedError("fit method is not implemented")

    def train_step(self, rollout_samples, global_step: int) -> Tuple[Dict, int]:
        # `rollout_samples` are lazy Experiences: each sample's heavy tensors (images / token ids /
        # rollout routing) sit in shared memory (the producing runner's object store) behind a handle.
        # The flow below is the ordinary single-controller RL step — balance, make experience, compute
        # advantages, push, optimize — and only the ranks that consume a sample fetch its heavy tensors
        # (Experience.reload()). So the controller works with light handles and a full image batch
        # never concentrates on one node; the transfer is transparent (Ray resolves each handle where
        # it is used).
        t0 = time.time()
        experiences = balance_experiences(rollout_samples, self.args)
        experiences = self.experience_maker.build_experiences(experiences)
        make_experience_time = time.time() - t0

        # Peek at the first sample's token ids for a sanity log. If it is still lazy, fetch its heavy
        # blob from shared memory and read sequences out — leaving heavy_ref set so the sample still
        # ships cheaply to its rank below; otherwise read the local tensor directly.
        first = experiences[0]
        if first.heavy_ref is not None:
            first_seq = ray.get(first.heavy_ref)["sequences"]
        else:
            first_seq = first.sequences
        sample0 = [
            self.tokenizer.decode(first_seq[0], skip_special_tokens=True),
            experiences[0].info["reward"][0].item(),
        ]
        logger.info(f"Sample: {sample0}")
        if os.environ.get("MOLT_DEBUG_ROLLOUT") == "1":
            debug_rows = []
            for exp in experiences:
                info = exp.info
                debug_rows.append(
                    {
                        "index": int(exp.index[0]) if exp.index else None,
                        "reward": info["reward"][0].item() if "reward" in info else None,
                        "return": info["return"][0].item() if "return" in info else None,
                        "group_reward_std": info["group_reward_std"][0].item() if "group_reward_std" in info else None,
                        "text": self.tokenizer.decode(exp.reload().sequences[0], skip_special_tokens=True),
                    }
                )
            logger.info(f"RolloutDebug: {debug_rows}")

        # Ground-truth rollout stats over the FULL generated set — from the lightweight fields present
        # on every rollout sample, and computed on `rollout_samples` (before balance_experiences drops
        # the trailing remainder), so num_samples and the means reflect everything we generated.
        all_rewards = torch.cat([s.info["reward"] for s in rollout_samples if "reward" in s.info])
        all_response_lengths = torch.cat([s.response_length for s in rollout_samples if s.response_length is not None])
        all_truncated = torch.cat([s.truncated for s in rollout_samples if s.truncated is not None])
        rollout_stats = {
            "rollout/reward_mean": all_rewards.float().mean().item(),
            "rollout/reward_std": all_rewards.float().std().item() if len(all_rewards) > 1 else 0.0,
            "rollout/response_length_mean": all_response_lengths.float().mean().item(),
            "rollout/truncated_rate": all_truncated.float().mean().item(),
            "rollout/num_samples": float(len(all_rewards)),
        }

        # Push the experiences to the actor shards (and the critic, which trains on the same batch
        # with values + returns) before optimization. Each rank fetches its samples' heavy tensors
        # from the producing runner via reload() — the images reach the rank straight from the
        # runner, never through the controller.
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs += self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
        ray.get(refs)

        # Perform policy optimization for the actor and gather metrics. During
        # critic warmup the actor is frozen — no policy update — while the value
        # model trains on the same rollouts.
        actor_frozen = global_step < self.freezing_actor_steps
        t0 = time.time()
        status = self.policy_train(train_actor=not actor_frozen)
        status["actor_frozen"] = float(actor_frozen)
        policy_train_time = time.time() - t0

        # Sync weights to vLLM (skipped while the actor is frozen: its weights are
        # unchanged, so the live vLLM copy is already current).
        t0 = time.time()
        # Reset per-step so a frozen/skipped broadcast reports 0 (not a stale
        # value); the TrainingActor.broadcast_to_vllm override sets these when
        # it runs (lock-wait vs actual transfer).
        self._broadcast_lock_wait_s = 0.0
        self._broadcast_transfer_s = 0.0
        if self.vllm_engines is not None and not actor_frozen:
            self.broadcast_to_vllm()
        broadcast_time = time.time() - t0

        # Log the KL coefficient applied to this step's loss (the value passed into
        # `fit`), captured before the update below mutates it. Without this, adaptive-KL
        # runs (--algo.kl.target set) surface the measured `kl` but not the coefficient
        # the AdaptiveKLController is driving in response — the feedback loop is invisible.
        status["kl_coef"] = self.kl_ctl.value

        # Refresh KL controller with the latest measurement (no-op for FixedKLController).
        if "kl" in status:
            self.kl_ctl.update(status["kl"], self.args.rollout.batch_size * self.args.rollout.n_samples_per_prompt)

        # Per-phase timing breakdown. timing/broadcast is the TOTAL; it splits into
        # broadcast_lock_wait (trainer blocked on the vllm_lock held by the overlapping
        # rollout generation — overlaps wall-clock, NOT a transfer cost) and
        # broadcast_transfer (the actual NCCL weight sync). Keep all three so the
        # total stays comparable while the lock-wait is no longer mistaken for transfer.
        status["timing/make_experience"] = make_experience_time
        status["timing/policy_train"] = policy_train_time
        status["timing/broadcast"] = broadcast_time
        status["timing/broadcast_lock_wait"] = self._broadcast_lock_wait_s
        status["timing/broadcast_transfer"] = self._broadcast_transfer_s

        # Merge rollout stats (ground-truth, pre-dynamic-batch)
        status.update(rollout_stats)

        status["generated_samples"] = sample0
        return status, global_step + 1

    def policy_train(self, train_actor: bool = True) -> Dict:
        """Run one actor optimization step (then the critic) and return merged status.

        Sequential, not concurrent: colocated actor and critic share GPUs, so running
        both trainings at once would double resident memory. Disaggregated setups pay
        a small no-overlap cost here. ``train_actor=False`` (critic warmup) skips the
        policy update but still drains the actor replay buffer.
        """
        refs = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value, train=train_actor)
        status: dict = {}
        for result in ray.get(refs):
            status.update(result)
        # Fail loudly when the vLLM-IS filter dropped every sequence: the policy
        # gradient is exactly zero and the run silently optimizes nothing. The
        # usual cause is rollout-vs-train forward mismatch, for MoE models most
        # often unreplayed expert routing — enable --train.routing_replay.
        if status.get("is_filter_ratio", 0.0) >= 0.999:
            logger.warning(
                f"is_filter_ratio={status['is_filter_ratio']:.3f}: the vLLM importance-sampling filter dropped "
                f"(nearly) every sequence — zero policy gradient this step (vllm_kl={status.get('vllm_kl')}). "
                "Rollout and training forwards disagree; for MoE models enable --train.routing_replay."
            )
        if self.critic_model_group is not None:
            # Colocated actor and critic are separate processes sharing the same GPUs.
            # Release the actor's cached GPU blocks back to the driver before the critic
            # trains so the critic's activation / cuDNN-attn-workspace allocations have
            # headroom (the per-fit empty_cache does this too; this makes the release
            # deterministic at the actor->critic boundary).
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))
            critic_refs = self.critic_model_group.async_run_method(method_name="fit")
            for result in ray.get(critic_refs):
                status.update(result)
            ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        return status

    def broadcast_to_vllm(self) -> None:
        """Broadcast actor weights to vLLM engines."""
        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        # NOTE: We keep vLLM in weights-only state after weight sync.
        # KV cache will be woken up before generation in SamplesGenerator.

    def save_best_checkpoint(self, eval_metrics, global_step, client_states=None):
        """Save checkpoint if eval metric is the best so far.

        When best_metric_key is 'none' or no eval_*_pass1 metric is present,
        this is a no-op — regular save_steps checkpoints still save the most recent.
        """
        if not eval_metrics or self.best_eval_metric_key == "none":
            return

        if self.best_eval_metric_key:
            metric_key = self.best_eval_metric_key if self.best_eval_metric_key in eval_metrics else None
        else:
            # Auto-detect: prefer eval_*_pass1 metric.
            metric_key = next((k for k in sorted(eval_metrics) if k.endswith("_pass1")), None)
            if metric_key is not None:
                self.best_eval_metric_key = metric_key
        if metric_key is None:
            return

        current_value = eval_metrics[metric_key]
        self._latest_eval_metric_value = current_value
        prev_best = self.best_eval_metric_value

        if current_value > self.best_eval_metric_value:
            self.best_eval_metric_value = current_value
            logger.info(
                f"New best eval metric: {metric_key}={current_value:.4f} at step {global_step} "
                f"(previous best: {prev_best if prev_best > float('-inf') else 'N/A'})"
            )

            client_states = client_states or {}
            client_states["best_eval_metric_key"] = metric_key
            client_states["best_eval_metric_value"] = current_value
            client_states["checkpoint_metric_key"] = metric_key

            tag = f"best_global_step{global_step}"
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint",
                tag=tag,
                client_states=client_states,
                metric_value=current_value,
                metric_key=metric_key,
            )
            if self.critic_model_group is not None:
                refs += self.critic_model_group.async_run_method(
                    method_name="save_checkpoint", tag=tag, metric_value=current_value, metric_key=metric_key
                )
            ray.get(refs)
            logger.info(f"Saved best checkpoint: {tag} ({metric_key}={current_value:.4f})")

    def save_logs_and_checkpoints(self, global_step: int, logs_dict=None, client_states=None) -> None:
        logs_dict = logs_dict or {}
        if global_step % self.args.logger.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)

        # save ckpt
        client_states = client_states or {}
        if global_step % self.args.ckpt.save_steps == 0:
            tag = f"global_step{global_step}"
            # Persist best-metric tracker on every rolling save so chain
            # successors can `restore_best_metric_tracker` from the latest
            # rolling DCP — not just from the `best_*` ckpt (which gets
            # rotated out by max_num cleanup).
            client_states["best_eval_metric_key"] = self.best_eval_metric_key
            client_states["best_eval_metric_value"] = self.best_eval_metric_value
            metric_value = self._latest_eval_metric_value
            metric_key = client_states.get("checkpoint_metric_key") or self.best_eval_metric_key or None
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint",
                tag=tag,
                client_states=client_states,
                metric_value=metric_value,
                metric_key=metric_key,
            )
            if self.critic_model_group is not None:
                refs += self.critic_model_group.async_run_method(
                    method_name="save_checkpoint", tag=tag, metric_value=metric_value, metric_key=metric_key
                )
            ray.get(refs)

    def load_checkpoint_states_or_default(self) -> Dict:
        ckpt_path = os.path.join(self.args.ckpt.path, "_actor")
        if self.args.ckpt.load_enable and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            # Log scalars only; never f-string the whole dict (a legacy checkpoint
            # can carry a multi-MB sub-dict that would OOM the driver on resume).
            logger.info(
                "checkpoint_states: %s",
                {k: v for k, v in checkpoint_states.items() if not isinstance(v, (dict, list))},
            )
            return checkpoint_states
        return {
            "episode": 0,
            "global_step": 0,
            "total_consumed_prompts": 0,
            "data_loader_state_dict": {},
        }


@ray.remote(num_cpus=0)
class VLLMLock:
    """Cross-actor mutex for vLLM critical sections."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def acquire(self):
        await self._lock.acquire()

    async def release(self):
        self._lock.release()


@ray.remote
class GenerateSamplesActor:
    def __init__(
        self,
        pretrain,
        strategy,
        *,
        vllm_lock,
        rollout_queue,
        rollout_slots,
        router_url=None,
        **generate_kwargs,
    ):
        # No vllm_engines here: generation runs through the vllm-router via the runner
        # actors below; only the TrainingActor touches the engines (pause/refit/resume).
        self.args = strategy.args

        tokenizer = get_tokenizer(pretrain, None, "left", use_fast=not strategy.args.data.disable_fast_tokenizer)
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)
        self.generate_kwargs = generate_kwargs

        # Rollout runs on a list of runner actors -> the shared vllm-router (generation),
        # grading in-process. Weight sync goes straight to the engines (bypasses the router).
        from molt.trainer.rollout.router import AgentRunnerActor

        num_runners = max(1, getattr(strategy.args.rollout, "num_runners", 2))
        agent_runners = [
            AgentRunnerActor.remote(strategy.args.train.agent_path, router_url, model_path=pretrain)
            for _ in range(num_runners)
        ]
        ray.get([r.ready.remote() for r in agent_runners])
        self.samples_generator = SamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            agent_runners=agent_runners,
        )

        self.vllm_lock = vllm_lock
        self._partial_rollout = getattr(strategy.args.train, "partial_rollout_enable", False)
        self.rollout_queue = rollout_queue
        self.rollout_slots = rollout_slots
        self._last_eval_step = -1
        # Eval fires once global_step crosses this threshold, then it advances by
        # eval_steps. Using `>=` (catch-up) instead of `% eval_steps == 0` is
        # robust to the async slot's global_step jumping past an exact multiple,
        # which silently skipped whole eval points (e.g. eval@20 never ran).
        self._next_eval_step = strategy.args.eval.steps
        # Optional baseline eval at global_step 0 (pre-RL model). Fresh runs only:
        # resume starts at a saved global_step > 0, so this never adds a redundant
        # eval on resume.
        self._eval_at_start = getattr(strategy.args.eval, "eval_at_start", False)

    def get_max_steps(self):
        return self.max_steps

    def load_dataloader_state_dict(self, state_dict, rollout_generator_state_dict=None):
        self.prompts_dataloader.load_state_dict(state_dict)
        self.samples_generator.load_state_dict(rollout_generator_state_dict)

    def fit(self, episode: int, total_consumed_prompts: int) -> None:
        eval_steps = self.args.eval.steps
        # eval_at_start must fire only on a genuinely FRESH run. The
        # GenerateSamplesActor transiently reads global_step 0 at startup even on
        # resume (the restored step propagates through rollout_slots a beat later),
        # so gating on global_step alone would also eval@0 on resume. The
        # consumed-prompt counter is 0 only on a fresh start (>0 once a checkpoint
        # is loaded), so it's the reliable fresh-run signal.
        fresh_start = total_consumed_prompts == 0
        # On resume, _next_eval_step was re-init to eval_steps (< the restored
        # global_step), so the catch-up `global_step >= _next_eval_step` would fire
        # one off-cadence eval at the resumed step every resubmit. Sync the threshold
        # past the resume point on the first real step so eval lands only on the
        # normal eval_steps multiples a continuous run would hit (no per-resume eval).
        # Fresh runs are already in sync (step-0 baseline + _next_eval_step=eval_steps).
        eval_synced = fresh_start
        for ep in range(episode, self.args.train.num_episodes):
            # Reshuffle prompts each episode (seed+epoch); without this every
            # episode replays the identical order. The generator rebuilds its
            # iterator at the episode boundary, so the new epoch takes effect on
            # the next iter() — and on resume `ep` matches the saved episode.
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(ep)
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{ep + 1}/{self.args.train.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                # Backpressure: slot token carries trainer's latest global_step for eval
                # timing. Time the block — in this async split, the generator stuck here
                # means the vLLM side is sitting IDLE waiting for the trainer to free a
                # slot (training slower than generation). This is the true vLLM-idle signal,
                # which the wall-clock generation_time alone cannot show.
                _slot_wait_t0 = time.time()
                global_step = self.rollout_slots.get(block=True)
                vllm_idle_wait = time.time() - _slot_wait_t0

                # Resume sync (once): advance the eval threshold to the next eval_steps
                # multiple after the restored step, so a resubmit doesn't trigger an
                # off-cadence eval at the resumed global_step.
                if not eval_synced and global_step > 0 and eval_steps > 0 and eval_steps != float("inf"):
                    self._next_eval_step = (global_step // eval_steps + 1) * eval_steps
                    eval_synced = True

                should_eval = (
                    self.eval_dataloader is not None
                    and eval_steps != float("inf")
                    and global_step != self._last_eval_step
                    and (
                        global_step >= self._next_eval_step
                        if global_step > 0
                        # step-0 baseline (fresh run only): measure the pre-RL model
                        # so later gains are attributable. After it fires, the normal
                        # cadence resumes (_next_eval_step advances to eval_steps).
                        else (self._eval_at_start and self._last_eval_step < 0 and fresh_start)
                    )
                )
                if should_eval:
                    self._last_eval_step = global_step
                    self._next_eval_step = (global_step // eval_steps + 1) * eval_steps
                    logger.info(f"Starting async evaluation at step {global_step}...")
                    # Independent eval sampling: each knob left unset (None) falls back to the
                    # rollout value in generate_kwargs; override only what's set for eval.
                    eval_n = self.args.eval.n_samples_per_prompt
                    if eval_n is None:
                        eval_n = self.args.rollout.n_samples_per_prompt
                    eval_kwargs = {**self.generate_kwargs, "n_samples_per_prompt": eval_n}
                    for key in ("temperature", "top_p", "max_new_tokens"):
                        override = getattr(self.args.eval, key)
                        if override is not None:
                            eval_kwargs[key] = override
                    # Under partial rollout the rollout path (below) deliberately
                    # skips vllm_lock so the trainer's broadcast_to_vllm refit can
                    # interleave via pause/resume. Eval must follow the same
                    # contract: holding the lock across the whole eval generation
                    # (~1hr at 32K) blocks the refit's acquire and wedges training
                    # (train_step never returns → no global_step advance).
                    if not self._partial_rollout:
                        ray.get(self.vllm_lock.acquire.remote())
                    try:
                        samples_list = self.samples_generator.generate_eval_samples(**eval_kwargs)
                    finally:
                        if not self._partial_rollout:
                            ray.get(self.vllm_lock.release.remote())
                    eval_metrics = compute_eval_metrics(self.eval_dataloader, samples_list, eval_n)
                    logger.info(f"Async evaluation completed: {eval_metrics}")
                    self.rollout_queue.put(("eval", global_step, eval_metrics), block=True)
                    continue

                if self.args.train.rollout_replay_dir:
                    # Debug: replay dumped rollout batches (train-only) to iterate on the
                    # training/refit path without regenerating. When the dumps run out, report
                    # exhaustion so the normal end-of-data path (below) stops the run cleanly.
                    replay_path = os.path.join(self.args.train.rollout_replay_dir, f"rollout_step{global_step}.pt")
                    rollout_metrics, prompts_consumed, generation_time = {}, 0, 0.0
                    is_exhausted = not os.path.exists(replay_path)
                    rollout_samples = None if is_exhausted else torch.load(replay_path, weights_only=False)
                    logger.info(
                        f"[rollout_replay] exhausted at {replay_path}"
                        if is_exhausted
                        else f"[rollout_replay] loaded {len(rollout_samples)} samples from {replay_path}"
                    )
                else:
                    if not self._partial_rollout:
                        ray.get(self.vllm_lock.acquire.remote())
                    try:
                        t0 = time.time()
                        rollout_samples, rollout_metrics, prompts_consumed, is_exhausted = (
                            self.samples_generator.generate_samples(**self.generate_kwargs)
                        )
                        generation_time = time.time() - t0
                        total_consumed_prompts += prompts_consumed
                    finally:
                        if not self._partial_rollout:
                            ray.get(self.vllm_lock.release.remote())
                    if self.args.train.rollout_dump_dir and rollout_samples:
                        os.makedirs(self.args.train.rollout_dump_dir, exist_ok=True)
                        dump_path = os.path.join(self.args.train.rollout_dump_dir, f"rollout_step{global_step}.pt")
                        torch.save(rollout_samples, dump_path)
                        logger.info(f"[rollout_dump] wrote {len(rollout_samples)} samples to {dump_path}")

                if rollout_samples:
                    client_states = {
                        "episode": ep,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                        "rollout_generator_state_dict": self.samples_generator.state_dict(),
                    }
                    self.rollout_queue.put(
                        (rollout_samples, client_states, rollout_metrics, generation_time, vllm_idle_wait),
                        block=True,
                    )
                    if prompts_consumed:
                        pbar.update(prompts_consumed)
                else:
                    # Nothing enqueued => trainer will never consume this slot.
                    self.rollout_slots.put(global_step, block=True)

                if is_exhausted:
                    break

            pbar.close()

        self.rollout_queue.put("done", block=True)


@ray.remote
class TrainingActor(BaseRLTrainer):
    def __init__(
        self,
        pretrain,
        strategy,
        actor_model_group,
        reference_model_group,
        vllm_engines,
        *,
        vllm_lock,
        rollout_queue,
        rollout_slots,
        critic_model_group=None,
    ):
        tokenizer = get_tokenizer(pretrain, None, "left", use_fast=not strategy.args.data.disable_fast_tokenizer)

        super().__init__(
            strategy,
            actor_model_group,
            reference_model_group,
            vllm_engines,
            tokenizer,
            critic_model_group=critic_model_group,
        )

        self.vllm_lock = vllm_lock
        self._prefix_caching_enabled = getattr(strategy.args.vllm, "enable_prefix_caching", False)
        self.rollout_queue = rollout_queue
        self.rollout_slots = rollout_slots

    def fit(self, global_step: int = 0) -> None:
        step_start_time = time.time()
        self._latest_client_states = {}
        while True:
            # Time the block — in this async split, the trainer stuck here means the
            # actor side is sitting IDLE waiting for a rollout to be produced (e.g. during
            # a long eval, or if generation becomes the bottleneck). True actor-idle signal.
            _queue_wait_t0 = time.time()
            payload = self.rollout_queue.get(block=True)
            actor_idle_wait = time.time() - _queue_wait_t0
            if payload == "done":
                break

            if payload[0] == "eval":
                _, eval_step, eval_metrics = payload
                self.rollout_slots.put(global_step, block=True)
                logger.info(f"Eval at step {eval_step}: {eval_metrics}")
                if self.wandb_logger:
                    self.wandb_logger.log_eval(eval_step, eval_metrics)
                if self.tensorboard_logger:
                    self.tensorboard_logger.log_eval(eval_step, eval_metrics)
                client_states = dict(self._latest_client_states)
                client_states["global_step"] = global_step
                self.save_best_checkpoint(eval_metrics, eval_step, client_states)
                step_start_time = time.time()
                continue

            rollout_samples, client_states, rollout_metrics, generation_time, vllm_idle_wait = payload

            # Batch consumed => free one token to allow generator to produce next batch.
            # --train.force_sync_mode defers this until AFTER train_step (which updates the
            # actor and refits vLLM), so the generator waits for the fresh weights before
            # producing the next batch -> the next rollout is generated with the same
            # weights the trainer recomputes it under (strictly on-policy). This removes the
            # 1-step-stale rollout that inflates vllm_kl on routing-sensitive MoE
            # checkpoints, at the cost of the generate/train overlap. Default off.
            force_sync = getattr(self.args.train, "force_sync_mode", False)
            if not force_sync:
                self.rollout_slots.put(global_step, block=True)

            status, global_step = self.train_step(rollout_samples, global_step)
            if force_sync:
                self.rollout_slots.put(global_step, block=True)
            status["timing/generation"] = generation_time
            # Async idle accounting (the real "which side is wasted" signal): in the split
            # actor/vLLM topology these directly attribute the idle the reaper sees.
            #   vllm_idle_wait  = generator blocked for a train slot  -> vLLM GPUs idle
            #   actor_idle_wait = trainer blocked for a rollout       -> actor GPUs idle
            # gen<<train => vllm_idle_wait large (vLLM wasted); long eval => actor_idle_wait large.
            status["timing/vllm_idle_wait"] = vllm_idle_wait
            status["timing/actor_idle_wait"] = actor_idle_wait
            status["timing/step_total"] = time.time() - step_start_time
            step_start_time = time.time()

            # rollout/dropped/<reason> counts + dynamic_filtering_pass_rate (when enabled).
            status.update(rollout_metrics)

            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"Global step {global_step}: {log_status}")

            client_states.update({"global_step": global_step})
            self._latest_client_states = client_states
            self.save_logs_and_checkpoints(global_step, status, client_states)

        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def broadcast_to_vllm(self):
        # Keep new generation calls out while existing requests are paused and
        # refitted. Report lock wait separately from the weight transfer. The lock
        # release sits in finally so a failed refit (NCCL errors do happen mid-run)
        # crashes cleanly instead of leaving vllm_lock held, which would silently
        # hang the trainer and eval (both acquire it).
        _t0 = time.time()
        ray.get(self.vllm_lock.acquire.remote())
        self._broadcast_lock_wait_s = time.time() - _t0
        _t0 = time.time()
        try:
            batch_vllm_engine_call(self.vllm_engines, "pause_generation")
            super().broadcast_to_vllm()
            if self._prefix_caching_enabled:
                batch_vllm_engine_call(self.vllm_engines, "reset_prefix_cache")
            batch_vllm_engine_call(self.vllm_engines, "resume_generation")
        finally:
            ray.get(self.vllm_lock.release.remote())
            self._broadcast_transfer_s = time.time() - _t0


@ray.remote
class RLTrainer:
    """Async-only RL controller."""

    def __init__(
        self,
        pretrain: str,
        strategy: FsdpStrategy,
        actor_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines,
        critic_model_group: RayActorGroup = None,
        router_url: str = None,
        **generate_kwargs,
    ) -> None:
        if strategy.args.eval.steps == -1:
            strategy.args.eval.steps = float("inf")
        if strategy.args.ckpt.save_steps == -1:
            strategy.args.ckpt.save_steps = float("inf")

        queue_size = getattr(strategy.args.train, "async_queue_size", 1)
        if queue_size <= 0:
            raise ValueError(f"async_queue_size must be positive, got {queue_size}")
        logger.info(f"async_queue_size={queue_size}")

        self.rollout_queue = Queue(maxsize=queue_size)
        self.rollout_slots = Queue(maxsize=queue_size)
        for _ in range(queue_size):
            self.rollout_slots.put(0, block=True)

        vllm_lock = VLLMLock.remote()

        self.generator_actor = GenerateSamplesActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            vllm_lock=vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            router_url=router_url,
            **generate_kwargs,
        )

        self.trainer_actor = TrainingActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            actor_model_group=actor_model_group,
            reference_model_group=reference_model_group,
            vllm_engines=vllm_engines,
            vllm_lock=vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            critic_model_group=critic_model_group,
        )

    def fit(self) -> None:
        checkpoint_states = ray.get(self.trainer_actor.load_checkpoint_states_or_default.remote())
        ray.get(self.trainer_actor.restore_best_metric_tracker.remote(checkpoint_states))

        # .get with defaults: an interrupted save can leave model/ without extra_state.pt, so
        # load_ckpt returns states={} (weights load, scalars empty) — resume at 0, not KeyError.
        start_episode = checkpoint_states.get("episode", 0)
        global_step = checkpoint_states.get("global_step", 0)
        total_consumed_prompts = checkpoint_states.get("total_consumed_prompts", 0)
        if global_step > 0:
            ray.get(
                [
                    self.generator_actor.load_dataloader_state_dict.remote(
                        checkpoint_states["data_loader_state_dict"],
                        checkpoint_states.get("rollout_generator_state_dict"),
                    ),
                    self.trainer_actor.broadcast_to_vllm.remote(),
                ]
            )

        ray.get(
            [
                self.generator_actor.fit.remote(episode=start_episode, total_consumed_prompts=total_consumed_prompts),
                self.trainer_actor.fit.remote(global_step=global_step),
            ]
        )

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
