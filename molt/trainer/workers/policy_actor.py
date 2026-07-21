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

import os
import socket
import time
from contextlib import ExitStack
from dataclasses import fields
from typing import Dict, List

import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from molt.models import Actor, PolicyLoss, agg_loss
from molt.models.utils import compute_approx_kl, masked_mean, split_moe_aux_loss
from molt.trainer.algorithm.experience import Experience, get_model_parallel_size
from molt.trainer.fsdp import FsdpStrategy
from molt.trainer.fsdp.refit import gather_full_param
from molt.utils import get_tokenizer
from molt.utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync
from molt.utils.logging_utils import init_logger
from molt.utils.vlm_utils import merge_mm_train_inputs

from ..algorithm import NaiveReplayBuffer
from .actor_group import BaseModelActor

logger = init_logger(__name__)


class PolicyTrainer:
    """Owns actor-side policy optimization on each Ray/FSDP worker."""

    def __init__(
        self,
        strategy,
        actor: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: List = None,
        **kwargs,
    ):
        """Policy trainer for Ray actor workers.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        # MOLT_DEFER_GRAD_SYNC=1 → defer the FSDP grad reduce-scatter to the last
        # microbatch of the accumulation window (AutoModel's get_sync_ctx /
        # defer_fsdp_grad_sync default: ~1 reduce-scatter/step instead of one per
        # microbatch). Mathematically identical (reduce-scatter is linear over the
        # accumulated grad). Default ON to align with AutoModel's default; it trades
        # comm for higher peak memory, so set =0 for memory-bound runs that OOM.
        self._defer_grad_sync = os.environ.get("MOLT_DEFER_GRAD_SYNC", "1") == "1"
        self.tokenizer = tokenizer
        self.dataloader_pin_memory = dataloader_pin_memory

        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.max_epochs = self.args.train.max_epochs

        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.actor.eps_clip_low_high[0],
            clip_eps_high=self.args.actor.eps_clip_low_high[1],
            dual_clip=self.args.actor.dual_clip,
            loss_mode=self.args.actor.loss_mode,
            is_correction_level=self.args.algo.advantage.is_correction_level,
            is_correction_mode=self.args.algo.advantage.is_correction_mode,
            is_correction_threshold=(
                self.args.algo.advantage.is_correction_threshold
                if self.args.algo.advantage.is_correction_level != "off"
                else None
            ),
        )

        # Add the MoE router load-balancing aux loss only when its coefficient is set.
        self.aux_loss = self.args.actor.aux_loss_coef > 1e-8

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            dynamic_batch=self.args.train.dynamic_batch_enable,
        )

        # Init torch group for weights sync (NCCL only — async-split topology
        # has actor and vLLM on different nodes, so CUDA IPC is not applicable).
        backend = getattr(self.strategy.args.vllm, "sync_backend", "nccl")
        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            self._init_vllm_sync_group(backend)

        # AutoModel's MFU calculator (owns the per-arch FLOP model + device-peak
        # table); None if AutoModel/arch unsupported -> we then report memory only.
        self._mfu = None
        try:
            from nemo_automodel._transformers.mfu import AutoMFU

            self._mfu = AutoMFU.from_config(self.actor.model, device=torch.cuda.get_device_name())
        except Exception as exc:
            logger.warning(f"perf: MFU unavailable ({exc!r}); reporting memory only.")

        torch_dist_barrier_and_cuda_sync()

    def _init_vllm_sync_group(self, backend: str):
        """Create a torch process group between trainer rank 0 and all vLLM engine ranks.

        Layout example (3 engines, TP=4):
            [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
            |train rank|  engine-0  |  engine-1  |   engine-2   |

        One rank per vLLM worker GPU, so an engine occupies
        `tensor_parallel_size * pipeline_parallel_size` consecutive ranks; a
        TP/PP group spanning nodes still contributes one rank per GPU. The packed
        broadcast reaches every worker; each loads only the shards/experts/PP-stage
        layers it owns (vLLM's name-based load_weights skips the rest). FSDP2/TP
        params are materialized before broadcasting.
        """
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        vllm_num_engines = self.strategy.args.vllm.num_engines
        vllm_tensor_parallel_size = self.strategy.args.vllm.tensor_parallel_size
        vllm_pipeline_parallel_size = getattr(self.strategy.args.vllm, "pipeline_parallel_size", 1)
        vllm_data_parallel_size = getattr(self.strategy.args.vllm, "data_parallel_size", 1)
        # One rank per vLLM worker GPU. DP replicates the full (EP-sharded) model,
        # so every replica's TP*PP workers take the weight broadcast too.
        engine_world = vllm_tensor_parallel_size * vllm_pipeline_parallel_size * vllm_data_parallel_size
        world_size = vllm_num_engines * engine_world + 1

        group_name = "molt"
        refs = [
            engine.init_process_group.remote(
                master_address,
                master_port,
                i * engine_world + 1,
                world_size,
                group_name,
                backend=backend,
            )
            for i, engine in enumerate(self.vllm_engines)
        ]
        self._model_update_group = stateless_init_process_group(
            master_address, master_port, 0, world_size, torch.cuda.current_device()
        )

        ray.get(refs)

    def _record_status(self, status, status_list, pbar):
        metrics = status["metrics"]
        weights = status["weights"]
        n_tokens = status["num_action_tokens"]
        n_samples = status["num_samples"]

        # Weighted mean of each metric across DP ranks: scale this rank's value by
        # its local token/sample count, all-reduce the weighted sums together with
        # the counts (carried as "_num_*" so they ride the same reduce), then divide
        # by the reduced totals. weight is None means "report as-is" (e.g. lr) -> kept
        # out of the reduce in last_metrics.
        reduced_status = {"_num_action_tokens": n_tokens, "_num_samples": n_samples}
        last_metrics = {}
        for k, value in metrics.items():
            weight = weights[k]
            if weight is None:
                last_metrics[k] = value
                continue
            scale = n_tokens if weight == "token" else n_samples
            reduced_status[k] = (
                value.float().mean().item() * scale if isinstance(value, torch.Tensor) else value * scale
            )

        reduced_status = self.strategy.all_reduce(reduced_status)

        # n_tokens/n_samples above were this rank's local counts; these are the
        # cross-rank totals that serve as the weighted-mean denominators.
        total_tokens = reduced_status.pop("_num_action_tokens")
        total_samples = reduced_status.pop("_num_samples")
        merged_status = {}
        for k, value in reduced_status.items():
            denom = total_tokens if weights[k] == "token" else total_samples
            merged_status[k] = value / denom

        merged_status.update(last_metrics)
        merged_status["_num_samples"] = total_samples
        merged_status["_num_action_tokens"] = total_tokens
        merged_status["_weights"] = weights

        short_status = {
            "act_loss": merged_status["policy_loss"],
            "reward": merged_status.get("reward", 0),
            "return": merged_status.get("return", 0),
            "gen_len": merged_status.get("response_length", 0),
            "tot_len": merged_status.get("total_length", 0),
            "kl": merged_status.get("kl", 0),
            "act_lr": merged_status.get("actor_lr", 0),
            "grad_norm": merged_status.get("actor_grad_norm", 0),
        }
        if "entropy_loss" in merged_status:
            short_status["ent_loss"] = merged_status["entropy_loss"]

        status_list.append(merged_status)
        pbar.set_postfix(short_status)

    def policy_train(self, kl_ctl: float):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.train.dynamic_batch_enable:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        should_shuffle = get_model_parallel_size(self.args) <= 1 and not self.args.train.dynamic_batch_enable
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=should_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        # Perf accounting: peak memory + FLOPs/MFU for this optimization phase.
        torch.cuda.reset_peak_memory_stats(device)
        perf_t0 = time.time()
        local_seq_count = 0.0
        local_token_sum = 0.0

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            dynamic = self.args.train.dynamic_batch_enable
            force_on_policy = self.args.train.force_on_policy
            accum_steps = self.strategy.accumulated_gradient
            max_steps = len(dataloader)
            if force_on_policy and not dynamic:
                # On-policy: the whole rollout is ONE accumulation window —
                # accumulate every microbatch and step once at the end, so the
                # update is computed from exactly the data the current weights
                # generated. max_epochs is asserted == 1, so this is one step.
                accum_steps = max(max_steps, 1)
            elif not dynamic:
                # Only run complete accumulation windows; partial windows leave
                # gradients live because optimizer_step() has not stepped yet.
                remainder = max_steps % accum_steps
                if remainder:
                    max_steps -= remainder
                    self.strategy.print(
                        f"[PolicyRL] dropping {remainder} trailing actor microbatches "
                        f"(< grad_accum={accum_steps}) to avoid partial gradients."
                    )
            # slime global token-mean: every microbatch of one optimizer-step
            # batch ("window") shares a single token denominator summed over all
            # its microbatches and all DP ranks. Buffer the window, count its
            # action tokens on the full pre-CP mask, then run its microbatches.
            window = []
            for step, experience in enumerate(pbar):
                if step >= max_steps:
                    break
                window.append(experience)
                if dynamic:
                    window_end = bool(self.replay_buffer.dynamic_optimizer_step[step])
                else:
                    window_end = len(window) == accum_steps
                if not window_end:
                    continue
                local_tokens = sum(exp.action_mask.sum() for exp in window)
                batch_num_tokens = self.strategy.global_token_count(local_tokens)
                for idx, exp in enumerate(window):
                    exp.to_device(device)
                    # Full per-sequence lengths drive the FLOP estimate (forward
                    # processes the whole sequence, not just action tokens).
                    seqlens = exp.attention_mask.sum(dim=-1)
                    local_seq_count += float(seqlens.numel())
                    local_token_sum += float(seqlens.sum())
                    is_optimizer_step = idx == len(window) - 1
                    status = self.training_step(exp, kl_ctl, batch_num_tokens, len(window), is_optimizer_step)
                    self._record_status(status, status_list, pbar)
                    if force_on_policy and self.replay_buffer.cpu_offload:
                        # The window spans the whole rollout; offload each
                        # microbatch back to CPU after use so peak GPU holds one
                        # microbatch, not the entire buffer (matters for VLM).
                        exp.to_device(torch.device("cpu"))
                window = []
            assert not window, "actor train window not flushed at epoch end"

        if status_list:
            total_tokens = sum(s["_num_action_tokens"] for s in status_list)
            total_samples = sum(s["_num_samples"] for s in status_list)
            for k in set().union(*(s.keys() for s in status_list)):
                if k in ("_num_samples", "_num_action_tokens", "_weights"):
                    continue
                if k == "actor_grad_norm":
                    vals = [s[k] for s in status_list if k in s]
                    status_mean[k] = sum(vals) / len(vals) if vals else 0.0
                elif k == "actor_lr":
                    vals = [s[k] for s in status_list if k in s]
                    status_mean[k] = vals[-1] if vals else 0.0
                elif status_list[0].get("_weights", {}).get(k) == "token":
                    status_mean[k] = sum(s.get(k, 0) * s["_num_action_tokens"] for s in status_list) / total_tokens
                else:
                    status_mean[k] = sum(s.get(k, 0) * s["_num_samples"] for s in status_list) / total_samples

        status_mean.update(
            self.strategy.compute_perf_metrics(self._mfu, local_seq_count, local_token_sum, time.time() - perf_t0)
        )
        return status_mean

    def training_step(
        self,
        experience: Experience,
        kl_ctl: float,
        batch_num_tokens: torch.Tensor,
        num_microbatches: int,
        is_optimizer_step: bool,
    ) -> Dict[str, object]:
        self.actor.train()

        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs
        rollout_log_probs = experience.rollout_log_probs

        # Stage 1: prepare tensors and optional multimodal inputs.
        multimodal_inputs = {}
        if experience.mm_train_inputs and getattr(self.actor, "is_vlm", False):
            multimodal_inputs = merge_mm_train_inputs(experience.mm_train_inputs, sequences.device)

        # AutoModel CP train context must cover both forward and backward.
        cp_context_stack = ExitStack()

        # Stage 2: forward actor. CP is only an implementation detail here:
        # log-probs are restored to the dense token axis before losses run.
        # The loss uses slime's global token-mean: batch_num_tokens is the action
        # token count of the whole optimizer-step batch (all microbatches, all DP
        # ranks), so FSDP's dp_cp reduce-scatter cannot bias gradients toward DP
        # shards with fewer action tokens. The dp_size scale stays DP-only (CP
        # ranks share the sample); the CP gradient compensation for FSDP's extra
        # dp_cp averaging is applied in FsdpStrategy.backward (loss *= cp_size).
        loss_data_parallel_size = self.strategy.dp_size
        model_output = self.actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            cp_context_stack=cp_context_stack,
            # entropy_coef=0.0 disables the entropy term in loss. Skip the
            # entropy forward path entirely in that case.
            return_entropy=bool(self.args.actor.entropy_coef),
            # R3: replay the rollout's expert selection (None when routing replay off).
            routed_experts=experience.routed_experts,
            **multimodal_inputs,
        )
        action_log_probs = model_output["action_log_probs"]
        if old_action_log_probs is None:
            # force_on_policy: experience_maker skipped the redundant old-logprob forward.
            # The batch is trained for one on-policy step, so old == this forward -> PPO
            # ratio 1 -> REINFORCE gradient; the IS correction still runs vs rollout_log_probs.
            # Taking old FROM this forward also guarantees old and action share the exact
            # R3-replayed routing (they are the same forward) — the importance ratio needs
            # both log-probs computed under the rollout's expert selection.
            old_action_log_probs = action_log_probs.detach()

        # Debug observability: MOLT_DUMP_ROLLOUT_LOGPROBS=<path> dumps per-position
        # token_id / rollout(vLLM) logprob / actor recomputed logprob for the first
        # sequence of the first microbatch (rank 0, once) — the token-level evidence
        # needed to diagnose rollout-vs-actor logprob misalignment (e.g. a one-token
        # shift shows up as actor_logp[i] ~ vllm_logp[i+1]).
        dump_path = os.environ.get("MOLT_DUMP_ROLLOUT_LOGPROBS")
        if dump_path and rollout_log_probs is not None and not getattr(self, "_rollout_logprob_dumped", False):
            self._rollout_logprob_dumped = True
            if torch.distributed.get_rank() == 0:
                with open(dump_path, "w") as f:
                    f.write("pos\ttoken_id\tvllm_logp\tactor_logp\tmask\n")
                    rows = zip(
                        sequences[0, 1:].tolist(),
                        rollout_log_probs[0].float().tolist(),
                        action_log_probs[0].detach().float().tolist(),
                        action_mask[0].long().tolist(),
                    )
                    for j, (t, v, a, m) in enumerate(rows):
                        f.write(f"{j}\t{t}\t{v:.6f}\t{a:.6f}\t{m}\n")
                logger.info(f"MOLT_DUMP_ROLLOUT_LOGPROBS: wrote token-level logprob dump to {dump_path}")

        # Stage 3: compute policy loss and metric-only policy diagnostics.
        # reported_actor_loss is a plain per-token mean for logging, decoupled
        # from the global token-mean used for the gradient (actor_loss).
        actor_loss, reported_actor_loss, clip_ratio, policy_kl, vllm_kl, is_filter_ratio = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=action_mask,
            rollout_log_probs=rollout_log_probs,
            dp_size=loss_data_parallel_size,
            batch_num_tokens=batch_num_tokens,
        )
        experience.info["policy_clip_ratio"] = clip_ratio.detach()
        experience.info["policy_kl"] = policy_kl.detach()
        if vllm_kl is not None:
            experience.info["vllm_kl"] = vllm_kl.detach()
        if is_filter_ratio is not None:
            experience.info["is_filter_ratio"] = is_filter_ratio.detach()

        # Stage 4: add optional KL-as-loss, MoE aux loss, and entropy regularization.
        if self.args.algo.kl.use_loss:
            if self.args.algo.kl.init_coef > 0:
                approx_kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.algo.kl.estimator,
                )
                logprob_diff = torch.nan_to_num(
                    action_log_probs.float() - base_action_log_probs.float(),
                    nan=0.0,
                    posinf=30.0,
                    neginf=-30.0,
                )
            else:
                approx_kl = torch.zeros_like(action_log_probs)
                logprob_diff = torch.zeros_like(action_log_probs)
            kl_loss = agg_loss(
                approx_kl,
                action_mask,
                "token-mean",
                dp_size=loss_data_parallel_size,
                batch_num_tokens=batch_num_tokens,
            )
            mean_logprob_diff = masked_mean(logprob_diff, action_mask)
            # Report a plain per-token mean (kl_loss itself is global-token-mean
            # normalized for the gradient, i.e. a fraction-of-window, not a mean).
            experience.info["kl"] = masked_mean(approx_kl, action_mask).detach()
            experience.info["logprobs_diff"] = mean_logprob_diff.detach()
        else:
            kl_loss = 0

        total_loss = actor_loss + kl_loss * kl_ctl
        # MoE balancing loss. It is a per-microbatch mean, so divide by the
        # window's microbatch count to average it over the optimizer-step batch
        # (the token-level terms above are already whole-batch normalized).
        if self.aux_loss:
            aux_loss, _ = split_moe_aux_loss(model_output, self.aux_loss)
            total_loss += aux_loss * self.args.actor.aux_loss_coef / num_microbatches
        # entropy loss — only computed when entropy is actually wanted; matches
        # the return_entropy gating on the forward call above. With coef=0/None,
        # model_output.entropy was never materialized. entropy_loss is the plain
        # per-token mean reported in metrics; the gradient term uses the same
        # global token-mean as the policy loss.
        if bool(self.args.actor.entropy_coef):
            entropy = model_output.entropy[:, -experience.action_mask.shape[1] :]
            entropy_loss = masked_mean(entropy, action_mask)
            entropy_term = agg_loss(
                entropy,
                action_mask,
                "token-mean",
                dp_size=loss_data_parallel_size,
                batch_num_tokens=batch_num_tokens,
            )
            total_loss -= entropy_term * self.args.actor.entropy_coef
        else:
            entropy_loss = None

        # Stage 5: backward and optimizer step. The global token denominator
        # already normalizes over the whole optimizer-step batch, so the loss
        # must NOT be re-divided by the accumulation count (scale_loss_by_
        # accumulation=False) — same contract as the SFT trainer.
        try:
            self.strategy.backward(
                total_loss,
                self.actor,
                self.actor_optim,
                name="actor",
                accumulate=not self.args.train.dynamic_batch_enable,
                scale_loss_by_accumulation=False,
                # Defer the grad reduce-scatter to the optimizer-step microbatch when
                # enabled; otherwise sync every microbatch (default). See _defer_grad_sync.
                sync_gradients=(is_optimizer_step if self._defer_grad_sync else True),
            )
        finally:
            cp_context_stack.close()

        # The window loop owns the optimizer-step boundary (is_optimizer_step) for
        # every batching mode — grad-accum, dynamic token-budget, and
        # force-on-policy alike. Accumulate until then; step on the final
        # microbatch, where .grad is the fully reduced batch gradient.
        if is_optimizer_step:
            self.strategy.optimizer_step(
                self.actor_optim, self.actor, self.actor_scheduler, name="actor", accumulate=False
            )

        # Stage 6: collect weighted metrics from this microbatch.
        metrics = {"policy_loss": reported_actor_loss.detach()}
        weights = {"policy_loss": "token"}
        if entropy_loss is not None:
            metrics["entropy_loss"] = entropy_loss.detach()
            weights["entropy_loss"] = "token"

        # Action-token-weighted mean of the policy advantages. The aggregate is
        # near zero when batch whitening is enabled; otherwise it is the raw
        # policy-signal mean.
        metrics["advantage_mean"] = masked_mean(advantages, action_mask).detach()
        weights["advantage_mean"] = "token"

        metrics["actor_lr"] = self.actor_scheduler.get_last_lr()[0]
        weights["actor_lr"] = None
        # grad_norm is meaningful only on the optimizer-step microbatch, where
        # .grad is the fully reduced batch gradient.
        if is_optimizer_step:
            # True clip-time norm of the accumulated gradient. Normalization now
            # comes from the global token denominator (whole optimizer-step
            # batch), not a /accum factor, so after accumulation .grad is already
            # the correctly normalized batch gradient. This matches the value
            # clipped against max_norm and the grad_norm convention on wandb.
            metrics["actor_grad_norm"] = self.strategy.get_grad_norm(self.actor)
            weights["actor_grad_norm"] = None

        for k, v in experience.info.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v
                weights[k] = "token" if v.dim() == 0 else "sample"
            elif isinstance(v, list):
                metrics[k] = torch.tensor(v, dtype=torch.float)
                weights[k] = "sample"

        for f in fields(Experience):
            if f.name in {"rewards", "scores"} or not Experience.is_episode_tensor_field(f.name):
                continue
            value = getattr(experience, f.name)
            if isinstance(value, torch.Tensor) and f.name not in metrics:
                metrics[f.name] = value
                weights[f.name] = "sample"

        return {
            "metrics": metrics,
            "weights": weights,
            "num_samples": float(experience.action_mask.shape[0]),
            "num_action_tokens": float(action_mask.sum().item()),
        }

    def broadcast_to_vllm(self):
        torch.cuda.empty_cache()
        # FSDP2/AutoModel: `actor.model` is the FSDP2-wrapped HF model directly
        # (no DS-engine `.module` indirection); params are DTensors when sharded.
        model = self.actor.model

        from torch.distributed.tensor import DTensor

        ep_size = getattr(self.strategy.args.fsdp, "ep_size", 1) or 1

        # Pack many small per-tensor broadcasts into large batched broadcasts.
        # Inspired by vLLM's `vllm.distributed.weight_transfer.packed_tensor`
        # — same idea, simplified (no double-buffer streams) since FSDP
        # `gather_full_param` already serializes on the default stream so
        # overlap gain is small. Cuts a 30B+ MoE refit from thousands of
        # RPC+broadcast pairs to ~tens.
        #
        # Only trainer rank 0 holds `_model_update_group`; non-rank-0 ranks
        # still call `gather_full_param` (an FSDP collective) but drop the
        # gathered tensor immediately — no point staging the batch on every rank.
        is_rank0 = torch.distributed.get_rank() == 0
        # --train.check_weight_update_equal: rank 0 arms the check, evaluated after the last flush.
        check_weight_update = is_rank0 and getattr(self.strategy.args.train, "check_weight_update_equal", False)
        if check_weight_update:
            ray.get([engine.reset_weight_update_check.remote() for engine in self.vllm_engines])
        # 512 MiB flushes, matching slime's `--update-weight-buffer-size` default
        # (512 * 1024**2). vLLM runs at high gpu_memory_utilization (~0.9-0.95) with
        # little free VRAM, and the receiver allocates a contiguous
        # `torch.empty(sum(sizes))` per flush — the old 1 GiB batch OOMed the engine.
        packed_threshold_bytes = 512 * 1024**2  # 512 MiB (slime default)

        pending_metas: list[tuple[str, torch.dtype, tuple[int, ...]]] = []
        pending_tensors: list[torch.Tensor] = []
        pending_bytes = 0

        def _flush():
            nonlocal pending_bytes
            if not pending_metas:
                return
            refs = [engine.update_weights_packed.remote(pending_metas) for engine in self.vllm_engines]
            flat = torch.cat([t.view(torch.uint8).view(-1) for t in pending_tensors], dim=0)
            self._model_update_group.broadcast(flat, src=0, stream=torch.cuda.current_stream())
            ray.get(refs)
            del flat
            pending_metas.clear()
            pending_tensors.clear()
            pending_bytes = 0

        for name, tensor in model.state_dict().items():
            # Refit EVERY state_dict entry (each converted to HF names below). vLLM's
            # load_weights matches by name and ignores what it doesn't have, so the
            # "which weights to accept" decision lives on the vLLM side. We deliberately
            # do NOT pre-filter by a named_parameters requires_grad map: its FQNs differ
            # from state_dict's (custom-AutoModel / FSDP naming), so that filter silently
            # skipped ~all trained weights and left vLLM stuck on the base checkpoint
            # (vllm_kl then grew with training as the actor drifted from the stale engine).
            # Skip TE `_extra_state` (fp8 amax bookkeeping — a uint8 tensor in recent
            # TE, a BytesIO in older) and any other non-tensor. It is never a real
            # weight and vLLM has no param for it; the HF adapter drops it via
            # `exclude_key_regex` anyway (NeMo-RL relies on that same regex). Skip it
            # here so a tensor-valued `_extra_state` can't trip the expert guard below.
            if not torch.is_tensor(tensor) or name.endswith("_extra_state"):
                continue

            # EP-sharded experts must be DTensors so `gather_full_param`'s
            # `full_tensor()` collects every expert across the EP mesh. The
            # default torch_mm experts (GroupedExperts + ExpertParallel) are
            # DTensors, so this holds; but the TE GroupedLinear layout
            # (BackendConfig experts="te") keeps only this rank's local experts
            # as plain tensors, which would silently broadcast just rank-0's
            # slice. The train and rollout engines must share this DTensor
            # invariant — enforce it loudly rather than let their policies diverge.
            if ep_size > 1 and "expert" in name and not isinstance(tensor, DTensor):
                raise RuntimeError(
                    f"Refit: expert weight {name!r} is not a DTensor under ep_size={ep_size}; "
                    "EP-sharded experts would not be gathered (only rank-0's experts reach "
                    "vLLM). Build the MoE with BackendConfig experts='torch_mm' (the default)."
                )

            # Collective; every rank participates.
            weight, _ = gather_full_param(tensor)
            if not is_rank0:
                del weight
                continue
            adapter = getattr(model, "state_dict_adapter", None)
            if adapter is None:
                hf_pairs = [(name, weight)]
            elif getattr(adapter, "convert_single_tensor_to_hf", None) is not None:
                hf_pairs = adapter.convert_single_tensor_to_hf(
                    name, weight, exclude_key_regex=r".*_extra_state.*", quantization=False
                )
            elif getattr(adapter, "to_hf", None) is not None:
                hf_pairs = list(adapter.to_hf({name: weight}, exclude_key_regex=r".*_extra_state.*").items())
            else:
                raise RuntimeError(
                    f"{type(model).__name__} uses an AutoModel state_dict_adapter without "
                    "`convert_single_tensor_to_hf`; vLLM refit cannot safely map this custom "
                    "weight layout to HuggingFace/vLLM names."
                )
            for hf_name, hf_weight in hf_pairs:
                if not torch.is_tensor(hf_weight) or not hf_weight.is_floating_point():
                    continue
                # Dtype-faithful refit: keep each param in its native/compute dtype
                # instead of force-casting everything to a single `param_dtype`. vLLM's
                # `load_weights` casts each tensor to *that param's own* target dtype via
                # `param.data.copy_()`, so an fp32-kept weight (e.g. a fp32 MoE
                # router/gate) round-trips as fp32 rather than being silently
                # bf16-downcast (which previously corrupted routing). For bf16-target
                # params the final vLLM value is identical to the old forced-bf16 path
                # (the cast just happens on the receiver); the only cost is ~2x transfer
                # bytes for fp32 masters — negligible next to the broadcast lock-wait.
                hf_weight = hf_weight.to(
                    device=torch.device("cuda", torch.cuda.current_device()),
                    non_blocking=True,
                ).contiguous()
                nbytes = hf_weight.numel() * hf_weight.element_size()
                # slime's `_chunk_by_size`: flush the accumulated batch BEFORE adding a
                # weight that would take it to/over the buffer size, so each broadcast
                # buffer stays bounded (an oversized lone tensor forms its own batch). The
                # old post-append check let a big tensor land on an already-near-threshold
                # batch, ballooning the receiver's contiguous buffer and OOM-ing vLLM. The
                # trailing `_flush()` after the loop sends the final partial batch.
                if pending_bytes and pending_bytes + nbytes >= packed_threshold_bytes:
                    _flush()
                pending_metas.append((hf_name, hf_weight.dtype, tuple(hf_weight.shape)))
                pending_tensors.append(hf_weight)
                pending_bytes += nbytes
            del weight

        if is_rank0:
            _flush()

        if check_weight_update:
            per_engine = ray.get([e.weight_update_missing.remote() for e in self.vllm_engines])
            missing = sorted({name for lst in per_engine for name in lst})
            if missing:
                logger.warning(
                    f"[check_weight_update] {len(missing)} vLLM params NOT refreshed by this broadcast "
                    f"(stale rollout weights); sample: {missing[:10]}"
                )
            else:
                logger.info("[check_weight_update] all vLLM params refreshed by this broadcast")

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: FsdpStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.ckpt.save_hf
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        # Only set NCCL_CUMEM_ENABLE=0 on vLLM < 0.16; on >= 0.16 it causes
        # ncclCommInitRank to fail ("unhandled cuda error") under NCCL 2.27+.
        if getattr(args.vllm, "sync_backend", "nccl") == "nccl":
            import vllm
            from packaging import version as pkg_version

            if pkg_version.parse(vllm.__version__) < pkg_version.parse("0.16"):
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.fsdp.attn_implementation,
            param_dtype=strategy.args.fsdp.param_dtype,
            device_mesh=strategy.device_mesh,
            moe_mesh=strategy.moe_mesh,
            distributed_config=strategy.distributed_config,
            moe_config=strategy.moe_config,
            activation_checkpointing=args.actor.gradient_checkpoint,
            packing_samples=strategy.args.fsdp.packing_samples,
            temperature=strategy.args.rollout.temperature,
            freeze_visual_encoder=getattr(strategy.args.actor, "freeze_visual_encoder", False),
            freeze_moe_router=getattr(strategy.args.actor, "freeze_moe_router", False),
            moe_aux_loss_coef=args.actor.aux_loss_coef,
            routing_replay=getattr(args.train, "routing_replay", False),
        )
        if vllm_engines is not None:
            adapter = getattr(actor.model, "state_dict_adapter", None)
            if (
                adapter is not None
                and getattr(adapter, "convert_single_tensor_to_hf", None) is None
                and getattr(adapter, "to_hf", None) is None
            ):
                raise RuntimeError(
                    f"{type(actor.model).__name__} uses an AutoModel state_dict_adapter without "
                    "`convert_single_tensor_to_hf`; vLLM refit cannot safely map this custom "
                    "weight layout to HuggingFace/vLLM names."
                )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", use_fast=not strategy.args.data.disable_fast_tokenizer
        )

        actor_cfg = dict(
            optim=args.actor.optim,
            muon=vars(args.actor.muon),
            adam=vars(args.actor.adam),
            lr_scheduler=args.actor.lr_scheduler,
            lr_warmup_ratio=args.actor.lr_warmup_ratio,
            min_lr_ratio=args.actor.min_lr_ratio,
            max_norm=args.actor.max_norm,
            scheduler_steps=max_steps,
        )
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare((actor, actor_cfg))

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt.path, "_actor")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            # LOAD_MODEL_ONLY=1: skip optim/scheduler state to allow switching optimizer kinds
            # (e.g. Adam ckpt → Muon resume). LambdaLR's strict-zip on get_lr breaks when
            # base_lrs (loaded, 1 entry for Adam) mismatches lr_lambdas (fresh, 4 entries for Muon).
            model_only = os.environ.get("LOAD_MODEL_ONLY", "0") == "1"
            _, states = strategy.load_ckpt(
                self.actor.model,
                ckpt_path,
                optimizer=None if model_only else self.actor_optim,
                scheduler=None if model_only else self.actor_scheduler,
            )
            self.checkpoint_states = states

        # configure Trainer
        self.trainer = PolicyTrainer(
            strategy,
            self.actor,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.train.micro_batch_size,
            tokenizer=self.tokenizer,
            vllm_engines=self.vllm_engines,
        )

    def fit(self, kl_ctl: float = 0, train: bool = True):
        """Train actor model with the replay buffer.

        ``train=False`` (critic warmup / actor freeze) skips the policy update but
        still drains the replay buffer, so it does not accumulate across the frozen
        steps while the critic keeps training on the same rollouts.
        """
        torch.cuda.empty_cache()
        status = {}
        if train:
            self.actor.train()
            status = self.trainer.policy_train(kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def export_hf_model(self):
        """Export HF safetensors snapshot to ckpt.output_dir.

        This is NOT a resumable checkpoint — for that, see ``save_checkpoint``
        (DCP rolling save). Called once at end of training to produce a
        deployable HF model directory.
        """
        args = self.strategy.args

        self.strategy.save_model(
            self.actor,
            self.tokenizer,
            args.ckpt.output_dir,
        )

    def forward(self, experience) -> torch.Tensor:
        """Old actor action log-probs for one rollout Experience — the old log-probs used off-policy
        and the student side of the KL-as-reward path (on_policy_distill / reinforce-KL). reload()
        first fetches the sample's heavy tensors (token ids / images / routing) from the producing
        runner's shared-memory store — they reach this rank straight from the runner, never through
        the controller. Called per sample by execute_batch; the controller attaches the result as
        action_log_probs."""
        experience = experience.reload()
        device = torch.cuda.current_device()

        # VLM: merge pre-processed multimodal inputs.
        mm_inputs = {}
        if experience.mm_train_inputs and getattr(self.actor, "is_vlm", False):
            mm_inputs = merge_mm_train_inputs(experience.mm_train_inputs, device)

        routed_experts = experience.routed_experts
        self.actor.eval()
        with torch.no_grad():
            output = self.actor(
                experience.sequences.to(device),
                experience.action_mask.to(device),
                experience.attention_mask.to(device),
                # R3: replay rollout routing so old picks the same experts as training.
                routed_experts=routed_experts.to(device) if routed_experts is not None else None,
                **mm_inputs,
            )
        self.actor.train()  # reset model state
        return output["action_log_probs"].to("cpu")

    def broadcast_to_vllm(self):
        self.trainer.broadcast_to_vllm()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        # reload() fetches the sample's heavy tensors (images / token ids / routing) from the
        # producing runner's shared-memory store — they reach this rank straight from the runner,
        # never through the controller. A no-op for an already-local experience.
        self.trainer.replay_buffer.append(experience.reload())

    def save_checkpoint(self, tag, client_states=None, metric_value=None, metric_key=None):
        args = self.strategy.args
        client_states = client_states or {}
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt.path, "_actor"),
            tag,
            args.ckpt.max_num,
            args.ckpt.max_mem,
            client_states,
            metric_value=metric_value,
            metric_key=metric_key,
            optimizer=self.actor_optim,
            scheduler=self.actor_scheduler,
        )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt.path, f"{tag}_hf")
            self.strategy.save_model(
                self.actor,
                self.tokenizer,
                save_path,
            )
        torch_dist_barrier_and_cuda_sync()
