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

"""Critic (PPO value model) Ray worker.

A first-class sibling to ``PolicyModelActor`` / ``ReferenceModelActor``: its own Ray
actor holding the value model, its own optimizer, and its own value-only training
loop. It is colocated on the actor's GPUs by default (shared placement group) but,
being a separate group, can be disaggregated onto its own GPUs.

The training loop mirrors ``PolicyTrainer``'s window / grad-accumulation /
global-token-mean contract so the value update is DP-invariant and aligned with the
actor's batching — the only loss is the clipped value loss (no vLLM sync, entropy,
or KL). The scalar value head is replicated (not FSDP-wrapped), so its accumulated
gradient is mean-all-reduced over the DP(+CP) group once, right before the step.
"""

import os
import time
from contextlib import ExitStack
from typing import Dict

import ray
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from molt.models import Critic, ValueLoss
from molt.trainer.algorithm.experience import Experience, get_model_parallel_size
from molt.trainer.fsdp import FsdpStrategy
from molt.utils import get_tokenizer
from molt.utils.distributed_util import torch_dist_barrier_and_cuda_sync
from molt.utils.logging_utils import init_logger
from molt.utils.vlm_utils import merge_mm_train_inputs

from ..algorithm import NaiveReplayBuffer
from .actor_group import BaseModelActor

logger = init_logger(__name__)


class CriticTrainer:
    """Value optimization on each critic worker (replay buffer + value loss)."""

    def __init__(
        self,
        strategy,
        critic: Critic,
        critic_optim: Optimizer,
        critic_scheduler,
        micro_train_batch_size: int = 8,
        buffer_cpu_offload: bool = True,
        dataloader_pin_memory: bool = True,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self._defer_grad_sync = os.environ.get("MOLT_DEFER_GRAD_SYNC", "1") == "1"
        self.dataloader_pin_memory = dataloader_pin_memory
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_scheduler = critic_scheduler
        # The critic can fit the value function with more passes per RL step than the actor
        # (--critic.max_epochs); falls back to the shared --train.max_epochs when unset.
        self.max_epochs = getattr(self.args.critic, "max_epochs", None) or self.args.train.max_epochs
        self.value_loss_fn = ValueLoss(value_clip=self.args.critic.value_clip)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            0,
            buffer_cpu_offload,
            dynamic_batch=self.args.train.dynamic_batch_enable,
        )
        # AutoModel's MFU calculator over the value model (same backbone as the
        # actor -> ~same FLOP/token); None if AutoModel/arch unsupported, then we
        # report memory only. The critic is a SEPARATE process colocated on the
        # actor's GPUs, so its peak memory adds to the actor's — reported under a
        # distinct perf/critic_* prefix so neither overwrites the other.
        self._mfu = None
        try:
            from nemo_automodel._transformers.mfu import AutoMFU

            self._mfu = AutoMFU.from_config(self.critic.model, device=torch.cuda.get_device_name())
        except Exception as exc:
            logger.warning(f"perf: critic MFU unavailable ({exc!r}); reporting memory only.")
        torch_dist_barrier_and_cuda_sync()

    def value_train(self) -> Dict[str, float]:
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

        # Perf accounting: peak memory + FLOPs/MFU for this value-optimization phase.
        torch.cuda.reset_peak_memory_stats(device)
        perf_t0 = time.time()
        local_seq_count = 0.0
        local_token_sum = 0.0

        # Token-weighted accumulators for the reported value-loss metrics.
        loss_sum = clip_sum = 0.0
        token_total = 0.0
        last_lr = last_grad_norm = 0.0
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Critic epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            dynamic = self.args.train.dynamic_batch_enable
            accum_steps = self.strategy.accumulated_gradient
            max_steps = len(dataloader)
            if self.args.train.force_on_policy and not dynamic:
                accum_steps = max(max_steps, 1)
            elif not dynamic:
                remainder = max_steps % accum_steps
                if remainder:
                    max_steps -= remainder

            # Same window / global-token-mean contract as PolicyTrainer.policy_train.
            window = []
            for step, experience in enumerate(pbar):
                if step >= max_steps:
                    break
                window.append(experience)
                window_end = (
                    bool(self.replay_buffer.dynamic_optimizer_step[step]) if dynamic else len(window) == accum_steps
                )
                if not window_end:
                    continue
                local_tokens = sum(exp.action_mask.sum() for exp in window)
                batch_num_tokens = self.strategy.global_token_count(local_tokens)
                for idx, exp in enumerate(window):
                    exp.to_device(device)
                    # Full per-sequence lengths drive the FLOP estimate (the forward
                    # processes the whole sequence, not just action tokens).
                    seqlens = exp.attention_mask.sum(dim=-1)
                    local_seq_count += float(seqlens.numel())
                    local_token_sum += float(seqlens.sum())
                    is_optimizer_step = idx == len(window) - 1
                    value_loss, clip_frac, grad_norm = self.training_step(exp, batch_num_tokens, is_optimizer_step)
                    n_tok = float(exp.action_mask.sum().item())
                    loss_sum += value_loss * n_tok
                    clip_sum += clip_frac * n_tok
                    token_total += n_tok
                    last_lr = self.critic_scheduler.get_last_lr()[0]
                    if grad_norm is not None:
                        last_grad_norm = grad_norm
                    if self.args.train.force_on_policy and self.replay_buffer.cpu_offload:
                        exp.to_device(torch.device("cpu"))
                window = []
            assert not window, "critic train window not flushed at epoch end"

        # DP-reduce the token-weighted sums into global means.
        reduced = self.strategy.all_reduce({"loss": loss_sum, "clip": clip_sum, "tokens": token_total}, op="sum")
        tokens = reduced["tokens"] or 1.0
        status = {
            "value_loss": reduced["loss"] / tokens,
            "value_clip_frac": reduced["clip"] / tokens,
            "critic_lr": last_lr,
            "critic_grad_norm": last_grad_norm,
        }
        # perf/critic_* (peak mem + MFU), distinct from the actor's perf/* so the
        # last-wins status merge keeps both; the two peaks add to GPU pressure.
        status.update(
            self.strategy.compute_perf_metrics(
                self._mfu, local_seq_count, local_token_sum, time.time() - perf_t0, prefix="perf/critic_"
            )
        )
        return status

    def training_step(self, experience: Experience, batch_num_tokens, is_optimizer_step: bool):
        self.critic.train()

        multimodal_inputs = {}
        if experience.mm_train_inputs and getattr(self.critic, "is_vlm", False):
            multimodal_inputs = merge_mm_train_inputs(experience.mm_train_inputs, experience.sequences.device)

        cp_context_stack = ExitStack()
        try:
            output = self.critic(
                experience.sequences,
                experience.action_mask,
                attention_mask=experience.attention_mask,
                cp_context_stack=cp_context_stack,
                **multimodal_inputs,
            )
            value_loss, reported_value_loss, value_clip_frac = self.value_loss_fn(
                output["action_values"],
                experience.values,
                experience.returns,
                action_mask=experience.action_mask,
                dp_size=self.strategy.dp_size,
                batch_num_tokens=batch_num_tokens,
            )
            self.strategy.backward(
                value_loss,
                self.critic,
                self.critic_optim,
                name="critic",
                accumulate=not self.args.train.dynamic_batch_enable,
                scale_loss_by_accumulation=False,
                sync_gradients=(is_optimizer_step if self._defer_grad_sync else True),
            )
        finally:
            cp_context_stack.close()

        grad_norm = None
        if is_optimizer_step:
            # The replicated value head is not covered by FSDP's reduce — sync it
            # over the DP(+CP) group before stepping (mean commutes with accum).
            self.strategy.sync_replicated_grads(self.critic.value_head_parameters())
            self.strategy.optimizer_step(
                self.critic_optim, self.critic, self.critic_scheduler, name="critic", accumulate=False
            )
            grad_norm = self.strategy.get_grad_norm(self.critic)

        clip = value_clip_frac.item() if value_clip_frac is not None else 0.0
        return reported_value_loss.item(), clip, grad_norm


@ray.remote(num_gpus=1)
class CriticModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: FsdpStrategy, pretrain, max_steps=None):
        args = strategy.args
        self._setup_distributed(strategy)
        # The scalar value head reads its (replicated) input via _ValueHead.to_local();
        # under sequence parallelism the post-norm hidden is seq-sharded, so to_local()
        # would silently read only the local shard -> wrong V(s). Fail fast until handled.
        assert not getattr(strategy, "sequence_parallel", False), (
            "PPO critic value head is not sequence-parallel-safe (critic.py _ValueHead); "
            "run without --fsdp.sequence_parallel or extend the head to gather the seq dim."
        )

        # Init from the critic checkpoint (a reward model / value model) when given,
        # else from the actor checkpoint. `pretrain` is already the actor path.
        critic_pretrain = args.critic.model_name_or_path or pretrain
        critic = Critic(
            critic_pretrain,
            attn_implementation=args.fsdp.attn_implementation,
            param_dtype=args.fsdp.param_dtype,
            device_mesh=strategy.device_mesh,
            moe_mesh=strategy.moe_mesh,
            distributed_config=strategy.distributed_config,
            moe_config=strategy.moe_config,
            activation_checkpointing=args.actor.gradient_checkpoint,
            packing_samples=args.fsdp.packing_samples,
            temperature=args.rollout.temperature,
            freeze_visual_encoder=getattr(args.actor, "freeze_visual_encoder", False),
            # Critic router freeze: its own --critic.freeze_moe_router, or inherit the actor's flag.
            freeze_moe_router=getattr(args.critic, "freeze_moe_router", False)
            or getattr(args.actor, "freeze_moe_router", False),
            moe_aux_loss_coef=args.actor.aux_loss_coef,
        )
        strategy.print(critic)
        self.tokenizer = get_tokenizer(
            critic_pretrain,
            critic.model,
            "left",
            use_fast=not args.data.disable_fast_tokenizer,
        )

        # Independent critic optimizer / scheduler / grad-clip — the full --critic.*
        # group (add_optimizer_args(prefix="critic.")), so the value model can use its
        # own optimizer kind and LR (PPO critics often want a higher LR than the policy).
        critic_cfg = dict(
            optim=args.critic.optim,
            muon=vars(args.critic.muon),
            adam=vars(args.critic.adam),
            lr_scheduler=args.critic.lr_scheduler,
            lr_warmup_ratio=args.critic.lr_warmup_ratio,
            min_lr_ratio=args.critic.min_lr_ratio,
            max_norm=args.critic.max_norm,
            scheduler_steps=max_steps,
        )
        self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare((critic, critic_cfg))

        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt.path, "_critic")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the critic checkpoint: {ckpt_path}")
            model_only = os.environ.get("LOAD_MODEL_ONLY", "0") == "1"
            _, states = strategy.load_ckpt(
                self.critic.model,
                ckpt_path,
                optimizer=None if model_only else self.critic_optim,
                scheduler=None if model_only else self.critic_scheduler,
            )
            self.checkpoint_states = states

        self.trainer = CriticTrainer(
            strategy,
            self.critic,
            self.critic_optim,
            self.critic_scheduler,
            micro_train_batch_size=args.train.micro_batch_size,
        )

    def fit(self):
        """Train the value model on the replay buffer."""
        torch.cuda.empty_cache()
        self.critic.train()
        status = self.trainer.value_train()
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def forward(self, experience) -> torch.Tensor:
        """Per-token value V(s) on the action span (collection-time old_values) for one rollout
        Experience. reload() first fetches the sample's heavy tensors from the producing runner's
        shared-memory store. Called per sample by execute_batch; the controller attaches values."""
        experience = experience.reload()
        device = torch.cuda.current_device()

        mm_inputs = {}
        if experience.mm_train_inputs and getattr(self.critic, "is_vlm", False):
            mm_inputs = merge_mm_train_inputs(experience.mm_train_inputs, device)

        self.critic.eval()
        with torch.no_grad():
            output = self.critic(
                experience.sequences.to(device),
                experience.action_mask.to(device),
                experience.attention_mask.to(device),
                **mm_inputs,
            )
        self.critic.train()  # reset model state
        return output["action_values"].to("cpu")

    def append(self, experience: Experience):
        # reload() fetches the sample's heavy tensors from the producing runner's shared-memory
        # store (a no-op if already local); mirrors PolicyModelActor.append.
        self.trainer.replay_buffer.append(experience.reload())

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def save_checkpoint(self, tag, client_states=None, metric_value=None, metric_key=None):
        args = self.strategy.args
        # Resumable DCP checkpoint only — the critic is a value model, never an
        # HF-exported servable policy.
        self.strategy.save_ckpt(
            self.critic.model,
            os.path.join(args.ckpt.path, "_critic"),
            tag,
            args.ckpt.max_num,
            args.ckpt.max_mem,
            client_states or {},
            # Forward the actor's eval metric so the critic's retention/pruning
            # (sorted by metric in _prune_checkpoints) makes the SAME keep/drop
            # decisions as the actor — otherwise the critic prunes by recency
            # only and the two checkpoint sets desync (a step the actor keeps for
            # its metric may have its _critic dir pruned).
            metric_value=metric_value,
            metric_key=metric_key,
            optimizer=self.critic_optim,
            scheduler=self.critic_scheduler,
        )
        torch_dist_barrier_and_cuda_sync()
