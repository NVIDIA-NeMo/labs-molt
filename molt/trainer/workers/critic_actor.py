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

The training loop converts each replay-buffer microbatch into an AutoModel Datum.
Engine owns the complete accumulation window, global-token normalization, backward,
FSDP finalization, clipping, and optimizer update. The scalar value head is installed
before FSDP and follows the same lifecycle as the backbone.
"""

import os
import time
from typing import Dict

import ray
import torch
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.engine import Engine, LossFnOutputBatch, PerTokenOutput
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from molt.models import Critic, ValueLoss
from molt.trainer.algorithm.experience import Experience, get_model_parallel_size
from molt.trainer.fsdp import FsdpStrategy
from molt.trainer.fsdp.packing import unshard_dtensor
from molt.utils import get_tokenizer
from molt.utils.distributed_util import torch_dist_barrier_and_cuda_sync
from molt.utils.logging_utils import init_logger

from ..algorithm import NaiveReplayBuffer
from .actor_group import BaseModelActor
from .engine_utils import (
    extract_model_logits,
    prepare_rl_engine_datum,
    resolve_rl_engine_collation,
    run_rl_engine_forward,
)

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
        tokenizer=None,
        dataloader_pin_memory: bool = True,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self._defer_grad_sync = os.environ.get("MOLT_DEFER_GRAD_SYNC", "1") == "1"
        self.tokenizer = tokenizer
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
        raw_model = self.critic.model
        padding_token_id = getattr(getattr(raw_model, "config", None), "pad_token_id", None) or 0
        max_grad_norm = self.args.critic.max_norm
        collate_fn, engine_microbatch_size = resolve_rl_engine_collation(
            self.critic, self.tokenizer, strategy, micro_train_batch_size
        )
        self.engine = Engine(
            raw_model,
            device=torch.device("cuda", torch.cuda.current_device()),
            mesh_context=MeshContext.from_meshes(strategy.device_mesh, strategy.moe_mesh),
            microbatch_size=engine_microbatch_size,
            collate_fn=collate_fn,
            padding_token_id=padding_token_id,
            batch_context_fn=getattr(self.critic, "_routing_replay_adapter", None),
            defer_fsdp_grad_sync=self._defer_grad_sync,
            optimizers=self.critic_optim,
            max_grad_norm=max_grad_norm if max_grad_norm and max_grad_norm > 0 else None,
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
                prepared_window = []
                for exp in window:
                    # Full per-sequence lengths drive the FLOP estimate (the forward
                    # processes the whole sequence, not just action tokens).
                    seqlens = exp.attention_mask.sum(dim=-1)
                    local_seq_count += float(seqlens.numel())
                    local_token_sum += float(seqlens.sum())
                    prepared_window.append(
                        prepare_rl_engine_datum(
                            exp,
                            self.critic,
                            loss_fields={"old_values": exp.values, "returns": exp.returns},
                            routed_experts=exp.routed_experts,
                        )
                    )

                engine_datums = [datum for prepared in prepared_window for datum in prepared.datums]
                if self.dataloader_pin_memory and window[0].sequences.device.type == "cpu":
                    for datum in engine_datums:
                        datum.pin_memory()
                result = self.engine.forward_backward(
                    engine_datums,
                    self._engine_loss,
                    microbatch_sizes=[prepared.num_datums for prepared in prepared_window],
                )
                self.strategy._maybe_debug_grad_stats(self.critic, "critic")
                optim_result = self.engine.optim_step()
                # Transformers LambdaLR.step() takes an absolute epoch when passed
                # an argument; AutoModel schedulers use step(1) as an increment.
                self.critic_scheduler.step()
                last_grad_norm = float(optim_result.grad_norm)
                last_lr = self.critic_scheduler.get_last_lr()[0]

                expected_outputs = sum(prepared.num_datums for prepared in prepared_window)
                if len(result.loss_fn_outputs) != expected_outputs:
                    raise RuntimeError(
                        f"Critic Engine returned {len(result.loss_fn_outputs)} outputs for {expected_outputs} Datums"
                    )
                output_offset = 0
                for exp, prepared in zip(window, prepared_window):
                    output_records = result.loss_fn_outputs[output_offset : output_offset + prepared.num_datums]
                    output_offset += prepared.num_datums
                    exp.to_device(device)
                    action_values = prepared.restore_token_outputs(
                        [record["action_values"] for record in output_records]
                    )
                    _, reported_value_loss, value_clip_frac = self.value_loss_fn(
                        action_values,
                        exp.values,
                        exp.returns,
                        action_mask=exp.action_mask,
                    )
                    n_tok = float(exp.action_mask.sum().item())
                    loss_sum += float(reported_value_loss) * n_tok
                    clip_sum += (float(value_clip_frac) if value_clip_frac is not None else 0.0) * n_tok
                    token_total += n_tok
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

    def _engine_loss(self, output, loss_inputs):
        values = unshard_dtensor(extract_model_logits(output)).squeeze(-1).float()
        weights = loss_inputs["weights"]
        local_tokens = weights.sum()
        value_loss, _, _ = self.value_loss_fn(
            values,
            loss_inputs["old_values"],
            loss_inputs["returns"],
            action_mask=weights.bool(),
            dp_size=1,
            batch_num_tokens=local_tokens,
        )
        # ValueLoss returns a local token mean. Engine expects a scalar local
        # weighted numerator and applies the one global window denominator.
        numerator = value_loss * local_tokens
        return numerator, LossFnOutputBatch(
            per_token={"action_values": PerTokenOutput(values * weights, fill_value=0.0)}
        )


@ray.remote(num_gpus=1)
class CriticModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: FsdpStrategy, pretrain, max_steps=None):
        args = strategy.args
        # Init from the critic checkpoint (a reward model / value model) when given,
        # else from the actor checkpoint. `pretrain` is already the actor path.
        critic_pretrain = args.critic.model_name_or_path or pretrain
        if getattr(args.train, "routing_replay", False) and critic_pretrain != pretrain:
            raise ValueError("critic routing replay requires the critic to use the actor checkpoint")
        self._setup_distributed(strategy)
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
            routing_replay=getattr(args.train, "routing_replay", False),
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
        critic_cfg = {
            "optim": args.critic.optim,
            "muon": vars(args.critic.muon),
            "adam": vars(args.critic.adam),
            "lr_scheduler": args.critic.lr_scheduler,
            "lr_warmup_ratio": args.critic.lr_warmup_ratio,
            "min_lr_ratio": args.critic.min_lr_ratio,
            "max_norm": args.critic.max_norm,
            "scheduler_steps": max_steps,
        }
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
            tokenizer=self.tokenizer,
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
        prepared = prepare_rl_engine_datum(
            experience,
            self.critic,
            loss_fields={},
            routed_experts=experience.routed_experts,
        )
        self.critic.eval()
        try:
            output = run_rl_engine_forward(
                self.trainer.engine,
                prepared,
                "action_values",
                lambda model_output, _loss_inputs: (
                    unshard_dtensor(extract_model_logits(model_output)).squeeze(-1).float()
                ),
            )
        finally:
            self.critic.train()
        return output.to("cpu")

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
            args.ckpt.dcp_max_num,
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
