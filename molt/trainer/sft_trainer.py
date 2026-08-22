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
from functools import partial

import torch
from nemo_automodel._transformers.utils import resolve_get_rope_index
from nemo_automodel.components.datasets.datum import collate_datums, collate_vlm_datums
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.engine import Engine
from torch.optim import Optimizer
from tqdm import tqdm

from molt.utils.distributed_sampler import DistributedSampler


class SFTTrainer:
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        max_epochs: int = 2,
        tokenizer=None,
        save_hf_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.save_hf_ckpt = save_hf_ckpt
        clip_norm = max_norm if max_norm and max_norm > 0 else None
        raw_model = model.model
        processor = tokenizer if hasattr(tokenizer, "image_processor") else None
        packing_samples = strategy.args.fsdp.packing_samples
        if packing_samples and getattr(model, "_packing_style", "automodel") != "automodel":
            raise NotImplementedError(
                "Engine THD packing requires an AutoModel-native THD model; the Hugging Face fallback's "
                "FlashAttention packing adapter lives in Molt's legacy wrapper and is not used by Engine."
            )
        mesh_names = getattr(strategy.device_mesh, "mesh_dim_names", ()) or ()
        cp_size = strategy.device_mesh["cp"].size() if "cp" in mesh_names else 1
        if processor is not None:
            get_rope_index = resolve_get_rope_index(raw_model) if packing_samples else None
            if packing_samples and cp_size > 1 and get_rope_index is not None:
                raise NotImplementedError(
                    "AutoModel does not yet support multi-axis mRoPE with packed THD context parallelism; "
                    "use cp_size=1 or disable VLM packing."
                )
            if (
                packing_samples
                and cp_size > 1
                and not bool(getattr(raw_model, "supports_cp_with_sequence_packing", False))
            ):
                raise NotImplementedError(
                    f"{type(raw_model).__name__} does not support VLM sequence packing with "
                    f"context parallelism (cp_size={cp_size}) on its active attention backend."
                )
            collate_fn = partial(
                collate_vlm_datums,
                processor=processor,
                packed=packing_samples,
                get_rope_index=get_rope_index,
                sequence_alignment=2 * cp_size if packing_samples and cp_size > 1 else 1,
            )
        else:
            collate_fn = partial(collate_datums, packed=packing_samples)
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        padding_token_id = getattr(text_tokenizer, "pad_token_id", None)
        if padding_token_id is None:
            padding_token_id = getattr(getattr(raw_model, "config", None), "pad_token_id", None) or 0
        self.engine = Engine(
            raw_model,
            device=next(raw_model.parameters()).device,
            mesh_context=MeshContext.from_meshes(strategy.device_mesh, strategy.moe_mesh),
            microbatch_size=strategy.args.train.micro_batch_size,
            collate_fn=collate_fn,
            padding_token_id=padding_token_id,
            defer_fsdp_grad_sync=os.environ.get("MOLT_DEFER_GRAD_SYNC", "1") == "1",
            optimizers=optim,
            max_grad_norm=clip_norm,
        )
        self.loss_fn = MaskedCrossEntropy(reduction="sum")
        self.strategy.print("[SFT] backend=engine")

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.logger.wandb.key and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.logger.wandb.key)
            wandb.init(
                entity=strategy.args.logger.wandb.org,
                project=strategy.args.logger.wandb.project,
                group=strategy.args.logger.wandb.group,
                name=strategy.args.logger.wandb.run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.logger.tensorboard_dir and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.logger.tensorboard_dir, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.logger.tensorboard_dir, strategy.args.logger.wandb.run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def _engine_loss(self, output, loss_inputs):
        if not torch.is_tensor(output):
            output = output["logits"] if isinstance(output, dict) else output.logits
        return self.loss_fn(output, loss_inputs["labels"])

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # Infer num_update_steps_per_epoch from dataloader if not provided
        if num_update_steps_per_epoch is None:
            num_update_steps_per_epoch = len(self.train_dataloader) // self.strategy.accumulated_gradient
        if num_update_steps_per_epoch <= 0:
            raise ValueError(
                f"num_update_steps_per_epoch must be positive, got {num_update_steps_per_epoch}. "
                "Check that your dataset is not smaller than train_batch_size."
            )

        # get eval and save steps
        if args.eval.steps == -1:
            args.eval.steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.ckpt.save_steps == -1:
            args.ckpt.save_steps = float("inf")  # do not save ckpt

        # Restore the completed optimizer-step count and epoch boundary.
        completed_steps = consumed_samples // args.train.batch_size
        start_epoch = completed_steps // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train.batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            accum_window = []
            accum_microbatches = 0
            accum_steps = self.strategy.accumulated_gradient
            for batch in self.train_dataloader:
                accum_window.extend(batch)
                accum_microbatches += 1
                if accum_microbatches < accum_steps:
                    continue

                window_size = accum_microbatches
                result = self.engine.forward_backward(accum_window, self._engine_loss)
                self.strategy._maybe_debug_grad_stats(self.model, "model")
                optim_result = self.engine.optim_step()
                self.scheduler.step()

                logs_dict = {
                    "sft_loss": result.loss.item(),
                    "lr": self.engine.optimizers[0].param_groups[0]["lr"],
                    "grad_norm": float(optim_result.grad_norm),
                }
                step_bar.set_postfix(logs_dict)
                step_bar.update(window_size)
                accum_window = []
                accum_microbatches = 0

                completed_steps += 1
                global_step = completed_steps
                client_states = {"consumed_samples": global_step * args.train.batch_size}
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

            # Preserve the configured optimizer-window boundary across epochs.
            if accum_microbatches:
                self.strategy.print(
                    f"[SFT] dropping {accum_microbatches} trailing microbatches "
                    f"(< accum_steps={accum_steps}) at end of epoch."
                )

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict=None, client_states=None):
        logs_dict = logs_dict or {}
        client_states = client_states or {}
        if global_step % args.logger.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval — an empty eval_dataloader would zero-divide inside evaluate()
        if global_step % args.eval.steps == 0 and self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.ckpt.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model,
                args.ckpt.path,
                tag,
                args.ckpt.dcp_max_num,
                args.ckpt.max_mem,
                client_states,
                optimizer=self.engine.optimizers[0],
                scheduler=self.scheduler,
            )
            if self.save_hf_ckpt:
                hf_root = os.path.join(args.ckpt.path, "_hf")
                self.strategy.save_model(self.model, self.tokenizer, os.path.join(hf_root, tag))
                self.strategy.prune_checkpoints(hf_root, tag, args.ckpt.max_num, args.ckpt.max_mem)

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        try:
            loss_sum = None
            token_sum = None
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            with torch.no_grad():
                for batch in eval_dataloader:
                    result = self.engine.forward(batch, self._engine_loss)
                    batch_loss_sum = result.loss_sum
                    batch_token_sum = result.weight_sum
                    loss_sum = batch_loss_sum if loss_sum is None else loss_sum + batch_loss_sum
                    token_sum = batch_token_sum if token_sum is None else token_sum + batch_token_sum
                    step_bar.update()
                    step_bar.set_postfix({"eval sft_loss": (batch_loss_sum / batch_token_sum.clamp_min(1)).item()})

            if loss_sum is None or token_sum is None:
                raise ValueError("evaluation dataloader produced no batches")
            loss_sum, token_sum = self.strategy.all_reduce(torch.stack((loss_sum, token_sum)), op="sum")
            if token_sum.item() <= 0:
                raise ValueError("evaluation produced no supervised tokens")
            last_logs = {"eval sft_loss": (loss_sum / token_sum).item()}
            step_bar.set_postfix(last_logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    wandb_logs = {"eval/%s" % k: v for k, v in {**last_logs, "global_step": steps}.items()}
                    self._wandb.log(wandb_logs)
                elif self._tensorboard is not None:
                    for k, v in last_logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        finally:
            self.model.train()
