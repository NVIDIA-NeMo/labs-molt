# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

import argparse
import math
import os


def train(args):
    from molt.datasets import SFTDataset
    from molt.datasets.utils import blending_datasets
    from molt.models import Actor
    from molt.trainer.sft_trainer import SFTTrainer
    from molt.utils import get_strategy, get_tokenizer

    strategy = get_strategy(args)
    strategy.setup_distributed()

    model = Actor(
        args.model.model_name_or_path,
        attn_implementation=args.fsdp.attn_implementation,
        param_dtype=args.fsdp.param_dtype,
        device_mesh=strategy.device_mesh,
        moe_mesh=strategy.moe_mesh,
        distributed_config=strategy.distributed_config,
        moe_config=strategy.moe_config,
        activation_checkpointing=args.model.gradient_checkpoint,
        packing_samples=args.fsdp.packing_samples,
        freeze_visual_encoder=args.model.freeze_visual_encoder,
        moe_aux_loss_coef=args.model.aux_loss_coef,
    )
    tokenizer = get_tokenizer(
        args.model.model_name_or_path, model.model, "right", use_fast=not args.data.disable_fast_tokenizer
    )
    strategy.print(model)

    train_data = blending_datasets(
        args.data.dataset,
        args.data.dataset_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.dataset_split,
    )
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.data.max_len,
        strategy,
        image_key=args.data.image_key,
        max_images_per_prompt=args.data.max_images_per_prompt,
        train_on_last_turn_only=args.data.train_on_last_turn_only,
    )
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.train.micro_batch_size,
        True,
        True,
        train_dataset.collate_fn,
        num_workers=args.data.dataloader_num_workers,
    )

    eval_dataloader = None
    if getattr(args.eval, "dataset", None):
        eval_data = blending_datasets(
            args.eval.dataset,
            None,
            strategy,
            dataset_split=args.eval.split,
        )
        eval_dataset = SFTDataset(
            eval_data,
            tokenizer,
            args.data.max_len,
            strategy,
            image_key=args.data.image_key,
            max_images_per_prompt=args.data.max_images_per_prompt,
            train_on_last_turn_only=args.data.train_on_last_turn_only,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.train.micro_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
            num_workers=args.data.dataloader_num_workers,
        )

    num_update_steps_per_epoch = len(train_dataset) // args.train.batch_size
    max_steps = math.ceil(args.train.max_epochs * num_update_steps_per_epoch)

    cfg = dict(
        optim=args.optim,
        muon=vars(args.muon),
        adam=vars(args.adam),
        lr_scheduler=args.lr_scheduler,
        lr_warmup_ratio=args.lr_warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        max_norm=args.max_norm,
        scheduler_steps=max_steps,
    )
    model, optim, scheduler = strategy.prepare((model, cfg))

    consumed_samples = 0
    if args.ckpt.load_enable and os.path.exists(args.ckpt.path):
        load_path, states = strategy.load_ckpt(model, args.ckpt.path, optimizer=optim, scheduler=scheduler)
        if load_path is not None:
            consumed_samples = states.get("consumed_samples", 0)
            strategy.print(f"Loaded the checkpoint: {args.ckpt.path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.ckpt.output_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        batch_size=args.train.batch_size,
        max_epochs=args.train.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.ckpt.save_hf,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)
    if not args.ckpt.disable_final_save:
        strategy.save_model(model, tokenizer, args.ckpt.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from molt.cli.common_args import add_ckpt_args, add_fsdp_args, add_logger_args, add_optimizer_args

    # ====================== Shared blocks (same surface as train_rl_ray) ======================
    # FSDP2 / AutoModel backend.
    add_fsdp_args(parser)
    # Optimizer + scheduler + grad clip (unprefixed flags + lr default 5e-6).
    add_optimizer_args(parser, prefix="", default_adam_lr=5e-6)
    # Checkpoints; SFT evaluates once per epoch when --eval.steps is -1.
    add_ckpt_args(parser, default_ckpt_path="./ckpt/checkpoints_sft")
    # wandb + TensorBoard + logging cadence.
    add_logger_args(parser, default_wandb_project="molt_train_sft", run_name_prefix="sft")

    # ====================== SFT-specific arguments ======================
    # Model
    parser.add_argument("--model.model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--model.gradient_checkpoint",
        nargs="?",
        const="full",
        default="full",
        help="Activation-checkpointing mode (string): 'full' = full-block AC (AutoModel "
        "recipe default), 'selective' = TorchTitan per-op AC, 'none'/'off'/'' = disable.",
    )
    parser.add_argument("--model.aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--model.freeze_visual_encoder",
        action="store_true",
        default=False,
        help="VLM only: freeze the vision encoder + projector and train the language backbone only "
        "(AutoModel's finetune recipe defaults to freezing the vision tower). cp_size>1 forces this on.",
    )

    # Data
    parser.add_argument("--data.dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument(
        "--data.dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets"
    )
    parser.add_argument("--data.dataset_split", type=str, default="train")
    parser.add_argument("--data.max_samples", type=int, default=1000000, help="Maximum number of samples to use")
    parser.add_argument("--data.max_len", type=int, default=2048, help="Max total sequence length (prompt + response)")
    parser.add_argument("--data.input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument(
        "--data.output_key", type=str, default=None, help="Dataset column holding the assistant reply (SFT target)."
    )
    parser.add_argument("--data.tokenizer_chat_template", type=str, default=None)
    parser.add_argument(
        "--data.train_on_last_turn_only",
        action="store_true",
        default=False,
        help="Supervise only the last assistant turn (per-turn-flattened SFT; history assistants are context).",
    )
    parser.add_argument(
        "--data.image_key",
        type=str,
        default=None,
        help="Dataset key for per-sample image list. Set to enable VLM SFT; leave None for text-only.",
    )
    parser.add_argument(
        "--data.max_images_per_prompt",
        type=int,
        default=4,
        help="Cap on images per sample; over-budget samples are filtered out.",
    )
    parser.add_argument("--data.disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--data.dataloader_num_workers", type=int, default=0, help="Number of dataloader workers for IO"
    )

    # Training (loop hyperparameters; optimizer/scheduler/grad-clip are in the shared block above)
    parser.add_argument("--train.micro_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train.batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--train.max_epochs", type=int, default=2)
    parser.add_argument("--train.seed", type=int, default=42)
    parser.add_argument(
        "--train.full_determinism_enable",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )

    # Eval
    parser.add_argument("--eval.dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--eval.split", type=str, default="train")
    parser.add_argument(
        "--eval.steps", type=int, default=-1, help="Evaluate every N steps; -1 evaluates once per epoch."
    )

    # Runtime / misc
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank from torchrun")
    parser.add_argument("--use_ms", action="store_true", default=False, help="Resolve models from ModelScope hub.")

    args = parser.parse_args()
    from molt.utils.config import hierarchize

    args = hierarchize(args)

    # ============================ Validate arguments ============================
    # --- Required inputs ---
    if not args.model.model_name_or_path:
        raise ValueError("--model.model_name_or_path is required")

    if not args.data.dataset:
        raise ValueError("--data.dataset is required")

    # --- Parallelism / FSDP ---
    if args.fsdp.pp_size > 1:
        raise NotImplementedError("Molt trainers are not pipeline-parallel aware yet; set --fsdp.pp_size 1")

    if args.data.image_key and args.fsdp.packing_samples:
        raise ValueError(
            "VLM SFT does not support --fsdp.packing_samples (packing is text-only here); "
            "use --fsdp.cp_size with AutoModel TE native CP for long VLM sequences instead."
        )

    if args.fsdp.cp_size > 1 and args.fsdp.packing_samples:
        raise ValueError(
            "--fsdp.cp_size > 1 is incompatible with --fsdp.packing_samples; disable packing for CP runs."
        )

    if args.fsdp.packing_samples and args.fsdp.attn_implementation not in {"te", "flash_attention_2"}:
        raise ValueError("--fsdp.packing_samples requires --fsdp.attn_implementation te or flash_attention_2.")

    # --- Runtime ---
    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        patch_hub()

    train(args)
