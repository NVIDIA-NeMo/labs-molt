# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared argparse blocks used by ``train_sft`` and ``train_rl_ray``.

Keeping these knobs in one place avoids two-CLI drift where the RL launcher
adds a flag and the SFT launcher silently keeps a stale default — exactly the
``--adam.lr`` vs ``--actor.adam.lr`` default skew this module now pins down.
"""

from datetime import datetime


def add_fsdp_args(parser) -> None:
    """FSDP2 / AutoModel backend args. Identical surface for SFT + RL."""
    parser.add_argument("--fsdp.tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--fsdp.cp_size", type=int, default=1, help="Context parallel size (replaces ring-attn)")
    parser.add_argument("--fsdp.ep_size", type=int, default=1, help="Expert parallel size (MoE)")
    parser.add_argument("--fsdp.pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--fsdp.sequence_parallel",
        action="store_true",
        default=False,
        help="Opt into sequence parallelism within the TP region. Off by default "
        "(matches AutoModel's omni / Qwen3.5-MoE recipes).",
    )
    parser.add_argument(
        "--fsdp.offload",
        type=str,
        default="none",
        choices=["none", "optimizer", "full"],
        help="CPU-offload level — a nested progression, not orthogonal toggles (FSDP "
        "param offload inherently offloads the optimizer too). "
        "'none': everything on GPU. "
        "'optimizer': run the AdamW step on CPU so the fp32 master + Adam moments never "
        "occupy GPU during the step (shrinks the optimizer-step peak, the binding one at "
        "long context); params stay on GPU for the forward, so it's safe on Qwen3.6 MoE; "
        "AdamW only; numerically equivalent to a GPU step. "
        "'full': FSDP2 CPUOffloadPolicy also streams the *params* to CPU (maximal saving, "
        "but breaks Qwen3.6 MoE and slows the forward).",
    )
    parser.add_argument(
        "--fsdp.param_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Model data type"
    )
    parser.add_argument(
        "--fsdp.attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., sdpa, eager, flex, te, flash_attention_2)",
    )
    parser.add_argument("--fsdp.packing_samples", action="store_true", default=False)


def add_ckpt_args(parser, default_ckpt_path: str) -> None:
    """Resumable-checkpoint retention + final HF-export knobs. Shared by SFT + RL.

    Only ``--ckpt.path``'s default differs between the two CLIs (sft vs rl_ray
    subdir), so it is a parameter. RL defines ``--ckpt.best_metric_key`` at its
    own call site (best-checkpoint selection is RL-eval-only).
    """
    parser.add_argument(
        "--ckpt.output_dir",
        type=str,
        default="./ckpt",
        help="Directory for the final consolidated HF safetensors export.",
    )
    parser.add_argument(
        "--ckpt.save_steps", type=int, default=-1, help="Save a resumable checkpoint every N steps; -1 disables it."
    )
    parser.add_argument(
        "--ckpt.save_hf",
        action="store_true",
        default=False,
        help="Also export an HF-format snapshot alongside each resumable checkpoint.",
    )
    parser.add_argument(
        "--ckpt.disable_final_save",
        action="store_true",
        default=False,
        help="Skip the final consolidated HF export after training. Useful for smoke tests.",
    )
    parser.add_argument(
        "--ckpt.path",
        type=str,
        default=default_ckpt_path,
        help="Directory for resumable FSDP/DCP training checkpoints (optimizer + scheduler state).",
    )
    parser.add_argument(
        "--ckpt.max_num", type=int, default=3, help="Keep at most this many resumable checkpoints (older ones pruned)."
    )
    parser.add_argument(
        "--ckpt.max_mem",
        type=float,
        default=float("inf"),
        help="Disk budget in GB for retained checkpoints; inf = no limit.",
    )
    parser.add_argument(
        "--ckpt.load_enable",
        action="store_true",
        default=False,
        help="Resume from the latest checkpoint in --ckpt.path if present.",
    )


def add_optimizer_args(parser, prefix: str = "", default_adam_lr: float = 5e-6) -> None:
    """Optimizer + LR-scheduler + grad-clip block.

    ``prefix`` namespaces the flags: "" for SFT (``--optim``, ``--adam.lr``),
    "actor." for RL (``--actor.optim``, ``--actor.adam.lr``). ``default_adam_lr``
    legitimately diverges by recipe (SFT 5e-6, RL 1e-6) so it is explicit here
    rather than skewing silently across the two launchers.
    """
    g = f"--{prefix}"
    parser.add_argument(f"{g}optim", type=str, default="adam", choices=["adam", "muon"])
    # Muon 2D-weight group
    parser.add_argument(f"{g}muon.lr", type=float, default=0.02, help="LR for the Muon 2D-weight group")
    parser.add_argument(f"{g}muon.momentum", type=float, default=0.95)
    parser.add_argument(
        f"{g}muon.weight_decay", type=float, default=None, help="Weight decay for the Muon 2D-weight group"
    )
    parser.add_argument(f"{g}muon.ns_steps", type=int, default=5, help="Newton-Schulz steps for Muon updates")
    parser.add_argument(f"{g}muon.nesterov", action="store_true", default=True)
    parser.add_argument(f"{g}muon.no_nesterov", dest=f"{prefix}muon.nesterov", action="store_false")
    # AdamW (pure-AdamW when optim=adam; Muon's aux-Adam subgroup when =muon)
    parser.add_argument(f"{g}adam.lr", type=float, default=default_adam_lr)
    parser.add_argument(f"{g}adam.betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument(f"{g}adam.eps", type=float, default=1e-8)
    parser.add_argument(f"{g}adam.weight_decay", type=float, default=0.0)
    # Scheduler
    parser.add_argument(f"{g}lr_scheduler", type=str, default="constant")
    parser.add_argument(f"{g}lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument(f"{g}min_lr_ratio", type=float, default=0.1)
    # Gradient clip
    parser.add_argument(f"{g}max_norm", type=float, default=1.0, help="Gradient clipping")


def add_logger_args(parser, default_wandb_project: str, run_name_prefix: str) -> None:
    """wandb + TensorBoard sinks and the metric-logging cadence. Shared by SFT + RL."""
    parser.add_argument("--logger.logging_steps", type=int, default=1, help="Log metrics every N steps.")
    parser.add_argument("--logger.wandb.key", type=str, default=None)
    parser.add_argument("--logger.wandb.org", type=str, default=None)
    parser.add_argument("--logger.wandb.group", type=str, default=None)
    parser.add_argument("--logger.wandb.project", type=str, default=default_wandb_project)
    parser.add_argument(
        "--logger.wandb.run_name",
        type=str,
        default="%s_%s" % (run_name_prefix, datetime.now().strftime("%m%dT%H:%M")),
    )
    parser.add_argument("--logger.tensorboard_dir", type=str, default=None, help="TensorBoard logging path")
