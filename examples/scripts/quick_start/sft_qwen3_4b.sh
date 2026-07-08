#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single-node quick-start: Qwen3-4B (dense) text SFT on proRL_text_rl.
#
# 8 GPUs on one machine via torchrun --standalone — no slurm.
# Mirrors slurm/sft_qwen3_4b.sh's recipe for single-node use.
#
#   MODEL_PATH=/path/to/Qwen3-4B bash examples/scripts/quick_start/sft_qwen3_4b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/path/to/models/Qwen3/Qwen3-4B-Instruct-2507}"

SFT_DATASET="${SFT_DATASET:-$REPO_ROOT/.tmp/proRL_text_rl/train}"
EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/proRL_text_rl/eval}"
test -e "$SFT_DATASET" || { echo "SFT_DATASET not found: $SFT_DATASET — prepare proRL_text_rl first."; exit 1; }
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-sft-qwen3-4b/run}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
MAX_LEN="${MAX_LEN:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-4096}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" -m molt.cli.train_sft \
  --data.max_len "$MAX_LEN" \
  --data.dataset "$SFT_DATASET" \
  --data.input_key prompt \
  --data.output_key response \
  --data.max_samples "$MAX_SAMPLES" \
  --model.model_name_or_path "$MODEL_PATH" \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps "${SAVE_STEPS:-50}" \
  --logger.logging_steps 1 \
  --eval.dataset "$EVAL_DATASET" \
  --eval.steps "${EVAL_STEPS:-20}" \
  --train.max_epochs "${MAX_EPOCHS:-1}" \
  --train.batch_size "$TRAIN_BATCH_SIZE" \
  --train.micro_batch_size "$MICRO_BATCH_SIZE" \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation "${FSDP_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --fsdp.tp_size "$TP_SIZE" \
  --fsdp.ep_size "$EP_SIZE" \
  --fsdp.cp_size "$CP_SIZE" \
  --model.gradient_checkpoint full \
  --adam.lr "${LR:-1e-6}" \
  --logger.wandb.project "${WANDB_PROJECT:-molt_quickstart_sft_qwen3_4b}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_4b_sft_quickstart_$$}"
