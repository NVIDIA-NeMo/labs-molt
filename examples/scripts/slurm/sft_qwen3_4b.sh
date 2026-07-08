#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=your_slurm_account
#SBATCH --partition=interactive
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=molt-sft-qwen3-4b
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

# Qwen3-4B (dense) text-only SFT on the proRL_text_rl math set. Same
# dataset family used by the matching RL recipe (rl_qwen3_4b.sh) so SFT
# checkpoints can be passed straight into RL.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$REPO_ROOT/images/molt-cu13.sqsh}"

MODEL_PATH="${MODEL_PATH:-/path/to/models/Qwen3/Qwen3-4B-Instruct-2507}"

SFT_DATASET="${SFT_DATASET:-$REPO_ROOT/.tmp/proRL_text_rl/train}"
EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/proRL_text_rl/eval}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/sft-qwen3-4b/$SLURM_JOB_ID}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Dense 8B: no expert parallelism. Default FSDP-only.
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"

MAX_LEN="${MAX_LEN:-4096}"
MAX_SAMPLES="${MAX_SAMPLES:-8192}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"

test -f "$CONTAINER_IMAGE"
test -e "$SFT_DATASET"
mkdir -p "$SAVE_ROOT"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RAY_USAGE_STATS_ENABLED=0

MOUNTS="${CONTAINER_MOUNTS:-$REPO_ROOT:/molt,/lustre:/lustre,$HOME/.cache:/root/.cache,/dev/shm:/dev/shm}"
CONTAINER_ARGS=(--no-container-mount-home --container-image="$CONTAINER_IMAGE" --container-mounts="$MOUNTS")

DEFAULT_AUTOMODEL_PATH=/path/to/Automodel
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
if [ -z "$EXTRA_PYTHONPATH" ] && [ -d "$DEFAULT_AUTOMODEL_PATH" ]; then
  EXTRA_PYTHONPATH="$DEFAULT_AUTOMODEL_PATH"
fi
PYTHONPATH_PREFIX=""
if [ -n "$EXTRA_PYTHONPATH" ]; then
  PYTHONPATH_PREFIX="PYTHONPATH=${EXTRA_PYTHONPATH}:\${PYTHONPATH:-} "
fi

head_node="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
head_addr="$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')"

TRAIN_ARGS=(
  --data.max_len "$MAX_LEN"
  --data.dataset "$SFT_DATASET"
  --data.input_key prompt
  --data.output_key response
  --data.max_samples "$MAX_SAMPLES"
  --model.model_name_or_path "$MODEL_PATH"
  --ckpt.output_dir "$SAVE_ROOT/hf"
  --ckpt.path "$SAVE_ROOT/state"
  --ckpt.save_steps "${SAVE_STEPS:-50}"
  --logger.logging_steps 1
  --eval.dataset "$EVAL_DATASET"
  --eval.steps "${EVAL_STEPS:-20}"
  --train.max_epochs "${MAX_EPOCHS:-1}"
  --train.batch_size "$TRAIN_BATCH_SIZE"
  --train.micro_batch_size "$MICRO_BATCH_SIZE"
  --fsdp.param_dtype bf16
  --fsdp.attn_implementation "${FSDP_ATTN_IMPLEMENTATION:-flash_attention_2}"
  --fsdp.tp_size "$TP_SIZE"
  --fsdp.ep_size "$EP_SIZE"
  --fsdp.cp_size "$CP_SIZE"
  --model.gradient_checkpoint full
  --adam.lr "${LR:-1e-6}"
  --logger.wandb.project "${WANDB_PROJECT:-molt_sft_qwen3_4b}"
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_4b_sft_$SLURM_JOB_ID}"
)

if [ "${DISABLE_FINAL_SAVE:-0}" = "1" ]; then
  TRAIN_ARGS+=(--ckpt.disable_final_save)
fi

printf -v TRAIN_ARGS_Q " %q" "${TRAIN_ARGS[@]}"

srun \
  --nodes="$SLURM_JOB_NUM_NODES" \
  --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 \
  "${CONTAINER_ARGS[@]}" \
  bash -lc "cd /molt && ${PYTHONPATH_PREFIX}torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$GPUS_PER_NODE \
    --node-rank=\$SLURM_NODEID \
    --master-addr=$head_addr \
    --master-port=$MASTER_PORT \
    -m molt.cli.train_sft$TRAIN_ARGS_Q"
