#!/bin/bash
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

#SBATCH --account=your_slurm_account
#SBATCH --partition=interactive
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=molt-sft-qwen3-6
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

# Qwen3.6-35B-A3B VLM SFT on geo3k.
# 2-node default: TP=1 EP=8 CP=1. Override TP_SIZE/EP_SIZE/CP_SIZE for other
# parallelism layouts; for 4-node prod scale, submit with --nodes=4 and adjust
# TRAIN_BATCH_SIZE accordingly.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$REPO_ROOT/images/molt-cu13.sqsh}"

MODEL_PATH="${MODEL_PATH:-/path/to/models/Qwen3.6-35B-A3B}"

# geo3k VLM dataset (same format as the RL recipe). Auto-prep if missing.
SFT_DATASET="${SFT_DATASET:-$REPO_ROOT/.tmp/geo3k/train}"
EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/geo3k/eval}"
if [ ! -d "$SFT_DATASET" ]; then
  python3 "$REPO_ROOT/examples/python/utils/prepare_geo3k.py" \
    --max-eval 256 --num-proc 8 --out-dir "$REPO_ROOT/.tmp/geo3k"
fi
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/sft-qwen3-6/$SLURM_JOB_ID}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Qwen3.5-MoE family: TP=1 required by Automodel's custom MoE parallelizer.
# EP=8 maps experts across one node; CP=1 keeps the recipe portable.
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-1}"

# Standard SFT defaults: bf16, geo3k answers are short — 4K context covers
# prompt (image placeholder + text) + boxed answer + small CoT.
MAX_LEN="${MAX_LEN:-4096}"
MAX_SAMPLES="${MAX_SAMPLES:-8192}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

test -f "$CONTAINER_IMAGE"
test -e "$SFT_DATASET"
mkdir -p "$SAVE_ROOT"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RAY_USAGE_STATS_ENABLED=0

MOUNTS="${CONTAINER_MOUNTS:-$REPO_ROOT:/molt,/lustre:/lustre,$HOME/.cache:/root/.cache,/dev/shm:/dev/shm}"
CONTAINER_ARGS=(--no-container-mount-home --container-image="$CONTAINER_IMAGE" --container-mounts="$MOUNTS")

# Prepend sibling Automodel checkout if present (matches the RL launcher).
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
  --data.image_key images
  --data.max_images_per_prompt 1
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
  --model.aux_loss_coef "${MOE_AUX_LOSS_COEF:-0.001}"
  --logger.wandb.project "${WANDB_PROJECT:-molt_sft_qwen3_6}"
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_6_sft_$SLURM_JOB_ID}"
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
