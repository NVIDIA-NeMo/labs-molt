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

# Single-node quick-start: Qwen3-4B SFT warmstart on AlfWorld expert demos.
#
# Warmstart is required, not optional: AlfWorld reward is a 0/1 terminal signal
# and the base model wins ~0%, so pure RL has no in-group win/loss contrast to
# learn from. SFT on the handcoded-expert demos (alfworld-download --extra) lifts
# the win rate to ~10-30%, giving GRPO groups a spread to bootstrap from.
#
# Prereq (one-time):
#   pip install alfworld textworld[gym]
#   export ALFWORLD_DATA=$HOME/.molt_data/alfworld
#   alfworld-download            # game .tw-pddl + traj_data.json
#   alfworld-download --extra    # seq2seq expert demos (consumed below)
#   python examples/python/utils/prepare_alfworld.py sft
#
#   MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 bash examples/scripts/quick_start/sft_qwen3_4b_alfworld.sh
# Point the follow-up RL run at this run's checkpoint (outputs/.../hf).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/path/to/models/Qwen3/Qwen3-4B-Instruct-2507}"

SFT_DATASET="${SFT_DATASET:-$REPO_ROOT/.tmp/alfworld_sft/train.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/alfworld_sft/eval.jsonl}"
test -e "$SFT_DATASET" || { echo "SFT_DATASET not found: $SFT_DATASET — run: python examples/python/utils/prepare_alfworld.py sft"; exit 1; }
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-sft-qwen3-4b-alfworld/run}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
MAX_LEN="${MAX_LEN:-8192}"            # multi-turn demos: task + up to ~40 obs/action turns
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"   # long multi-turn sequences
MAX_SAMPLES="${MAX_SAMPLES:-4096}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Offline by default — flip WANDB_MODE=online (and set WANDB_API_KEY) to log remotely.
export WANDB_MODE="${WANDB_MODE:-offline}"
# wandb appends its own wandb/ subdir -> point WANDB_DIR at $SAVE_ROOT, not .../wandb.
export WANDB_DIR="${WANDB_DIR:-$SAVE_ROOT}"

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
  --train.max_epochs "${MAX_EPOCHS:-3}" \
  --train.batch_size "$TRAIN_BATCH_SIZE" \
  --train.micro_batch_size "$MICRO_BATCH_SIZE" \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation "${FSDP_ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --fsdp.tp_size "$TP_SIZE" \
  --fsdp.ep_size "$EP_SIZE" \
  --fsdp.cp_size "$CP_SIZE" \
  --model.gradient_checkpoint full \
  --adam.lr "${LR:-1e-6}" \
  --logger.wandb.key "${WANDB_KEY:-local}" \
  --logger.wandb.project "${WANDB_PROJECT:-molt_alfworld_sft_qwen3_4b}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_4b_alfworld_sft_$$}" \
  --logger.tensorboard_dir "$SAVE_ROOT/tb"
