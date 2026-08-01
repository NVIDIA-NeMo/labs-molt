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

# Single-node quick-start: Qwen3-30B-A3B (stock HF Qwen3MoeForCausalLM) text LoRA SFT.
#
# The stock HF MoE runs on the same custom MoE + expert-parallel path as the
# Qwen3.5/3.6 recipes, so this mirrors sft_qwen3_5_35b_lora.sh minus the vision
# columns (text-only model). Only the adapters train; adapters land on `*_proj`
# -- add `--model.lora.target_modules '*'` to adapt the grouped experts too. MoE
# + expert-parallel + full recompute needs `--data.pad_to_max_len` (PR #61 /
# AutoModel#3325) to avoid a CheckpointError.
#
#   MODEL_PATH=/path/to/Qwen3-30B-A3B bash examples/scripts/quick_start/sft_qwen3_30b_a3b_lora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen3-30B-A3B checkpoint.}"

# Reuse geo3k's text columns (prompt/response) — auto-prep if missing.
DATA_DIR="$REPO_ROOT/.tmp/geo3k"
if [ ! -d "$DATA_DIR/train" ]; then
  echo "[quickstart] preparing geo3k (VeraIsHere/geo3k_imgurl_processed) — one-time"
  python3 "$REPO_ROOT/examples/python/utils/prepare_geo3k.py" \
    --max-eval 256 --num-proc 8 --out-dir "$DATA_DIR"
fi
SFT_DATASET="${SFT_DATASET:-$DATA_DIR/train}"
EVAL_DATASET="${EVAL_DATASET:-$DATA_DIR/eval}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-sft-qwen3-30b-a3b-lora/run}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-1}"
MAX_LEN="${MAX_LEN:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
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
  --model.lora.rank "${LORA_RANK:-16}" \
  --model.lora.alpha "${LORA_ALPHA:-32}" \
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
  --adam.lr "${LR:-1e-4}" \
  --model.aux_loss_coef "${MOE_AUX_LOSS_COEF:-0.001}" \
  --logger.wandb.project "${WANDB_PROJECT:-molt_quickstart_sft_qwen3_30b_a3b_lora}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_30b_a3b_sft_lora_quickstart_$$}"
