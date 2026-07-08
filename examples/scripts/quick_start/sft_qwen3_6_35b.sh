#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single-node quick-start: Qwen3.6-35B-A3B VLM SFT on geo3k.
#
# 8 GPUs on one machine via torchrun --standalone — no slurm.
# Mirrors slurm/sft_qwen3_6_35b.sh's recipe for single-node use.
#
#   MODEL_PATH=/path/to/Qwen3.6-35B-A3B bash examples/scripts/quick_start/sft_qwen3_6_35b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen3.6-35B-A3B checkpoint.}"

# geo3k VLM SFT data — auto-prep if missing.
DATA_DIR="$REPO_ROOT/.tmp/geo3k"
if [ ! -d "$DATA_DIR/train" ]; then
  echo "[quickstart] preparing geo3k VLM (VeraIsHere/geo3k_imgurl_processed) — one-time"
  python3 "$REPO_ROOT/examples/python/utils/prepare_geo3k.py" \
    --max-eval 256 --num-proc 8 --out-dir "$DATA_DIR"
fi
SFT_DATASET="${SFT_DATASET:-$DATA_DIR/train}"
EVAL_DATASET="${EVAL_DATASET:-$DATA_DIR/eval}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-sft-qwen3-6/run}"

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
  --data.image_key images \
  --data.max_images_per_prompt 1 \
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
  --model.aux_loss_coef "${MOE_AUX_LOSS_COEF:-0.001}" \
  --logger.wandb.project "${WANDB_PROJECT:-molt_quickstart_sft_qwen3_6}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_6_sft_quickstart_$$}"
