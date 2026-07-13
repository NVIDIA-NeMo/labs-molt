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
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=molt-sft-omni3
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

# Nemotron Omni3 (NemotronH_Nano_Omni_Reasoning_V3) VLM SFT.
# Default config = 32K + CP8 + EP8 + deepep + AC: the trainer delegates CP to the
# Actor (RL contract); 2 nodes (16 GPUs) → CP8 shards the 32K sequence to 4K/rank,
# DP=2. Mirrors slurm/rl_omni3_30b.sh:
#   - Native AutoModel path (NemotronOmniForConditionalGeneration). TE is the
#     fused-attention backend (flash_attention_2 silently falls to sdpa) and is
#     required for CP>1.
#   - Gradient checkpointing ON: the deepep MoE dispatcher makes the recompute
#     deterministic under AC (the torch dispatcher raises CheckpointError).

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$REPO_ROOT/images/molt-cu13.sqsh}"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/.tmp/nemotron_omni_v3_shim/iter_0001926_mcore_to_hf}"

# geo3k VLM dataset (same SFT format as the qwen3.6 recipe). Auto-prep if missing.
SFT_DATASET="${SFT_DATASET:-$REPO_ROOT/.tmp/geo3k/train}"
EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/geo3k/eval}"
if [ ! -d "$SFT_DATASET" ]; then
  python3 "$REPO_ROOT/examples/python/utils/prepare_geo3k.py" \
    --max-eval 256 --num-proc 8 --out-dir "$REPO_ROOT/.tmp/geo3k"
fi
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/sft-omni3/$SLURM_JOB_ID}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"

# AutoModel custom MoE asserts TP=1; EP=8 maps experts across one node.
# CP_SIZE=8 shards each sample's 32K sequence to 4K/rank (2 nodes → DP=2).
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-8}"

# 32K long-context default (omni3); short datasets simply don't fill it.
MAX_LEN="${MAX_LEN:-32768}"
MAX_SAMPLES="${MAX_SAMPLES:-8192}"
# batch = micro * dp_size * grad_accum, dp_size = world / (tp*cp).
# 2 nodes CP8 → dp=2; batch=8 → grad_accum=4 (micro=1).
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

# Omni3 native path: TE native cuDNN-fused attention. Grad-ckpt ON by default —
# the deepep dispatcher makes the MoE recompute deterministic under AC.
FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-te}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT-full}"

test -f "$CONTAINER_IMAGE"
test -e "$SFT_DATASET"
if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "[omni3] MODEL_PATH not found: $MODEL_PATH" >&2
  echo "[omni3] build the shim first, e.g.:" >&2
  echo "        python3 $REPO_ROOT/.tmp/scripts/build_nemotron_omni_v3_shim.py" >&2
  exit 1
fi
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
  --fsdp.attn_implementation "$FSDP_ATTN_IMPLEMENTATION"
  --fsdp.tp_size "$TP_SIZE"
  --fsdp.ep_size "$EP_SIZE"
  --fsdp.cp_size "$CP_SIZE"
  --adam.lr "${LR:-5e-6}"
  --model.aux_loss_coef "${MOE_AUX_LOSS_COEF:-0.001}"
  --logger.wandb.project "${WANDB_PROJECT:-molt_sft_omni3}"
  --logger.wandb.run_name "${WANDB_RUN_NAME:-omni3_sft_$SLURM_JOB_ID}"
)

# Grad-ckpt mode from GRAD_CHECKPOINT (default full; '' disables, 'selective' for
# per-op AC). Safe under the deepep dispatcher (deterministic MoE recompute).
TRAIN_ARGS+=(--model.gradient_checkpoint "$GRAD_CHECKPOINT")

if [ "${DISABLE_FINAL_SAVE:-0}" = "1" ]; then
  TRAIN_ARGS+=(--ckpt.disable_final_save)
fi

printf -v TRAIN_ARGS_Q " %q" "${TRAIN_ARGS[@]}"

srun \
  --nodes="$SLURM_JOB_NUM_NODES" \
  --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 \
  "${CONTAINER_ARGS[@]}" \
  bash -lc "cd /molt && export CUDNN_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MOLT_MOE_RESHARD_AFTER_FWD=${MOLT_MOE_RESHARD_AFTER_FWD:-1} MOLT_DEFER_GRAD_SYNC=${MOLT_DEFER_GRAD_SYNC:-1} MOLT_MOE_DISPATCHER=${MOLT_MOE_DISPATCHER:-hybridep} && ${PYTHONPATH_PREFIX}torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$GPUS_PER_NODE \
    --node-rank=\$SLURM_NODEID \
    --master-addr=$head_addr \
    --master-port=$MASTER_PORT \
    -m molt.cli.train_sft$TRAIN_ARGS_Q"
