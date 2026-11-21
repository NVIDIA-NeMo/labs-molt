#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=your_slurm_account
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=molt-rl-inkling2-mini
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
export MOLT_PATH="$REPO_ROOT"

# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini): 26B MoE, 128 experts, layers 6-17 run three
# times per token. Text math RL on two 4-GPU Blackwell nodes: one node trains (EP4), the other
# hosts four TP1 vLLM rollout engines. Needs a vLLM with the model (vllm-project/vllm#69028) and
# an AutoModel with the looped checkpoint (NVIDIA-NeMo/Automodel#4861).
export MODEL_PATH="${MODEL_PATH:-/path/to/models/Inkling-2-Mini}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export ACTOR_NODES="${ACTOR_NODES:-1}"
export ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-4}"
export TP_SIZE=1
export EP_SIZE="${EP_SIZE:-4}"
export CP_SIZE=1
# The AutoModel implementation runs FlexAttention with torch linear layers and norms.
export FSDP_ATTN_IMPLEMENTATION=flex
export MOLT_RMS_NORM=torch
export GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-full}"
export COLOCATE=0
export ROUTING_REPLAY=0
export MOE_AUX_LOSS_COEF=0

# DAPO-Math-17k graded by the math agent. The released chat template asks for thinking effort
# 0.9, at which about a third of the answers do not finish within 4096 new tokens.
DATA_DIR="${DATA_DIR:-$REPO_ROOT/.tmp/dapo}"
if [ -z "${PROMPT_DATASET:-}" ] && [ ! -e "$DATA_DIR/train" ]; then
  python3 "$REPO_ROOT/examples/python/utils/prepare_dapo.py" --out-dir "$DATA_DIR"
fi
export PROMPT_DATASET="${PROMPT_DATASET:-$DATA_DIR/train}"
export EVAL_DATASET="${EVAL_DATASET:-$DATA_DIR/eval}"
export AGENT_PATH="${AGENT_PATH:-/molt/examples/python/agents/math.py}"
export MAX_AGENT_TURNS=1
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export MAX_LENGTH="${MAX_LENGTH:-5120}"

# 50 updates of 20 prompts x 8 samples.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-20}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-160}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export MAX_SAMPLES="${MAX_SAMPLES:-1000}"
export ASYNC_QUEUE_SIZE=1
export ESTIMATOR="${ESTIMATOR:-grpo}"
export LR="${LR:-3e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export KL_COEF=0
export ENABLE_DYNAMIC_FILTERING=0
export EVAL_STEPS="${EVAL_STEPS:-10}"
export EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-2}"
export EVAL_AT_START="${EVAL_AT_START:-1}"
export DISABLE_FINAL_SAVE="${DISABLE_FINAL_SAVE:-1}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"

export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"
export VLLM_TP_SIZE=1
export VLLM_DP_SIZE=1
export VLLM_ENABLE_EXPERT_PARALLEL=0
export VLLM_ENFORCE_EAGER=0
export MTP_NUM_SPECULATIVE_TOKENS=0

export SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/rl-inkling2-mini/$SLURM_JOB_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-molt_inkling2_mini_rl}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-inkling2_mini_rl_$SLURM_JOB_ID}"

# Slurm executes a spool copy of this wrapper, so resolve the shared launcher
# from the submitted repository rather than from this file's runtime directory.
exec bash "$REPO_ROOT/examples/scripts/slurm/rl_qwen3_6_35b.sh" "$@"
