#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=your_slurm_account
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=molt-vrl-muse-glimmer
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
export MOLT_PATH="$REPO_ROOT"

# Dense Muse Glimmer actor: TP is capped at 2 by its two KV heads. Two actor
# nodes give DP=8, which fits fp32 master weights; the third node hosts four
# independent TP2 vLLM rollout engines.
export MODEL_PATH="${MODEL_PATH:-meta-models/Muse-Glimmer-30B}"
export ACTOR_NODES="${ACTOR_NODES:-2}"
export ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
export TP_SIZE="${TP_SIZE:-2}"
export EP_SIZE=1
export CP_SIZE="${CP_SIZE:-1}"
export FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-sdpa}"
export GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-full}"
export OFFLOAD_OPTIMIZER=0
export FSDP_CPU_OFFLOAD=0

export MAX_LENGTH="${MAX_LENGTH:-8192}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-5}"
export MAX_SAMPLES="${MAX_SAMPLES:-65536}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export ROLLOUT_GENERATE_BATCH_SIZE="${ROLLOUT_GENERATE_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export MICRO_BATCH_SIZE=1
export ASYNC_QUEUE_SIZE="${ASYNC_QUEUE_SIZE:-1}"

export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
export VLLM_DP_SIZE=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export VLLM_ENABLE_EXPERT_PARALLEL=0
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export VLLM_MM_ENCODER_ATTN_BACKEND="${VLLM_MM_ENCODER_ATTN_BACKEND:-TORCH_SDPA}"
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_MAMBA_SSM_CACHE_DTYPE=auto
export MTP_NUM_SPECULATIVE_TOKENS=0
export ENABLE_PREFIX_CACHING=0

export ROUTING_REPLAY=0
export FREEZE_VISUAL_ENCODER="${FREEZE_VISUAL_ENCODER:-1}"
export FREEZE_MOE_ROUTER=0
export MOE_AUX_LOSS_COEF=0
export FORCE_ON_POLICY=1
export ENABLE_DYNAMIC_FILTERING="${ENABLE_DYNAMIC_FILTERING:-0}"
export KL_COEF="${KL_COEF:-0.0}"
export LR="${LR:-1e-6}"

export SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/async-visual-rl-muse-glimmer/$SLURM_JOB_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-molt_muse_glimmer_rl}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-muse_glimmer_rl_$SLURM_JOB_ID}"

# Keep the public 60 GB checkpoint off the small home filesystem. The generic
# launcher serves this mount at the HF_HOME it sets inside the container.
HF_CACHE_DIR="${HF_CACHE_DIR:-$REPO_ROOT/.tmp/hf-cache}"
mkdir -p "$HF_CACHE_DIR"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$REPO_ROOT:/molt,/lustre:/lustre,$HF_CACHE_DIR:/root/.cache/huggingface,/dev/shm:/dev/shm}"

# The base recipe keeps vllm_kl enabled and logged.
# Slurm executes a spool copy of this wrapper, so resolve the shared launcher
# from the submitted repository rather than from this file's runtime directory.
exec bash "$REPO_ROOT/examples/scripts/slurm/rl_qwen3_6_35b.sh" "$@"
