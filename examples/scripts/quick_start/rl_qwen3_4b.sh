#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single-node quick-start: Qwen3-4B dense math RL.
#
# 8 GPUs on one machine, split 4 actor + 4 vLLM rollout. No slurm. Same
# standard batch / context / dataset as slurm/qwen3_4b.sh — only the topology
# differs (1 node here, 2 nodes there).
#
#   MODEL_PATH=/path/to/Qwen3-4B bash examples/scripts/quick_start/qwen3_4b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen3-4B checkpoint.}"

# Same dataset path as slurm/qwen3_4b.sh — the proRL text-math RL split.
PROMPT_DATASET="${PROMPT_DATASET:-$REPO_ROOT/.tmp/proRL_text_rl/train}"
EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/proRL_text_rl/eval}"
test -e "$PROMPT_DATASET" || { echo "PROMPT_DATASET not found: $PROMPT_DATASET — prepare proRL_text_rl first."; exit 1; }
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-qwen3-4b/run}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
VLLM_TP="${VLLM_TP:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS-}"  # unset = per-turn unlimited (bounded by max_len); set N to cap

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP16=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_AGENT_TURNS=1

if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus="$GPUS_PER_NODE" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && ray stop --force >/dev/null 2>&1 || true' EXIT

cd "$REPO_ROOT"
# Standard config: MAX_LEN=16384, ROLLOUT=16, N=8, TRAIN_BATCH=128, dataset=proRL.
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$PROMPT_DATASET" \
  --data.input_key prompt \
  --data.label_key reward_model \
  --data.apply_chat_template \
  --data.max_samples 4800 \
  --data.max_len 16384 \
  ${MAX_NEW_TOKENS:+--rollout.max_new_tokens=$MAX_NEW_TOKENS} \
  --rollout.batch_size 16 \
  --rollout.vllm_generate_batch_size 8 \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 8 \
  --rollout.temperature 1.0 \
  --train.batch_size 128 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes 1 \
  --train.async_queue_size 2 \
  --train.colocate_fsdp_models \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$ACTOR_GPUS" \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$ACTOR_GPUS" \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size "$VLLM_TP" \
  --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.8 \
  --vllm.distributed_executor_backend mp \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation flash_attention_2 \
  --fsdp.tp_size 1 \
  --fsdp.ep_size 1 \
  --fsdp.cp_size 1 \
  --fsdp.packing_samples \
  --actor.gradient_checkpoint full \
  --actor.adam.lr 1e-6 \
  --actor.eps_clip_low_high 0.2 0.27 \
  --actor.dual_clip 10.0 \
  --algo.advantage.estimator reinforce_baseline \
  --algo.advantage.is_correction_enable \
  --algo.advantage.is_correction_type seq-mask-tis \
  --algo.advantage.is_correction_threshold 0.99 1.01 \
  --algo.kl.use_loss \
  --algo.kl.estimator k2 \
  --algo.kl.init_coef 0.001 \
  --algo.dynamic_filtering_enable \
  --algo.dynamic_filtering_range 0.01 0.99 \
  --reward.clip_range -10 10 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/math.py" \
  --eval.dataset "$EVAL_DATASET" \
  --eval.steps 5 \
  --eval.n_samples_per_prompt 1 \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps 5 \
  --logger.logging_steps 1 \
  --logger.wandb.project "${WANDB_PROJECT:-molt_quickstart_qwen3_4b}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_4b_quickstart_$$}"
