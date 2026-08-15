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

# Single-node quick-start: Qwen3-4B multi-turn RL on AlfWorld (TextWorld).
#
# Point MODEL_PATH at the SFT warmstart checkpoint (sft_qwen3_4b_alfworld.sh) —
# pure-RL cold start has no signal here (0/1 terminal reward, ~0% base win rate).
# Each turn the agent emits one NL command; the env steps the game and feeds the
# new observation back as a ChatML user turn. n_samples_per_prompt=8 rolls the
# SAME game 8 times so a GRPO group has a win/loss spread to contrast on.
#
# Prereq (one-time):
#   pip install alfworld textworld[gym]
#   export ALFWORLD_DATA=$HOME/.molt_data/alfworld
#   alfworld-download
#   python examples/python/utils/prepare_alfworld.py rl
#
#   MODEL_PATH=<sft checkpoint> bash examples/scripts/quick_start/rl_qwen3_4b_alfworld.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the AlfWorld SFT checkpoint (run sft_qwen3_4b_alfworld.sh first).}"

PROMPT_DATASET="${PROMPT_DATASET:-$REPO_ROOT/.tmp/alfworld_rl/train.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/alfworld_rl/eval.jsonl}"
test -e "$PROMPT_DATASET" || { echo "PROMPT_DATASET not found: $PROMPT_DATASET — run: python examples/python/utils/prepare_alfworld.py rl"; exit 1; }
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-rl-qwen3-4b-alfworld/run}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
VLLM_TP="${VLLM_TP:-4}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP16=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Agent reads MAX_AGENT_TURNS; AlfWorld games need ~15-30 commands.
export MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-30}"
# Dedicated TextWorld thread pool — raise toward rollout concurrency if steps stall.
export ALFWORLD_TW_WORKERS="${ALFWORLD_TW_WORKERS:-8}"
# Offline by default — flip WANDB_MODE=online to log remotely.
export WANDB_MODE="${WANDB_MODE:-offline}"
# wandb appends its own wandb/ subdir -> point WANDB_DIR at $SAVE_ROOT, not .../wandb.
export WANDB_DIR="${WANDB_DIR:-$SAVE_ROOT}"

# Ray session logs land at /tmp/ray/session_latest/logs/ (not under $SAVE_ROOT):
# ray's plasma_store socket lives under --temp-dir and is capped at 107 bytes
# (AF_UNIX), so a deep outputs/... path overruns it and ray start fails.
if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus="$GPUS_PER_NODE" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && ray stop --force >/dev/null 2>&1 || true' EXIT

cd "$REPO_ROOT"
# n_samples_per_prompt=8 forms a GRPO group per game (sparse-reward signal source);
# max_new_tokens=32 (commands are short); max_len=16384 fits ~50 obs/action turns.
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$PROMPT_DATASET" \
  --data.input_key prompt \
  --data.label_key label \
  --data.apply_chat_template \
  --data.max_samples "${MAX_SAMPLES:-4800}" \
  --data.max_len "${MAX_LEN:-16384}" \
  --rollout.batch_size "${ROLLOUT_BATCH:-16}" \
  --rollout.vllm_generate_batch_size "${VLLM_GEN_BATCH:-4}" \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 8 \
  --rollout.max_new_tokens "${MAX_NEW_TOKENS:-32}" \
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
  --vllm.disable_custom_all_reduce \
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
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_threshold 0.99 1.01 \
  --algo.kl.use_loss \
  --algo.kl.estimator k2 \
  --algo.kl.init_coef 0.001 \
  --algo.dynamic_filtering_enable \
  --algo.dynamic_filtering_range 0.01 0.99 \
  --reward.clip_range -10 10 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/alfworld.py" \
  --eval.dataset "$EVAL_DATASET" \
  --eval.steps 5 \
  --eval.n_samples_per_prompt 1 \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps 5 \
  --logger.logging_steps 1 \
  --logger.wandb.key "${WANDB_KEY:-local}" \
  --logger.wandb.project "${WANDB_PROJECT:-molt_alfworld_rl_qwen3_4b}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_4b_alfworld_rl_$$}" \
  --logger.tensorboard_dir "$SAVE_ROOT/tb"
