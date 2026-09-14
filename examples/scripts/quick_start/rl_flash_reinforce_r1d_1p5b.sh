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

# Single-node quick-start: FlashREINFORCE — critic-free, single-rollout RL for agentic language models
# (paper: https://www.researchgate.net/publication/414274571_FlashREINFORCE_FLASHREINFORCE_CRITIC-FREE_SINGLE-ROLLOUT_ASYNCHRONOUS_RL_FOR_AGENTIC_LANGUAGE_MODELS)
# on the FP16 sanity test (arXiv:2510.26788): DeepSeek-R1-Distill-Qwen-1.5B on sail/Sanity-Test-R1D-1.5B
# (1,460 MATH problems), where BF16 PG-IS is known to drift. 1,000 episodes x 12 rounds = 12,000 updates; one
# rollout per prompt, 128 rollouts per update, lr 1e-6 cosine, wd 0.1, temperature 1.0, 8k responses, no KL;
# 512 rollouts stay in flight and up to 8 finished batches queue ahead of the trainer, so vLLM never waits
# for training; sequences are packed up to MAX_TOKENS_PER_GPU tokens per micro-batch (DYNAMIC_BATCH=0 trains
# one sequence per micro-batch). FlashREINFORCE is a composition of configs:
#   --train.force_on_policy                           PPO ratio == 1 -> plain REINFORCE gradient
#   --algo.advantage.is_correction_level seq          IS weight pi/mu against the vLLM behavior logprobs,
#   --algo.advantage.is_correction_gating binary_kl     gated per sequence by the mean sampled-token
#   --algo.advantage.is_correction_threshold 5e-3       binary KL (two-sided trust region, delta = 5e-3)
#   --actor.loss_agg_mode seq-mean-token-mean         sample mean: every rollout weighs the same
#   --algo.advantage.estimator flash_reinforce        reward minus the rollout-batch mean, no whitening
# Eval: AIME 2024 + 2025 avg@32 (temperature 0.6 / top-p 0.95) every 128 updates. 8 GPUs: 1 actor + 7 vLLM.
#
#   MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B bash examples/scripts/quick_start/rl_flash_reinforce_r1d_1p5b.sh
#   The data is fetched on the first run (examples/python/utils/prepare_dapo.py, see its docstring).
#   TRUST_REGION_DELTA=inf disables the trust region (the PG-IS ablation); NUM_EPISODES sets the horizon;
#   extra arguments are forwarded to molt.cli.train_rl_ray.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a DeepSeek-R1-Distill-Qwen-1.5B checkpoint.}"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/.tmp/prep_sanity_r1d}"
PROMPT_DATASET="${PROMPT_DATASET:-$DATA_DIR/train}"
EVAL_DATASET="${EVAL_DATASET:-$DATA_DIR/eval}"
# First run: fetch sail/Sanity-Test-R1D-1.5B (1,460 MATH train problems; AIME 2024 + 2025 as eval).
[ -e "$PROMPT_DATASET" ] || python3 "$REPO_ROOT/examples/python/utils/prepare_dapo.py" \
  --train-source sail/Sanity-Test-R1D-1.5B --eval-source sail/Sanity-Test-R1D-1.5B --eval-split test --out-dir "$DATA_DIR"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-flash-reinforce-r1d-1p5b/run}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-1}"
VLLM_ENGINES="${VLLM_ENGINES:-7}"
DYNAMIC_BATCH="${DYNAMIC_BATCH:-1}"; [ "$DYNAMIC_BATCH" = "0" ] && DYNAMIC_BATCH=""
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"
# Packed sequences need a varlen attention kernel: Transformer Engine when installed, else HF FA2.
ATTN_IMPL="${FSDP_ATTN_IMPLEMENTATION:-$(python3 -c "import transformer_engine" 2>/dev/null && echo te || echo flash_attention_2)}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
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
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$PROMPT_DATASET" \
  --data.input_key prompt \
  --data.label_key reward_model \
  --data.apply_chat_template \
  --data.max_samples 1460 \
  --data.max_len 9216 \
  --rollout.batch_size 128 \
  --rollout.vllm_generate_batch_size 512 \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 1 \
  --rollout.max_new_tokens 8192 \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --train.batch_size 128 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes "${NUM_EPISODES:-1000}" \
  --train.async_queue_size 8 \
  --train.force_on_policy \
  --train.colocate_fsdp_models \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$ACTOR_GPUS" \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$ACTOR_GPUS" \
  --vllm.num_engines "$VLLM_ENGINES" \
  --vllm.tensor_parallel_size 1 \
  --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.9 \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation "$ATTN_IMPL" \
  --fsdp.packing_samples \
  ${DYNAMIC_BATCH:+--train.dynamic_batch_enable --train.max_tokens_per_gpu "$MAX_TOKENS_PER_GPU"} \
  --actor.gradient_checkpoint full \
  --actor.adam.lr 1e-6 \
  --actor.adam.weight_decay 0.1 \
  --actor.lr_scheduler cosine_with_min_lr \
  --actor.loss_agg_mode seq-mean-token-mean \
  --algo.advantage.estimator flash_reinforce \
  --algo.advantage.is_correction_level seq \
  --algo.advantage.is_correction_gating binary_kl \
  --algo.advantage.is_correction_threshold "${TRUST_REGION_DELTA:-5e-3}" \
  --algo.kl.init_coef 0 \
  --reward.clip_range -10 10 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/math.py" \
  --eval.dataset "$EVAL_DATASET" \
  --eval.steps 128 \
  --eval.n_samples_per_prompt 32 \
  --eval.temperature 0.6 \
  --eval.top_p 0.95 \
  --eval.eval_at_start \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps 50 \
  --logger.logging_steps 1 \
  --logger.wandb.project "${WANDB_PROJECT:-molt_flash_reinforce_r1d_sanity}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-flash_reinforce_r1d_1p5b_$$}" \
  "$@"
