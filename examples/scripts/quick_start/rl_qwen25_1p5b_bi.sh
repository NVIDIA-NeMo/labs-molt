#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Four-GPU, single-node batch-invariance smoke test for Qwen2.5-1.5B.
#
# MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct \
# PROMPT_DATASET=/path/to/dapo-math.parquet \
# bash examples/scripts/quick_start/rl_qwen25_1p5b_bi.sh
#
# The standard training metric `vllm_kl` is the masked mean of
# rollout_log_probs - old_log_probs. Require it to remain zero for this
# deterministic smoke test; no custom logprob dump is needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen2.5-1.5B-Instruct checkpoint.}"
PROMPT_DATASET="${PROMPT_DATASET:?Set PROMPT_DATASET to a math-RL parquet dataset.}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-qwen25-1p5b-bi/run}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
VLLM_ENGINES="${VLLM_ENGINES:-2}"
if (( GPUS_PER_NODE != 4 || ACTOR_GPUS + VLLM_ENGINES != GPUS_PER_NODE )); then
  echo "This launcher requires 4 GPUs: ACTOR_GPUS + VLLM_ENGINES must equal 4." >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_BATCH_INVARIANT=1

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
  --data.max_samples "${MAX_SAMPLES:-4}" \
  --data.max_len "${MAX_LEN:-1024}" \
  --rollout.batch_size "${ROLLOUT_BATCH_SIZE:-4}" \
  --rollout.vllm_generate_batch_size "${ROLLOUT_BATCH_SIZE:-4}" \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 1 \
  --rollout.max_new_tokens "${MAX_NEW_TOKENS:-128}" \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --train.batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes 1 \
  --train.async_queue_size 1 \
  --train.seed "${TRAIN_SEED:-42}" \
  --train.force_on_policy \
  --train.force_sync_mode \
  --train.full_determinism_enable \
  --train.colocate_fsdp_models \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$ACTOR_GPUS" \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$ACTOR_GPUS" \
  --vllm.num_engines "$VLLM_ENGINES" \
  --vllm.tensor_parallel_size 1 \
  --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.8 \
  --vllm.enable_prefix_caching \
  --vllm.max_num_batched_tokens 8192 \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation sdpa \
  --fsdp.tp_size 1 \
  --fsdp.ep_size 1 \
  --fsdp.cp_size 1 \
  --actor.gradient_checkpoint none \
  --actor.adam.lr 1e-6 \
  --actor.loss_agg_mode seq-mean-token-mean \
  --algo.advantage.estimator flash_reinforce \
  --algo.advantage.is_correction_level token \
  --algo.advantage.is_correction_gating ratio \
  --algo.advantage.is_correction_mode clip \
  --algo.advantage.is_correction_threshold 0.0 1000.0 \
  --algo.kl.init_coef 0 \
  --reward.clip_range -10 10 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/math.py" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.save_steps -1 \
  --ckpt.disable_final_save \
  --eval.steps -1 \
  --logger.logging_steps 1 \
  --logger.wandb.project "${WANDB_PROJECT:-molt_qwen25_bi}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen25_1p5b_bi_smoke_$$}" \
  "$@"
