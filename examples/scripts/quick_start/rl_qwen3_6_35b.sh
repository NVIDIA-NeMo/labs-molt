#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single-node quick-start: Qwen3.6-35B-A3B (Qwen3.5-MoE) VLM RL on geo3k.
#
# 8 GPUs on one machine, split 4 actor + 4 vLLM rollout. No slurm. Same recipe
# as slurm/rl_qwen3_6_35b.sh — only the topology differs (1 node here, 2+ there).
# Multi-turn rollout uses the `<tool_call>` geo3k math grader (with a boxed
# fallback that gives partial credit for raw `\boxed{}` answers).
#
#   MODEL_PATH=/path/to/Qwen3.6-35B-A3B bash examples/scripts/quick_start/rl_qwen3_6_35b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen3.6-35B-A3B checkpoint.}"

# geo3k VLM multi-turn — same default as slurm/rl_qwen3_6_35b.sh. Auto-prep if missing.
DATA_DIR="$REPO_ROOT/.tmp/geo3k"
if [ ! -d "$DATA_DIR/train" ]; then
  echo "[quickstart] preparing geo3k VLM (VeraIsHere/geo3k_imgurl_processed) — one-time"
  python3 "$REPO_ROOT/examples/python/utils/prepare_geo3k.py" \
    --max-eval 256 --num-proc 8 --out-dir "$DATA_DIR"
fi
PROMPT_DATASET="${PROMPT_DATASET:-$DATA_DIR/train}"
EVAL_DATASET="${EVAL_DATASET:-$DATA_DIR/eval}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-qwen3-6/run}"

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
export MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-4}"

if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus="$GPUS_PER_NODE" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && ray stop --force >/dev/null 2>&1 || true' EXIT

cd "$REPO_ROOT"
# Smoke-friendly config: smaller batch + 16K ctx so it fits on a single node.
# Bump for stability runs on multi-node (slurm/rl_qwen3_6_35b.sh).
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$PROMPT_DATASET" \
  --data.input_key prompt \
  --data.label_key reward_model \
  --data.tools_key tools \
  --data.apply_chat_template \
  --data.image_key images \
  --data.max_images_per_prompt 1 \
  --data.max_samples 4096 \
  --data.max_len 16384 \
  ${MAX_NEW_TOKENS:+--rollout.max_new_tokens=$MAX_NEW_TOKENS} \
  --rollout.batch_size 8 \
  --rollout.vllm_generate_batch_size 8 \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 4 \
  --rollout.temperature 1.0 \
  --train.batch_size 32 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes 1 \
  --train.async_queue_size 2 \
  --train.routing_replay \
  --train.colocate_fsdp_models \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$ACTOR_GPUS" \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$ACTOR_GPUS" \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size "$VLLM_TP" \
  --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.7 \
  --vllm.distributed_executor_backend mp \
  --vllm.enable_expert_parallel \
  --vllm.mamba_ssm_cache_dtype float32 \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation te \
  --fsdp.tp_size 1 \
  --fsdp.ep_size 4 \
  --fsdp.cp_size 2 \
  --actor.gradient_checkpoint full \
  --actor.freeze_visual_encoder \
  --actor.adam.lr 2e-6 \
  --actor.eps_clip_low_high 0.2 0.28 \
  --actor.dual_clip 10.0 \
  --actor.aux_loss_coef 0.000 \
  --algo.advantage.estimator reinforce_baseline \
  --algo.advantage.is_correction_enable \
  --algo.advantage.is_correction_type seq-mask-tis \
  --algo.advantage.is_correction_threshold 0.99 1.01 \
  --algo.kl.use_loss \
  --algo.kl.estimator k2 \
  --algo.kl.init_coef 0.0 \
  --reward.clip_range -10 10 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/geo3k.py" \
  --eval.dataset "$EVAL_DATASET" \
  --eval.steps 5 \
  --eval.n_samples_per_prompt 1 \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps 5 \
  --logger.logging_steps 1 \
  --logger.wandb.project "${WANDB_PROJECT:-molt_quickstart_qwen3_6}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_6_quickstart_$$}"
