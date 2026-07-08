#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=your_slurm_account
#SBATCH --partition=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=molt-distill-omni3
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
export MOLT_PATH="$REPO_ROOT"

# On-policy distillation for Nemotron Omni3 (VLM MoE): student <- teacher.
#
# Same proven omni3 RL recipe (native AutoModel path, EP8/CP8/te/deepep, geo3k VLM
# multi-turn data), with ONE change: --algo.advantage.estimator on_policy_distill turns
# the reference model into a frozen teacher and makes the per-token reverse KL to it the
# entire training signal. k1 / kl.use_loss=off / unit coefficient are all derived — the
# only extra input is the teacher checkpoint (--ref.model_name_or_path / REF_MODEL_PATH).
#
# The teacher MUST share omni3's processor (RADIO vision tokenization + tokenizer), so the
# per-token logprobs align over the same vision-expanded sequence — i.e. a same-family
# Nemotron Omni checkpoint (a larger or more-trained one). It is colocated on the actor
# nodes (inference-only); for a much larger teacher, drop --train.colocate_fsdp_models and
# give it its own --ref.num_nodes.
#
# No reward function and no agent flag: on_policy_distill auto-selects the built-in
# generator (molt/agents/distill_agent.py), which samples one VLM completion per prompt and
# returns 0.0 (ignored by the estimator). Set AGENT_PATH to override (e.g. chat_geo3k.py for
# a multi-turn distribution). Watch `kl` / `logprobs_diff` in wandb fall toward 0 as the
# student matches the teacher (eval is off — a reward-based metric is meaningless here).

export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/.tmp/nemotron_omni_v3_shim/iter_0001926_mcore_to_hf}"
# Teacher checkpoint — REQUIRED. Must share omni3's processor/tokenizer.
export REF_MODEL_PATH="${REF_MODEL_PATH:-}"
export TP_SIZE="${TP_SIZE:-1}"        # AutoModel custom MoE asserts TP=1
export EP_SIZE="${EP_SIZE:-8}"        # only model-state shard knob for this MoE
export CP_SIZE="${CP_SIZE:-8}"        # shard sequence; CP8 leaves headroom for the colocated teacher
export GRAD_CHECKPOINT="${GRAD_CHECKPOINT-full}"
export FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-te}"

# 16K by default (vs 32K for plain RL): the colocated teacher adds a full model's params to
# the actor GPUs. Raise to 32000 with FSDP_CPU_OFFLOAD=1 or a separate teacher node if needed.
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS-}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export ROLLOUT_GENERATE_BATCH_SIZE="${ROLLOUT_GENERATE_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_EPISODES="${NUM_EPISODES:-5}"
export ASYNC_QUEUE_SIZE="${ASYNC_QUEUE_SIZE:-1}"
export MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-1}"   # dummy agent is single-turn (pure generation)
export MAX_SAMPLES="${MAX_SAMPLES:-65536}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"

export OPTIM="${OPTIM:-adam}"
export LR="${LR:-1e-5}"

# Distillation is monitored by kl/logprobs_diff, not task accuracy — eval off.
export SAVE_STEPS="${SAVE_STEPS:-2}"
export EVAL_STEPS="${EVAL_STEPS:--1}"
export EVAL_AT_START="${EVAL_AT_START:-0}"
export ENABLE_DYNAMIC_FILTERING="${ENABLE_DYNAMIC_FILTERING:-0}"

# Generate on the geo3k VLM subset (images exercise the vision path). The dummy agent
# ignores labels, so any image+text prompt set works.
export PROMPT_DATASET="${PROMPT_DATASET:-$REPO_ROOT/.tmp/geo3k_answer/train}"
export EVAL_DATASET="${EVAL_DATASET:-}"
if [ ! -d "$REPO_ROOT/.tmp/geo3k_answer/train" ]; then
  echo "[distill-omni3] preparing geo3k (<answer> format) — one-time"
  python3 "$REPO_ROOT/examples/python/utils/prepare_geo3k.py" \
    --answer-format answer --max-eval 256 --num-proc 8 \
    --out-dir "$REPO_ROOT/.tmp/geo3k_answer"
fi

if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "[distill-omni3] MODEL_PATH (student) not found: $MODEL_PATH" >&2
  echo "[distill-omni3] build the shim first, e.g.:" >&2
  echo "        python3 $REPO_ROOT/.tmp/scripts/build_nemotron_omni_v3_shim.py" >&2
  exit 1
fi
if [ -z "$REF_MODEL_PATH" ] || [ ! -e "$REF_MODEL_PATH/config.json" ]; then
  echo "[distill-omni3] REF_MODEL_PATH (teacher) not set or missing: '$REF_MODEL_PATH'" >&2
  echo "[distill-omni3] on_policy_distill needs a teacher that shares omni3's processor" >&2
  echo "        (a larger / more-trained Nemotron Omni checkpoint). Set REF_MODEL_PATH=..." >&2
  exit 1
fi

export SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/distill-omni3/run}"
export WANDB_PROJECT="${WANDB_PROJECT:-molt_distill_omni3}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-distill_omni3_$SLURM_JOB_ID}"

# Chain auto-resubmit (off by default): survive walltime/preemption by queueing the next
# slot with an afterany dependency. Set CHAIN_MAX>0 to bound; LOAD_ENABLE=1 resumes state.
CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
CHAIN_MAX="${CHAIN_MAX:-0}"
if [ "$CHAIN_DEPTH" -lt "$CHAIN_MAX" ]; then
  NEXT_DEPTH=$((CHAIN_DEPTH + 1))
  next_jobid=$(CHAIN_DEPTH="$NEXT_DEPTH" LOAD_ENABLE=1 \
    sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
    --account="$SLURM_JOB_ACCOUNT" \
    --partition="$SLURM_JOB_PARTITION" \
    ${SLURM_JOB_QOS:+--qos="$SLURM_JOB_QOS"} \
    ${SLURM_JOB_RESERVATION:+--reservation="$SLURM_JOB_RESERVATION"} \
    --nodes="$SLURM_JOB_NUM_NODES" \
    --comment='{"IdleGpuReaper":{"exemptIdleTimeMins":"120","reason":"other","description":"Async on-policy distillation; GPUs idle as train/rollout alternate"}}' \
    "$REPO_ROOT/examples/scripts/slurm/rl_distill_omni3_30b.sh")
  echo "[chain] depth=$NEXT_DEPTH/$CHAIN_MAX next_jobid=$next_jobid"
fi

# === Inlined launcher ===
set --
set -x

REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$REPO_ROOT/images/molt-cu13.sqsh}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the student checkpoint to train.}"
REF_MODEL_PATH="${REF_MODEL_PATH:?Set REF_MODEL_PATH to the teacher checkpoint.}"

PROMPT_DATASET="${PROMPT_DATASET:?Set PROMPT_DATASET}"
EVAL_DATASET="${EVAL_DATASET:-}"

SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/molt-distill-omni3/$SLURM_JOB_ID}"
# Empty by default → on_policy_distill uses the built-in generator. Override for multi-turn.
AGENT_PATH="${AGENT_PATH:-}"

DEFAULT_AUTOMODEL_PATH=/path/to/Automodel
if [ -z "${EXTRA_PYTHONPATH:-}" ] && [ -d "$DEFAULT_AUTOMODEL_PATH" ]; then
  EXTRA_PYTHONPATH="$DEFAULT_AUTOMODEL_PATH"
fi

# === Best-config defaults for 4-node H100, async split topology ===
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS-}"
MAX_SAMPLES="${MAX_SAMPLES:-8192}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
ROLLOUT_GENERATE_BATCH_SIZE="${ROLLOUT_GENERATE_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
ASYNC_QUEUE_SIZE="${ASYNC_QUEUE_SIZE:-2}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
# vLLM rollout side: dedicated nodes, TP+EP hybrid for MoE.
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-2}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MM_ENCODER_ATTN_BACKEND="${VLLM_MM_ENCODER_ATTN_BACKEND:-TORCH_SDPA}"
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
# omni3 is a hybrid Mamba2 model: force vLLM's SSM state cache to fp32 to match
# the fp32 training recompute. The vLLM fp16 default rounds over the rollout scan
# and drifts rollout log-probs from training — fatal here, since the rollout-vs-
# teacher per-token KL IS the distillation signal, not just a diagnostic.
VLLM_MAMBA_SSM_CACHE_DTYPE="${VLLM_MAMBA_SSM_CACHE_DTYPE:-float32}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"
# AutoModel actor side: 2 nodes (DP2), EP+CP for MoE actors (teacher colocated here).
ACTOR_NODES="${ACTOR_NODES:-2}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-8}"
FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-te}"
FREEZE_VISUAL_ENCODER="${FREEZE_VISUAL_ENCODER:-1}"
FREEZE_MOE_ROUTER="${FREEZE_MOE_ROUTER:-1}"
# Algo — distillation strength (reverse-KL reward coefficient) defaults to 1.0; tune via LR.
MOE_AUX_LOSS_COEF="${MOE_AUX_LOSS_COEF:-0}"
LR="${LR:-1e-5}"
DUAL_CLIP="${DUAL_CLIP:-10.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
EPS_CLIP_LOW="${EPS_CLIP_LOW:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DISABLE_FINAL_SAVE="${DISABLE_FINAL_SAVE:-0}"
FORCE_ON_POLICY="${FORCE_ON_POLICY:-1}"

test -f "$CONTAINER_IMAGE"
test -e "$PROMPT_DATASET"
mkdir -p "$SAVE_ROOT"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export -n VLLM_NUM_ENGINES VLLM_TP_SIZE VLLM_GPU_MEMORY_UTILIZATION VLLM_MM_ENCODER_ATTN_BACKEND VLLM_GDN_PREFILL_BACKEND VLLM_ATTENTION_BACKEND VLLM_ENFORCE_EAGER VLLM_DISTRIBUTED_EXECUTOR_BACKEND VLLM_ENABLE_EXPERT_PARALLEL

MOUNTS="${CONTAINER_MOUNTS:-$REPO_ROOT:/molt,/lustre:/lustre,$HOME/.cache:/root/.cache,/dev/shm:/dev/shm}"
CONTAINER_ARGS=(--overlap --no-container-mount-home --container-image="$CONTAINER_IMAGE" --container-mounts="$MOUNTS")

EXTRA_PYTHONPATH_EXPORT=""
if [ -n "${EXTRA_PYTHONPATH:-}" ]; then
  EXTRA_PYTHONPATH_EXPORT=" PYTHONPATH=${EXTRA_PYTHONPATH}:\${PYTHONPATH:-}"
fi

nodes="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
nodes_array=($nodes)
head_node="${nodes_array[0]}"
head_ip="$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')"
ip_head="$head_ip:$RAY_PORT"

# CUDNN_PATH/LD_LIBRARY_PATH pin TE to the bundled cuDNN 9.17.x (avoids the stale apt
# cuDNN that breaks the CP fused-attn kernel). MOLT_MOE_DISPATCHER=deepep makes the MoE
# AC recompute deterministic at long context.
ray_env="unset VLLM_NUM_ENGINES VLLM_TP_SIZE VLLM_GPU_MEMORY_UTILIZATION VLLM_MM_ENCODER_ATTN_BACKEND VLLM_GDN_PREFILL_BACKEND VLLM_ATTENTION_BACKEND VLLM_ENFORCE_EAGER VLLM_DISTRIBUTED_EXECUTOR_BACKEND VLLM_ENABLE_EXPERT_PARALLEL; cd /molt && export HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=true RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_DOCKER_CPU_WARNING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_MOE_FP16=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDNN_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-} MAX_AGENT_TURNS=${MAX_AGENT_TURNS:-1} LOAD_MODEL_ONLY=${LOAD_MODEL_ONLY:-0} NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1} NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-0} MOLT_MOE_RESHARD_AFTER_FWD=${MOLT_MOE_RESHARD_AFTER_FWD:-1} MOLT_DEFER_GRAD_SYNC=${MOLT_DEFER_GRAD_SYNC:-1} MOLT_MOE_DISPATCHER=${MOLT_MOE_DISPATCHER:-deepep}${EXTRA_PYTHONPATH_EXPORT}"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" bash -lc "
set -e
python -c 'import torch; torch.cuda.init(); print(\"torch=\" + torch.__version__ + \" device=\" + torch.cuda.get_device_name(0))' || {
  echo 'GPU preflight failed before Ray launch'
  exit 1
}
python - <<'PY'
import importlib

modules = ['timm', 'open_clip', 'ftfy', 'wcwidth', 'mamba_ssm', 'einops']
missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f'{module}: {exc}')
try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn  # noqa: F401
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
except Exception as exc:
    missing.append(f'mamba_ssm triton ops: {exc}')
if missing:
    raise SystemExit('container missing required Python runtime deps: ' + '; '.join(missing))
print('[preflight] container Python deps ready', flush=True)
PY
"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" \
  bash -lc "$ray_env && ray start --head --node-ip-address=$head_ip --port=$RAY_PORT --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT --num-gpus=$GPUS_PER_NODE --block --disable-usage-stats" &

sleep 20

for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
  node_i="${nodes_array[$i]}"
  srun --nodes=1 --ntasks=1 -w "$node_i" "${CONTAINER_ARGS[@]}" \
    bash -lc "$ray_env && ray start --address=$ip_head --num-gpus=$GPUS_PER_NODE --block --disable-usage-stats" &
done

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" bash -lc "$ray_env && python - <<PY
import subprocess
import time

expected = $SLURM_JOB_NUM_NODES
while True:
    result = subprocess.run(['ray', 'status'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    count = 0
    active = False
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped == 'Active:':
            active = True
            continue
        if stripped == 'Pending:':
            active = False
            continue
        if active:
            parts = stripped.split()
            if parts and parts[0].isdigit():
                count += int(parts[0])
    if count == expected:
        break
    print(result.stdout, flush=True)
    time.sleep(5)
PY"

RL_ARGS=(
  --actor.model_name_or_path "$MODEL_PATH"
  --ref.model_name_or_path "$REF_MODEL_PATH"
  --data.prompt_dataset "$PROMPT_DATASET"
  --data.input_key "${INPUT_KEY:-prompt}"
  --data.label_key "${LABEL_KEY:-reward_model}"
  --data.tools_key "${TOOLS_KEY:-tools}"
  --data.apply_chat_template
  --data.image_key "${IMAGE_KEY:-images}"
  --data.max_images_per_prompt "$MAX_IMAGES_PER_PROMPT"
  --data.max_samples "$MAX_SAMPLES"
  --data.max_len "$MAX_LENGTH"
  ${MAX_NEW_TOKENS:+--rollout.max_new_tokens=$MAX_NEW_TOKENS}
  --rollout.batch_size "$ROLLOUT_BATCH_SIZE"
  --rollout.vllm_generate_batch_size "$ROLLOUT_GENERATE_BATCH_SIZE"
  --rollout.micro_batch_size 1
  --rollout.n_samples_per_prompt "$N_SAMPLES_PER_PROMPT"
  --rollout.temperature "$TEMPERATURE"
  --rollout.top_p "${TOP_P:-1.0}"
  --train.batch_size "$TRAIN_BATCH_SIZE"
  --train.micro_batch_size "$MICRO_BATCH_SIZE"
  --train.max_epochs "${MAX_EPOCHS:-1}"
  --train.num_episodes "${NUM_EPISODES:-1}"
  --train.async_queue_size "$ASYNC_QUEUE_SIZE"
  --train.colocate_fsdp_models
  --actor.num_nodes "$ACTOR_NODES"
  --actor.num_gpus_per_node "$ACTOR_GPUS_PER_NODE"
  --ref.num_nodes "$ACTOR_NODES"
  --ref.num_gpus_per_node "$ACTOR_GPUS_PER_NODE"
  --vllm.num_engines "$VLLM_NUM_ENGINES"
  --vllm.tensor_parallel_size "$VLLM_TP_SIZE"
  --vllm.sync_backend nccl
  --vllm.gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm.mm_encoder_attn_backend "$VLLM_MM_ENCODER_ATTN_BACKEND"
  --vllm.gdn_prefill_backend "$VLLM_GDN_PREFILL_BACKEND"
  --vllm.attention_backend "$VLLM_ATTENTION_BACKEND"
  --vllm.mamba_ssm_cache_dtype "$VLLM_MAMBA_SSM_CACHE_DTYPE"
  --vllm.distributed_executor_backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND"
  --fsdp.param_dtype bf16
  --fsdp.attn_implementation "$FSDP_ATTN_IMPLEMENTATION"
  --fsdp.tp_size "$TP_SIZE"
  --fsdp.ep_size "$EP_SIZE"
  --fsdp.cp_size "$CP_SIZE"
  --actor.gradient_checkpoint "$GRAD_CHECKPOINT"
  --actor.optim "${OPTIM:-adam}"
  --actor.adam.lr "$LR"
  --actor.adam.betas "$ADAM_BETA1" "$ADAM_BETA2"
  --actor.adam.weight_decay "$WEIGHT_DECAY"
  --actor.eps_clip_low_high "$EPS_CLIP_LOW" "$EPS_CLIP_HIGH"
  --actor.dual_clip "$DUAL_CLIP"
  --actor.aux_loss_coef "$MOE_AUX_LOSS_COEF"
  --actor.entropy_coef "${ENTROPY_COEF:-0.0}"
  # On-policy distillation: one switch. The teacher is --ref.model_name_or_path (above);
  # k1 reverse-KL-as-advantage + kl.use_loss=off are derived. Strength defaults to 1.0;
  # override with --algo.kl.init_coef.
  --algo.advantage.estimator on_policy_distill
  # TIS still matters: tokens are sampled by vLLM but scored with FSDP-recomputed logprobs.
  --algo.advantage.is_correction_enable
  --algo.advantage.is_correction_type seq-mask-tis
  --algo.advantage.is_correction_threshold "${IS_LOW:-0.5}" "${IS_HIGH:-2.0}"
  --reward.clip_range -1 1
  --ckpt.output_dir "$SAVE_ROOT/hf"
  --ckpt.path "$SAVE_ROOT/state"
  --ckpt.save_steps "${SAVE_STEPS:-5}"
  --ckpt.max_num "${CKPT_MAX_NUM:-50}"
  --logger.logging_steps 1
  --logger.wandb.project "${WANDB_PROJECT:-molt_distill_omni3}"
  --logger.wandb.run_name "${WANDB_RUN_NAME:-distill_omni3_$SLURM_JOB_ID}"
)

# Only pass an agent if overridden; otherwise on_policy_distill defaults to the built-in generator.
if [ -n "$AGENT_PATH" ]; then
  RL_ARGS+=(--train.agent_path "$AGENT_PATH")
fi

if [ "${PARTIAL_ROLLOUT:-0}" = "1" ]; then
  RL_ARGS+=(--train.partial_rollout_enable)
fi

if [ "$FORCE_ON_POLICY" = "1" ]; then
  RL_ARGS+=(--train.force_on_policy)
fi

if [ "$VLLM_ENFORCE_EAGER" = "1" ]; then
  RL_ARGS+=(--vllm.enforce_eager)
fi

# Raise MAX_LENGTH to 32K under the colocated teacher by streaming optimizer+grads to CPU.
# --fsdp.offload level: FSDP_CPU_OFFLOAD=1 -> full (params to CPU; breaks MoE),
# OFFLOAD_OPTIMIZER=1 -> optimizer (AdamW step on CPU, params stay on GPU; MoE-safe).
FSDP_OFFLOAD=none
[ "${FSDP_CPU_OFFLOAD:-0}" = "1" ] && FSDP_OFFLOAD=full
[ "${OFFLOAD_OPTIMIZER:-0}" = "1" ] && FSDP_OFFLOAD=optimizer
[ "$FSDP_OFFLOAD" != "none" ] && RL_ARGS+=(--fsdp.offload "$FSDP_OFFLOAD")

if [ "$VLLM_ENABLE_EXPERT_PARALLEL" = "1" ]; then
  RL_ARGS+=(--vllm.enable_expert_parallel)
fi

if [ "$FREEZE_VISUAL_ENCODER" = "1" ]; then
  RL_ARGS+=(--actor.freeze_visual_encoder)
fi

if [ "$FREEZE_MOE_ROUTER" = "1" ]; then
  RL_ARGS+=(--actor.freeze_moe_router)
fi

# omni3 is MoE -> R3 (routing replay) on by default: replay the rollout's per-token expert
# selection so the student's training router matches its rollout router (lower vllm_kl). Set
# ROUTING_REPLAY=0 to disable. Incompatible with partial rollout (keep PARTIAL_ROLLOUT off).
if [ "${ROUTING_REPLAY:-1}" != "0" ]; then
  RL_ARGS+=(--train.routing_replay)
fi

if [ "${LOAD_ENABLE:-0}" = "1" ]; then
  RL_ARGS+=(--ckpt.load_enable)
fi

if [ -n "$EVAL_DATASET" ]; then
  RL_ARGS+=(--eval.dataset "$EVAL_DATASET" --eval.steps "${EVAL_STEPS:-5}" --eval.n_samples_per_prompt "${EVAL_N_SAMPLES_PER_PROMPT:-4}")
  if [ "${EVAL_AT_START:-0}" = "1" ]; then
    RL_ARGS+=(--eval.eval_at_start)
  fi
else
  RL_ARGS+=(--eval.steps -1)
fi

if [ "${ENABLE_DYNAMIC_FILTERING:-0}" = "1" ]; then
  RL_ARGS+=(--algo.dynamic_filtering_enable --algo.dynamic_filtering_range "${FILTER_MIN:-0.01}" "${FILTER_MAX:-0.99}")
fi

if [ "$DISABLE_FINAL_SAVE" = "1" ]; then
  RL_ARGS+=(--ckpt.disable_final_save)
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  RL_ARGS+=(--logger.wandb.key "$WANDB_API_KEY")
fi

RL_ARGS+=("$@")

printf -v RL_ARGS_Q " %q" "${RL_ARGS[@]}"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" \
  bash -lc "$ray_env && ray job submit --address=http://localhost:$DASHBOARD_PORT -- bash -lc 'cd /molt && python3 -u -m molt.cli.train_rl_ray$RL_ARGS_Q'"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" bash -lc "$ray_env && ray stop --force" || true
