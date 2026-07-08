#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=your_slurm_account
#SBATCH --partition=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=12
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=molt-vrl-qwen35-397b
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
export MOLT_PATH="$REPO_ROOT"

# Qwen3.5-397B-A17B (model_type=qwen3_5_moe, Qwen3_5MoeForConditionalGeneration):
# ~397B VLM MoE, same family as Qwen3.6-35B-A3B (rl_qwen3_6_35b.sh) but far larger.
# AutoModel's custom MoE parallelizer asserts TP=1. 12-node split: 8 actor/ref
# nodes (64 GPUs) + 4 vLLM rollout nodes (32 GPUs). vLLM 0.23 natively serves it.
export MODEL_PATH="${MODEL_PATH:-/path/to/models/Qwen3.5-397B-A17B}"
# TP MUST be 1 (custom MoE parallelizer). EP=64 = full non-PP world (dp4*cp16*tp1=64);
# 512 experts / 64 = 8 experts/rank (512 % 64 == 0).
export TP_SIZE="${TP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-64}"
# Activation checkpointing ON (safe under the hybridep MoE dispatcher; deterministic
# recompute).
export GRAD_CHECKPOINT="${GRAD_CHECKPOINT-full}"
# CP=16, te-native CP path (attn=te) for this 12-node run at 32K context:
#  * dp = world 64 / cp16 = 4, and train.batch_size=8 >= dp=4 passes the
#    "num sample batches >= actor processes" assert (CP=1 → dp=64 would fail it).
#  * CP=16 shards 32K to 2048 tok/rank — more activation headroom than CP=8 for the
#    397B at 32K. cp is the innermost mesh axis, so cp16 = 16 ranks = 2 nodes: the
#    GDN linear-attn full-seq all-gather now crosses one node boundary. If it hangs
#    on that all-gather (as CP=32 across 4 nodes did), drop to CP=8 (intra-node/NVLink).
#  * 2*cp=32 divides MAX_LENGTH (32768/32 = 1024).
export CP_SIZE="${CP_SIZE:-16}"
# 32K context. CP=16 shards non-GDN activations to 32768/16=2048 tok/rank
# (GDN layers all-gather the full sequence).
export MAX_LENGTH="${MAX_LENGTH:-32768}"
# Adam optimizer offload (fp32 master + Adam moments on CPU during the step).
# Essential to fit a 397B optimizer state off-GPU on 64 GPUs without PP.
export OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-1}"
# FSDP param CPU offload OFF (Qwen3.5-MoE has upstream device-mismatch bugs under
# full offload). Control GPU memory via EP + CP + adam-offload + AC.
export FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-0}"
# VLM+CP requires a frozen visual encoder (base.py:503).
export FREEZE_VISUAL_ENCODER="${FREEZE_VISUAL_ENCODER:-1}"
# Backend: attn=te enables the qwen3_5_moe CP path (see CP_SIZE); linear=torch per
# the recipe; experts=torch_mm because grouped_gemm (gmm) isn't in molt-cu13 —
# torch_mm uses the same GroupedExpertsDeepEP dispatcher (correct, a bit slower);
# dispatcher=hybridep (needs deep_ep, which is in molt-cu13).
export FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-te}"
export MOLT_LINEAR_BACKEND="${MOLT_LINEAR_BACKEND:-torch}"
export MOLT_MOE_EXPERTS="${MOLT_MOE_EXPERTS:-torch_mm}"
export MOLT_MOE_DISPATCHER="${MOLT_MOE_DISPATCHER:-hybridep}"
# Qwen3.5-MoE has GDN (gated-delta-net) linear-attention layers; vLLM needs a GDN
# prefill backend for them.
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
# 50-step stability/correctness smoke. max_steps =
# len(dataset)//rollout_batch * num_episodes = 50//1 * 1 = 50.
export LR="${LR:-2e-6}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export MAX_SAMPLES="${MAX_SAMPLES:-50}"
export NUM_EPISODES="${NUM_EPISODES:-1}"
export ASYNC_QUEUE_SIZE="${ASYNC_QUEUE_SIZE:-1}"
# R3 rollout routing replay ON (automodel-r3 PR#2797 + molt --train.routing_replay).
export ROUTING_REPLAY="${ROUTING_REPLAY:-1}"
# Chat agent (loopback OpenAI SDK harness), NOT the STEP runner.
export AGENT_PATH="${AGENT_PATH:-/molt/examples/python/agents/chat_geo3k.py}"
# 50-step smoke: don't persist a 397B checkpoint (skip intermediate + final saves).
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export DISABLE_FINAL_SAVE="${DISABLE_FINAL_SAVE:-1}"
# Sparse pass@1 eval so the 50-step run isn't dominated by eval rollouts.
export EVAL_STEPS="${EVAL_STEPS:-25}"
export EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
# Dynamic filtering OFF (zero-advantage groups pass through; stable throughput).
export ENABLE_DYNAMIC_FILTERING="${ENABLE_DYNAMIC_FILTERING:-0}"
# Per-turn generation cap (the multi-turn chat agent accumulates within MAX_LENGTH).
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
# Actor/ref: 8 dedicated training nodes (64 GPUs).
export ACTOR_NODES="${ACTOR_NODES:-8}"
# vLLM rollout on the other 4 nodes (32 GPUs). 397B bf16 (~807GB) does NOT fit 1
# node (807/8=101GB/GPU); default = 2 engines × TP16 (EP16), each spanning 2 nodes
# (807/16=50GB/GPU), ray executor for cross-node TP. qwen3_5_moe has no MSA block
# constraint, so standard TP16 works. kv_heads=2 < 16 → vLLM replicates KV (fine).
export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-2}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-16}"
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-ray}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

# Stable SAVE_ROOT (no $SLURM_JOB_ID) so resubmits can resume via LOAD_ENABLE=1.
export SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/async-visual-rl-qwen35-397b/run}"
export WANDB_PROJECT="${WANDB_PROJECT:-molt_visual_rl_qwen35_397b}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen35_397b_visual_$SLURM_JOB_ID}"

# Idle-reaper exemption: async RL idles the actor GPUs during eval@0 and between
# train steps, so without this the idle-GPU reaper kills the job. Pass the same
# --comment on the initial submit and every chain successor so the whole chain is
# exempt (EVAL_AT_START=1 especially needs it).
IDLE_EXEMPT_MINS="${IDLE_EXEMPT_MINS:-120}"
SBATCH_COMMENT="${SBATCH_COMMENT:-$(printf '{"IdleGpuReaper":{"exemptIdleTimeMins":"%s","reason":"other","description":"Async RL VLM: actor GPUs idle during eval@0 and rollout between train steps"}}' "$IDLE_EXEMPT_MINS")}"

# Chain auto-resubmit: queue the next slot with afterany dependency so the
# run survives walltime / preemption. LOAD_ENABLE=1 makes the successor
# resume from $SAVE_ROOT/state. Bound the chain with CHAIN_MAX.
CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
CHAIN_MAX="${CHAIN_MAX:-0}"
if [ "$CHAIN_DEPTH" -lt "$CHAIN_MAX" ]; then
  NEXT_DEPTH=$((CHAIN_DEPTH + 1))
  next_jobid=$(CHAIN_DEPTH="$NEXT_DEPTH" LOAD_ENABLE=1 SBATCH_COMMENT="$SBATCH_COMMENT" \
    sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
    --account="$SLURM_JOB_ACCOUNT" \
    --partition="$SLURM_JOB_PARTITION" \
    ${SLURM_JOB_QOS:+--qos="$SLURM_JOB_QOS"} \
    ${SLURM_JOB_RESERVATION:+--reservation="$SLURM_JOB_RESERVATION"} \
    --nodes="$SLURM_JOB_NUM_NODES" \
    --comment="$SBATCH_COMMENT" \
    "$REPO_ROOT/examples/scripts/slurm/rl_qwen3_5_397b.sh")
  echo "[chain] depth=$NEXT_DEPTH/$CHAIN_MAX next_jobid=$next_jobid"
fi

# Use $REPO_ROOT, not $SCRIPT_DIR — Slurm copies the submitted wrapper to its
# spool directory, so $SCRIPT_DIR points there and the launcher isn't co-located.
# === Inlined launcher ===
set --
set -x

REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
# molt-cu13.sqsh = vLLM 0.23 (+vllm_router, deep_ep, TE) — natively serves
# qwen3_5_moe. Default points at $REPO_ROOT/images (override CONTAINER_IMAGE).
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$REPO_ROOT/images/molt-cu13.sqsh}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the VLM checkpoint to train.}"

# Default to the prepared geo3k VLM subset (math multi-turn).
# Override PROMPT_DATASET / EVAL_DATASET to swap in your own data.
DEFAULT_DATA_DIR="$REPO_ROOT/.tmp/geo3k"
if [ -z "${PROMPT_DATASET:-}" ] && [ ! -d "$DEFAULT_DATA_DIR/train" ]; then
  echo "[launcher] preparing geo3k VLM (VeraIsHere/geo3k_imgurl_processed) — one-time"
  python3 "$REPO_ROOT/examples/python/utils/prepare_geo3k.py" \
    --max-eval 256 --num-proc 8 --out-dir "$DEFAULT_DATA_DIR"
fi
PROMPT_DATASET="${PROMPT_DATASET:-$DEFAULT_DATA_DIR/train}"
EVAL_DATASET="${EVAL_DATASET:-$DEFAULT_DATA_DIR/eval}"

SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/molt-async-visual-rl/$SLURM_JOB_ID}"
AGENT_PATH="${AGENT_PATH:-/molt/examples/python/agents/geo3k.py}"

# R3 (rollout routing replay, PR#2797) + qwen3_5_moe te-native CP live on the
# automodel-r3 checkout — point PYTHONPATH there so it wins over the container's
# baked AutoModel. (molt-cu13 already ships vllm_router, so no extra pylib needed.)
DEFAULT_AUTOMODEL_PATH=/path/to/Automodel
if [ -z "${EXTRA_PYTHONPATH:-}" ] && [ -d "$DEFAULT_AUTOMODEL_PATH" ]; then
  EXTRA_PYTHONPATH="$DEFAULT_AUTOMODEL_PATH"
fi

# === Best-config defaults for 2-node interactive H100, async split topology ===
# Override any of these from the wrapper (model-specific) or sbatch env (per-run).
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
# MAX_LENGTH is the shared total-context budget (prompt + generation), used as
# both data.max_len and vLLM max_model_len; the 64k default gives headroom for
# vision tokens + multi-turn history. Longest prompt must satisfy
# prompt_len + MAX_NEW_TOKENS <= MAX_LENGTH.
MAX_LENGTH="${MAX_LENGTH:-65536}"
# MAX_NEW_TOKENS: per-turn generation cap. Unset = unlimited (bounded by
# MAX_LENGTH); multi-turn agents issue many turns, so this is a per-call max,
# not a context budget.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS-}"
MAX_SAMPLES="${MAX_SAMPLES:-8192}"
# rollout_batch_size = unique prompts dispatched per make_experience call.
# Keep rollout_batch_size * n_samples >= train_batch_size or the policy_train
# loop drops trailing microbatches; the default sizes one rollout per grad step.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
ROLLOUT_GENERATE_BATCH_SIZE="${ROLLOUT_GENERATE_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
# train_batch_size = trajectories per gradient step (rollout_batch × n_samples).
# Raise (e.g. 2048) for stability runs; step time scales linearly.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
# micro_batch_size=1 for large MoE actors: long prompts + generations push
# activation memory past 80GB even with grad checkpointing + colocated actor/ref.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
# Pure async + partial rollout: queue depth >= 2 so train overlaps next rollout.
ASYNC_QUEUE_SIZE="${ASYNC_QUEUE_SIZE:-1}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
# vLLM rollout side: dedicated full node, TP+EP hybrid for MoE.
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MM_ENCODER_ATTN_BACKEND="${VLLM_MM_ENCODER_ATTN_BACKEND:-TORCH_SDPA}"
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
# Qwen3.x GDN is a recurrent linear-attention state, the same class as omni3's
# Mamba2 SSM: force vLLM's recurrent state cache to fp32 to match the fp32
# training recompute. The bf16 default rounds per token across a long rollout,
# so rollout log-probs drift from training and vllm_kl climbs with steps.
VLLM_MAMBA_SSM_CACHE_DTYPE="${VLLM_MAMBA_SSM_CACHE_DTYPE:-float32}"
# Leave unset so vLLM auto-selects the fastest backend for the arch/driver
# (FlashAttention-3 / FlashInfer on Hopper) — matches verl (attention_backend=auto).
# Pin TRITON_ATTN on older drivers whose AOT-compiled FA2 PTX fails
# (cudaErrorUnsupportedPtxVersion), or FLASH_ATTN to force FA.
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
# TP_SIZE / EP_SIZE / CP_SIZE / FSDP_ATTN_IMPLEMENTATION / ACTOR_NODES /
# FREEZE_VISUAL_ENCODER are the model-specific values set at the top of this file
# (TP=1, EP=64, CP=16, attn=te, 8 actor nodes). TP=1 is mandatory (the custom MoE
# parallelizer asserts it); CP=16 uses the te-native CP path — cp is the innermost
# mesh axis, so the CP group spans two adjacent nodes.
# Router NOT frozen: R3 (routing_replay) already keeps rollout/train routing
# consistent by replaying the top-k SELECTION, while the router logits stay live
# so the gradient keeps flowing (the router keeps learning). Freezing is the
# heavier hammer that halts that learning — redundant with R3. Set =1 to freeze.
FREEZE_MOE_ROUTER="${FREEZE_MOE_ROUTER:-0}"
# Algo — defaults tuned for geo3k VLM math multi-turn.
KL_COEF="${KL_COEF:-0.0}"
MOE_AUX_LOSS_COEF="${MOE_AUX_LOSS_COEF:-0.000}"
LR="${LR:-2e-6}"
DUAL_CLIP="${DUAL_CLIP:-10.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
EPS_CLIP_LOW="${EPS_CLIP_LOW:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DISABLE_FINAL_SAVE="${DISABLE_FINAL_SAVE:-0}"
# On-policy: each multi-turn rollout is consumed in a single optimizer step
# (matches the omni3 run). Requires MAX_EPOCHS=1. Set FORCE_ON_POLICY=0 to fall
# back to grad-accum windows.
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
export -n VLLM_NUM_ENGINES VLLM_TP_SIZE VLLM_GPU_MEMORY_UTILIZATION VLLM_MM_ENCODER_ATTN_BACKEND VLLM_GDN_PREFILL_BACKEND VLLM_MAMBA_SSM_CACHE_DTYPE VLLM_ATTENTION_BACKEND VLLM_ENFORCE_EAGER VLLM_DISTRIBUTED_EXECUTOR_BACKEND VLLM_ENABLE_EXPERT_PARALLEL

# Mount the host's /dev/shm into the container. vLLM's multiproc executor uses
# shared memory for tensor-parallel broadcast (see vLLM docs/deployment/docker.md
# §33-35); without this, Pyxis's default tmpfs is too small and the EngineCore
# stalls on "No available shared memory broadcast block found in 60 seconds".
MOUNTS="${CONTAINER_MOUNTS:-$REPO_ROOT:/molt,/lustre:/lustre,$HOME/.cache:/root/.cache,/dev/shm:/dev/shm}"
CONTAINER_ARGS=(--overlap --no-container-mount-home --container-image="$CONTAINER_IMAGE" --container-mounts="$MOUNTS")

# Optional source override, e.g. a sibling AutoModel checkout. Runtime Python
# dependencies are expected to be baked into the container image.
EXTRA_PYTHONPATH_EXPORT=""
if [ -n "${EXTRA_PYTHONPATH:-}" ]; then
  EXTRA_PYTHONPATH_EXPORT=" PYTHONPATH=${EXTRA_PYTHONPATH}:\${PYTHONPATH:-}"
fi

nodes="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
nodes_array=($nodes)
head_node="${nodes_array[0]}"
head_ip="$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')"
ip_head="$head_ip:$RAY_PORT"

# vLLM >=0.20 ignores the legacy VLLM_ATTENTION_BACKEND env var — the backend is
# plumbed through --vllm.attention_backend instead (the MoE FlashInfer fallback
# still reads VLLM_USE_FLASHINFER_MOE_FP16=0). expandable_segments curbs
# fragmentation when long generations push the colocated actor+ref near 80GB.
ray_env="unset VLLM_NUM_ENGINES VLLM_TP_SIZE VLLM_GPU_MEMORY_UTILIZATION VLLM_MM_ENCODER_ATTN_BACKEND VLLM_GDN_PREFILL_BACKEND VLLM_MAMBA_SSM_CACHE_DTYPE VLLM_ATTENTION_BACKEND VLLM_ENFORCE_EAGER VLLM_DISTRIBUTED_EXECUTOR_BACKEND VLLM_ENABLE_EXPERT_PARALLEL; cd /molt && export HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=true RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_DOCKER_CPU_WARNING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_MOE_FP16=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAX_AGENT_TURNS=${MAX_AGENT_TURNS:-10} LOAD_MODEL_ONLY=${LOAD_MODEL_ONLY:-0} NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1} TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0} TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000} CUDNN_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-} NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1} NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-0} MOLT_MOE_DISPATCHER=${MOLT_MOE_DISPATCHER:-deepep} MOLT_LINEAR_BACKEND=${MOLT_LINEAR_BACKEND:-} MOLT_MOE_EXPERTS=${MOLT_MOE_EXPERTS:-} MOLT_MOE_RESHARD_AFTER_FWD=${MOLT_MOE_RESHARD_AFTER_FWD:-1} UCCL_IB_GID_INDEX=${UCCL_IB_GID_INDEX:-3} UCCL_IB_HCA=${UCCL_IB_HCA:-} NCCL_IB_HCA=${NCCL_IB_HCA:-} NVSHMEM_HCA_LIST=${NVSHMEM_HCA_LIST:-} NVSHMEM_IB_GID_INDEX=${NVSHMEM_IB_GID_INDEX:-} NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-} MOLT_FSDP_DEBUG_GRADS=${MOLT_FSDP_DEBUG_GRADS:-} MOLT_FSDP_DEBUG_GRADS_FILTER=${MOLT_FSDP_DEBUG_GRADS_FILTER:-} MOLT_FSDP_DEBUG_GRADS_TOPK=${MOLT_FSDP_DEBUG_GRADS_TOPK:-8}${EXTRA_PYTHONPATH_EXPORT}"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" bash -lc "
set -e
# GPU preflight.
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

# --data.apply_chat_template: the dataset rows are chat messages — one format for BOTH runner
# types. The STEP runner renders them dataset-side; a CHAT runner hands them through raw and the
# chat server renders once with the model's own template (Runner.PRERENDER_PROMPT decides).
RL_ARGS=(
  --actor.model_name_or_path "$MODEL_PATH"
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
  $([ "${ROUTING_REPLAY:-1}" != 0 ] && echo --train.routing_replay || true)
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
  --actor.muon.lr "${MUON_LR:-2e-4}"
  --actor.muon.momentum "${MUON_MOMENTUM:-0.95}"
  --actor.muon.weight_decay "${MUON_WEIGHT_DECAY:-0.0}"
  --actor.muon.ns_steps "${MUON_NS_STEPS:-5}"
  --actor.eps_clip_low_high "$EPS_CLIP_LOW" "$EPS_CLIP_HIGH"
  --actor.dual_clip "$DUAL_CLIP"
  --actor.aux_loss_coef "$MOE_AUX_LOSS_COEF"
  --actor.entropy_coef "${ENTROPY_COEF:-0.0}"
  --algo.advantage.estimator reinforce_baseline
  --algo.advantage.is_correction_enable
  --algo.advantage.is_correction_type seq-mask-tis
  --algo.advantage.is_correction_threshold "${IS_LOW:-0.99}" "${IS_HIGH:-1.01}"
  --algo.kl.use_loss
  --algo.kl.estimator k2
  --algo.kl.init_coef "$KL_COEF"
  --reward.clip_range -10 10
  --ckpt.output_dir "$SAVE_ROOT/hf"
  --ckpt.path "$SAVE_ROOT/state"
  --ckpt.save_steps "${SAVE_STEPS:-5}"
  --ckpt.max_num "${CKPT_MAX_NUM:-50}"
  --logger.logging_steps 1
  --logger.wandb.project "${WANDB_PROJECT:-molt_async_visual_rl}"
  --logger.wandb.run_name "${WANDB_RUN_NAME:-visual_rl_$SLURM_JOB_ID}"
)

RL_ARGS+=(--train.agent_path "$AGENT_PATH")

if [ "$VLLM_ENFORCE_EAGER" = "1" ]; then
  RL_ARGS+=(--vllm.enforce_eager)
fi

# Empty VLLM_ATTENTION_BACKEND = let vLLM auto-select (FA3/FlashInfer on Hopper);
# only pass the flag when pinned (argparse rejects an empty --vllm.attention_backend).
[ -n "$VLLM_ATTENTION_BACKEND" ] && RL_ARGS+=(--vllm.attention_backend "$VLLM_ATTENTION_BACKEND")

if [ "$FORCE_ON_POLICY" = "1" ]; then
  RL_ARGS+=(--train.force_on_policy)
fi

# Partial rollout (async gen/train overlap, ~2x throughput) is OFF by default: a
# rollout spanning a weight broadcast carries off-policy prefix tokens. Set
# PARTIAL_ROLLOUT=1 to enable — the off-policy prefix is then masked out of the
# loss (zero gradient, excluded from the token-mean denominator).
if [ "${PARTIAL_ROLLOUT:-0}" = "1" ]; then
  RL_ARGS+=(--train.partial_rollout_enable)
fi

# CPU-offload level (--fsdp.offload), from the env knobs (optimizer takes priority):
#   OFFLOAD_OPTIMIZER=1 -> 'optimizer': AdamW step on CPU, params stay on GPU (MoE-safe).
#   FSDP_CPU_OFFLOAD=1  -> 'full': also stream params to CPU (~15GB/rank, but breaks MoE).
FSDP_OFFLOAD=none
[ "${FSDP_CPU_OFFLOAD:-0}" = "1" ] && FSDP_OFFLOAD=full
[ "${OFFLOAD_OPTIMIZER:-0}" = "1" ] && FSDP_OFFLOAD=optimizer
[ "$FSDP_OFFLOAD" != "none" ] && RL_ARGS+=(--fsdp.offload "$FSDP_OFFLOAD")

# Sequence parallelism within the TP region is OFF by default (matches AutoModel's
# omni / Qwen3.5-MoE recipes; SP gives norm weights a _NormPartial placement that
# hangs the 2D TP+FSDP state-dict load on the HF-fallback path). Opt in with
# SEQUENCE_PARALLEL=1.
if [ "${SEQUENCE_PARALLEL:-}" = "1" ]; then
  RL_ARGS+=(--fsdp.sequence_parallel)
fi

if [ "$VLLM_ENABLE_EXPERT_PARALLEL" = "1" ]; then
  RL_ARGS+=(--vllm.enable_expert_parallel)
fi

if [ "$FREEZE_VISUAL_ENCODER" = "1" ]; then
  RL_ARGS+=(--actor.freeze_visual_encoder)
fi

if [ "$FREEZE_MOE_ROUTER" = "1" ]; then
  RL_ARGS+=(--actor.freeze_moe_router)
fi

# Resume from $SAVE_ROOT/state. Set LOAD_ENABLE=1 (with SAVE_ROOT pointing at a
# prior run) to chain RL across slurm allocations, since convergence typically
# exceeds a single walltime window.
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

if [ "${ENABLE_DYNAMIC_FILTERING:-1}" = "1" ]; then
  RL_ARGS+=(--algo.dynamic_filtering_enable --algo.dynamic_filtering_range "${FILTER_MIN:-0.01}" "${FILTER_MAX:-0.99}")
fi

if [ "$DISABLE_FINAL_SAVE" = "1" ]; then
  RL_ARGS+=(--ckpt.disable_final_save)
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  RL_ARGS+=(--logger.wandb.key "$WANDB_API_KEY")
fi

# Forward any positional args from caller wrappers (e.g. --fsdp.packing_samples).
RL_ARGS+=("$@")

printf -v RL_ARGS_Q " %q" "${RL_ARGS[@]}"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" \
  bash -lc "$ray_env && ray job submit --address=http://localhost:$DASHBOARD_PORT -- bash -lc 'cd /molt && python3 -u -m molt.cli.train_rl_ray$RL_ARGS_Q'"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" bash -lc "$ray_env && ray stop --force" || true
