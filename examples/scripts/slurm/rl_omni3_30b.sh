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
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=molt-rl-omni3
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
export MOLT_PATH="$REPO_ROOT"

# Nemotron Omni3 (NemotronH_Nano_Omni_Reasoning_V3) — VLM with a RADIO vision
# encoder + NemotronH hybrid Mamba2/Attention MoE LLM. Visual multi-turn tool-use
# RL on geo3k, 4 nodes (2 actor DP2 + 2 vLLM), Adam.

# --- Native AutoModel path ---------------------------------------------------
# Omni3 registers in AutoModel as the native NemotronOmniForConditionalGeneration
# (custom MoE/EP parallelizer + TE attn). If the "[AutoModel] WARNING: no native
# ... falling back to HuggingFace" line appears, bump requirements.txt / the
# sibling Automodel checkout so the architecture re-registers.
# Use the omni3 GA checkpoint (the aligned release), not an mcore->hf shim.
export MODEL_PATH="${MODEL_PATH:-/path/to/models/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
export TP_SIZE="${TP_SIZE:-1}"        # AutoModel custom MoE asserts TP=1
export EP_SIZE="${EP_SIZE:-8}"        # only model-state shard knob for this MoE
export CP_SIZE="${CP_SIZE:-8}"        # CP8 fits 32K on the 2-node DP2 actor (omni3 SFT-validated); hybrid-SSM CP fix is in
export GRAD_CHECKPOINT="${GRAD_CHECKPOINT-full}"   # all blocks activation-checkpointed
# Custom AutoModel models reject flash_attention_2 (it silently falls to sdpa);
# TE is the intended fused-attention backend for the native path.
export FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-te}"

# --- Sequence / batch shape -------------------------------------------------
# The deepep MoE dispatcher (actor.py default) makes expert routing/recompute
# deterministic, so activation checkpointing is safe at long context; the legacy
# torch dispatcher drifts the recompute and raises CheckpointError. Memory tiers
# on one 8-GPU actor node: CP=1 fits ~16K, 32K needs CP8.
export MAX_LENGTH="${MAX_LENGTH:-32000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS-}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export ROLLOUT_GENERATE_BATCH_SIZE="${ROLLOUT_GENERATE_BATCH_SIZE:-64}"  # = ROLLOUT_BATCH_SIZE: dispatch all prompts in one vLLM batch
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_EPISODES="${NUM_EPISODES:-5}"   # >=5 so RL doesn't cap at ~33 steps
export ASYNC_QUEUE_SIZE="${ASYNC_QUEUE_SIZE:-1}"
export MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-10}"   # multi-turn cap; trajectory also capped by MAX_LENGTH (32K)
export MAX_SAMPLES="${MAX_SAMPLES:-65536}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"

# --- Optimizer / algo (Adam; aligned with the qwen3.6 recipe) ---------------
export OPTIM="${OPTIM:-adam}"
export LR="${LR:-1e-5}"

# --- Eval / checkpoint ------------------------------------------------------
export SAVE_STEPS="${SAVE_STEPS:-2}"
export EVAL_STEPS="${EVAL_STEPS:-10}"
export EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
# Baseline eval at step 0 (pre-RL model) on fresh runs, so pass1 gains are
# attributable. No-op on resume (global_step starts > 0). Set EVAL_AT_START=0 to skip.
export EVAL_AT_START="${EVAL_AT_START:-1}"
export ENABLE_DYNAMIC_FILTERING="${ENABLE_DYNAMIC_FILTERING:-0}"

# --- Dataset: geo3k in <answer>...</answer> format --------------------------
# Omni3 was post-trained to answer in <answer>…</answer> (not \boxed{}); the
# geo3k grader accepts both. Build the answer-format subset on first run.
export PROMPT_DATASET="${PROMPT_DATASET:-$REPO_ROOT/.tmp/geo3k_answer/train}"
export EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/geo3k_answer/eval}"
if [ ! -d "$REPO_ROOT/.tmp/geo3k_answer/train" ]; then
  echo "[omni3] preparing geo3k (<answer> format) — one-time"
  python3 "$REPO_ROOT/examples/python/utils/prepare_geo3k.py" \
    --answer-format answer --max-eval 256 --num-proc 8 \
    --out-dir "$REPO_ROOT/.tmp/geo3k_answer"
fi

# Fail early with a clear hint if the Omni3 shim checkpoint isn't built yet.
if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "[omni3] MODEL_PATH not found: $MODEL_PATH" >&2
  echo "[omni3] build the shim first, e.g.:" >&2
  echo "        python3 $REPO_ROOT/.tmp/scripts/build_nemotron_omni_v3_shim.py" >&2
  exit 1
fi

# Dedicated output dir so Omni3 checkpoints don't clobber the qwen3.6 run.
export SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/rl-omni3/run}"
export WANDB_PROJECT="${WANDB_PROJECT:-molt_visual_rl_omni3}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-rl_omni3_$SLURM_JOB_ID}"

# Chain auto-resubmit: queue the next slot with afterany dependency so the run
# survives walltime / preemption. LOAD_ENABLE=1 makes the successor resume from
# $SAVE_ROOT/state. Disabled by default (CHAIN_MAX=0); set CHAIN_MAX>0 to bound.
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
    --comment='{"IdleGpuReaper":{"exemptIdleTimeMins":"120","reason":"other","description":"Async RL split actor+vLLM; GPUs idle as train/rollout/eval alternate; omni3 32K MoE"}}' \
    "$REPO_ROOT/examples/scripts/slurm/rl_omni3_30b.sh")
  echo "[chain] depth=$NEXT_DEPTH/$CHAIN_MAX next_jobid=$next_jobid"
fi

# Use $REPO_ROOT, not $SCRIPT_DIR — Slurm copies the submitted wrapper to its
# spool directory, so $SCRIPT_DIR points there and the launcher isn't co-located.
# === Inlined launcher ===
# Stash caller-wrapper positional args (e.g. --ckpt.max_num) before `set --`
# clears $@, so the tail's RL_ARGS+=(...) can still forward them to the train CLI.
_FWD_ARGS=("$@")
set --
set -x

REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
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

# Default the AutoModel source override to the sibling checkout if it exists, so
# the latest main wins over the version baked into the container image.
DEFAULT_AUTOMODEL_PATH=/path/to/Automodel
if [ -z "${EXTRA_PYTHONPATH:-}" ] && [ -d "$DEFAULT_AUTOMODEL_PATH" ]; then
  EXTRA_PYTHONPATH="$DEFAULT_AUTOMODEL_PATH"
fi

# === Best-config defaults for 4-node H100, async split topology ===
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
ASYNC_QUEUE_SIZE="${ASYNC_QUEUE_SIZE:-2}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
# vLLM rollout side: dedicated full node, TP+EP hybrid for MoE.
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-2}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MM_ENCODER_ATTN_BACKEND="${VLLM_MM_ENCODER_ATTN_BACKEND:-TORCH_SDPA}"
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
# Default to Triton attention to avoid AOT-compiled FA2 PTX kernels that can
# fail on older drivers (cudaErrorUnsupportedPtxVersion). Set to FLASH_ATTN
# explicitly on machines whose driver supports the bundled vLLM FA2 PTX.
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
# omni3 is a hybrid Mamba2 model: force vLLM's SSM state cache to fp32 to match
# the fp32 training recompute. The vLLM fp16 default rounds over the rollout scan
# and drifts rollout log-probs from training, inflating vllm_kl so seq-mask-TIS
# over-filters.
VLLM_MAMBA_SSM_CACHE_DTYPE="${VLLM_MAMBA_SSM_CACHE_DTYPE:-float32}"
# Eager stays ON for GDN-hybrid engines: CUDA-graph capture fails at engine init on
# this vLLM build (worker death -> shm cancelled). Dense-attention models can run 0.
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"
# AutoModel actor side: 2 nodes (DP2), EP+CP for MoE actors.
ACTOR_NODES="${ACTOR_NODES:-2}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
# Backend constraints with current Automodel pin:
#   TP>1  → "Tensor parallelism not supported for custom MoE models"
#   CP>1  → needs the hybrid-SSM pre-embed CP fix (now in); CP8 SFT-validated @32K
# EP shards model state; CP shards sequence to fit 32K. TE attention preferred
# (faster than FA2 for the Automodel custom MoE layers).
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-8}"
FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-te}"
FREEZE_VISUAL_ENCODER="${FREEZE_VISUAL_ENCODER:-1}"
FREEZE_MOE_ROUTER="${FREEZE_MOE_ROUTER:-1}"
# Algo — defaults tuned for geo3k VLM math multi-turn.
KL_COEF="${KL_COEF:-0.0}"
MOE_AUX_LOSS_COEF="${MOE_AUX_LOSS_COEF:-0}"   # RL: no load-balancing aux (don't perturb the policy grad)
LR="${LR:-1e-5}"
DUAL_CLIP="${DUAL_CLIP:-10.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
EPS_CLIP_LOW="${EPS_CLIP_LOW:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DISABLE_FINAL_SAVE="${DISABLE_FINAL_SAVE:-0}"
# On-policy: accumulate the whole multi-turn rollout into one optimizer step
# (the flattened per-rollout sample count is variable, so a fixed train batch
# would make later steps off-policy and drop the tail). Requires MAX_EPOCHS=1.
# Set FORCE_ON_POLICY=0 to fall back to grad-accum windows.
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
# MOLT_DEFER_GRAD_SYNC=1 defers the FSDP grad reduce-scatter to the last microbatch
# (~1 cross-node reduce-scatter/step instead of one per microbatch), the big
# training win (~6x on policy_train, grad_norm unchanged).
# MOLT_MOE_RESHARD_AFTER_FWD=1 reshards experts after forward to offset the higher
# peak defer holds across the accum window, keeping the combo OOM-safe. reshard=0
# is ~22% faster standalone; set it to 0 for max speed when memory allows.
ray_env="unset VLLM_NUM_ENGINES VLLM_TP_SIZE VLLM_GPU_MEMORY_UTILIZATION VLLM_MM_ENCODER_ATTN_BACKEND VLLM_GDN_PREFILL_BACKEND VLLM_ATTENTION_BACKEND VLLM_ENFORCE_EAGER VLLM_DISTRIBUTED_EXECUTOR_BACKEND VLLM_ENABLE_EXPERT_PARALLEL; cd /molt && export HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=true RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_DOCKER_CPU_WARNING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_MOE_FP16=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDNN_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-} MAX_AGENT_TURNS=${MAX_AGENT_TURNS:-3} LOAD_MODEL_ONLY=${LOAD_MODEL_ONLY:-0} NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1} TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0} TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000} MOLT_FSDP_DEBUG_GRADS=${MOLT_FSDP_DEBUG_GRADS:-} MOLT_FSDP_DEBUG_GRADS_FILTER=${MOLT_FSDP_DEBUG_GRADS_FILTER:-} MOLT_FSDP_DEBUG_GRADS_TOPK=${MOLT_FSDP_DEBUG_GRADS_TOPK:-8} NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1} NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-0} MOLT_MOE_RESHARD_AFTER_FWD=${MOLT_MOE_RESHARD_AFTER_FWD:-1} MOLT_DEFER_GRAD_SYNC=${MOLT_DEFER_GRAD_SYNC:-1} MOLT_MOE_DISPATCHER=${MOLT_MOE_DISPATCHER:-hybridep}${EXTRA_PYTHONPATH_EXPORT}"

# Optional vLLM audio extra (torchaudio): the GA omni3 model carries a Parakeet
# sound_encoder whose weights vLLM only builds (instead of asserting
# `self.sound_encoder is not None`) when audio support is present. Audio is never
# used for the vision-only RL task and the encoder stays frozen. Gated by
# MOLT_INSTALL_AUDIO; prepended to ray_env so it runs in every container.
if [ "${MOLT_INSTALL_AUDIO:-0}" = "1" ]; then
  # torchaudio is already in the base image, so add the rest of vllm[audio], pinning
  # vllm to the installed version in the same command so pip can't swap the custom
  # build. Guarded on soundfile (no-op once present). Permanent fix = bake into the Dockerfile.
  ray_env="$ray_env && (python3 -c 'import soundfile' 2>/dev/null || python3 -m pip install -q 'vllm[audio]')"
fi

# The rollout routes through the Rust vllm-router, absent from the base image;
# install it (idempotent) from the wheel pre-cached under /molt so it works on
# air-gapped nodes. Permanent fix = bake it into the Dockerfile.
ray_env="$ray_env && (python3 -c 'import vllm_router' 2>/dev/null || python3 -m pip install -q --break-system-packages /molt/.tmp/vllm_router_wheel/vllm_router-*.whl)"

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
  --rollout.num_runners "${NUM_RUNNERS:-8}"
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
  --actor.muon.lr "${MUON_LR:-2e-4}"
  --actor.muon.momentum "${MUON_MOMENTUM:-0.95}"
  --actor.muon.weight_decay "${MUON_WEIGHT_DECAY:-0.0}"
  --actor.muon.ns_steps "${MUON_NS_STEPS:-5}"
  --actor.eps_clip_low_high "$EPS_CLIP_LOW" "$EPS_CLIP_HIGH"
  --actor.dual_clip "$DUAL_CLIP"
  --actor.aux_loss_coef "$MOE_AUX_LOSS_COEF"
  --actor.entropy_coef "${ENTROPY_COEF:-0.0}"
  --actor.freezing_steps "${FREEZING_STEPS:-0}"
  --algo.advantage.estimator "${ESTIMATOR:-reinforce_baseline}"
  --algo.advantage.is_correction_enable
  --algo.advantage.is_correction_type seq-mask-tis
  --algo.advantage.is_correction_threshold "${IS_LOW:-0.95}" "${IS_HIGH:-1.05}"
  --algo.kl.use_loss
  --algo.kl.estimator k2
  --algo.kl.init_coef "$KL_COEF"
  --reward.clip_range -1 1
  --ckpt.output_dir "$SAVE_ROOT/hf"
  --ckpt.path "$SAVE_ROOT/state"
  --ckpt.save_steps "${SAVE_STEPS:-5}"
  --ckpt.max_num "${CKPT_MAX_NUM:-50}"
  --logger.logging_steps 1
  --logger.wandb.project "${WANDB_PROJECT:-molt_async_visual_rl}"
  --logger.wandb.run_name "${WANDB_RUN_NAME:-visual_rl_$SLURM_JOB_ID}"
)

RL_ARGS+=(--train.agent_path "$AGENT_PATH")

# A rollout spanning a weight broadcast can mix policy versions. The HTTP path
# cannot mark that boundary, so PARTIAL_ROLLOUT=1 requires per-token IS correction;
# it does not mask the old prefix.
if [ "${PARTIAL_ROLLOUT:-0}" = "1" ]; then
  RL_ARGS+=(--train.partial_rollout_enable)
fi

if [ "$FORCE_ON_POLICY" = "1" ]; then
  RL_ARGS+=(--train.force_on_policy)
fi

# Warm-resume the async rollout buffer across segment restarts (opt-in; off by default).
if [ "${WARM_RESUME_ROLLOUTS:-0}" = "1" ]; then
  RL_ARGS+=(--ckpt.warm_resume_rollouts)
fi

if [ "$VLLM_ENFORCE_EAGER" = "1" ]; then
  RL_ARGS+=(--vllm.enforce_eager)
fi

# CPU-offload level (--fsdp.offload): FSDP_CPU_OFFLOAD=1 -> full (params to CPU,
# ~15GB/rank but breaks MoE); OFFLOAD_OPTIMIZER=1 -> optimizer (AdamW step on CPU,
# params stay on GPU; MoE-safe).
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

# Forward any positional args from caller wrappers (e.g. --ckpt.max_num,
# --fsdp.packing_samples). Uses the pre-reset snapshot ($@ was cleared by
# `set --`); the `[@]+` guard expands to nothing under set -u when empty.
RL_ARGS+=("${_FWD_ARGS[@]+"${_FWD_ARGS[@]}"}")

printf -v RL_ARGS_Q " %q" "${RL_ARGS[@]}"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" \
  bash -lc "$ray_env && ray job submit --address=http://localhost:$DASHBOARD_PORT -- bash -lc 'cd /molt && python3 -u -m molt.cli.train_rl_ray$RL_ARGS_Q'"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" bash -lc "$ray_env && ray stop --force" || true
