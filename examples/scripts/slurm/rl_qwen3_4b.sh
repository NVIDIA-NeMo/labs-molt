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
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=molt-rl-qwen3-4b-packing
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive

# Qwen3-4B dense math RL with FA2 + cu_seq_lens packing.
# Thin wrapper over slurm/_launcher.sh that strips the VLM/MoE knobs
# (EP=1, text-only single-turn math agent) and appends
# --fsdp.packing_samples to exercise the HF FA2 packed path
# (cu_seq_lens_q/k kwargs from utils/fsdp/packing.py:182).
#
# Dense Qwen3 has no nemo_automodel native impl (HF Qwen3ForCausalLM only). That's fine: this
# recipe runs EP=1, and molt permits the HF path whenever EP is off; the fallback is forbidden
# only under expert parallelism (EP>1, e.g. the omni3 MoE), which HF transformers can't shard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
export MOLT_PATH="$REPO_ROOT"

export MODEL_PATH="${MODEL_PATH:-/path/to/models/Qwen3/Qwen3-4B-Instruct-2507}"
export TP_SIZE="${TP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS-}"
# FA2 is required for HF packing (cu_seq_lens path); init-time validation
# in Actor.from_pretrained refuses other attn impls when packing is on.
export FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-flash_attention_2}"

export VLLM_ENABLE_EXPERT_PARALLEL=0
export FREEZE_VISUAL_ENCODER=0

export AGENT_PATH="${AGENT_PATH:-/molt/examples/python/agents/math.py}"
export MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-1}"

export PROMPT_DATASET="${PROMPT_DATASET:-$REPO_ROOT/.tmp/proRL_text_rl/train}"
export EVAL_DATASET="${EVAL_DATASET:-$REPO_ROOT/.tmp/proRL_text_rl/eval}"
export MAX_SAMPLES="${MAX_SAMPLES:-4800}"
export ENABLE_DYNAMIC_FILTERING="${ENABLE_DYNAMIC_FILTERING:-1}"
export EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"

export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

export SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/rl-qwen3-4b-packing/run}"
export WANDB_PROJECT="${WANDB_PROJECT:-molt_rl_qwen3_4b_packing}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3_4b_packing_$SLURM_JOB_ID}"

# === Inlined launcher (was slurm/_launcher.sh) ===
# Packing on by default (this is the FA2 packed-path recipe); set PACKING_SAMPLES=0
# to run the plain AutoModel path (e.g. sdpa, no cu_seq_lens) for A/B debugging.
# Snapshot caller-wrapper positional args before `set --` clears $@ (the line below
# then re-injects only the packing flag).
_FWD_ARGS=("$@")
set --
[ "${PACKING_SAMPLES:-1}" = "1" ] && set -- --fsdp.packing_samples
set -x

REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$REPO_ROOT/images/molt-cu13.sqsh}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the VLM checkpoint to train.}"

# PROMPT_DATASET / EVAL_DATASET are exported above (proRL_text_rl by default).
# Override either env var to swap in your own text-only math data.
test -n "${PROMPT_DATASET:-}"

SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/molt-async-visual-rl/$SLURM_JOB_ID}"
AGENT_PATH="${AGENT_PATH:-/molt/examples/python/agents/math.py}"

# Default the AutoModel source override to the sibling checkout if it exists, so
# the latest main wins over the version baked into the container image.
DEFAULT_AUTOMODEL_PATH=/path/to/Automodel
if [ -z "${EXTRA_PYTHONPATH:-}" ] && [ -d "$DEFAULT_AUTOMODEL_PATH" ]; then
  EXTRA_PYTHONPATH="$DEFAULT_AUTOMODEL_PATH"
fi

# === Best-config defaults for 2-node interactive H100, async split topology ===
# Override any of these from the wrapper (model-specific) or sbatch env (per-run).
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
# Sequence + batch shape: 16 prompts × 8 samples = 128 sequences/train batch.
# MAX_LENGTH is the SHARED total-context budget (prompt + generation), used
# both as data.max_len and vLLM's max_model_len. Visual prompts expand to
# thousands of vision tokens once the chat template applies the `<image>`
# placeholder, and multi-turn agents accumulate history — default to 64k
# headroom. MAX_NEW_TOKENS is the per-request generation cap within that
# budget; longest prompt must satisfy `prompt_len + MAX_NEW_TOKENS <= MAX_LENGTH`.
MAX_LENGTH="${MAX_LENGTH:-65536}"
# MAX_NEW_TOKENS is the per-turn generation cap. Defaults to 8192 — enough for
# CoT + answer on math reasoning tasks while keeping rollout
# wall-time and activation memory bounded. Multiturn agents can issue many
# turns within MAX_LENGTH, so this isn't a context budget — it's a per-call
# max. Set to MAX_LENGTH explicitly if you really want each turn to consume
# the full remaining context.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS-}"
MAX_SAMPLES="${MAX_SAMPLES:-8192}"
# rollout_batch_size = unique prompts the trainer dispatches per
# `make_experience` call. The trainer's policy_train loop drops trailing
# microbatches when `microbatches_per_rollout < grad_accum`, so size
# `rollout_batch_size * n_samples >= train_batch_size` to avoid no-op steps.
# Default sized for one rollout = one full grad-accum window:
#   rollout_batch_size * n_samples = train_batch_size  (1 rollout per step)
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
ROLLOUT_GENERATE_BATCH_SIZE="${ROLLOUT_GENERATE_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
# train_batch_size = trajectories per gradient step (rollout_batch * n_samples).
# Default 128 trajectories/step (rollout_batch × n_samples). Override to a
# larger value (e.g. 2048) for stability runs; step time scales linearly and
# the 4h interactive slurm limit caps achievable batch sizes.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
# micro_batch_size=1 for large MoE actors: long visual prompts + generations
# push activation memory past 80GB even with gradient checkpointing + colocated
# actor/ref. With expandable_segments allocator, mb=1 fits comfortably.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
# Pure async + partial rollout: queue depth >= 2 so train overlaps next rollout.
ASYNC_QUEUE_SIZE="${ASYNC_QUEUE_SIZE:-1}"
# vLLM rollout side: dedicated full node, TP+EP hybrid for MoE.
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-8}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MM_ENCODER_ATTN_BACKEND="${VLLM_MM_ENCODER_ATTN_BACKEND:-TORCH_SDPA}"
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
# Default to Triton attention to avoid AOT-compiled FA2 PTX kernels that can
# fail on older drivers (cudaErrorUnsupportedPtxVersion). Set to FLASH_ATTN
# explicitly on machines whose driver supports the bundled vLLM FA2 PTX.
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"
# AutoModel actor side: 1 dedicated node, TP+EP+CP for MoE actors.
ACTOR_NODES="${ACTOR_NODES:-1}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-2}"
FSDP_ATTN_IMPLEMENTATION="${FSDP_ATTN_IMPLEMENTATION:-te}"
FREEZE_VISUAL_ENCODER="${FREEZE_VISUAL_ENCODER:-1}"
# Algo
KL_COEF="${KL_COEF:-0.001}"
MOE_AUX_LOSS_COEF="${MOE_AUX_LOSS_COEF:-0.000}"
LR="${LR:-1e-6}"
DUAL_CLIP="${DUAL_CLIP:-10.0}"
DISABLE_FINAL_SAVE="${DISABLE_FINAL_SAVE:-0}"

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

# vLLM >=0.20 ignores the legacy VLLM_ATTENTION_BACKEND env var; the attention backend
# is plumbed through --vllm.attention_backend → EngineArgs instead. The MoE-side
# FlashInfer fallback still uses an env var (VLLM_USE_FLASHINFER_MOE_FP16=0) for now.
# expandable_segments reduces fragmentation when long generations push the
# 30B colocated actor+ref close to 80GB.
ray_env="unset VLLM_NUM_ENGINES VLLM_TP_SIZE VLLM_GPU_MEMORY_UTILIZATION VLLM_MM_ENCODER_ATTN_BACKEND VLLM_GDN_PREFILL_BACKEND VLLM_ATTENTION_BACKEND VLLM_ENFORCE_EAGER VLLM_DISTRIBUTED_EXECUTOR_BACKEND VLLM_ENABLE_EXPERT_PARALLEL; cd /molt && export HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=true RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_DOCKER_CPU_WARNING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_MOE_FP16=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAX_AGENT_TURNS=${MAX_AGENT_TURNS:-2}${EXTRA_PYTHONPATH_EXPORT}"

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

RL_ARGS=(
  --actor.model_name_or_path "$MODEL_PATH"
  --data.prompt_dataset "$PROMPT_DATASET"
  --data.input_key "${INPUT_KEY:-prompt}"
  --data.label_key "${LABEL_KEY:-reward_model}"
  --data.apply_chat_template
  --data.image_key "${IMAGE_KEY:-images}"
  --data.max_samples "$MAX_SAMPLES"
  --data.max_len "$MAX_LENGTH"
  ${MAX_NEW_TOKENS:+--rollout.max_new_tokens=$MAX_NEW_TOKENS}
  --rollout.batch_size "$ROLLOUT_BATCH_SIZE"
  --rollout.vllm_generate_batch_size "$ROLLOUT_GENERATE_BATCH_SIZE"
  --rollout.micro_batch_size 1
  --rollout.n_samples_per_prompt "$N_SAMPLES_PER_PROMPT"
  --rollout.temperature "${TEMPERATURE:-1.0}"
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
  --vllm.distributed_executor_backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND"
  --fsdp.param_dtype bf16
  --fsdp.attn_implementation "$FSDP_ATTN_IMPLEMENTATION"
  --fsdp.tp_size "$TP_SIZE"
  --fsdp.ep_size "$EP_SIZE"
  --fsdp.cp_size "$CP_SIZE"
  --actor.gradient_checkpoint full
  --actor.adam.lr "$LR"
  --actor.eps_clip_low_high 0.2 0.27
  --actor.dual_clip "$DUAL_CLIP"
  --actor.aux_loss_coef "$MOE_AUX_LOSS_COEF"
  --actor.entropy_coef "${ENTROPY_COEF:-0.0}"
  --algo.advantage.estimator reinforce_baseline
  --algo.advantage.is_correction_level geo
  --algo.advantage.is_correction_threshold "${IS_LOW:-0.99}" "${IS_HIGH:-1.01}"
  --algo.kl.use_loss
  --algo.kl.estimator k2
  --algo.kl.init_coef "$KL_COEF"
  --reward.clip_range -10 10
  --ckpt.output_dir "$SAVE_ROOT/hf"
  --ckpt.path "$SAVE_ROOT/state"
  --ckpt.save_steps "${SAVE_STEPS:-5}"
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

if [ "$VLLM_ENFORCE_EAGER" = "1" ]; then
  RL_ARGS+=(--vllm.enforce_eager)
fi

# FSDP2 CPUOffloadPolicy: streams optimizer + grads to CPU per layer. Saves
# ~15GB / rank for 30B MoE in bf16 — required to fit MAX_LENGTH=32k under
# colocated actor/ref with EP=8 CP=2.
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

# Resume from the last checkpoint at $SAVE_ROOT/state. Set LOAD_ENABLE=1 (and
# point SAVE_ROOT to a prior run's directory) to chain RL across slurm
# allocations — convergence on a 30B agentic VLM RL run typically exceeds the
# 4h interactive partition limit, so resubmissions are expected.
if [ "${LOAD_ENABLE:-0}" = "1" ]; then
  RL_ARGS+=(--ckpt.load_enable)
fi

if [ -n "$EVAL_DATASET" ]; then
  RL_ARGS+=(--eval.dataset "$EVAL_DATASET" --eval.steps "${EVAL_STEPS:-5}" --eval.n_samples_per_prompt "${EVAL_N_SAMPLES_PER_PROMPT:-4}")
  # Baseline eval at step 0 (fresh runs only; no-op on resume). Needed to read
  # the pre-RL accuracy so step-N gains are attributable. The actor GPUs idle
  # during this rollout-only phase, so pair EVAL_AT_START=1 with an idle-reaper
  # exemption (see --comment at submit time).
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

# Forward any positional args from caller wrappers — e.g. the dense
# Qwen3-4B packing wrapper appends `--fsdp.packing_samples` to opt into
# the FA2 THD path.
RL_ARGS+=("$@")
# Forward the caller-wrapper positional args snapshotted before `set --`.
RL_ARGS+=("${_FWD_ARGS[@]+"${_FWD_ARGS[@]}"}")

printf -v RL_ARGS_Q " %q" "${RL_ARGS[@]}"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" \
  bash -lc "$ray_env && ray job submit --address=http://localhost:$DASHBOARD_PORT -- bash -lc 'cd /molt && python3 -u -m molt.cli.train_rl_ray$RL_ARGS_Q'"

srun --nodes=1 --ntasks=1 -w "$head_node" "${CONTAINER_ARGS[@]}" bash -lc "$ray_env && ray stop --force" || true
