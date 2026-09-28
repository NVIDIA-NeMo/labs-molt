#!/bin/bash
# Two-GPU asynchronous RL end-to-end check (the CI "E2E RL" workflow; runnable by hand on any 2-GPU box
# inside the molt image): Qwen2.5-Math-1.5B on DAPO-Math-17k, one vLLM engine + one actor GPU, 4 prompts x
# 4 samples per update with 8 prompts in flight (the surplus crosses each weight refit as partial rollouts),
# 4k context, 10 updates, one batch queued ahead of training. FlashREINFORCE loss (--train.force_on_policy:
# PPO ratio == 1; sequence-level IS gated by binary KL corrects the off-policy tokens) with the Dr. GRPO advantage.
# Passes when the driver exits 0, update 10 is logged, every weight refit was verified on the engine, the
# final HF export is written and summarize_metrics.py raises no alert (its table of every metric's first /
# last / min / max / change lands in metrics_summary.md, which the workflow posts as the job summary).
# MODEL_ID, MAX_LEN, MAX_NEW_TOKENS and FSDP_OFFLOAD select the model size (see Launch_RL_2GPU_30B.sh).
set -xeuo pipefail
cd "$(dirname "$0")/../../.."
WORK="${E2E_WORK_DIR:-.tmp/e2e_rl_2gpu}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Math-1.5B}"
MODEL_PATH="${MODEL_PATH:-$WORK/${MODEL_ID##*/}}"
mkdir -p "$WORK"
[ -f "$MODEL_PATH/config.json" ] || python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', local_dir='$MODEL_PATH')"
[ -e "$WORK/data/train" ] || python3 examples/python/utils/prepare_dapo.py --max-train 64 --max-eval 8 --out-dir "$WORK/data"

export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAX_AGENT_TURNS=1
export RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=true
# What the container actually gets (the runner pod's limits may differ from the node): GPUs, RAM, cgroup cap.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2
echo "cgroup memory limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown)"
ray start --head --num-gpus="$(nvidia-smi -L | wc -l)" --disable-usage-stats
# On any exit: stop Ray, and write the metrics summary of whatever ran (a failed run still gets its table).
trap 'ray stop --force >/dev/null 2>&1 || true; [ -f "$WORK/metrics_summary.md" ] || python3 tests/e2e/summarize_metrics.py "$WORK/train.log" --out "$WORK/metrics_summary.md" || true' EXIT

python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$WORK/data/train" \
  --data.input_key prompt \
  --data.label_key reward_model \
  --data.apply_chat_template \
  --data.max_samples 40 \
  --data.max_len "${MAX_LEN:-4096}" \
  --rollout.batch_size 4 \
  --rollout.vllm_generate_batch_size 8 \
  --rollout.n_samples_per_prompt 4 \
  --rollout.micro_batch_size 1 \
  --rollout.max_new_tokens "${MAX_NEW_TOKENS:-3072}" \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --train.batch_size 16 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes 1 \
  --train.async_queue_size 2 \
  --train.partial_rollout_enable \
  --train.force_on_policy \
  --train.colocate_fsdp_models \
  --train.check_weight_update_equal \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node 1 \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node 1 \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size 1 \
  --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.9 \
  --fsdp.param_dtype bf16 \
  --fsdp.offload "${FSDP_OFFLOAD:-none}" \
  --fsdp.attn_implementation "${FSDP_ATTN_IMPLEMENTATION:-te}" \
  --fsdp.packing_samples \
  --actor.gradient_checkpoint full \
  --actor.adam.lr 1e-6 \
  --actor.loss_agg_mode seq-mean-token-mean \
  --algo.advantage.estimator dr_grpo \
  --algo.advantage.is_correction_level seq \
  --algo.advantage.is_correction_gating binary_kl \
  --algo.advantage.is_correction_threshold 1e-2 \
  --algo.kl.init_coef 0 \
  --reward.clip_range -10 10 \
  --train.agent_path examples/python/agents/math.py \
  --ckpt.output_dir "$WORK/hf" \
  --logger.tensorboard_dir "$WORK/tb" \
  "$@" 2>&1 | tee "$WORK/train.log"

grep -q 'Global step 10:' "$WORK/train.log"
[ "$(grep -c 'the broadcast landed on every vLLM weight' "$WORK/train.log")" -ge 9 ]
[ -f "$WORK/hf/config.json" ] && ls "$WORK/hf"/*.safetensors
python3 tests/e2e/summarize_metrics.py "$WORK/train.log" --out "$WORK/metrics_summary.md"
