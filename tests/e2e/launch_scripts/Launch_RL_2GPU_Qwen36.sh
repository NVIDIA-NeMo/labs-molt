#!/bin/bash
# The daily (GPU-only) recipe: Qwen3.6-35B-A3B (Qwen3.5-MoE VLM, 256 experts; the chat template's
# enable_thinking=false keeps answers short) on geo3k with the multi-turn Python-tool agent, one vLLM
# engine + one actor GPU, 4 prompts x 4 samples per update, 8k context, 5 updates, one batch queued
# ahead, routing replay (R3) on -- hence pool == batch, R3 and partial rollout are exclusive. The actor
# is a single process, so the MoE runs without expert parallelism (AutoModel's plain GroupedExperts);
# the bf16 parameters and gradients of 35B fit one GB200 with the AdamW states on the CPU
# (--fsdp.offload optimizer). Same FlashREINFORCE loss + Dr. GRPO advantage and the same pass criteria
# as Launch_RL_2GPU.sh.
set -xeuo pipefail
cd "$(dirname "$0")/../../.."
WORK="${E2E_WORK_DIR:-.tmp/e2e_rl_2gpu_qwen36}"
MODEL_PATH="${MODEL_PATH:-$WORK/Qwen3.6-35B-A3B}"
mkdir -p "$WORK"
[ -f "$MODEL_PATH/config.json" ] || python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-35B-A3B', local_dir='$MODEL_PATH')"
[ -e "$WORK/data/train" ] || python3 examples/python/utils/prepare_geo3k.py --max-train 40 --max-eval 8 --num-proc 4 --out-dir "$WORK/data"

export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_MOE_FP16=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_AGENT_TURNS=4 RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=true
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
  --data.tools_key tools \
  --data.apply_chat_template \
  --data.chat_template_kwargs '{"enable_thinking": false}' \
  --data.image_key images \
  --data.max_images_per_prompt 1 \
  --data.max_samples 20 \
  --data.max_len 8192 \
  --rollout.batch_size 4 \
  --rollout.vllm_generate_batch_size 4 \
  --rollout.n_samples_per_prompt 4 \
  --rollout.micro_batch_size 1 \
  --rollout.max_new_tokens 4096 \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --train.batch_size 16 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes 1 \
  --train.async_queue_size 2 \
  --train.routing_replay \
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
  --vllm.gpu_memory_utilization 0.85 \
  --vllm.distributed_executor_backend mp \
  --vllm.mamba_ssm_cache_dtype float32 \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation te \
  --fsdp.offload optimizer \
  --actor.gradient_checkpoint full \
  --actor.freeze_visual_encoder \
  --actor.adam.lr 1e-6 \
  --actor.loss_agg_mode seq-mean-token-mean \
  --algo.advantage.estimator dr_grpo \
  --algo.advantage.is_correction_level seq \
  --algo.advantage.is_correction_gating binary_kl \
  --algo.advantage.is_correction_threshold 1e-2 \
  --algo.kl.init_coef 0 \
  --reward.clip_range -10 10 \
  --train.agent_path examples/python/agents/geo3k.py \
  --ckpt.output_dir "$WORK/hf" \
  --logger.tensorboard_dir "$WORK/tb" \
  "$@" 2>&1 | tee "$WORK/train.log"

grep -q 'Global step 5:' "$WORK/train.log"
[ "$(grep -c 'the broadcast landed on every vLLM weight' "$WORK/train.log")" -ge 4 ]
[ -f "$WORK/hf/config.json" ] && ls "$WORK/hf"/*.safetensors
python3 tests/e2e/summarize_metrics.py "$WORK/train.log" --out "$WORK/metrics_summary.md"
