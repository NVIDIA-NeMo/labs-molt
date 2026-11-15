# `python -m molt.cli.train_rl_ray`

> Generated from `molt/cli/train_rl_ray.py` by `docs/guide/build_reference.py`. Edit the source, then rerun the script.

## `--fsdp.*`

| Flag | Default | Description |
|---|---|---|
| `--fsdp.tp_size` | `1` | Tensor parallel size |
| `--fsdp.cp_size` | `1` | Context parallel size (replaces ring-attn) |
| `--fsdp.ep_size` | `1` | Expert parallel size (MoE) |
| `--fsdp.pp_size` | `1` | Pipeline parallel size |
| `--fsdp.sequence_parallel` | `False` | Opt into sequence parallelism within the TP region. Off by default (matches AutoModel's omni / Qwen3.5-MoE recipes). |
| `--fsdp.offload` | `"none"` | CPU-offload level — a nested progression, not orthogonal toggles (FSDP param offload inherently offloads the optimizer too). 'none': everything on GPU. 'optimizer': run the AdamW step on CPU so the fp32 master + Adam moments never occupy GPU during the step (shrinks the optimizer-step peak, the binding one at long context); params stay on GPU for the forward, so it's safe on Qwen3.6 MoE; AdamW only; numerically equivalent to a GPU step. 'full': FSDP2 CPUOffloadPolicy also streams the \*params\* to CPU (maximal saving, but breaks Qwen3.6 MoE and slows the forward). |
| `--fsdp.param_dtype` | `"bf16"` | Model data type Choices: `bf16`, `fp16`. |
| `--fsdp.attn_implementation` | `"flash_attention_2"` | Attention implementation (e.g., sdpa, eager, flex, te, flash_attention_2) |
| `--fsdp.packing_samples` | `False` |  |

## `--actor.*`

| Flag | Default | Description |
|---|---|---|
| `--actor.optim` | `"adam"` |  Choices: `adam`, `muon`. |
| `--actor.muon.lr` | `0.02` | LR for the Muon 2D-weight group |
| `--actor.muon.momentum` | `0.95` |  |
| `--actor.muon.weight_decay` |  | Weight decay for the Muon 2D-weight group |
| `--actor.muon.ns_steps` | `5` | Newton-Schulz steps for Muon updates |
| `--actor.muon.nesterov` | `True` |  |
| `--actor.muon.no_nesterov` | `True` |  |
| `--actor.adam.lr` | `1e-06` |  |
| `--actor.adam.betas` | `(0.9, 0.95)` |  |
| `--actor.adam.eps` | `1e-08` |  |
| `--actor.adam.weight_decay` | `0.0` |  |
| `--actor.lr_scheduler` | `"constant"` |  |
| `--actor.lr_warmup_ratio` | `0.03` |  |
| `--actor.min_lr_ratio` | `0.1` |  |
| `--actor.max_norm` | `1.0` | Gradient clipping |
| `--actor.model_name_or_path` |  | HF model name or path |
| `--actor.gradient_checkpoint` | `"full"` | Activation-checkpointing mode (string): 'full' = full-block AC (AutoModel recipe default), 'selective' = TorchTitan per-op AC, 'none'/'off'/'' = disable. |
| `--actor.aux_loss_coef` | `0` | MoE balancing loss |
| `--actor.freezing_steps` | `0` | Critic warmup: freeze the actor's policy update for the first N optimizer steps so the value model fits the initial rollouts before its early, high-variance advantages move the policy. The critic keeps training while frozen. 0 disables; only applies with --algo.advantage.estimator gae. |
| `--actor.freeze_visual_encoder` | `False` | Freeze vision encoder weights (only train language model). Reduces memory and weight sync overhead. |
| `--actor.freeze_moe_router` | `False` | Freeze the MoE router/gate weights so routing is held fixed during training, which stabilizes MoE training and keeps the vLLM-vs-actor routing identical. |
| `--actor.lora_dim` | `0` | LoRA rank; 0 disables LoRA. |
| `--actor.lora_alpha` | `32` | LoRA alpha; scale = alpha / dim. |
| `--actor.lora_target_modules` |  | Wildcard patterns matched against FULL module names (AutoModel ModuleMatcher), e.g. '\*.q_proj' '\*.v_proj' — bare leaf names like 'q_proj' match nothing. Default: all linear layers (match_all_linear). On MoE models, name targets explicitly to keep the router gate out of the patch set. |
| `--actor.eps_clip_low_high` |  | policy clip low and high |
| `--actor.dual_clip` |  | Dual-clip policy objective |
| `--actor.loss_mode` | `"ppo"` | Policy-gradient surrogate: ppo (clipped min(surr1,surr2), optionally --actor.dual_clip) or cispo (https://arxiv.org/abs/2506.13585 — clips only the upper side of the IS ratio, stop-gradient through that weight, gradient flows through log-probs only; pass --actor.eps_clip_low_high's high value as the absolute ratio ceiling, e.g. 1.2, not a +offset; --actor.dual_clip is rejected in this mode) or gspo (https://arxiv.org/abs/2507.18071 — clips ONE ratio per sequence, its geometric mean, so a single outlier token cannot clip the whole update; aggregated with molt's global token-mean denominator, not the paper's per-sequence 1/\|y\|; --actor.dual_clip is rejected). |
| `--actor.entropy_coef` |  | Entropy coefficient. Any nonzero value enables entropy computation and logs entropy_loss; positive values encourage higher entropy. 0 or unset skips entropy computation. |
| `--actor.loss_agg_mode` | `"token-mean"` | Policy-loss aggregation: token-mean (global token mean), seq-mean-token-mean (every sequence weighs the same) or prompt-mean-token-mean (token mean inside each prompt's rollouts, then every prompt weighs the same). |
| `--actor.num_nodes` | `1` | number of nodes for actor |
| `--actor.num_gpus_per_node` | `8` | number of gpus per node for actor |

## `--ckpt.*`

| Flag | Default | Description |
|---|---|---|
| `--ckpt.output_dir` | `"./ckpt"` | Directory for the final consolidated HF safetensors export. |
| `--ckpt.save_steps` | `-1` | Save a resumable checkpoint every N steps; -1 disables it. |
| `--ckpt.save_hf` | `False` | Also export an HF-format snapshot alongside each resumable checkpoint. |
| `--ckpt.disable_final_save` | `False` | Skip the final consolidated HF export after training. Useful for smoke tests. |
| `--ckpt.path` | `"./ckpt/checkpoints_rl_ray"` | Directory for resumable FSDP/DCP training checkpoints (optimizer + scheduler state). |
| `--ckpt.max_num` | `3` | How many checkpoints to keep; older ones pruned (\<=0 keeps all). Applies to the HF exports (--ckpt.save_hf, versioned under ckpt.path/_hf/\<step\>) and, unless --ckpt.dcp_max_num is set, to the DCP resumable checkpoints too. |
| `--ckpt.dcp_max_num` |  | Override --ckpt.max_num for DCP resumable checkpoints only; unset -\> --ckpt.max_num. DCP checkpoints carry optimizer state and are large, so keep this small (e.g. 2) while --ckpt.max_num keeps many cheap HF exports for eval. |
| `--ckpt.max_mem` | `inf` | Disk budget in GB for retained checkpoints; inf = no limit. |
| `--ckpt.load_enable` | `False` | Resume from the latest checkpoint in --ckpt.path if present. |
| `--ckpt.best_metric_key` | `""` | Eval metric key for best checkpoint saving (e.g., eval_default_pass1). Empty string auto-detects first pass1 metric. Set to 'none' to disable best checkpoint saving. |
| `--ckpt.warm_resume_rollouts` | `False` | At each train step, save the rollout groups that finished but were not trained yet, so a resumed async run trains them immediately instead of idling \~one generation while the pipeline refills. Opt-in/experimental; best-effort — a failed save or missing file falls back to a normal resume. |

## `--logger.*`

| Flag | Default | Description |
|---|---|---|
| `--logger.logging_steps` | `1` | Log metrics every N steps. |
| `--logger.wandb.key` |  |  |
| `--logger.wandb.org` |  |  |
| `--logger.wandb.group` |  |  |
| `--logger.wandb.project` | `"molt_train_rl"` |  |
| `--logger.wandb.run_name` | `"rl_<MMDD>T<HH:MM>"` |  |
| `--logger.tensorboard_dir` |  | TensorBoard logging path |

## `--ref.*`

| Flag | Default | Description |
|---|---|---|
| `--ref.model_name_or_path` |  | Reference/teacher checkpoint. Defaults to the actor checkpoint (standard KL-to-init RL). Set to a different (e.g. larger, same-tokenizer) model for on-policy distillation. |
| `--ref.num_nodes` | `1` | number of nodes for reference |
| `--ref.num_gpus_per_node` | `8` | number of gpus per node for reference |

## `--critic.*`

| Flag | Default | Description |
|---|---|---|
| `--critic.model_name_or_path` |  | Critic init checkpoint (a reward model or the policy). Defaults to the actor checkpoint. |
| `--critic.value_clip` | `0.2` | PPO value-clip range for the value loss. |
| `--critic.freeze_moe_router` | `False` | Freeze the critic's MoE router/gate weights, independently of --actor.freeze_moe_router (also inherited when the actor's flag is set). |
| `--critic.max_epochs` |  | Optimization epochs the critic runs per RL step; defaults to --train.max_epochs. Set higher to fit the value function harder each step (a PPO critic often wants more passes). |
| `--critic.optim` | `"adam"` |  Choices: `adam`, `muon`. |
| `--critic.muon.lr` | `0.02` | LR for the Muon 2D-weight group |
| `--critic.muon.momentum` | `0.95` |  |
| `--critic.muon.weight_decay` |  | Weight decay for the Muon 2D-weight group |
| `--critic.muon.ns_steps` | `5` | Newton-Schulz steps for Muon updates |
| `--critic.muon.nesterov` | `True` |  |
| `--critic.muon.no_nesterov` | `True` |  |
| `--critic.adam.lr` | `5e-06` |  |
| `--critic.adam.betas` | `(0.9, 0.95)` |  |
| `--critic.adam.eps` | `1e-08` |  |
| `--critic.adam.weight_decay` | `0.0` |  |
| `--critic.lr_scheduler` | `"constant"` |  |
| `--critic.lr_warmup_ratio` | `0.03` |  |
| `--critic.min_lr_ratio` | `0.1` |  |
| `--critic.max_norm` | `1.0` | Gradient clipping |

## `--data.*`

| Flag | Default | Description |
|---|---|---|
| `--data.prompt_dataset` |  | HF dataset name or path |
| `--data.prompt_probs` |  | sampling probs for datasets |
| `--data.prompt_split` | `"train"` |  |
| `--data.max_samples` | `100000000` | Max number of samples |
| `--data.max_len` | `2048` | Max total sequence length (prompt + response) |
| `--data.input_key` | `"input"` | JSON dataset key |
| `--data.label_key` |  | Dataset column holding the ground-truth answer/label used for reward scoring. |
| `--data.tools_key` |  | Dataset key whose value is a list of OpenAI function-call schemas; rendered into the chat template (see Qwen `tools=`) so the model is taught the native \<tool_call\>{...}\</tool_call\> emission format without manual prompt engineering. |
| `--data.apply_chat_template` | `False` | Dataset rows are chat messages. Step runners render them here with the HF chat template; chat agents require this flag and hand the raw messages to the chat server, which renders them once with the model's own template. |
| `--data.image_key` | `"images"` | Dataset key for image paths/URLs |
| `--data.max_images_per_prompt` | `0` | Max images per prompt for vLLM (0 = text-only) |
| `--data.disable_fast_tokenizer` | `False` |  |
| `--data.dataloader_num_workers` | `0` | Number of dataloader workers for IO (for Ray training, ensure sufficient CPU resources per actor) |

## `--algo.*`

| Flag | Default | Description |
|---|---|---|
| `--algo.advantage.estimator` | `"reinforce"` | Advantage estimation method: reinforce, rloo, reinforce_baseline, grpo, dr_grpo, on_policy_distill (per-token reverse KL to the --ref.model_name_or_path teacher), gae (PPO value baseline — builds a colocated critic; see --critic.\*), or flash_reinforce (FlashREINFORCE: reward minus the rollout-batch mean, no group, no whitening — the n_samples_per_prompt=1 estimator; see examples/scripts/quick_start/rl_flash_reinforce_r1d_1p5b.sh) |
| `--algo.advantage.gamma` | `1` | discount factor |
| `--algo.advantage.lam` | `1.0` | GAE lambda (PPO only). 1.0 = Monte-Carlo return minus the value baseline. `length_adaptive` (VAPO, https://arxiv.org/abs/2504.05118): each sample's lambda is max(0, 1 - 1/(alpha \* H)), H its generated tokens, and the value target uses lambda 1 (decoupled GAE). |
| `--algo.advantage.length_adaptive_alpha` | `0.5` | alpha of `--algo.advantage.lam length_adaptive`. The outcome reward keeps about exp(-1/alpha) of the weight at the first token whatever the length (0.1: 5e-5, 0.5: 0.14, 1.5: 0.51). |
| `--algo.advantage.no_whiten` | `False` | skip cross-batch advantage whitening — use raw returns (no mean-center, no std). Affects the whitening estimators (reinforce / reinforce_baseline / gae); for a no-std GRPO use --algo.advantage.estimator dr_grpo. Useful in single-rollout / async where the batch mean/std couple samples and are a noisy moving target. |
| `--algo.advantage.is_correction_level` | `"off"` | Granularity of the gated statistic: off (correction disabled), token (each token), seq (ratio: product = exp(sum), unbiased/high-variance; binary_kl/tv: per-sequence mean), geo (per-seq geometric mean ratio = exp(mean), balanced). seq/geo are rejection filters and require --is_correction_mode mask. |
| `--algo.advantage.is_correction_gating` | `"ratio"` | Statistic the IS-correction gate reads: ratio (the importance weight itself, TIS) or the sampled-token binary_kl / tv divergence between rollout and train policy (a trust region; FlashREINFORCE = binary_kl at level seq, the per-sequence mean). Divergences only mask, with --is_correction_threshold delta. |
| `--algo.advantage.is_correction_mode` | `"mask"` | Bound treatment (token level only for clip/trunc): mask (drop out-of-band units, zero gradient), clip (clamp the weight into [low, high]), trunc (clamp only the upper tail). |
| `--algo.advantage.score_centering_top_k` | `0` | Score centering (https://arxiv.org/abs/2609.20807): record the rollout sampler's top-k log-probs per generated token and subtract its expected score from the policy loss, so vLLM/trainer drift cancels without an importance ratio. 0 = off; the paper uses 128 and reports 32 within noise at a quarter of the payload. Needs one update per batch (--train.max_epochs 1), --rollout.top_p 1.0 and fsdp tp_size 1; composes with is_correction_level token (mode trunc/clip/mask = the paper's TIS/MIS variants). |
| `--algo.advantage.is_correction_threshold` | `[0.5, 5.0]` | Bounds on the gated statistic: LOW HIGH is the [low, high] band on the off-policy IS ratio pi_train/pi_rollout; a single HIGH is an upper bound only (no lower bound, e.g. a trust-region delta). |
| `--algo.kl.use_loss` | `False` | whether to use KL loss from GRPO |
| `--algo.kl.estimator` | `"k1"` | In GRPO, k3 is utilized as the loss function, while k2, when used as the loss, is nearly equivalent to k1. The KL runs from the distribution that sampled the tokens to the other one: the policy with --algo.kl.ref model, the rollout policy with --algo.kl.ref rollout (k3 takes the matching form). |
| `--algo.kl.ref` |  | What the KL term compares the policy to. rollout (default): the policy that generated the samples, read from the log-probs vLLM returns with them -- no reference workers, so no extra memory; the trainer-generator KL of https://huggingface.co/blog/NormalUhr/sfc-kl (pair with --algo.kl.use_loss --algo.kl.estimator k3). model: a frozen reference model on its own GPUs (--ref.\*); on_policy_distill always uses it (the teacher). |
| `--algo.kl.init_coef` |  | KL coefficient. Defaults to 0 (no KL term, no reference workers); on_policy_distill defaults to 1.0. |
| `--algo.kl.target` |  | Target KL for adaptive control. Set to a positive value to adjust the KL coefficient so measured KL moves toward this target; leave unset to keep the coefficient fixed at --algo.kl.init_coef. |
| `--algo.kl.horizon` | `10000` | Adaptive-KL horizon in rollout samples; larger values produce slower coefficient updates. Used only when --algo.kl.target is set. |
| `--algo.dynamic_filtering_enable` | `False` | Enable dynamic filtering |
| `--algo.dynamic_filtering_range` | `(0, 1)` | Dynamic filtering rewards range |

## `--reward.*`

| Flag | Default | Description |
|---|---|---|
| `--reward.clip_range` | `(-10, 10)` | Reward clip range |
| `--reward.overlong_buffer_len` |  | DAPO-style context penalty: subtract factor \* clamp((peak_context_length - (data.max_len - buffer)) / buffer, 0, 1). Peak context includes prompt, generated tokens, tool feedback and expanded image tokens; take the maximum across segments after history compaction. Uses data.max_len, independently of the per-turn rollout.max_new_tokens (which may be unset). Unset disables it. |
| `--reward.overlong_penalty_factor` | `1.0` | Maximum magnitude of the overlong penalty. No effect without --reward.overlong_buffer_len. |
| `--reward.stop_properly_penalty_coef` |  | ProRL-style stop-properly penalty: scale truncated-rollout rewards by this coefficient in [0, 1], or set them to this value if negative. Note: 'truncated' in Molt covers vLLM finish_reason=length AND context exhaustion AND env turn caps. Unset disables it. |

## `--train.*`

| Flag | Default | Description |
|---|---|---|
| `--train.agent_path` |  | Agent script path |
| `--train.batch_size` | `128` | Global training batch size |
| `--train.micro_batch_size` | `1` | batch size per GPU |
| `--train.max_tokens_per_gpu` | `16192` |  |
| `--train.max_epochs` | `1` |  |
| `--train.num_episodes` | `1` |  |
| `--train.seed` | `42` |  |
| `--train.full_determinism_enable` | `False` | Enable reproducible behavior during distributed training |
| `--train.dynamic_batch_enable` | `False` | Group packed samples by token budget; reuses the loaded model's packed forward path. |
| `--train.force_on_policy` | `False` | Force true on-policy updates: accumulate gradients over the ENTIRE flattened rollout batch and run a single optimizer step at the final microbatch, instead of splitting the rollout into several train.batch_size / grad-accum windows. Needed for multi-turn flatten, where the per-rollout sample count is variable and a fixed train.batch_size would make every step after the first off-policy and drop the trailing samples. Requires --train.max_epochs 1. |
| `--train.colocate_fsdp_models` | `False` | Colocate the FSDP models (actor, reference, critic) on the actor's GPUs (they time-slice the same GPUs). |
| `--train.async_queue_size` | `1` | Queue size for async sampler\<-\>trainer |
| `--train.partial_rollout_enable` | `False` | Use vLLM pause/resume during weight sync so generation can overlap with training. |
| `--train.force_sync_mode` | `False` | Strictly on-policy: free the rollout slot only AFTER train_step refits vLLM, so the next batch is generated with the same weights the trainer recomputes it under. Removes the 1-step-stale rollout that inflates vllm_kl on routing-sensitive MoE checkpoints, at the cost of the generate/train overlap. |
| `--train.routing_replay` | `False` | R3: capture the rollout router's per-token expert selection (vLLM) and replay it in the training forward (AutoModel RouterReplay) so MoE training/rollout routing match. MoE models only; turns on --vllm.enable_prefix_caching (the capture requires it); incompatible with --train.partial_rollout_enable (preemption drops routing). |
| `--train.rollout_dump_dir` |  | Save each rollout batch to \<dir\>/rollout_step{N}.pt for later train-only replay. |
| `--train.rollout_replay_dir` |  | Load \<dir\>/rollout_step{N}.pt instead of generating (train-only replay). |
| `--train.check_weight_update_equal` | `False` | After each vLLM weight broadcast, warn which params vLLM did NOT refresh (stale rollout weights). |

## `--vllm.*`

| Flag | Default | Description |
|---|---|---|
| `--vllm.num_engines` |  | number of vLLM engines (required, \> 0) |
| `--vllm.tensor_parallel_size` | `1` | tensor parallel size of vLLM Engine for multi-GPU inference |
| `--vllm.pipeline_parallel_size` | `1` | pipeline parallel size per vLLM engine. For a giant MoE that overflows a node, set TP to the node GPU count and PP to the node span (TP\*PP GPUs/engine, ray executor): PP hands off between stages point-to-point across nodes instead of a cross-node TP all-reduce every layer. |
| `--vllm.data_parallel_size` | `1` | data parallel size per vLLM engine (mp DP backend, single node). vLLM has no standalone expert-parallel size: EP = TP \* DP, so raise DP to decouple EP from TP (DeepSeek-V3-style TP8+DP4 attention -\> EP32 experts). An engine spans TP\*DP GPUs; requires --vllm.enable_expert_parallel and the mp executor, and excludes --vllm.pipeline_parallel_size \> 1. |
| `--vllm.sync_backend` | `"nccl"` | trainer -\> vLLM weight sync backend |
| `--vllm.enforce_eager` | `False` | Disable CUDA graph in vLLM |
| `--vllm.router_policy` | `"cache_aware"` | vllm-router policy (default cache_aware: token-prefix affinity keeps a rollout's turns on one engine and falls back to the shortest queue when engines are imbalanced; consistent_hash \| round_robin \| power_of_two \| random) |
| `--vllm.router_balance_abs` | `10` | cache_aware keeps a rollout's turns on one engine until (max_in_flight - min_in_flight) exceeds this AND max \> --vllm.router_balance_rel \* min. The router's own 64 is sized for a serving fleet and is unreachable when one batch puts rollout.generate_batch_size requests in flight; vime runs this workload at 10. |
| `--vllm.router_balance_rel` | `1.2` | The ratio half of the cache_aware load-balance trigger (router default 1.5; vime 1.2). |
| `--vllm.mtp_num_speculative_tokens` | `0` | MTP speculative-decoding tokens for rollout (0=off). \>0 enables multi-token prediction in vLLM; the draft is auto-detected from the checkpoint's MTP head (Qwen3.6-MoE). Lossless (target verifies every token). 1 is a good default. |
| `--vllm.dtype` | `"bfloat16"` | vLLM inference dtype |
| `--vllm.gpu_memory_utilization` | `0.95` | vLLM gpu_memory_utilization |
| `--vllm.mm_encoder_attn_backend` |  | Optional vLLM vision encoder attention backend, e.g. TORCH_SDPA. |
| `--vllm.gdn_prefill_backend` |  | Optional vLLM GDN prefill backend for Qwen3-style linear attention layers. Choices: `flashinfer`, `triton`. |
| `--vllm.attention_backend` |  | Optional vLLM attention backend (FLASH_ATTN/FLASHINFER/TRITON_ATTN/FLEX_ATTENTION). Pass TRITON_ATTN to bypass AOT-compiled FA2/FlashInfer kernels on older drivers. |
| `--vllm.moe_backend` |  | Optional vLLM fused-MoE kernel backend (triton, flashinfer_trtllm, flashinfer_cutlass; default: vLLM's choice, FlashInfer TRT-LLM for bf16 MoE on Blackwell). |
| `--vllm.block_size` |  | vLLM KV cache block size (tokens). MiniMax-M3 MSA sparse attention requires 128; vLLM's default (16) raises 'No common block size' across M3's dense+sparse layers. |
| `--vllm.mamba_ssm_cache_dtype` |  | vLLM Mamba SSM state-cache dtype. Force 'float32' for hybrid Mamba2 models (NemotronH/omni3): vLLM defaults NemotronH to float16, so the recurrent SSM scan accumulates error over the rollout and rollout log-probs drift from the fp32 training recompute (inflates vllm_kl / seq-mask-TIS filtering). Matches opd-rl/nemo-rl. Choices: `auto`, `float32`, `float16`. |
| `--vllm.distributed_executor_backend` |  | Optional vLLM distributed executor backend override. Choices: `ray`, `mp`, `uni`. |
| `--vllm.enable_expert_parallel` | `False` | Enable vLLM TP+EP hybrid: experts EP-sharded across the TP ranks (Qwen3.5/3.6 MoE). |
| `--vllm.disable_custom_all_reduce` | `False` | Disable vLLM's custom all-reduce kernel and fall back to NCCL. Useful when custom all-reduce fails because GPU P2P or CUDA IPC is unavailable or incompatible. |
| `--vllm.enable_prefix_caching` / `--no-vllm.enable_prefix_caching` | `False` | vLLM prefix KV cache. Multi-turn rollouts re-prefill the growing history each turn; prefix caching cuts that cost. The trainer invalidates the cache after every weight broadcast, including the blocks held by requests that straddle it — without that, rollouts get served KV computed under the previous policy. Off by default: it makes the rollout/train logprob drift arrive sooner and larger. Incompatible with MTP speculative decoding. |
| `--vllm.enable_chunked_prefill` / `--no-vllm.enable_chunked_prefill` |  | vLLM chunked prefill (default: vLLM auto — True for non-encoder-decoder models). |
| `--vllm.max_num_batched_tokens` |  | vLLM scheduler token budget per iteration (default: vLLM auto, \~2048 with chunked prefill). Set \>= max_model_len so every prefill fits in one chunk and a recurrent-state model (Mamba2/GDN) never hands its state across chunk boundaries. |
| `--vllm.async_scheduling` / `--no-vllm.async_scheduling` |  | vLLM async scheduling (default: vLLM auto — True for mp/uniproc executors with no spec-decode). |
| `--vllm.decode_context_parallel_size` | `1` | vLLM decode-context-parallel size; shards KV cache across decode workers for long contexts. |

## `--rollout.*`

| Flag | Default | Description |
|---|---|---|
| `--rollout.batch_size` | `1024` | Batch size for make experience |
| `--rollout.num_runners` | `2` | Router path: number of runner-pool actors (rollouts + in-process reward grading). Rollouts are dispatched one per call, so up to vllm_generate_batch_size \* n_samples_per_prompt runners stay busy. |
| `--rollout.vllm_generate_batch_size` |  | Batch size for vLLM generating samples |
| `--rollout.micro_batch_size` | `1` |  |
| `--rollout.max_new_tokens` |  | Max tokens to generate per sample. If None, dynamically computed as max_len - prompt_len per sample. |
| `--rollout.max_tokens_per_gpu` |  |  |
| `--rollout.n_samples_per_prompt` | `1` | number of responses for each prompt in generation |
| `--rollout.top_p` | `1.0` |  |
| `--rollout.temperature` | `1.0` |  |

## `--eval.*`

| Flag | Default | Description |
|---|---|---|
| `--eval.dataset` |  | Path to the evaluation dataset |
| `--eval.batch_size` |  | Concurrent eval prompt groups; defaults to --rollout.batch_size when unset. |
| `--eval.split` | `"train"` |  |
| `--eval.steps` | `-1` | Evaluate every N steps; -1 disables eval. |
| `--eval.temperature` |  | Eval temperature; falls back to --rollout.temperature when unset. |
| `--eval.top_p` |  | Eval top-p; falls back to --rollout.top_p when unset. |
| `--eval.max_new_tokens` |  | Eval max new tokens; falls back to --rollout.max_new_tokens when unset. |
| `--eval.n_samples_per_prompt` |  | Eval samples per prompt; falls back to --rollout.n_samples_per_prompt when unset. |
| `--eval.eval_at_start` | `False` | Run one baseline eval at global_step 0 (before any update) to measure the pre-RL model. Fresh runs only — gated on the consumed-prompt counter being 0, so a resume (which loads a non-zero step) does not add a redundant eval. |
| `--eval.eval_only` | `False` | Score --eval.dataset once and exit — no training. vLLM already holds the HF weights, so the policy/ref/critic FSDP actors are never built and their GPUs go to the eval (use all nodes for vLLM engines / env runners, or fewer nodes). Requires --eval.dataset. |

## Other flags

| Flag | Default | Description |
|---|---|---|
| `--local_rank` | `-1` | local_rank from torchrun |
| `--use_ms` | `False` | Resolve models from ModelScope hub. |
