# molt integration

"Where consistency lives" describes `main`. From "The aligned mode" on, the file describes the reference
implementation: branch `jianh/batch-invariant`, built against vLLM 0.28 for Qwen2/Qwen3 dense and Qwen3-MoE,
not on `main`. Every engine fact that depends on the installed vLLM lives in one table, [facts bound to the
pinned vLLM](#facts-bound-to-the-pinned-vllm); the rest of the file is method. What each row must make
equal: [contracts.md](contracts.md); how to build the trainer side: [training-side.md](training-side.md).

## Where consistency lives in molt (`main`)

### Which vLLM am I on

1. The pin is `setup.py`'s `vllm==` extra and the image installs it; read `vllm.__version__` inside a
   worker container, never a Dockerfile comment (the comments and the install line have disagreed before).
2. `python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"` in the container and
   `PYTHONPATH` the Ray workers use — an overlay imports before the wheel.
3. An image tag is not a pin: record the OCI digest next to the version
   ([provenance](acceptance.md#environment-and-provenance)).
4. Compare with the "Verified on" column of the [pinned facts](#facts-bound-to-the-pinned-vllm) and with
   the reference implementation's pin; tag every fact you reuse with the version it was verified on.

### Trainer kernel choice

- Place: `molt/models/base.py`, the `BackendConfig` dict (`attn`, `linear`, `experts`, `rms_norm`,
  `gate_precision`, `dispatcher`) and `_resolve_custom_backend_attn`.
- AutoModel custom models; env overrides `MOLT_LINEAR_BACKEND`, `MOLT_MOE_EXPERTS`, `MOLT_RMS_NORM`,
  `MOLT_GATE_PRECISION`, `MOLT_MOE_DISPATCHER`. Architectures without an AutoModel definition fall back to
  HF transformers.

### Master weights and dtype

- Place: fp32 masters, FSDP2 `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`.
- `--fsdp.param_dtype` has no fp32. The MoE router computes in fp32 through `gate_precision=float32`;
  `MOLT_RMS_NORM=torch_fp32` is a compute dtype, norm weights follow `param_dtype` unless the AutoModel
  class keeps them fp32.

### Refit

- Place: `molt/trainer/workers/policy_actor.py` broadcast, `molt/trainer/vllm/vllm_worker_wrap.py`.
- The broadcast sends the fp32 master; vLLM rounds it to bf16 on `load_weights` with the same
  round-to-nearest cast FSDP2's `param_dtype` applies in the training forward. For parameters vLLM loads
  with a plain `copy_` the engine bytes therefore equal the training-forward bytes; derived parameters
  (Mamba `A = -exp(A_log)`, calibration scales) are not a cast — hash them.
- `--train.check_weight_update_equal` lists vLLM parameters the broadcast never assigned (stale weights); it
  does not compare bytes. Prove refit bytes by hashing the same tensor on both sides after `load_weights`,
  or by the logprob comparison itself.
- The broadcast runs inside vLLM's layerwise-reload bracket (`WorkerWrap.begin_weight_update` /
  `end_weight_update`): `load_weights` copies in place, so graph-captured addresses stay valid, and the
  bracket re-runs vLLM's post-load weight processing (kernel layouts such as the FlashInfer MoE block
  layout) once the whole update has landed, never on half-updated weights.
- Under `--actor.lora_dim > 0` the refit merges `W + scale·B·A` in fp32 while the training forward runs the
  unmerged `W·x + scale·B(A·x)` — one rounding apart by construction, so LoRA is off the bitwise contract.

### Trainer scorer

- Place: `molt/models/utils.py::log_probs_from_logits`: bf16 logits upcast per 256-row chunk to fp32,
  divided by `--rollout.temperature` in fp32, torch `gather − logsumexp`.
- The scorer dispatches on the logits dtype. With `--actor.entropy_coef` ≠ 0 (no CP, no score-centering
  top-k targets, flash-attn importable) the actor passes fp32 logits and the Triton cross-entropy kernel
  runs instead — two trainer scorer kernels to inventory. The TP variant
  `log_probs_from_vocab_parallel_logits` in `molt/trainer/fsdp/packing.py` has an all-reduce order that is
  part of the contract.
- The engine scorer is a third kernel: the sampler's own logprob kernel over the sampler's logits, which
  batch-invariant mode's aten overrides do not reach, and whose logits dtype depends on whether a logits
  processor is active (molt's default `min_tokens=1` is one while a stop token exists). Call the kernel the
  engine's sampler actually uses ([pinned facts](#facts-bound-to-the-pinned-vllm); [engine scorer
  path](debugging.md#isolate-the-side-first)).

### Rollout logprobs

- Always requested: every molt rollout path sends `logprobs=1` (the `/inference/v1/generate` client in
  `molt/trainer/rollout/router.py`, and the chat server forces it on), `logprobs=k` under score centering
  (`--algo.advantage.score_centering_top_k`), with `logprobs_mode=processed_logprobs`. `mismatch/*` therefore
  exist in every arm; the IS setting decides only what the loss does with the ratio.
- `--algo.advantage.is_correction_level off` leaves the ratio out of the loss. Any other level multiplies
  the loss by the per-token ratio π_old/μ — a `token` gate with threshold `0 1e9` never masks but still
  weights — which is exactly 1 only when `bitwise_frac` = 0: under default kernels it changes the BI=0
  gradient. Both arms run the same correction settings and the report declares the confounder
  ([arms](acceptance.md#one-recipe-one-switch)).
- Sequence gates: `--algo.advantage.is_correction_level seq|geo` accept only
  `--algo.advantage.is_correction_mode mask` (`loss.py`). One flag per sequence zeroes every token of an
  out-of-band sequence, and the survivors still carry the per-token ratio. On the BI=0 arm a whole batch can
  be masked; molt then warns `is_filter_ratio=…: the vLLM importance-sampling filter dropped (nearly) every
  sequence — zero policy gradient this step` (`rl_trainer.py`). A step with that warning made no update — a
  false green for "two effective updates".
- Temperature applied in vLLM; `--rollout.top_p 1.0` (default), no top-k ([object of
  comparison](contracts.md#object-of-comparison)).

### Mismatch metrics

- Place: `mismatch_metrics` in `molt/models/utils.py`, logged by `policy_actor.py` from the grad-enabled
  training forward whenever rollout logprobs exist: `mismatch/bitwise_frac` (fraction of valid tokens whose
  fp32 logprob bytes differ, `view(int32)`), `mismatch/abs_diff_mean`, `mismatch/k3_kl`. `vllm_kl` in
  `molt/models/loss.py` is the older signed masked mean of `log μ − log π_old`.
- The pass signal is `mismatch/bitwise_frac` = 0 at every step; the replay comparison in debugging.md adds
  the count, max and first failing position. `vllm_kl` cannot tell a last-bit difference from 0 (positive
  and negative differences cancel) and is diagnostic only.
- Absent is not zero: `mismatch/*` missing from the log means no rollout logprob reached the trainer (a
  replay without them, an agent path of its own) — find out why before reading a 0.
  `policy_kl` is 0 by construction under `--train.force_on_policy` with `--algo.kl.init_coef 0` (then the
  old logprob is the detached training forward) and is not evidence. Check the producer before trusting any
  0.

### Dumps and replay

- `MOLT_DUMP_ROLLOUT_LOGPROBS`: one sequence, rank 0, once per process, 6 decimals — triage for a position
  shift or a gross gap, never byte evidence.
- `--train.rollout_dump_dir` saves `rollout_step{N}.pt` with the samples materialized (the heavy tensors are
  reloaded from the runners' object store before the save, so the file replays in another job —
  [debugging.md](debugging.md#save-and-compare-protocol)). `--train.rollout_replay_dir` trains on saved
  rollouts without sampling (a full training step: old-logprob forward, update, refit). A dump is bound to
  its weight version by step number only, so keep that step's pre-update checkpoint beside it.
- Byte evidence = replay plus an int-view comparison over all valid tokens
  ([debugging.md](debugging.md#compare-per-token-logprobs)).

### Synchronous run

- `--train.force_sync_mode --train.async_queue_size 1 --train.max_epochs 1`, no
  `--train.partial_rollout_enable`, `--rollout.vllm_generate_batch_size` equal to `--rollout.batch_size` (a
  larger value turns partial rollout on).
- One optimizer step per rollout batch: `--train.batch_size` (samples per optimizer step) =
  `--rollout.batch_size × --rollout.n_samples_per_prompt`, or `--train.force_on_policy`. With several
  minibatch updates per rollout the microbatches after the first update compare against updated weights and
  legitimately read ≈ 1 — `mismatch/*` is logged per grad-enabled microbatch.
- Anything else is lag ≥ 1 ([layout rules](#layout-rules)).

### Engine environment and worker hook

- Place: `molt/trainer/vllm/vllm_engine.py::_configure_vllm_env` runs inside the Ray actor before
  `AsyncEngineArgs` and exports `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, `RAY_ADDRESS` and, under
  `--train.full_determinism_enable`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`; the aligned mode adds
  `VLLM_BATCH_INVARIANT=1` and the workspace/NCCL pins here.
- `worker_extension_cls="molt.trainer.vllm.vllm_worker_wrap.WorkerWrap"` is imported in every worker before
  CUDA-graph capture: the place for kernel patches that must be identical to the trainer's, and for switching
  off a kernel the installed vLLM selects by GPU architecture (the sm_103 router row of the [pinned
  facts](#facts-bound-to-the-pinned-vllm)). A kernel choice that differs by architecture is a closure field:
  record the architecture with every result.
- Standalone engine for C3: export the same env, then `vllm.LLM(model,
  worker_extension_cls="molt.trainer.vllm.vllm_worker_wrap.WorkerWrap", seed=<--train.seed + i>,
  logprobs_mode="processed_logprobs", **<the recipe's --vllm.* flags>)` and
  `SamplingParams(temperature=<--rollout.temperature>, top_p=1, logprobs=1, min_tokens=1)`; any other env or
  flag set measures another engine.

### Engine knobs

- `--vllm.attention_backend TRITON_ATTN`: the only backend the aligned attention kernel patches;
  auto-selection picks FA3/FlashInfer and aligns nothing, without an error.
- `--vllm.enforce_eager`; `--vllm.enable_prefix_caching` (default off, on under `--train.routing_replay`;
  molt resets the cache after every refit — [pinned facts](#facts-bound-to-the-pinned-vllm));
  `--vllm.enable_chunked_prefill` (vLLM's default; a hybrid may be unable to turn it off — pinned facts);
  `--vllm.max_num_batched_tokens`; `--vllm.mamba_ssm_cache_dtype`; `--vllm.block_size` (KV page; read the
  resolved value in the engine log). Their interaction on a state-layer model: [Mamba
  hybrid](#mamba-hybrid); engine-side pitfalls: [prefix cache](debugging.md#prefix-cache-semantics),
  [chunked prefill](debugging.md#chunked-prefill), [lifecycle](debugging.md#lifecycle-memory-metrics).
- `--vllm.mtp_num_speculative_tokens 0`: MTP on puts the draft in both arms and needs spec-verify logprob ==
  full forward; molt's weight-sync does not refresh the draft block (it shares only embed/lm_head with the
  target), so acceptance drifts across updates ([MTP](debugging.md#speculative-decoding--mtp)).
- Graph mode: fix capture; eager is diagnostic only, and the C3 graph cell must pass before a graph path is
  claimed. Never claim a backend by bypassing vLLM's batch-invariant support check in the backend selector:
  align the backend first, then let the selector declare it.

### Routing replay (R3)

- `--train.routing_replay` → vLLM `enable_return_routed_experts`; the trainer replays the engine's expert
  ids. Under the switch R3 is on for MoE (startup guard) and the report lists the disagreement rate it masks
  ([reference measurements](#reference-measurements)). It may be switched off only after C3/C5 show the
  router bitwise without it (a batch-invariant fp32-output gate GEMM and a fixed tie-break — the vLLM blog
  post's DeepSeek-V4 recipe).
- Incompatible with partial rollout (preemption drops the routing) and MTP; molt raises at startup. Tokens
  the engine did not sample (tool outputs) have no record and fall back to natural routing (molt fills them
  with a −1 sentinel that keeps the live selection). Engine-side checks of the returned ids: [R3
  pitfalls](debugging.md#routing-replay-r3).
- vLLM indexes the returned ids by global decoder-layer id (non-MoE layers are zero rows); molt's gates
  replay their own global-id row, parsed from the module path `layers.N`, and raise when the capture has
  fewer layer rows than the highest MoE gate's global id needs. The trainer has no pipeline parallelism, so
  no stage offset applies.
- The engine returns the ids through a connector whose requirements are version-bound (a model runner, a
  MoE generate model, prefix caching — [pinned facts](#facts-bound-to-the-pinned-vllm)). `main` turns
  `--vllm.enable_prefix_caching` on under `--train.routing_replay` (`train_rl_ray.py` prints
  `[routing_replay] enabling --vllm.enable_prefix_caching`), so an R3 recipe always runs with the cache and
  C3 must verify hit == miss.

### Seed determinism

`--train.full_determinism_enable` calls `transformers.enable_full_determinism(seed)` —
`torch.use_deterministic_algorithms(True, warn_only=False)` (an op without a deterministic implementation
raises), cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG=:16:8`, `CUDA_LAUNCH_BLOCKING=1`,
`FLASH_ATTENTION_DETERMINISTIC=1` — and runs the engine single-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`).
It is main-thread only ([contracts.md](contracts.md#determinism-only-on-the-main-thread)) and does not make
kernels batch-invariant.

## Facts bound to the pinned vLLM

The pin is `setup.py`'s vllm version; the image installs it; read `vllm.__version__` inside a worker
([which vLLM am I on](#which-vllm-am-i-on)). When it differs from the "Verified on" column, re-run the
check before relying on the row. Checks run inside a worker container against the installed package,
`V=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")`; a row marked "kernel" is a
numerical property no grep proves — re-verify it with the probe named.

| Fact | Verified on | How to re-verify | Where it matters |
|---|---|---|---|
| Sampler logprob kernel = `compute_token_logprobs` (row-max + logsumexp Triton), not `log_softmax`; logits bf16, or an fp32 batch copy when a logits processor is active (molt's default `min_tokens=1` is one) | 0.31.0 | `grep -rn -e "def compute_token_logprobs" -e "float32).copy_" $V/v1/worker/gpu/sample/` | [trainer scorer](#trainer-scorer), [engine scorer path](debugging.md#isolate-the-side-first) |
| Batch-invariant mode overrides aten ops (`mm`, `addmm`, `_log_softmax`, …); the sampler kernel above is not one, so the override does not reach it | 0.31.0 | `grep -n "aten::_log_softmax" $V/model_executor/determinism/batch_invariant.py`; `grep -n "torch.*log_softmax" $V/v1/worker/gpu/sample/logprob.py` (empty) | [trainer scorer](#trainer-scorer), [operator details](#operator-details) |
| Routed-expert ids (R3) travel through the AuxOutput connector, which requires Model Runner V2, a MoE generate model and prefix caching; engine init raises otherwise | 0.31.0 | `grep -n "AuxOutput Connector" $V/config/vllm.py` | [routing replay](#routing-replay-r3), [Mamba hybrid](#mamba-hybrid) |
| Mamba hybrid + prefix caching: cache mode `align` asserts chunked prefill on; `mamba_block_size` = attention block, raised to the Mamba page; without prefix caching it is `max_model_len` | 0.31.0 | `grep -n -e align -e mamba_block_size $V/model_executor/models/config.py`; `grep -n "attn_block_size =" $V/platforms/interface.py`; resolved sizes: engine log | [Mamba hybrid](#mamba-hybrid), [chunked prefill](debugging.md#chunked-prefill) |
| CUDA-graph capture fails at engine init for gated delta net and Mamba hybrids, so they run eager | 0.31.0 | start the engine with `--vllm.enforce_eager` off; the init error reproduces, or the row is stale | [Mamba hybrid](#mamba-hybrid) |
| `GateLinear` (MoE router GEMM) switches kernel by M: a small-M CuteDSL kernel at M ≤ 16, cuBLAS above | 0.31.0 | `grep -n "LL_BF16_MAX_TOKENS" $V/model_executor/layers/fused_moe/router/gate_linear.py` | [operator details](#operator-details) |
| On sm_103 (GB300) the small-M CuteDSL router kernel (`cute_dsl/ll_bf16.py`) fails to compile at engine warmup (compiler ICE) and every MoE engine dies at init; `ll_bf16._cutedsl_available = False` set from the worker hook sends every M through cuBLAS | 0.32.0, sm_103 | start one engine on the target GPU: the warmup error reproduces, or the row is stale; `grep -n _cutedsl_available $V/model_executor/kernels/linear/cute_dsl/ll_bf16.py` | [engine environment and worker hook](#engine-environment-and-worker-hook) |
| Stock `_C.rms_norm` switches its CUDA block size at 256 tokens (`num_tokens < 256 ? 1024 : 256`) | 0.31.0 | compiled, not in the wheel: `grep -rn "num_tokens < 256" csrc/` at the pinned tag, or compare 255/256/257 rows alone vs batched | [operator details](#operator-details), [norm contract](contracts.md#per-operator-contract) |
| TRITON_ATTN takes its 3D split only in decode-only steps with at most `seq_threshold_3D` sequences, never under `VLLM_BATCH_INVARIANT=1` | 0.31.0 | `grep -n -A 10 "use_3d = not" $V/v1/attention/ops/triton_unified_attention.py` | [default-kernel A/A](debugging.md#isolate-the-side-first) |
| A `prompt_logprobs` request skips the prefix-cache read by default (`skip_reading_prefix_cache`) and scores only the positions this prefill computed, so cached positions are omitted on a hit | 0.31.0 | `grep -n -A 4 "skip_reading_prefix_cache =" $V/sampling_params.py`; `grep -n num_computed_prefill_tokens $V/v1/worker/gpu/sample/prompt_logprob.py` | [prompt-logprob scoring](debugging.md#isolate-the-side-first) |
| In-engine DP ranks (`--vllm.data_parallel_size`) share `model_config.seed`; no per-rank offset | 0.31.0 | `grep -n set_random_seed $V/v1/worker/gpu_worker.py`; `grep -n seed $V/config/parallel.py` (empty = no offset) | [seed collision](debugging.md#sampling-seed-collision) |
| ReplaySSM (`--use-replayssm`) is a third Mamba2 decode path with its own launch configs, bitwise with neither the SSD prefill nor the stock decode | 0.31.0 | `ls $V/model_executor/layers/mamba/ops/replayssm_config.py`; `grep -n use_replayssm $V/config/cache.py` | [Mamba hybrid](#mamba-hybrid), [SSM chunk vs decode](contracts.md#ssm-chunk-vs-per-token-recurrence) |
| Kernel: the chunked Mamba2 scan (SSD) is batch-invariant and split-invariant at 128-aligned splits only | 0.31.0 | re-verify with the probe: prefill one prompt whole and split at 128k and 128k ± 1 tokens, compare bytes | [Mamba hybrid](#mamba-hybrid), [state-layer contract](contracts.md#state-layer-contract) |

## The aligned mode

One switch, `--train.batch_invariant`; with the switch off the default path must stay byte-identical. The
reference implementation takes vLLM's generic batch-invariant mode (`VLLM_BATCH_INVARIANT=1`: Triton matmul,
segmented attention) and recovers speed in C10; the recipes in the vLLM blog post instead make the
deployment kernels themselves invariant (DeepGEMM batch-invariant mode, fixed split-KV, single-tactic
grouped GEMM). Both satisfy the contracts; the choice is a C1/C10 cost decision per model.

Process environment, in the trainer and in every engine worker (they share the weight-sync NCCL group, so
NCCL settings must match): `VLLM_BATCH_INVARIANT=1`, vLLM's `override_envs_for_invariance()` (NCCL
algo/proto/channel pins), `CUBLAS_WORKSPACE_CONFIG=:16:8` and `CUBLASLT_WORKSPACE_SIZE=1` set before the
first cuBLAS handle; whitelisted into the Ray runtime env.

Engine: `CompilationConfig(mode=NONE, cudagraph_mode=FULL)` (no Inductor fusion, graphs allowed),
`TRITON_ATTN` with the segmented kernel, prefix caching off or verified bitwise on hit and miss, `top_p=1`,
logprobs requested, one seed per engine. Worker import hook: the same kernel patches the trainer applies
(matmul tile table, RMSNorm kernel, MoE tile pin), before graph capture.

Trainer: after distributed init, `setup_process()` (same env,
`torch.backends.cuda.matmul.fp32_precision="ieee"`), then patch the model class forwards. The patch is
process-wide (class forwards and every `nn.Linear`, lm_head and vision towers included): build the BI=0
arm's model before patching, or an A/B reads 0 because both sides were aligned. Every swapped operator is a
`torch.autograd.Function` ([wiring](training-side.md#backward-reuse-glue-or-new)); gradients are "the same
function up to rounding", and activation checkpointing recomputes bitwise because the forward is
deterministic.

## Operator table

H100, reference implementation; version-bound cells link to the [pinned
facts](#facts-bound-to-the-pinned-vllm); re-verify every row against the installed vLLM before relying on it.

| Operator | vLLM kernel under `VLLM_BATCH_INVARIANT=1` | Trainer forward | Backward |
|---|---|---|---|
| Linear (q/k/v/o, gate/up/down, lm_head) | `linear_batch_invariant` → Triton persistent matmul; tile M independent of the K reduction | Same kernel on the gathered FSDP2 parameter | `g @ W`, `gᵀ @ x` |
| RMSNorm / residual-add + RMSNorm | `torch.ops._C.rms_norm`, `fused_add_rms_norm`; block size pinned by the kernel patch | Same ops; keep the `(hidden, residual)` pair | fp32 native math |
| SiLU · up | `torch.ops._C.silu_and_mul` on `cat(gate, up)` | Same op; `cat(gate_proj(x), up_proj(x))` is bitwise the merged GEMM | fp32 analytic |
| RoPE (neox) | `torch.ops._C.rotary_embedding`, bf16 cos/sin built on GPU | Same op | Same op with `(cos, −sin)` |
| Attention | Triton unified attention over absolute KV segments ([details](#operator-details)) | Paged KV + a block table per packed sequence; causal GQA only | flash-attn varlen recompute |
| MoE router + experts | `fused_topk` + `fused_experts`, pinned tile config; ids from R3 ([details](#operator-details)) | Same kernels on strided views of the AutoModel `[E, …]` weights; replay overrides the ids | `grouped_mm` native math |
| Logits / logprobs | fp32 processed logprobs; engine scorer kernel ([details](#operator-details)) | fp32 `gather − logsumexp` in 256-row chunks — a different kernel | Native autograd |

### Operator details

- RMSNorm: the stock `_C.rms_norm` switches block size with the token count ([pinned
  facts](#facts-bound-to-the-pinned-vllm)); the kernel patch pins it ([norm
  contract](contracts.md#per-operator-contract)).
- Attention: fixed keys per segment, so decode and chunked prefill reduce in the same order; segment
  constants are shared by both sides and accepted on the end-to-end A/B only ([changing the arithmetic
  baseline](contracts.md#changing-the-arithmetic-baseline)); fp32 partial buffers are sized by a fixed
  context ceiling and never freed (widening the ceiling changes numerics).
- MoE: routing weights are re-gathered from the fp32 softmax after replay overrides the ids; the router gate
  GEMM may switch kernel by M ([pinned facts](#facts-bound-to-the-pinned-vllm)) — pin one GEMM path on both
  sides; grouped top-k is a fused CUDA kernel whose bytes torch sigmoid + renormalisation cannot reproduce —
  call the kernel.
- Logits: the engine scorer is the sampler's logprob kernel, outside batch-invariant mode's overrides
  ([trainer scorer](#trainer-scorer), [pinned facts](#facts-bound-to-the-pinned-vllm)); the trainer's
  `gather − logsumexp` is a different kernel: call vLLM's or prove equal bytes.

### Not covered

Not covered by upstream batch-invariant mode and not in the reference implementation: Mamba2/SSM layers
(table below), sliding window, logit softcap, attention sinks, fp8 KV, multimodal encoders, mRoPE,
multi-turn/tool runs under R3 ([R3](#routing-replay-r3)). Each needs its own row in the C0 inventory with a
verdict before the model is declared supportable. For a multimodal model the compared path starts at the
image bytes: the trainer's processor must reproduce the engine's pixel tensors and placeholder layout byte
for byte (hash both sides), and the engine's multimodal processor cache is a batch-composition dependency
to inventory — with prefix caching and that cache off, vLLM 0.31 keyed images by a per-renderer counter
rather than by content, and engines reused each other's image embeddings (observed live; re-verify on the pin).

### Mamba hybrid

The shipped hybrid recipe (`examples/scripts/slurm/rl_omni3_30b.sh`) and what each engine setting does to
alignment; the version-bound reasons are rows of the [pinned facts](#facts-bound-to-the-pinned-vllm).

| Setting | State | Why | Consequence |
|---|---|---|---|
| R3 `--train.routing_replay` | on (MoE) | the router is not batch-invariant; replay takes it off the compared path | may require prefix caching (see pinned facts); engine init fails otherwise |
| Prefix caching `--vllm.enable_prefix_caching` | off by default; on for R3 | the routed-ids store may be keyed by prefix-cache blocks (see pinned facts) | Mamba cache mode becomes `align` (see pinned facts); C3 proves hit == miss; molt resets the cache after every refit |
| Chunked prefill `--vllm.enable_chunked_prefill` | on; `align` may refuse to turn it off (see pinned facts) | `align` materialises SSM state at page boundaries | prefill splits at the Mamba page, not at a 128-aligned SSD chunk; test the page edges |
| Graph mode `--vllm.enforce_eager` | eager while capture fails at engine init for GDN and Mamba hybrids (see pinned facts) | a capture bug, not a numerics choice | an eager result proves eager only; the aligned target is `cudagraph_mode=FULL` once the C3 graph cell passes |
| Mamba cache dtype `--vllm.mamba_ssm_cache_dtype` | `float32` | the default follows the model dtype and rounds the state across the rollout scan; the trainer recomputes in fp32 | the boundary-state dtype of the [state-layer contract](contracts.md#state-layer-contract) |

- Page: under prefix caching the attention block is raised to the Mamba page and the prefill splits there
  (see pinned facts); read the resolved block size in the engine log and let C3 test page − 1 / page /
  page + 1 and a prompt spanning two pages — the SSD scan is split-invariant only at 128-aligned splits
  (pinned facts, kernel). Without prefix caching the page is `max_model_len`, and
  `--vllm.max_num_batched_tokens` ≥ `max_model_len` keeps a prompt in one chunk.
- Decode and prefill stay two rounding paths until decode replays the chunked arithmetic with an fp32 state
  (upstream PR vllm-project/vllm#55905) or the trainer replays both phases ([SSM chunk vs per-token
  recurrence](contracts.md#ssm-chunk-vs-per-token-recurrence)); ReplaySSM is a third path, not that fix
  (pinned facts). Gated delta net (GDN) layers follow the same rows with their own state list.

## Layout rules

- Packing: THD `cu_seqlens` on the AutoModel path, per-sequence block tables for attention; batch
  composition must not change bytes (check alone vs packed with neighbours).
- Parallelism: the reduction tree is part of the contract — trainer TP (vocab-shard logprob reduction),
  engine TP (all-reduce order; vLLM's custom all-reduce kernel unless `--vllm.disable_custom_all_reduce`
  sends it through the pinned NCCL), EP (combine order). Verify the actual layout. The reference
  implementation guards `cp_size == 1`; CP > 1 is in
  [training-side.md](training-side.md#layout-length-and-world-size).
- Only identical weights are bitwise. Under any lag ≥ 1 the rollout came from older weights and sits on the
  lag floor whatever the learning rate or kernels — `bitwise_frac` reads ≈ 1 ([reference
  measurements](#reference-measurements)); alignment removes kernel noise, not staleness. An asynchronous
  recipe is accepted on quality and performance and reads the lag floor, never 0. In a multi-turn agentic
  recipe with episodes in flight across refits, one episode's turns were sampled under several weight
  versions; its `mismatch/*` mixes lag with kernel noise and is kernel evidence only from a synchronous run.

## Adding an architecture

1. **Inventory (C0)** from vLLM's model file, not from the HF reference, with the [module
   inventory](training-side.md#module-inventory) columns.
2. **One `autograd.Function` per operator** that produces a value; reuse the existing ones; keep vLLM's op
   order and fusion boundaries (concatenate gate/up before `silu_and_mul`; fused norm returns the residual;
   q/k norm position relative to RoPE).
3. **Patch point**: the model's AutoModel classes, listed in an explicit table (no implicit dispatch — a
   half-patched model aligns by accident on some ops and not others); the HF classes for HF-fallback models.
4. **Offline proof**: fixed weights and prompts, dump both sides per layer, first diverging layer; every
   compared tensor byte-equal, not "close".
5. **Live proof**: a [synchronous run](#synchronous-run) with [rollout logprobs](#rollout-logprobs)
   requested at non-binding thresholds: `mismatch/bitwise_frac` 0 over all valid tokens of every step; then
   the asynchronous recipe with its correction, expecting the lag floor.
6. **Guards at startup**: `--vllm.attention_backend TRITON_ATTN` forced, `cp_size == 1` (until the CP note
   is proven), context ≤ the segment ceiling, MoE ⇒ R3 on, `top_p == 1`, `--vllm.mtp_num_speculative_tokens
   0`; fail loudly, never fall back.
7. **Cost**: expect the aligned arm clearly slower at first ([reference
   measurements](#reference-measurements)); remove structural defects in C4 and recover the rest in C10 with
   the profile until the rollout deficit is ≤ 20 % ([performance budget](acceptance.md#performance-budget)),
   never by loosening the contract.

## Reference measurements

Expectations only; every new model, version or layout measures its own. Source `ref` = the reference
implementation (H100, Qwen3-8B dense / Qwen3-30B-A3B MoE); `post` = the vLLM blog post's
Nemotron-H / MiMo runs.

| Quantity | Value | Source | Use |
|---|---|---|---|
| `vllm_kl` of the aligned arm, synchronous | ≈ 1e-7 (a last-ulp residue on a few tokens; not byte-proven) | ref | Why the pass signal is `bitwise_frac`, not `vllm_kl` |
| Lag floor (identical kernels, rollout one update stale) | binary KL ≈ 1.2–1.8e-4 per sequence, `vllm_kl` ≈ 3–4e-4 | ref | Any asynchronous recipe reads this, not 0 |
| Natural-routing disagreement masked by R3 | ≈ 50 % of MoE positions | ref | Report it next to every MoE result |
| Attention segment ceiling / trainer KV block | 16k tokens / 16 tokens | ref | Boundary lengths to test |
| Microbenchmark-best segment constants vs end-to-end | lost 8–11 % end to end | ref | Accept constants on end-to-end A/B only |
| Aligned cost before optimisation | rollout ≈ 2× slower; trainer attention ≈ 4×, linear ≈ 2.5× | ref | The C10 starting point; the budget caps the rollout deficit at 20 % |
| Same-prompt siblings, healthy sampling | ≈ 0.1 % identical answers (≤ 8 tokens); 1–3 % share a ≥ 16-token prefix | post | More means seed correlation |
| Gradient difference from a microbatch re-split | ≈ 2e-3 relative | post | Gradient tolerance denominator, not a bug |
| Proxy vs full model mismatch magnitude, same defect | 12-layer proxy ≈ 1e-12, full model ≈ 1e-2 | post | Small-model magnitudes do not extrapolate |
