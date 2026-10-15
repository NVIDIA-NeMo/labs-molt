# molt integration

The first section describes `main`. Everything from "The aligned mode" on describes the reference
implementation on branch `jianh/batch-invariant` (built against vLLM 0.28; Qwen2/Qwen3 dense and
Qwen3-MoE), which is not on `main` yet and is being upstreamed as its own PR. What each row must
make equal is in [contracts.md](contracts.md); how to build the trainer side is in
[training-side.md](training-side.md).

## Where consistency lives in molt (`main`)

### Trainer kernel choice

- Place: `molt/models/base.py`, the `BackendConfig` dict (`attn`, `linear`, `experts`, `rms_norm`,
  `gate_precision`, `dispatcher`) and `_resolve_custom_backend_attn`.
- AutoModel custom models; env overrides `MOLT_LINEAR_BACKEND`, `MOLT_MOE_EXPERTS`, `MOLT_RMS_NORM`,
  `MOLT_GATE_PRECISION`, `MOLT_MOE_DISPATCHER`. Architectures without an AutoModel definition fall
  back to HF transformers.

### Master weights and dtype

- Place: fp32 masters, FSDP2 `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`.
- `--fsdp.param_dtype` has no fp32. The MoE router computes in fp32 through
  `gate_precision=float32`; `MOLT_RMS_NORM=torch_fp32` is a compute dtype, norm weights follow
  `param_dtype` unless the AutoModel class keeps them fp32.

### Refit

- Place: `molt/trainer/workers/policy_actor.py` broadcast, `molt/trainer/vllm/vllm_worker_wrap.py`.
- The broadcast sends the fp32 master; vLLM rounds it to bf16 on `load_weights` with the same
  round-to-nearest cast FSDP2's `param_dtype` applies in the training forward, so for parameters
  vLLM loads with a plain `copy_` the engine bytes equal the training-forward bytes; derived
  parameters (Mamba `A = -exp(A_log)`, calibration scales) are not a cast — hash them.
- `--train.check_weight_update_equal` lists vLLM parameters the broadcast never assigned (stale
  weights); it does not compare bytes — prove refit bytes by hashing the same tensor on both sides
  after `load_weights`, or by the logprob comparison itself.
- `load_weights` copies in place (`param.data.copy_`), so graph-captured addresses stay valid.
- Under `--actor.lora_dim > 0` the refit merges `W + scale·B·A` in fp32 while the training forward
  runs the unmerged `W·x + scale·B(A·x)` — one rounding apart by construction, so LoRA is off the
  bitwise contract.

### Trainer scorer

- Place: `molt/models/utils.py::log_probs_from_logits`: bf16 logits upcast per 256-row chunk to
  fp32, divided by `--rollout.temperature` in fp32, torch `gather − logsumexp`.
- The scorer dispatches on the logits dtype: with `--actor.entropy_coef` ≠ 0 (no CP, no
  score-centering top-k targets, flash-attn importable) the actor passes fp32 logits and the Triton
  cross-entropy kernel runs instead — two scorer kernels to inventory. TP variant
  `log_probs_from_vocab_parallel_logits` in `molt/trainer/fsdp/packing.py`, whose all-reduce order
  is part of the contract.
- A distinct kernel from vLLM's sampler: vLLM 0.31 (Model Runner V2) computes the sampler logprob
  with a row max + logsumexp Triton kernel (`compute_token_logprobs`) on the sampler's logits — bf16
  on the stock path, an fp32 copy whenever a logits processor is active, which molt's default
  `min_tokens=1` guarantees while a stop token exists ([engine scorer
  path](debugging.md#isolate-the-side-first)) — not `log_softmax`, and batch-invariant mode's
  `aten::_log_softmax` override does not reach it — inventory both.

### Rollout logprobs

- Requested only when `--algo.advantage.is_correction_level` ≠ `off` or score centering is on: then
  `logprobs=1` (IS correction) or `logprobs=--algo.advantage.score_centering_top_k` (score
  centering), with `logprobs_mode=processed_logprobs`; otherwise vLLM returns no logprobs and no
  mismatch metric exists.
- To get them: `--algo.advantage.is_correction_level token --algo.advantage.is_correction_threshold
  0 1e9` — the gate never masks, but the loss is still multiplied by the per-token ratio π_old/μ,
  which is exactly 1 only when `bitwise_frac` = 0; under default kernels it changes the BI=0
  gradient, so both arms run the same correction settings and the report declares the confounder
  ([arms](acceptance.md#one-recipe-one-switch)).
- Temperature applied in vLLM; `--rollout.top_p 1.0` (default), no top-k
  ([object of comparison](contracts.md#object-of-comparison)).

### Mismatch metrics

- Place: `mismatch_metrics` in `molt/models/utils.py`, logged by `policy_actor.py` from the
  grad-enabled training forward whenever rollout logprobs exist: `mismatch/bitwise_frac` (fraction
  of valid tokens whose fp32 logprob bytes differ, `view(int32)`), `mismatch/abs_diff_mean`,
  `mismatch/k3_kl`. `vllm_kl` in `molt/models/loss.py` is the older signed masked mean of
  `log μ − log π_old`.
- The pass signal is `mismatch/bitwise_frac` = 0 at every step; the replay comparison in
  debugging.md adds the count, max and first failing position.
- `vllm_kl` cannot tell a last-bit difference from 0 (positive and negative differences cancel) and
  is diagnostic only.
- Absent is not zero: `mismatch/*` missing from the log means rollout logprobs were not requested.
  `policy_kl` is 0 by construction under `--train.force_on_policy` with `--algo.kl.init_coef 0`
  (then the old logprob is the detached training forward) and is not evidence. Check the producer
  before trusting any 0.

### Dumps and replay

- `MOLT_DUMP_ROLLOUT_LOGPROBS`: one sequence, rank 0, once per process, 6 decimals — triage for a
  position shift or a gross gap, never byte evidence.
- `--train.rollout_dump_dir` saves `rollout_step{N}.pt` (lazy `Experience`s — heavy tensors stay in
  the runners' object store; materialize before saving for cross-job replay,
  [debugging.md](debugging.md#save-and-compare-protocol)); `--train.rollout_replay_dir` trains on
  saved rollouts without regenerating (a full training step: old-logprob forward, update, refit); a
  dump is bound to its weight version by step number only, so keep that step's pre-update checkpoint
  beside it.
- Byte evidence = replay plus an int-view comparison over all valid tokens
  ([debugging.md](debugging.md#compare-per-token-logprobs)).

### Synchronous run

- `--train.force_sync_mode --train.async_queue_size 1 --train.max_epochs 1`, no
  `--train.partial_rollout_enable`, `--rollout.vllm_generate_batch_size` equal to
  `--rollout.batch_size` (a larger value turns partial rollout on).
- One optimizer step per rollout batch: `--train.batch_size` (samples per optimizer step) =
  `--rollout.batch_size × --rollout.n_samples_per_prompt`, or `--train.force_on_policy`. With several
  minibatch updates per rollout the microbatches after the first update compare against updated
  weights and legitimately read ≈ 1 — `mismatch/*` is logged per grad-enabled microbatch.
- Anything else is lag ≥ 1 ([layout rules](#layout-rules)).

### Engine environment and worker hook

- Place: `molt/trainer/vllm/vllm_engine.py::_configure_vllm_env` runs inside the Ray actor before
  `AsyncEngineArgs` and exports `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, `RAY_ADDRESS` and, under
  `--train.full_determinism_enable`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`; the aligned mode adds
  `VLLM_BATCH_INVARIANT=1` and the workspace/NCCL pins here.
- `worker_extension_cls="molt.trainer.vllm.vllm_worker_wrap.WorkerWrap"` is imported in every worker
  before CUDA-graph capture: the place for kernel patches that must be identical to the trainer's.
- Standalone engine for C3: export the same env, then
  `vllm.LLM(model, worker_extension_cls="molt.trainer.vllm.vllm_worker_wrap.WorkerWrap",
  seed=<--train.seed + i>, logprobs_mode="processed_logprobs", **<the recipe's --vllm.* flags>)` and
  `SamplingParams(temperature=<--rollout.temperature>, top_p=1, logprobs=1, min_tokens=1)`; any other
  env or flag set measures another engine.

### Engine knobs

- `--vllm.attention_backend TRITON_ATTN`: the only backend the aligned attention kernel patches;
  auto-selection picks FA3/FlashInfer and aligns nothing, without an error.
- `--vllm.enforce_eager`; `--vllm.enable_prefix_caching` (default off; molt resets the cache after
  every refit; required by [R3](#routing-replay-r3) on vLLM 0.31); `--vllm.enable_chunked_prefill`
  (a Mamba hybrid with prefix caching cannot turn it off, [hybrid chain](#not-covered));
  `--vllm.mamba_ssm_cache_dtype`; `--vllm.block_size` (KV page; vLLM's default is 16).
- `--vllm.mtp_num_speculative_tokens 0`: MTP on puts the draft in both arms and needs spec-verify
  logprob == full forward; molt's weight-sync does not refresh the draft block (it shares only
  embed/lm_head with the target), so acceptance drifts across updates.
- GDN/Mamba-hybrid recipes run eager because graph capture fails at init; prefer fixing capture
  over paying eager. Never claim a backend by bypassing vLLM's batch-invariant support check in the
  backend selector: align the backend first, then let the selector declare it.

### Routing replay (R3)

- `--train.routing_replay` → vLLM `enable_return_routed_experts`; the trainer replays the engine's
  expert ids. Under the switch R3 is on for MoE (startup guard) and the report lists the
  disagreement rate it masks ([reference measurements](#reference-measurements)); it may be switched
  off only after C3/C5 show the router bitwise without it (a batch-invariant fp32-output gate GEMM
  and a fixed tie-break — the vLLM blog post's DeepSeek-V4 recipe).
- Incompatible with partial rollout (preemption drops the routing) and speculative decoding; tokens
  the engine did not generate (tool outputs) have no record and fall back to natural routing (molt
  fills them with a −1 sentinel that keeps the live selection).
- vLLM indexes the returned ids by global decoder-layer id (non-MoE layers are zero rows); molt's
  gates replay their own global-id row, parsed from the module path `layers.N`, and raise when the
  capture has fewer layer rows than the highest MoE gate's global id needs. The trainer has no
  pipeline parallelism, so no stage offset applies.
- On vLLM 0.31 the routed ids come back through the AuxOutput store keyed by prefix-cache block
  hashes, so prefix caching must be on (the engine refuses otherwise; the model must be MoE and run
  Model Runner V2, vLLM's CUDA default) and C3 must verify cache hit == miss.

### Seed determinism

- `--train.full_determinism_enable` calls `transformers.enable_full_determinism(seed)` —
  `torch.use_deterministic_algorithms(True, warn_only=False)` (an op without a deterministic
  implementation raises), cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG=:16:8`,
  `CUDA_LAUNCH_BLOCKING=1`, `FLASH_ATTENTION_DETERMINISTIC=1` — and runs the engine single-process
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`). It is main-thread only
  ([contracts.md](contracts.md#determinism-only-on-the-main-thread)) and does not make kernels
  batch-invariant.

## The aligned mode

One switch, `--train.batch_invariant`; with the switch off the default path must stay byte-identical.
The reference takes vLLM's generic batch-invariant mode (`VLLM_BATCH_INVARIANT=1`: Triton matmul,
segmented attention) and recovers speed in C10; the recipes in the vLLM blog post instead make the
deployment kernels themselves invariant (DeepGEMM batch-invariant mode, fixed split-KV,
single-tactic grouped GEMM). Both satisfy the contracts; the choice is a C1/C10 cost decision per
model.

Process environment, in the trainer and in every engine worker (they share the weight-sync NCCL
group, so NCCL settings must match): `VLLM_BATCH_INVARIANT=1`, vLLM's `override_envs_for_invariance()`
(NCCL algo/proto/channel pins), `CUBLAS_WORKSPACE_CONFIG=:16:8` and `CUBLASLT_WORKSPACE_SIZE=1`
set before the first cuBLAS handle; whitelisted into the Ray runtime env.

Engine: `CompilationConfig(mode=NONE, cudagraph_mode=FULL)` (no Inductor fusion, graphs allowed),
`TRITON_ATTN` with the segmented kernel, prefix caching off or verified bitwise on hit and miss,
`top_p=1`, logprobs requested, one seed per engine.
Worker import hook: the same kernel patches the trainer applies (matmul tile table, RMSNorm kernel,
MoE tile pin), before graph capture.

Trainer: after distributed init, `setup_process()` (same env,
`torch.backends.cuda.matmul.fp32_precision="ieee"`), then patch the model class forwards. The patch
is process-wide (class forwards and every `nn.Linear`, lm_head and vision towers included): build
any unpatched control model before patching, or an A/B reads 0 because both sides were aligned.
Every swapped operator is a `torch.autograd.Function` ([wiring](training-side.md#backward-reuse-glue-or-new));
gradients are "the same function up to rounding", and activation checkpointing recomputes bitwise
because the forward is deterministic.

## Operator table

H100, vLLM 0.28–0.31; re-verify every row against the pinned vLLM before relying on it.

| Operator | vLLM kernel under `VLLM_BATCH_INVARIANT=1` | Trainer forward | Backward |
|---|---|---|---|
| Linear, incl. q/k/v/o, gate/up/down, lm_head | `linear_batch_invariant` → Triton persistent matmul; tile M is independent of the K reduction, so small-M tile tuning keeps bytes | Same kernel on the gathered FSDP2 parameter | `g @ W`, `gᵀ @ x` |
| RMSNorm / residual-add + RMSNorm | `torch.ops._C.rms_norm`, `fused_add_rms_norm` (the fused op returns the new residual; keep the `(hidden, residual)` pair); the kernel patch pins the block size the stock kernel switches at 256 tokens ([norm contract](contracts.md#per-operator-contract)) | Same ops | fp32 reference |
| SiLU · up | `torch.ops._C.silu_and_mul` on `cat(gate, up)` | Same op; `cat(gate_proj(x), up_proj(x))` is bitwise the merged GEMM | fp32 analytic |
| RoPE (neox) | `torch.ops._C.rotary_embedding`, bf16 cos/sin built on GPU | Same op | Same op with `(cos, −sin)` |
| Attention | Triton unified attention over absolute KV segments (fixed keys per segment), so decode and chunked prefill reduce in the same order; segment constants shared by both sides, accepted on end-to-end A/B only; fp32 partial buffers sized by a fixed context ceiling, never freed (widening the ceiling changes numerics) | Paged KV + a block table per packed sequence; causal GQA only | flash-attn varlen recompute |
| MoE router + experts | `fused_topk` + `fused_experts` with a pinned tile config; expert ids from routing replay; vLLM 0.31 gate GEMM and grouped top-k: [below](#moe-router-and-experts-on-vllm-031) | Same kernels on strided views of the AutoModel `[E, …]` weights; replay overrides the ids, weights re-gathered from the fp32 softmax | `grouped_mm` reference |
| Logits / logprobs | fp32 processed logprobs; on vLLM 0.31 a row-logsumexp Triton kernel (`compute_token_logprobs`) over the sampler's logits — bf16 stock, an fp32 copy under any logits processor ([trainer scorer](#trainer-scorer)) — outside batch-invariant mode's overrides | fp32 `gather − logsumexp` in 256-row chunks — a different kernel: call vLLM's or prove equal bytes | Native autograd |

### MoE router and experts on vLLM 0.31

- `GateLinear` picks the gate GEMM by M (a specialised small-M kernel at M ≤ 16, cuBLAS above) —
  pin one GEMM path on both sides.
- Sigmoid grouped top-k runs the `ops.grouped_topk` CUDA kernel, whose bytes torch sigmoid +
  renormalisation cannot reproduce — call the kernel.

### Not covered

Not covered by upstream batch-invariant mode and not in the reference implementation: Mamba2/SSM
layers (chain below), sliding window, logit softcap, attention sinks, fp8 KV, multimodal encoders,
mRoPE, multi-turn/tool runs under routing replay ([R3](#routing-replay-r3)). Each needs its own row
in the C0 inventory with a verdict before the model is declared supportable.

Mamba hybrid chain: R3 ⇒ prefix caching on (vLLM 0.31) ⇒ the hybrid keeps chunked prefill on (mamba
cache mode `align`) ⇒ align mode materializes state at the attention block = Mamba page (2096 tokens
for the 30B config; model dependent) ⇒ prefill splits there, not at a 128-aligned boundary ⇒ the
chunked SSD is split-invariant only at 128-aligned splits ⇒ keep prompts below the block or record
the split points. Decode and prefill stay two rounding paths until decode replays the chunked
arithmetic with an fp32 state (the Nemotron recipe, upstream PR vllm-project/vllm#55905) or the
trainer replays both phases ([state-layer contract](contracts.md#state-layer-contract)).

## Layout rules

- Packing: THD `cu_seqlens` on the AutoModel path, per-sequence block tables for attention; batch
  composition must not change bytes (check alone vs packed with neighbours).
- Parallelism: the reduction tree is part of the contract — trainer TP (vocab-shard logprob
  reduction), engine TP (all-reduce order; vLLM's custom all-reduce kernel unless
  `--vllm.disable_custom_all_reduce` sends it through the pinned NCCL), EP (combine order). Verify the
  actual layout; `cp_size == 1` ([verified pairs, CP](training-side.md#layout-length-and-world-size)).
- Only identical weights are bitwise. Under any lag ≥ 1 the rollout came from older weights and sits
  on the bf16 requantization floor whatever the learning rate or kernels — `bitwise_frac` reads ≈ 1
  ([reference measurements](#reference-measurements)); alignment removes kernel noise, not
  staleness. An asynchronous recipe is accepted on quality and performance and reads the lag floor,
  never 0.

## Adding an architecture

1. **Inventory (C0)** from vLLM's model file, not from the HF reference, with the
   [module inventory](training-side.md#module-inventory) columns.
2. **One `autograd.Function` per operator** that produces a value; reuse the existing ones; keep
   vLLM's op order and fusion boundaries (concatenate gate/up before `silu_and_mul`; fused norm
   returns the residual; q/k norm position relative to RoPE).
3. **Patch point**: the model's AutoModel classes, listed in an explicit table (no implicit
   dispatch — a half-patched model aligns by accident on some ops and not others); the HF classes for
   HF-fallback models.
4. **Offline proof**: fixed weights and prompts, dump both sides per layer, first diverging layer;
   every compared tensor byte-equal, not "close".
5. **Live proof**: a [synchronous run](#synchronous-run) with
   [rollout logprobs](#rollout-logprobs) requested at non-binding thresholds:
   `mismatch/bitwise_frac` 0 over all valid tokens of every step; then the asynchronous recipe with
   its correction, expecting the lag floor.
6. **Guards at startup**: `--vllm.attention_backend TRITON_ATTN` forced, `cp_size == 1`, context ≤
   the segment ceiling, MoE ⇒ routing replay on, `top_p == 1`, `--vllm.mtp_num_speculative_tokens 0`;
   fail loudly, never fall back.
7. **Cost**: expect the aligned arm clearly slower at first
   ([reference measurements](#reference-measurements)); remove structural defects in C4 and recover
   the rest in C10 with the profile, not by loosening the contract.

## Reference measurements

Expectations only; every new model, version or layout measures its own. Source `ref` = the reference
branch (H100, vLLM 0.28, Qwen3-8B dense / Qwen3-30B-A3B MoE); `upstream` = the vLLM skill's
Nemotron-H / MiMo runs.

| Quantity | Value | Source | Use |
|---|---|---|---|
| `vllm_kl` of the aligned arm, synchronous | ≈ 1e-7 (a last-ulp residue on a few tokens; `bitwise_frac` was not yet logged, so the reference is not proven bitwise) | ref | Why the pass signal is `bitwise_frac`, not `vllm_kl` |
| Lag-1 floor (identical kernels, rollout one update stale) | binary KL ≈ 1.2–1.8e-4 per sequence, `vllm_kl` ≈ 3–4e-4 | ref | Any asynchronous recipe reads this, not 0 |
| Natural-routing disagreement masked by routing replay | ≈ 50 % of MoE positions | ref | Report it next to every MoE result |
| Attention segment ceiling / trainer KV block | 16k tokens / 16 tokens | ref | Boundary lengths to test |
| Microbenchmark-best segment constants vs end-to-end | lost 8–11 % end to end, twice | ref | Accept constants on end-to-end A/B only |
| Aligned cost before optimisation | rollout ≈ 2× slower; trainer attention ≈ 4×, linear ≈ 2.5× | ref | C4/C10 budget |
| Same-prompt siblings, healthy sampling | ≈ 0.1 % identical answers (≤ 8 tokens); 1–3 % share a ≥ 16-token prefix | upstream | More means seed correlation |
| Gradient difference from a microbatch re-split | ≈ 2e-3 relative | upstream | Gradient tolerance denominator, not a bug |
| Proxy vs full model mismatch magnitude, same defect | 12-layer proxy ≈ 1e-12, full model ≈ 1e-2 | upstream | Small-model magnitudes do not extrapolate |
