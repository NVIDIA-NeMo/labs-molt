# molt integration

The first section describes `main`. Everything from "The aligned mode" on describes the reference
implementation on branch `jianh/batch-invariant` (built against vLLM 0.28; Qwen2/Qwen3 dense and
Qwen3-MoE), which is not on `main` yet and is being upstreamed as its own PR. What each row must
make equal is in [contracts.md](contracts.md); how to build the trainer side is in
[training-side.md](training-side.md).

## Where consistency lives in molt (`main`)

| Concern | Place | Notes |
|---|---|---|
| Trainer kernel choice | `molt/models/base.py`, the `BackendConfig` dict (`attn`, `linear`, `experts`, `rms_norm`, `gate_precision`, `dispatcher`) and `_resolve_custom_backend_attn` | AutoModel custom models; env overrides `MOLT_LINEAR_BACKEND`, `MOLT_MOE_EXPERTS`, `MOLT_RMS_NORM`, `MOLT_GATE_PRECISION`, `MOLT_MOE_DISPATCHER`. Architectures without an AutoModel definition fall back to HF transformers |
| Master weights and dtype | fp32 masters, FSDP2 `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` | `--fsdp.param_dtype` has no fp32. The MoE router computes in fp32 through `gate_precision=float32`; `MOLT_RMS_NORM=torch_fp32` is a compute dtype, norm weights follow `param_dtype` unless the AutoModel class keeps them fp32 |
| Refit (trainer → vLLM) | `molt/trainer/workers/policy_actor.py` broadcast, `molt/trainer/vllm/vllm_worker_wrap.py` | The broadcast sends the fp32 master; vLLM rounds it to bf16 on `load_weights` with the same round-to-nearest cast FSDP2's `param_dtype` applies in the training forward, so for parameters vLLM loads with a plain `copy_` the engine bytes equal the training-forward bytes; derived parameters (Mamba `A = -exp(A_log)`, calibration scales) are not a cast — hash them. `--train.check_weight_update_equal` lists vLLM parameters the broadcast never assigned (stale weights); it does not compare bytes — prove refit bytes by hashing the same tensor on both sides after `load_weights`, or by the logprob comparison itself |
| Trainer scorer | `molt/models/utils.py::log_probs_from_logits`: bf16 logits upcast per 256-row chunk to fp32, divided by `--rollout.temperature` in fp32, torch `gather − logsumexp`; the scorer dispatches on the logits dtype: with `--actor.entropy_coef` ≠ 0 (no CP, no score-centering top-k targets, flash-attn importable) the actor passes fp32 logits and the Triton cross-entropy kernel runs instead — two scorer kernels to inventory; TP variant `log_probs_from_vocab_parallel_logits` in `molt/trainer/fsdp/packing.py`, whose all-reduce order is part of the contract | A distinct kernel from vLLM's sampler — inventory it like any other op |
| Rollout logprobs | Requested only when `--algo.advantage.is_correction_level` ≠ `off` or score centering is on: then `logprobs=1` and `logprobs_mode=processed_logprobs`; otherwise vLLM returns no logprobs and no mismatch metric exists | To get them: `--algo.advantage.is_correction_level token --algo.advantage.is_correction_threshold 0 1e9` — the gate never masks, but the loss is still multiplied by the per-token ratio π_old/μ, which is exactly 1 only when `bitwise_frac` = 0; under default kernels it changes the BI=0 gradient, so declare it as a confounder or use the recipe's own correction in both arms. Temperature applied in vLLM; `top_p` must be 1 and `top_k` off or the candidate set differs from the trainer's full softmax |
| Mismatch metrics | `mismatch_metrics` in `molt/models/utils.py`, logged by `policy_actor.py` from the grad-enabled training forward whenever rollout logprobs exist: `mismatch/bitwise_frac` (fraction of valid tokens whose fp32 logprob bytes differ, `view(int32)`), `mismatch/abs_diff_mean`, `mismatch/k3_kl`. `vllm_kl` in `molt/models/loss.py` is the older signed masked mean of `log μ − log π_old` | The pass signal is `mismatch/bitwise_frac` = 0 at every step; the replay comparison in debugging.md adds the count, max and first failing position. `vllm_kl` cannot tell a last-bit difference from 0 (positive and negative differences cancel) and is diagnostic only; absent is not zero. Under lag ≥ 1 `bitwise_frac` ≈ 1 whatever the kernels: alignment is measured synchronously. `policy_kl` is 0 by construction under `--train.force_on_policy` with `--algo.kl.init_coef 0` (then the old logprob is the detached training forward) and is not evidence |
| Dumps and replay | `MOLT_DUMP_ROLLOUT_LOGPROBS`: one sequence, rank 0, once per process, 6 decimals — triage for a position shift or a gross gap, never byte evidence. `--train.rollout_dump_dir` saves `rollout_step{N}.pt`; `--train.rollout_replay_dir` trains on saved rollouts without regenerating (a full training step: old-logprob forward, update, refit) | Byte evidence = replay plus an int-view comparison over all valid tokens (see debugging.md) |
| Synchronous run | `--train.force_sync_mode --train.async_queue_size 1 --train.max_epochs 1`, no `--train.partial_rollout_enable`, `--rollout.vllm_generate_batch_size` equal to `--rollout.batch_size` (a larger value turns partial rollout on) | Anything else is lag ≥ 1: the rollout came from older weights and `bitwise_frac` reads ≈ 1 for any kernels |
| Engine environment | `molt/trainer/vllm/vllm_engine.py::_configure_vllm_env` runs inside the Ray actor before `AsyncEngineArgs` | The place to export `VLLM_BATCH_INVARIANT=1` and workspace/NCCL pins to the engine |
| Engine worker hook | `worker_extension_cls` → `vllm_worker_wrap.py`, imported in every worker before CUDA-graph capture | The place for kernel patches that must be identical to the trainer's |
| Engine knobs | `--vllm.attention_backend TRITON_ATTN` (the only backend the aligned attention kernel patches; auto-selection picks FA3/FlashInfer and aligns nothing, without an error), `--vllm.enforce_eager`, `--vllm.enable_prefix_caching` (default off; molt resets the cache after every refit; vLLM 0.31 requires it with `--train.routing_replay`, see below), `--vllm.enable_chunked_prefill` (a Mamba hybrid with prefix caching must keep it on: vLLM's mamba cache mode `align` demands it), `--vllm.mamba_ssm_cache_dtype` | GDN/Mamba-hybrid recipes run eager because graph capture fails at init; prefer fixing capture over paying eager |
| Routing replay (R3) | `--train.routing_replay` → vLLM `enable_return_routed_experts`, trainer replays the engine's expert ids | Needed while the router itself is not batch-invariant (the reference's route; report the disagreement rate replay masks, see [reference measurements](#reference-measurements)). With a batch-invariant fp32-output router GEMM and a fixed tie-break, rollout and training route identically under one update per step and replay becomes a diagnostic (the blog's DeepSeek-V4 recipe). Incompatible with partial rollout and speculative decoding; tokens the engine did not generate (tool outputs) have no record and fall back to natural routing. On vLLM 0.31 the routed ids come back through the AuxOutput store keyed by prefix-cache block hashes, so prefix caching must be on (the model must be MoE and the engine must run Model Runner V2, vLLM's CUDA default) and G3 must verify cache hit == miss |
| Seed determinism | `--train.full_determinism_enable` | Seeds, torch deterministic algorithms (`transformers.enable_full_determinism`) and a single-process engine; it does not make kernels batch-invariant |

## The aligned mode

One switch, `--train.batch_invariant`; with the switch off the default path must stay byte-identical.
The reference takes vLLM's generic batch-invariant mode (`VLLM_BATCH_INVARIANT=1`: Triton matmul, segmented
attention) and recovers speed in G8; the blog's recipes instead make the deployment kernels themselves
invariant (DeepGEMM batch-invariant mode, fixed split-KV, single-tactic grouped GEMM). Both satisfy the
contracts; the choice is a G1/G8 cost decision per model.

Process environment, in the trainer and in every engine worker (they share the weight-sync NCCL
group, so NCCL settings must match): `VLLM_BATCH_INVARIANT=1`, vLLM's `override_envs_for_invariance()`
(NCCL algo/proto/channel pins), `CUBLAS_WORKSPACE_CONFIG=:16:8` and `CUBLASLT_WORKSPACE_SIZE=1`
set before the first cuBLAS handle; whitelisted into the Ray runtime env.

Engine: `CompilationConfig(mode=NONE, cudagraph_mode=FULL)` (no Inductor fusion, graphs allowed),
`TRITON_ATTN` with the segmented kernel, prefix caching off or verified bitwise on hit and miss (on
whenever routing replay runs under vLLM 0.31),
`top_p=1`, logprobs requested, one seed per engine. Worker import hook: the same kernel patches the
trainer applies (matmul tile table, RMSNorm kernel, MoE tile pin), before graph capture.

Trainer: after distributed init, `setup_process()` (same env, `torch.backends.cuda.matmul.fp32_precision="ieee"`),
then patch the model class forwards. The patch is process-wide (class forwards and every
`nn.Linear`, lm_head and vision towers included): build any unpatched control model before patching,
or an A/B reads 0 because both sides were aligned. Every swapped operator is a `torch.autograd.Function`:
forward calls the vLLM op and saves inputs; backward recomputes a reference (flash-attn varlen,
`grouped_mm`, fp32 torch) and differentiates it. Gradients are "the same function up to rounding";
activation checkpointing recomputes bitwise because the forward is deterministic.

## Operator table

H100, vLLM 0.28–0.31; re-verify every row against the pinned vLLM before relying on it.

| Operator | vLLM kernel under `VLLM_BATCH_INVARIANT=1` | Trainer forward | Backward |
|---|---|---|---|
| Linear, incl. q/k/v/o, gate/up/down, lm_head | `linear_batch_invariant` → Triton persistent matmul; tile M is independent of the K reduction, so small-M tile tuning keeps bytes | Same kernel on the gathered FSDP2 parameter | `g @ W`, `gᵀ @ x` |
| RMSNorm / residual-add + RMSNorm | `torch.ops._C.rms_norm`, `fused_add_rms_norm` (the fused op returns the new residual; keep the `(hidden, residual)` pair) | Same ops | fp32 reference |
| SiLU · up | `torch.ops._C.silu_and_mul` on `cat(gate, up)` | Same op; `cat(gate_proj(x), up_proj(x))` is bitwise the merged GEMM | fp32 analytic |
| RoPE (neox) | `torch.ops._C.rotary_embedding`, bf16 cos/sin built on GPU | Same op | Same op with `(cos, −sin)` |
| Attention | Triton unified attention with absolute KV segments (fixed keys per segment) so decode and chunked prefill reduce in the same order; the segment constants are shared by both sides and accepted only on end-to-end A/B; fp32 partial buffers sized by a fixed context ceiling (widening it changes numerics) are never freed | Paged KV + a block table per packed sequence; causal GQA only | flash-attn varlen recompute |
| MoE router + experts | `fused_topk` + `fused_experts` with a pinned tile config; expert ids from routing replay | Same kernels on strided views of the AutoModel `[E, …]` weights; replay overrides the ids, weights re-gathered from the fp32 softmax | `grouped_mm` reference |
| Logits / logprobs | fp32 processed logprobs | fp32 `gather − logsumexp` | — |

Not covered by upstream batch-invariant mode and not in the reference implementation: Mamba2/SSM
layers (chunked scan in prefill vs per-token recurrence in decode are two rounding paths; the blog's
Nemotron recipe makes decode replay prefill's chunked arithmetic with an fp32 state), sliding
window, logit softcap, attention sinks, fp8 KV, multimodal encoders, mRoPE, multi-turn/tool runs
under routing replay (see above). Each needs its own row in the G0 inventory with a verdict before
the model is declared supportable.

## Layout rules

- Packing: THD `cu_seqlens` on the AutoModel path, per-sequence block tables for attention; batch
  composition must not change bytes (check alone vs packed with neighbours).
- Parallelism: the reduction tree is part of the contract — trainer TP (vocab-shard logprob
  reduction), engine TP (all-reduce order), EP (combine order). Verify the actual layout (the
  reference verified trainer EP=4 against engine TP=4) and guard `cp_size == 1` until a CP path exists.
- Only identical weights are bitwise. Any lag ≥ 1 puts rollouts on the bf16 requantization floor
  whatever the learning rate ([reference measurements](#reference-measurements)): alignment removes
  kernel noise, not staleness.

## Adding an architecture

1. **Inventory (G0)** from vLLM's model file, not from the HF reference: operator order, fusion
   boundaries, norm variant (`1 + w`?), position encoding (neox / mRoPE / none), router (softmax or
   sigmoid, bias, renormalisation, grouped top-k, tie-break), activation, shared experts, attention
   features (sliding window, sinks, GQA), state layers (Mamba, conv), encoders.
2. **One `autograd.Function` per operator** that produces a value; reuse the existing ones; keep
   vLLM's op order and fusion boundaries (concatenate gate/up before `silu_and_mul`; fused norm
   returns the residual; q/k norm position relative to RoPE).
3. **Patch point**: the model's AutoModel classes, listed in an explicit table (no implicit
   dispatch — a half-patched model aligns by accident on some ops and not others); the HF classes for
   HF-fallback models.
4. **Offline proof**: fixed weights and prompts, dump both sides per layer, first diverging layer;
   every compared tensor byte-equal, not "close".
5. **Live proof**: a synchronous run (table row above) with rollout logprobs requested at non-binding
   thresholds (table row above): `mismatch/bitwise_frac` 0 over all valid tokens of every step; then
   the asynchronous recipe with its correction, expecting the lag floor.
6. **Guards at startup**: `--vllm.attention_backend TRITON_ATTN` forced, `cp_size == 1`, context ≤
   the segment ceiling, MoE ⇒ routing replay on, `top_p == 1`, no speculative decoding; fail loudly,
   never fall back.
7. **Cost**: expect the aligned arm clearly slower at first ([reference measurements](#reference-measurements));
   recover in G8 with the profile, not by loosening the contract.

## Precision facts for the BF16 route

- Both sides read the same bf16 checkpoint bytes. Parameters the trainer computes in fp32 (router
  gate, norms) must be fp32 on the engine too or rounded identically before use.
- FP8 through vLLM online quantization is receiver-owned arithmetic and is not bitwise against bf16
  training unless the trainer calls the same FP8 GEMM; out of scope for the default route.

## Resume

A resume is consistent when the fp32 masters restore bitwise (DCP), the first refit after restore
sends the same bytes as the refit before the save, the first comparison reads 0, and one effective
update follows. molt documents the CPU AdamW of `--fsdp.offload optimizer` as numerically equivalent to
the GPU step but not bit-identical (CPU vs GPU rounding of sqrt and division), so never change `--fsdp.offload`
across a resume.

## Reference measurements

Numbers from the reference implementation (H100, vLLM 0.28, Qwen3-8B dense and Qwen3-30B-A3B MoE).
They set expectations only; every new model, version or layout measures its own.

| Quantity | Reference value | Use |
|---|---|---|
| `vllm_kl` of the aligned arm, synchronous | ≈ 1e-7 (a last-ulp residue on a few tokens; `bitwise_frac` was not yet logged, so the reference is not proven bitwise) | Why the pass signal is `bitwise_frac`, not `vllm_kl` |
| Lag-1 floor (identical kernels, rollout one update stale) | binary KL ≈ 1.2–1.8e-4 per sequence, `vllm_kl` ≈ 3–4e-4 | Any asynchronous recipe reads this, not 0 |
| Natural-routing disagreement masked by routing replay | ≈ 50 % of MoE positions | Report it next to every MoE result |
| Attention segment ceiling / trainer KV block | 16k tokens / 16 tokens | Boundary lengths to test |
| Microbenchmark-best segment constants vs end-to-end | lost 8–11 % end to end, twice | Accept constants on end-to-end A/B only |
| Aligned cost before optimisation | rollout ≈ 2× slower; trainer attention ≈ 4×, linear ≈ 2.5× | G8 budget |
| Same-prompt siblings, healthy sampling | ≈ 0.1 % identical answers (≤ 8 tokens); 1–3 % share a ≥ 16-token prefix | More means seed correlation |
| Tokens the engine did not generate (tool outputs) under replay | ≈ 4 % of tokens, logprob offsets O(0.1) | Scope MoE claims to single turn |
| Gradient difference from a microbatch re-split | ≈ 2e-3 relative | Gradient tolerance denominator, not a bug |
| Proxy vs full model mismatch magnitude, same defect | 12-layer proxy ≈ 1e-12, full model ≈ 1e-2 | Small-model magnitudes do not extrapolate |
