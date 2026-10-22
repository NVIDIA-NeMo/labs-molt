# Debugging a mismatch

Find the earliest divergence, not the largest; isolate the side before bisecting; classify before touching
precision. One variable per experiment; a path that did not execute is `inconclusive`
([workflow.md](workflow.md#evidence)).

## Triage

1. Is `--train.batch_invariant` on, and does this checkout have the aligned mode at all? `main` does not —
   the task there is a C0 port, not a fix ([SKILL.md](SKILL.md)); a non-zero mismatch is then the expected
   baseline.
2. Is every operator of the model supported? Mamba2/SSM, sliding window, sinks, mRoPE, fp8 KV, encoders and
   multi-turn under R3 are not ([not covered](molt-integration.md#not-covered)).
3. One optimizer step per rollout and the synchronous flags ([synchronous
   run](molt-integration.md#synchronous-run)).
4. `mismatch/*` present in the step log — molt requests rollout logprobs on every path, so absent means
   they never reached the trainer ([rollout logprobs](molt-integration.md#rollout-logprobs)).

Only then the controls below.

## Symptom → first step

| Symptom | Do first |
|---|---|
| Rollout and old-logprob differ at θ0 | [Isolate the side](#isolate-the-side-first): a side whose A/A and alone-vs-batched both pass is stable; the other side moved |
| One stable failing sample | Same-prefix sibling (μ differing over the shared prefix puts it on the engine side); save that rollout, replay it alone and with its original neighbours, then bisect by layer |
| Gap grows with training, θ0 fine | [Cheapest control](#isolate-the-side-first): score one saved batch repeatedly per side for its noise floor, then early vs late weights |
| `bitwise_frac` = k/n with k microbatches after the first update | More than one optimizer step per rollout: `--train.batch_size` below `--rollout.batch_size × --rollout.n_samples_per_prompt`, or `--train.max_epochs` > 1 ([synchronous run](molt-integration.md#synchronous-run)) |
| Identical operator inputs still differ | Kernel or arguments differ: executed kernel name, accumulation dtype, splits, tile, epilogue, fusion boundary, quantizer dispatch (compiled native vs CUDA op); [operator table](molt-integration.md#operator-table) |
| Constant logprob offset on every token | An fp32-kept parameter downcast by the FSDP `param_dtype` policy, or temperature divided in bf16 ([precision chain](contracts.md#weights-and-precision-chain)) |
| Difference survives a precision change | It was never quantization: back to the semantic / discrete / weights classes ([order](#classification-and-handling-order)) |
| Grad norm NaN/Inf while every logprob is finite | The per-token IS ratio overflowed: a long multi-turn rollout under mismatch or lag with unbounded `--algo.advantage.is_correction_*`; read the max of `logprobs_diff` (not its mean) and the ratio bounds before blaming a kernel |
| Only under graph / compile / cache | [Execution state](#execution-state) |
| Fails after changing packing / CP / TP / EP | Trainer packing A/C and the [layout rules](molt-integration.md#layout-rules); a C failure may be a page, tile, segment or expert boundary rather than the packing code |
| Many identical samples in a group | [Sampling seed collision](#sampling-seed-collision): a sampling defect the mismatch metric cannot see |
| Differs only after resume | [Resume](acceptance.md#checkpoint-resume-and-memory): masters restore bitwise, first refit bytes equal the pre-save refit, derived state reset |
| Multi-node will not start, NCCL errors | [Communication and transport](#communication-and-transport): world size per arm, transport per group, `NCCL_DEBUG` once per arm |
| Steps stopped, no error | [Stalls](#stalls-and-cuda-async-errors): worker and engine logs, checkpoint mtimes, the last step's `timing/*`; slow is not a hang |

## Isolate the side first

μ ≠ π (rollout vs trainer logprob) does not say which side moved. Same weights, same tokens, four controls:

| Control | How | Tells you |
|---|---|---|
| Engine A/A | Same request twice, separate processes when in doubt | Run-to-run nondeterminism |
| Engine alone vs neighbours and position | One request vs inside a full batch, and moved to another slot | Batch invariance |
| Engine prefill / chunked / decode | Score one history by full prefill (`SamplingParams(prompt_logprobs=0, max_tokens=1)` in a standalone script), chunked prefill and token-by-token decode | Prefill/decode invariance (attention splits, state-layer chunk grid) |
| Trainer A/A and packing A/C | Recompute twice; alone vs packed with others, permuted | Trainer determinism, packing invariance |

- **Prompt-logprob scoring.** `prompt_logprobs=0` means no extra top-k, not off; the first response token is
  predicted at the last prompt position — never shift. A `prompt_logprobs` request may bypass the prefix
  cache and omit cached positions on a hit ([pinned
  facts](molt-integration.md#facts-bound-to-the-pinned-vllm)): check that every response position got a
  score and reject missing ones.
- **Verdict.** Trainer A/C fails, engine stable → trainer layout first. Both stable alone → weights,
  operator semantics, scorer (run both scorers on one identical logits tensor). Same-prefix siblings
  (`n_samples_per_prompt` > 1) agree over the shared prefix only; μ equal, π differ → trainer layout/state.
- **Per-sequence pattern** (valid/mismatching tokens, first failing position, longest run, lengths, group,
  packing slot, rank). Check shift, mask and EOS first; a ±1/±2 offset is a comparison bug.
  - from token 0 → prefill, weights, scorer or a shift (fix that token first with a prompt-only recompute);
  - after a boundary (KV segment, chunk, page, packing edge) → state layer or split policy;
  - growing → propagation, only the first position is independent;
  - sparse spikes → discrete flip or a last bit across a cast;
  - all drift → weights (lag) or scorer dtype; only after an update → refit.
- **Cheapest control.** Save a small batch of real prompt+response ids, prompt lengths and mask; only score
  from then on (never resample or re-tokenize); score one version repeatedly per side for its noise floor,
  then early vs late weights when the gap grows with training (lag ≥ 1: [layout
  rules](molt-integration.md#layout-rules)).
- **Measurement basis.** Per valid token, the sampled token's fp32 logprob as bytes ([object of
  comparison](contracts.md#object-of-comparison)) — never a batch mean, `vllm_kl` or the 6-decimal dump.
  Like with like (`processed_logprobs` vs the scorer's fp32 temperature divide). k3 on an old trajectory is
  a diagnostic, not the policy KL; relative L2 / cosine are for hidden tensors only.
- **Engine scorer path.** A logits processor in a sampling batch (molt's default `min_tokens=1` with a stop
  token) can move the engine's sampler onto another logits dtype, and the processed logprob comes from that
  path ([pinned facts](molt-integration.md#facts-bound-to-the-pinned-vllm)): inventory it next to the two
  trainer scorers. k3 overflows on a large gap — count non-finite, never clip.
- **Default-kernel A/A fails by design.** Request arrival changes each step's token count and composition,
  and the default kernels are not step-invariant (the stock norm kernel's block size, cuBLAS and MoE tiles
  follow M; the attention backend may take another split plan in decode-only steps — [pinned
  facts](molt-integration.md#facts-bound-to-the-pinned-vllm)); Triton autotune (the Mamba SSD kernels among
  others) adds a per-process factor. A failing rollout-path A/A on default kernels is the BI=0 baseline, not
  a lead — isolate on fixed tokens.
- **Validity of controls.** Cache off may run a prefill that never ran; eager changes scheduling: count
  executions of the target branch, one round proves nothing, a switch that also changes scheduling is not
  attributable.

## Save and compare protocol

Archive by step; a conclusion that exists only in chat is an evidence gap ([failure
records](evidence.md#failure-records)).

| Item | Where |
|---|---|
| Index | `steps/INDEX.md`: the current candidate, the failed / excluded and the unverified items, the human-frozen BI=0 baseline id apart from candidates; every comparison cites that id |
| Run | `steps/<NN>-<gate>/<run-id>/{config,scripts,results,summary.md}`; new run = new id; invalidated conclusions are marked, not deleted |
| Per run | launch command, resolved args, image digest, molt / vLLM / nemo-automodel commits plus dirty patch, evidence of the executed backend (a flag is not execution), exit code |
| Large tensors | `results/<run-id>/`, not in git |

Replayable raw data: `--train.rollout_dump_dir` writes `rollout_step{N}.pt` with the samples materialized
(tokens, masks, rollout logprobs, routed ids as tensors — the lazy `Experience` list is reloaded before the
save), so the file replays in another job and reads back offline; a dump from an older molt that saved Ray
`heavy_ref` handles does not. Add the weight version (step N's rollout comes from the policy before update N; the
post-update checkpoint does not reproduce it), sampling params, resolved engine config, packing map,
parallel layout. Under partial rollout or async lag, segments of one sequence come from different policies
and molt stores no per-segment version: dump with partial rollout off and `--train.force_sync_mode`. Verify
the save fired, counts match the log and the file reads back.

### Compare per-token logprobs

Each side exports `{request_id: (logprobs, response_mask)}` as original-dtype tensors via `torch.save` — no
JSON, no printed logs. Rollout side: the materialized dump above; trainer side: save `action_log_probs` and
`action_mask` next to the `mismatch_metrics` call in `policy_actor.py`, keyed by the same request ids. Align
by request id, not row; request sets must match (error, not intersection); masks equal; ≥ 1 valid token per
sequence; hash the input tokens first. Positions after the first mismatch are not independent for
history-dependent layers
([contracts.md](contracts.md#history-dependent-operators-conv-scan-attention-over-a-cache)): count them,
localize on the first. Pass = `bitwise_count` 0 (the offline count; the live key is `mismatch/bitwise_frac`)
with `valid_tokens` > 0 and `nonfinite` 0; equal sampled-token logprobs do not prove equal logits.

```python
def compare_logprobs(a, b):  # per side {uid: (logprobs, mask)}, original dtype, same uid set
    assert a.keys() == b.keys(), "request sets differ"
    n = bad = nonfinite = 0; first = None; max_abs = sum_abs = 0.0
    for uid in sorted(a):
        (x, mx), (y, my) = a[uid], b[uid]
        assert x.dtype == y.dtype and torch.equal(mx, my) and mx.any(), f"{uid}: dtype, mask or no valid token"
        x, y = x[mx], y[my]
        itype = torch.int32 if x.dtype == torch.float32 else torch.int16
        diff = (x.view(itype) != y.view(itype)).nonzero()
        if len(diff) and first is None: first = (uid, int(diff[0]))
        d = (x.double() - y.double()).abs(); max_abs = max(max_abs, d.max().item()); sum_abs += d.sum().item()
        n += len(x); bad += len(diff); nonfinite += int((~torch.isfinite(x) | ~torch.isfinite(y)).sum())
    return {"dtype": str(x.dtype), "valid_tokens": n, "bitwise_count": bad, "nonfinite": nonfinite,
            "abs_diff_max": max_abs, "abs_diff_mean": sum_abs / n, "first_fail": first}
```

- **Per-layer trace.** Same semantic boundaries on both sides under the same packing (embedding, each
  layer's attention/scan output, router ids, residual, final norm, logits), keyed `(request_id, boundary,
  logical slice)` after stitching TP/EP shards by global coordinates and stripping padding. Hash first —
  `hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()` plus shape, dtype, execution
  index — after `clone()` (engine workspaces and graph output buffers are overwritten by the next call); raw
  tensors only at the first differing boundary, which is not the cause until its input hash is shown equal.
  For the final logits add full-vocab log-softmax, probability total variation and argmax as separate rows;
  keep one clean sibling as the negative control.
- **Replay controls.** One process per variant through a synchronous entry (no fire-and-forget `sbatch`), a
  new directory per variant with command, env, exit code, time; preview command and budget first and stop at
  the failing variant. Prove frozen: weight fingerprint before and after, no sampling, no optimizer step,
  the packing actually in effect. Variants touching compile, libraries, autotune or init env start a fresh
  process.

## Bisect and first divergence

- **Search axis**, one at a time:
  - version: good/bad on the same reproducer; save the full dirty patch, a SHA is not a program; a build
    failure or a non-reproduction is `inconclusive`, not a bad revision;
  - config: paired single factor, 2×2 when coupled;
  - data trigger: subset, length, neighbours, boundary remainder;
  - compute path: layer → attention/MoE/norm → operator → metadata, launch order.
- **Single-request prefill first.** Fixed-input scoring, same weights both ends, repeated for the noise
  floor; then a minimal forward — one request, one prefill, one output token, no cache hit,
  warmup/draft/decode calls told apart (eager does not stand for graph); earliest divergence at embedding,
  block outputs, logits at the last valid prompt position; same-input operator replay; then back to the
  original failing path at full depth, decode and graph
  ([ladder](workflow.md#cheapest-failure-criterion-first)).
- **Dump order inside the suspect layer:** norm in → norm out → q/k/v projections (combine a fused QKV
  before comparing with separate q/k/v) → position encoding (q/k after RoPE, q/k norm in vLLM's order) →
  attention core / SSM scan / conv → router logits, ids, weights → expert outputs → combine → output
  projection → residual add (the fused add+norm returns the new residual). Core and output projection get
  separate boundaries.
- **Align probes by semantics:** tensor meaning, dtype, global token coordinate, valid mask — not module
  names or tuple indices (FSDP wrapper names, `(hidden, residual)` pairs, THD vs padded, LSE ln vs log2,
  before/after norm or residual). EP dispatch reorders, duplicates and pads tokens: align expert tensors by
  (token, expert, slot), then compare reduction, shared+routed combine and post-compile rounding separately.
- **Credible observations:** a hook never replaces an output; under FSDP sample boundaries already
  materialized in this forward; count forward vs activation-recompute vs graph-replay so the recompute does
  not overwrite the forward; refuse comparisons with missing boundaries or incompatible shapes; non-finite
  fails explicitly.
- **Recommended log line** (reuse existing logs; terminal prints the summary only; unknown stays unknown):
  `[mismatch] run= step= weights=<digest> input=<sha> phase=<prefill|decode> rank= repeat=
  scope=<fixed|live> intervention=<none|name> metric= value= self_repeat=<noise> report=<path>
  class=<semantic|…|unknown>`.
- **Limits of bisect.** A difference can vanish downstream in rounding, masking or routing, so "layer k
  differs" is not monotonic in depth: coarse scan → suspect interval → fine scan → re-verify the boundaries
  before it. Stop at a discrete decision (top-k, expert id, tie-break): fix the routing, then continue. A
  cut point carries full state (residual pair, KV and block table, SSM/conv state, positions, routing ids,
  `cu_seqlens`). Only a same-input A/B separates propagation from a local error; injected oracle
  activations, forced routing or a swapped kernel prove causality but are diagnostics — accept the fix on
  the restored path.
- **Reviewable root cause:** minimal input, last-equal/first-unequal boundary, same-input control,
  mechanism, trigger, minimal change; a config difference is a lead, not a cause.
- **Stop rules:** three rounds of only "final score differs" → add probes or re-check the sample mapping, do
  not rerun; stop when this round's goal is met, no config changes along the way; output replacement is
  diagnosis, not a fix.

## Classification and handling order

| Class | Signature | Action |
|---|---|---|
| Semantic | One side lacks or applies a feature (sinks, sliding window, q/k-norm position, `1 + w` norm, RoPE variant) | Port the feature; no numerical work until both sides agree |
| Discrete | Same scores, different selection, order or tie-break | Fix the ordering contract; replay routing (R3) if the engine's choice must be reproduced |
| Reduction order | Same inputs, last bits differ, grows with batch or length | Make the kernel invariant (fixed splits/tiles) or call the same kernel from the trainer |
| Dtype / cast position | Appears at a cast or fp32 island | Align accumulation dtype and where the cast happens |
| State | Only after a chunk boundary, cache hit or decode step | Chunk grid, boundary-state dtype, cache reset, prefill/decode path |
| Execution state | Only under graph / compile / specific shape | Compile boundary, capture shape, autotune: pin the kernel on both sides |
| Weights | Bytes differ between trainer and engine | Refit dtype, fp32-kept params, requantization; same-tensor hash after `load_weights` ([refit](molt-integration.md#refit)) |

Order: semantic → discrete → weights → reduction/dtype → execution state; do not move on until the previous
class is ruled out; `unknown` is a valid label, "just rounding" needs same-input evidence. Delivery:
re-verify on the original reproducer at full depth, decode and graph, counts before and after, production
path restored; only then shrink. Write `side → first operator → cause → fix → re-verified on the original
reproducer` with counts before and after; keep failing samples and ruled-out hypotheses for the next round;
a fix that changes the sampled trajectory follows
[evidence.md](evidence.md#limited-exceptions-and-fallback).

## Execution state

Each state is its own closure row ([evidence.md](evidence.md#execution-state)).

- **Graph vs eager.** Hooks run at capture, not at replay ([graph
  mode](contracts.md#observation-in-graph-mode)); replay reads captured buffers (refit copies weights in
  place): dump the engine in eager and prove graph == eager bytes separately (C3). Piecewise graphs fall
  back to eager above the capture size — each path invariant, the two not equal — hence
  `CompilationConfig(mode=NONE, cudagraph_mode=FULL)` in aligned mode; `--vllm.enforce_eager` proves eager
  only. Graph-only faults: snapshot the existing output buffers after replay; never recapture or clear the
  graph to make a comparison pass.
- **Compile boundaries.** Inductor fusions round differently; a same-named module may run a compiled native
  forward on one side and a CUDA op on the other — check the executed kernel (profiler, nsys), not the
  Python name.
- **Autotune, determinism.** Autotune is per process and keyed by the first shapes seen (Triton matmul,
  `fused_moe`): pin configs on both sides, rerun a possibly re-tuned variant in a fresh process. Determinism
  switches are scoped and thread-local ([main-thread
  determinism](contracts.md#determinism-only-on-the-main-thread)).
- **Cache state.** Prefix hit == miss == off, bitwise; molt resets the cache after a refit only when caching
  is on; every cache or fingerprint binds weights, input and config.
- **Parallel layout.** Bitwise holds for the verified layout only — record it per row ([layout
  rules](molt-integration.md#layout-rules)).

## Rollout engine pitfalls

[Sampling seed collision](#sampling-seed-collision) · [Prefix cache semantics](#prefix-cache-semantics) ·
[Chunked prefill](#chunked-prefill) · [Speculative decoding / MTP](#speculative-decoding--mtp) · [Routing
replay (R3)](#routing-replay-r3) · [Lifecycle, memory, metrics](#lifecycle-memory-metrics) · [Replay
matrix](#replay-matrix).

### Sampling seed collision

Prove distinct samples per engine and per DP rank on the pinned version before using DP > 1, and seed any
engine you add yourself. Identically seeded engines sample identical tokens for their n-th request; molt
seeds `seed + i` per engine; in-engine DP ranks (`--vllm.data_parallel_size`) may share one seed ([pinned
facts](molt-integration.md#facts-bound-to-the-pinned-vllm)). Accept on the dump: distinct samples per group
and identical sibling pairs at the healthy rate ([reference
measurements](molt-integration.md#reference-measurements)); more is RNG correlation (zero-advantage groups
come from task difficulty). Greedy n repeats are identical by construction: a greedy A/B cannot see a
sampling fix. Never give one prompt's siblings the same seed.

### Prefix cache semantics

Premise: one request is bitwise identical on hit, miss and cache off.

- Pure attention gets this from KV block reuse; state layers (Mamba2, gated delta net) depend on which
  states the engine copies: vLLM's Mamba cache mode `align` copies only the conv and SSM state, so any extra
  state a decode kernel keeps must be in the layer's declared state list or hit ≠ miss; where `align` splits
  the prefill: [Mamba hybrid](molt-integration.md#mamba-hybrid).
- molt: off by default, on under R3; when on, `reset_prefix_cache(reset_running_requests=True,
  reset_connector=True)` after every refit (an engine that ignores them returns False and molt warns), so
  a request straddling the refit also loses old-policy KV; keep partial rollout off anyway so no rollout
  spans two policies.
- A hit changes the step's token count and may change the graph bucket. Diagnose with cache off; hit,
  preemption and recompute paths get their own gate, on every engine and DP rank and including chunks
  truncated by the token budget; two fixes passing alone may fail stacked.

### Chunked prefill

- On by default in vLLM; read the resolved token budget (`max_num_batched_tokens`) in the engine log; a
  hybrid may be unable to turn it off ([Mamba hybrid](molt-integration.md#mamba-hybrid), [pinned
  facts](molt-integration.md#facts-bound-to-the-pinned-vllm)).
- Segmented attention is chunk-invariant; state layers hand state across chunks in another rounding path:
  `--vllm.max_num_batched_tokens` ≥ `max_model_len` keeps a prompt in one chunk unless the step is already
  full, `--vllm.mamba_ssm_cache_dtype float32`.
- Test the chunk-budget edge (2047/2048/2049 for a 2048-token `--vllm.max_num_batched_tokens`), the segment
  edges (1023/1024/1025 … 16383/16384/16385 at the reference implementation's 16k ceiling) and, on a Mamba
  hybrid, the page edges (page − 1 / page / page + 1 at the Mamba page the engine log reports).

### Speculative decoding / MTP

- `--vllm.mtp_num_speculative_tokens` > 0: the verification logprob must be bitwise with the full forward —
  unproven, so off for bitwise. Both arms or neither; molt rejects it with R3 and with prefix caching. A
  no-MTP probe is a control and covers nothing of the MTP path.
- Weight-sync does not refresh the draft ([engine knobs](molt-integration.md#engine-knobs)); the verify path
  is a second numerical path to inventory.

### Routing replay (R3)

- Ids per token per MoE layer; 0 is a valid id — check range and distinctness, not non-zero (an engine MoE
  backend that bypasses the top-k capture returns all zeros).
- Proving the returned ids are the engine's: compare against a strictly serial single-request run of the
  same engine after measuring baseline-vs-baseline drift (default routing is not batch-invariant);
  concurrent greedy dumps prove nothing. Per-step return costs frontend time — profile before blaming
  kernels.
- Incompatibilities, tokens without a record and the engine's prefix-caching requirement:
  [R3](molt-integration.md#routing-replay-r3).

### Lifecycle, memory, metrics

- molt refits under the vLLM lock: `pause_generation(mode="keep")` → `begin_weight_update` → broadcast →
  `end_weight_update` (post-load processing re-run, [refit](molt-integration.md#refit)) →
  `reset_prefix_cache` (if on) → resume; in-flight must be 0 before pause, the queue empty after resume.
  Requests queued during pause run on the new weights and belong to no version.
- Sleep levels, if added: 1 drops KV and prefix cache and backs weights on host, 2 leaves GPU weights
  uninitialized until reloaded. Per-engine request counts follow `--vllm.router_policy` (default
  `cache_aware`: a rollout's turns stay on one engine while load is even): an idle engine is routing, not a
  fault, and the policy changes each engine's step composition — record it with the layout.

### Replay matrix

Use when a difference shows once in many steps — rerunning changes tokens, scheduling, lengths, cache and
packing at once. Save after sampling, before the old-logprob forward; replay with
`--train.rollout_replay_dir` from the same checkpoint, no sampling (molt's replay still updates and refits,
so compare the first replayed step before its update and restart every variant from the checkpoint); A/A
first (unstable A/A → RNG, buffers, streams, state).

| Variant | Change | Answers |
|---|---|---|
| A | Original packing | Does the frozen input reproduce it |
| B | Exact repeat of A | Run-to-run nondeterminism |
| C | Sequence permutation / packing | Layout dependence (or a page, tile, segment, expert boundary) |
| D | Micro-batch size or target position | Call shape / neighbour dependence |
| E | Fresh process or equivalent node | Initialization / device state |

Run only the variants that separate live hypotheses; compare by request id and position after reordering.
Record per matrix the rollout file, the pre-update checkpoint id, the closure and the budget (max runs,
timeout); per variant its name, the one change, command, output dir, exit code and result. Shrink keeping
the trigger: target sample and neighbours first, then length and count, rerunning the criterion after every
cut (graph fillers and boundary remainders are often the trigger). Frozen rollout and weights do not freeze
dropout, RNG, replay ids, caches or workspaces; verify A/A between a fresh and a reused process. Leaving
replay: back to sampling, real backward, next rollout — training-only success covers the frozen forward
only.

## Communication and transport

Both arms on the same transport. In aligned mode trainer and engine workers share the weight-sync NCCL
group, so the NCCL algo/proto/channel pins (vLLM `override_envs_for_invariance()`) must be set in both
processes and whitelisted into the Ray runtime env. World-size sanity: write the trainer groups from the
`[FSDP] world= dp= cp= tp= ep=` line and the engine groups (engines × TP) separately, count GPUs from the
runtime, not the config string (two DP4/EP4 groups are not EP8). Run `NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET` once per arm (`NCCL_DEBUG` propagates through the Ray env whitelist;
`NCCL_DEBUG_SUBSYS` must go into the recipe's node env) and confirm IB vs P2P/NVLink per group; a two-node
test must show that cross-node traffic actually flowed — initialization proves no bandwidth. Never compare
arms on different fabrics or nodes; `export VAR=` in a recipe overrides the container's `-e`, write
`${VAR:-default}`.

## Stalls and CUDA async errors

- Stalls: the driver log is a forwarded copy. When steps stop, read the worker and engine logs, the
  checkpoint mtimes and the last step's `timing/*` keys; one stack is not a hang and slow is not a hang.
  Never restart a job because forwarding stopped.
- CUDA async errors surface later than the faulting kernel: `CUDA_LAUNCH_BLOCKING=1` or scoped
  `torch.cuda.synchronize()`, one process per variant, keep every rank's log. A corrupted context yields no
  numerical evidence; swapping the attention backend is not a fix.
