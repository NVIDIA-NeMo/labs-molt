# Numerical contracts

What must be equal, operator by operator, for the trainer's recomputed logprob to be the bytes vLLM
returned. Model-independent: the G0 inventory fills each row with the kernels of the model at hand
([molt-integration.md](molt-integration.md) has the H100 table); this file says what "equal" means
per row and which gate checks it ([workflow.md](workflow.md#gate-table)).

## Object of comparison

- For every valid response token `y[t]` (inside the response mask), `log p(y[t] | prompt, y[:t])`: the
  engine's selected-token logprob vs the trainer's recompute under the same weight version, token
  history and mask, before that version's update; the first response token comes from the logits at
  the last prompt position. Compared as bytes in the original dtype (`view(int32)` / `view(int16)`): a
  dtype difference is a contract violation, never cast away; NaN/Inf at a valid position fails with
  identical bytes; zero valid tokens is `inconclusive`; a ±1 shift is a comparison bug, not a tolerance.
- The object is the raw fp32 logprob. vLLM returns `processed_logprobs` (temperature applied in the
  engine) and the trainer divides by `--rollout.temperature` after `.float()`: identical processing,
  hence `--rollout.top_p 1.0` and no top-k — any truncation makes the candidate set a discrete
  decision compared before the renormalised logprob ([below](#discrete-decisions)). Metric names and
  gating: [molt-integration.md](molt-integration.md#where-consistency-lives-in-molt-main); int-view
  code: [debugging.md](debugging.md#compare-per-token-logprobs).

## Decomposition

Each term guards one root cause — a different order of operations or a different computation path —
and one gate checks it; `mismatch = 0` means both sides compute the same thing, not that training is
good ([acceptance.md](acceptance.md#evaluation-contract) accepts quality separately).

| Term | Meaning | Checked by |
|---|---|---|
| Kernel determinism | same kernel, same input, same bytes in every run and process | ladder step 2 (one op: repeat, cross-process, graph vs eager); G3 engine A/A |
| Batch independence | a request's bytes do not depend on batch-mates, slot or the engine's merge/split; trainer: micro-batch, packing, per-expert token counts | G3 alone vs with neighbours; G4 packing A/A then A/C |
| Phase consistency | one token gets the same bytes in full prefill, chunked prefill and decode | G3 prefill vs chunked prefill vs token-by-token decode |
| Cross-GPU consistency | reductions (all-reduce, reduce-scatter, expert combine) fixed in order and dtype; with different layouts per side, a fixed tree or no reduction on the compared path | G3/G4 at the verified layout, [layout rules](molt-integration.md#layout-rules) |
| Same numerical contract | same accumulation dtype, rounding and kernel arithmetic ([table](#per-operator-contract)) | G0 inventory, G4 |
| Same discrete choices | top-k ties and order, routing, candidate set ([below](#discrete-decisions)) | G4 with routing replay; set-flip rate reported |
| One update per step | `--train.max_epochs 1`, synchronous; compare only within one weight version | G6, [two-step sequence](workflow.md#two-step-rl-sequence) |
| Correct backward | the forward terms say nothing about gradients | G5 |
| Consistent first step after restore | masters restore bitwise, first refit sends the pre-save bytes, first compare 0 | G7, [resume](molt-integration.md#resume) |

## Per-operator contract

"Same math" is not "same bytes": the same source executes differently through autotune, compile
specialisation, a different caller or GPU — verify the executed kernel and its arguments, not the
Python function. Semantics first: a feature inside the kernel on one side and outside or missing on the
other (sink, sliding window, `1 + w` norm, q/k-norm position, RoPE variant) never goes away with
precision settings; enable one feature at a time on a two-layer model.

| Operator class | What must match |
|---|---|
| Dense GEMM (q/k/v/o, gate/up/down) | One kernel and one split-K / tile plan for every M — no shape-dependent kernel switch (small-M branch, library heuristic); fp32 accumulation; FMA usage; tile M independent of the K reduction; fixed cuBLAS workspace; epilogue (bias, fused activation, where the output is rounded) |
| Attention | Fixed split / segment constants over keys, independent of batch and length (a length ceiling errors, never re-splits silently); QK, online softmax, PV and the merge order of partials; softmax scale and where it is applied; LSE base (ln vs log2) and dtype; mask, sink, sliding window and GQA head mapping inside the kernel on both sides; fp32 partial accumulators; KV page layout |
| Norm (RMSNorm, fused add + norm) | Same kernel both sides; fp32 variance accumulation and eps position; `1 + w` variant; weight dtype; the fused op returns the new residual — keep the `(hidden, residual)` pair; norm + activation + quant fusion only if identical on both sides |
| Activation (+ quant) | `silu(gate) * up` in fp32, rounded once to bf16 (rounding `silu` first can differ by one ulp, which a later FP8 quant turns into a code flip); `cat(gate, up)` order; same approximation (`erf` vs `tanh` GELU, exp implementation); if quantized: scale grouping, amax, rounding, clamp, aggregation order |
| Router / top-k | fp32-output gate GEMM at every batch size; softmax or sigmoid, bias, renormalisation and grouped top-k in fp32 with autocast off; stable top-k with tie-break by request-local index; output order fixed; invalid / padding index sentinel; routing weights re-gathered from the fp32 scores |
| Expert GEMM | Per-row result independent of layout (masked vs contiguous, capacity padding, token position); same input rounding across dispatch modes; one tile config, no split-K or atomics along K; expert weight dtype; padding rows never alter valid rows |
| Combine | fp32 accumulation on the token's owning rank in top-k slot order, independent of EP degree and arrival order; routing-weight dtype and multiplication order; shared-expert add position |
| KV read path | Prefill, chunked prefill, decode and the trainer read the same bytes through the same dequant (a dtype label is not alignment); page size and block-table semantics; prefix-cache hit = miss; cache reset after every refit |
| SSM scan / conv (state layers) | Chunk grid (chunk size, chunk boundaries at sequence boundaries); boundary-state dtype (`--vllm.mamba_ssm_cache_dtype`) and where it is rounded; gate applied inside the fp32 scan vs after rounding; cumsum tile; conv window alignment; the full declared state list copied on a cache hit; every packed sequence consumes `cu_seqlens`; chunked prefill vs per-token decode are two rounding paths ([below](#shape-process-and-state-boundaries)) |
| lm_head + scorer | Head GEMM as dense GEMM; logits dtype at the hand-over (no unconditional `.float()` on one side only); `.float()` then divide by temperature, even at temperature 1; fp32 log-softmax and its reduction order; vocab-shard reduction order under TP; the engine's sampler logprob and the trainer's `gather − logsumexp` are two kernels — inventory both |
| Embedding | Table bytes as in the checkpoint (a tied head shares them); exact gather; any scale (`sqrt(d)`) in the same dtype and position; vision / audio embeddings shipped from the engine, not recomputed |
| RoPE | cos/sin table built on the same device (CPU and GPU round differently) in the same dtype; neox vs interleaved; partial rotary dims; scaling variant; applied before or after q/k norm as vLLM does; positions identical under packing and CP; mRoPE streams |

<a id="discrete-decisions"></a>
## Discrete decisions before last-bit error

- Router: after the logits compare the selected set, its order and the tie-break — the same set in a
  different order changes downstream gather and reduction order. The tie-break defines the index order
  for equal scores (request-local index, never the global batch offset) and the invalid / padding index.
- Top-k / top-p: a last-bit logit difference moves tokens across the cutoff and the processed-logprob
  denominator jumps with them; compare the candidate set before the processed logprob. molt's contract
  is `--rollout.top_p 1.0`, no top-k; a candidate-set replay is legal only if the report says it masks
  the set difference and counts the set-flip rate separately.
- MoE: dispatch, expert capacity, grouped-GEMM geometry, quantization and combine are fixed or verified,
  not assumed. Routing replay (`--train.routing_replay`, R3 in molt) is a declared contract that takes
  natural routing off the compared path — report the rate it masks (the reference's figure is in the
  [reference measurements](molt-integration.md#reference-measurements)). Forcing ids ad hoc to pass a comparison isolates the downstream (a labelled
  diagnostic); it never fixes a router difference.

## Shape, process and state boundaries

For the same victim sequence build: batch 1 vs with neighbours, a different physical slot, lengths just
before and after every boundary (chunk, page, segment, capture bucket), a fresh process (autotune and
compile caches), prefill vs decode, graph vs eager. Launches differing when only M changes is not a
numerical finding — measure the output; no difference covers only the shapes that ran. Pinning graph,
compile, autotune and cache state: [debugging.md](debugging.md#execution-state).

| Boundary | Rule |
|---|---|
| SSM chunk vs per-token recurrence | Chunked prefill and token-by-token decode are two rounding paths today, so one token's bytes depend on how it was scheduled. The model is supportable only when both give the same bytes: match the chunk grid, the boundary-state precision and the gate order on both sides (a kernel change in vLLM, not a determinism switch), then prove prefill == chunked prefill == decode in G3 — decode is a cell that must pass, not a tolerated exception. `--vllm.mamba_ssm_cache_dtype float32` and a prompt-sized `--vllm.max_num_batched_tokens` fix the engine's state dtype and chunking; the trainer's scan walks the same grid |
| History-dependent operators (conv, scan, attention over a cache) | Localise over the full prompt batch: first differing token per layer, then the operators inside that layer in order ([bisect](debugging.md#bisect-and-first-divergence)). A single-position probe narrows candidates, but an upstream history difference looks like an error in the probed operator. Hooks in a compiled graph may hold stale warm-up data — check freshness with a sentinel (embedding, residual) first |
| Parallel layout | The reduction tree is part of the contract; a degenerate layout (TP = 1) matches a multi-rank side only by simulating its tree — group by rank, partial sums, same rounding points, same merge order ([layout rules](molt-integration.md#layout-rules), [training-side.md](training-side.md#layout-length-and-world-size)) |

## Kernel-level contracts

### Stable top-k, full output

- Take the tie rule from the current baseline, never invent one.
- Keep the comparison key for equal scores and the ±0 and NaN/Inf conventions.
- Keep the output indices *and* their order: swapping equal-score indices changes the later reduction order.
- Keep valid length, mask, invalid sentinel, K larger than the valid count, empty rows, padding, tail blocks.
- Keep prefill/decode index coordinates and the page mapping of the same semantic token.
- Audit each candidate for repeated allocation, sort input range, radix/merge passes, workspace, launch
  geometry; prove order preservation before deleting a pass.
- Verify the binding and temporary-storage lifetime of any CUB or custom sort; a pre-built index/workspace
  must not be overwritten under streams, concurrent requests or graph replay.

### Index / elementwise fusion

- Fuse only confirmed hotspots: gather lengths, row start/end, sparse index concatenation, invalid-row
  zeroing.
- Per row write down input coordinates, half-open interval, output capacity, valid range and who writes the
  padding.
- Shrink grid and loop stride together; poison-fill a reused buffer to catch unwritten tails.
- Integer indices are not automatically right: off-by-one, request offset, CP boundary, overflow and
  uninitialised rows bias a whole sequence.
- A fusion that crosses dequant or norm keeps the original dtype conversions and rounding positions.
- Test top-k and index post-processing separately, combined, at module level, then through the engine gates.
- Record whether the change also speeds BI=0 (then it is not BI-specific overhead).

### Split-K and the M of grouped GEMM

- One kernel and one split plan for every M; tile M never touches the K reduction; each row's result
  independent of layout.
- `expert_num_tokens.max().item()` is a host sync, not invariance, and cannot sit inside graph capture.
- Before replacing it read whether the parameter is a correctness bound, a capacity or a tuning hint, and
  cover heavy load and empty experts.
- A fixed `n_split` is a candidate, not a proof of prefill = decode.
- The K range per split, the order inside it, the accumulator dtype, the partial-reduce order, tail/padding
  and the actual specialisation must still be fixed or verified.
- Test identical operator inputs and the full semantic path.

### Attention coverage

- Invariance covers QK, online softmax, PV and the merge of splits.
- Tile scheduler, Q staging, cluster selection, page layout, mask/sink, score scale and accumulator can all
  depend on batch or length.
- Cover batch size, position, length (long context), chunked prefill, decode, graph buckets, mixed batches,
  page boundaries and prefix hit/miss.
- Record the coverage — graph plus cache being on does not mean one test executed every branch.

### Observation in graph mode

- Record the resolved graph mode, capture sizes, step types (prefill/decode/mixed) and kernel-call evidence.
  A successful capture does not make graph bytes equal eager bytes.
- Python hooks run at capture, not at replay — a capture dump is not a replay. Observe replay through
  device-side static buffers and state whether the instrumentation changed the capture.
- Localise in eager, re-check in the production graph mode, keep eager / compiled-without-graph / graph
  artifacts separately.
- A hook not firing is not a layer not executing.
- The graph-or-eager switch is "total tokens ≤ max capture size": each path is invariant on its own but not
  equal to the other.
- Turning graphs off changes speed, hence batching, hence the path a request takes.
- Fix tokens, neighbours, padding and cache state first, then run a 2×2 (mode × filler length).

### Determinism only on the main thread

- `torch.use_deterministic_algorithms(True)` and the Inductor deterministic config are thread-local in recent
  PyTorch.
- Autograd's CUDA thread still sees `deterministic=False`, a compiled VJP first compiled during backward
  autotunes its reduction block, and backward step 0 is not reproducible across runs (forward logprobs
  unaffected).
- Do both: `torch.compile(..., options={"deterministic": True})` at the compile site and
  `TORCHINDUCTOR_DETERMINISTIC=1` in the environment.
- A backward with atomics (SSM scan) needs its own deterministic mode.
- `--train.full_determinism_enable` sets seeds, torch's deterministic mode in the main thread and a
  single-process engine; it does not make kernels batch-invariant.

### Cost breakdown BI=0 → BI=1

- Draw the real call chain — default fast kernel, BI branch, extra pre/post-processing (sort, index, quant) —
  from source, binding, compiled artifacts and real launches, not from function names.
- Copy the BI=0 implementation into a standalone candidate and add one deterministic condition at a time for
  a bisectable A/B; the default path stays unchanged.
- Switching algorithms is allowed when BI=0's cannot meet the tie or reduction contract — state the cost.
- Fairness before any kernel conclusion: same node, image and transport, BI=0 not running BI kernels for
  nothing, no JIT stall, shared fixes in both arms, median over paired rounds
  ([acceptance.md](acceptance.md#performance)).
- Record per candidate:
  - source and binary hashes;
  - the one intended change (BI-specific or common) with kernel, shapes and time share;
  - same-input bitwise, BI composition and prefill/decode gates;
  - BI=0 unchanged plus real shape/tie/mask/tail and graph/workspace reuse;
  - paired latency, launch and allocation counts, repetitions;
  - verdict and remaining scope.
- On failure save the minimal counterexample, then withdraw from the default path.
- A correctness failure is never covered by a performance number, and a stable-sort patch is not removed to
  reproduce a long-length mismatch.

## Weights and precision chain

BF16 route (molt default): both sides read the same bf16 checkpoint bytes; the weight contract is three facts.

| Fact | Rule |
|---|---|
| fp32-kept parameters | Parameters the trainer computes in fp32 (router gate via `gate_precision`, whatever the AutoModel class keeps fp32) are fp32 on the engine too or rounded identically before use. FSDP2's `MixedPrecisionPolicy(param_dtype=bf16)` downcasts them unless excluded and a later `.float()` cannot restore them — measure dtype inside the forward under the real autocast |
| Refit dtype | The broadcast sends the fp32 master and vLLM rounds it to bf16 on `load_weights`, the same cast FSDP2's `param_dtype` applies for the training forward — equal bytes for plainly copied parameters, not for derived ones (Mamba `A = -exp(A_log)`, calibration scales); hash the same tensor on both sides after `load_weights` to prove the arriving bytes are the bytes the training forward saw (`--train.check_weight_update_equal` only lists unassigned parameters) |
| Mapping | Checked by value and semantics, not shape — fused QKV and gate/up order, expert numbering, tied weights, buffers, derived state; a written reason for every missing or extra item ([training-side.md](training-side.md#call-kernels-do-not-port-the-scheduler)) |

Four-stage chain — `A master (fp32 + declared exceptions) → B weights the training forward sees → C
exported bytes → D engine bytes after load_weights and post-load` — one conversion owner per tensor
family, converted exactly once. A: dtype recorded per parameter and buffer. B: bf16 (under QAT
`DQ(Q(W))`), equal to `dequant(C)` bitwise. C: computed on the full logical tensor (`full_tensor()`
then convert, never per shard — block boundaries and fusion-group scales cross shards), idempotent on
the same master. D: post-load only reorders; calibration scales (`input_scale`, `k_scale`, `v_scale`)
pass through unchanged. On the BF16 route B, C and D are the same bytes; in general deployed bytes =
`requantize(master)` with the exporter's own function. Dtype boundaries to record per model:

| Boundary | FSDP2 / AutoModel in molt | Must check |
|---|---|---|
| Source weights and conversion | HF loader, initialisation dtype, AutoModel state-dict adapter | Same revision; logical value, scale and layout match, not only the target dtype |
| Parameter storage | fp32 masters; class-level fp32 exceptions | An fp32 master does not mean an fp32 forward |
| Forward parameters | `MixedPrecisionPolicy(param_dtype=bf16|fp16, reduce_dtype=fp32)`, `cast_forward_inputs` | Measure inside forward, under the real autocast |
| Operator computation and output | `BackendConfig` kernels, `gate_precision`, `rms_norm` compute dtype | An fp32 output does not prove an fp32 product; a bf16 output does not prove bf16 accumulation |
| Gradient and optimizer | `reduce_dtype=fp32`, gradient accumulation, GPU or offloaded AdamW | fp32 reduction does not restore values rounded in forward; CPU and GPU AdamW are documented as not bit-identical (sqrt and division rounding) |
| Export and serving | DTensor `full_tensor()`, HF naming, dtype-faithful refit, engine cast on `load_weights`, DCP save | Compare values on the same logical layout; the first refit after a restore sends the pre-save bytes |

When quantization is in play (FP8 is the only case on H100; NVFP4-class formats are out of scope),
vLLM online quantization (`quantization="fp8"`) is receiver-owned — the trainer exports bf16, the
engine quantizes — and is not bitwise against a bf16 training forward unless the trainer calls the
same FP8 GEMM on `requantize(master)` with the same scale rule. That route follows the QAT/STE rules:

- The training forward sees `w + (DQ(Q(w)) − w).detach()` on the deployment grid — the exporter's own
  function (range, axis, block size, scale dtype and rounding, fusion-group scale); STE identity for
  the gradient, dX on the dequantized weight by default; check `QDQ(master) == dequant(deployed bytes)`
  bitwise. A training grid that differs from the deployment grid keeps the mismatch with QAT "on"; QAT
  closes the weight-value gap only — the same kernels close the rest.
- Verifying a scale rule with dequantized weights gives a false positive (they sit on the grid and
  match every rule): verify from the real bf16 release and require a known-wrong rule to fail.
- Dead zone: a master initialised from a dequantized checkpoint sits on grid points half a spacing from
  every decision boundary, so at RL learning rates the deployed bytes do not change for thousands of
  steps. Start the master from the real bf16 release and requantize every step — a human accepts that
  θ0 bytes then differ from the official checkpoint, with the cost measured on the proxy.

## Shared forward and backward boundary

The visible forward calls the vLLM kernel for the value (bitwise with inference) and saves inputs
without building a graph; the native backward is the training stack's same-math implementation and
gives the VJP ([reuse / STE then reuse / glue](training-side.md#backward-reuse-glue-or-new)). It is
not bitwise; it satisfies a written [gradient contract](training-side.md#gradient-contract-and-validation)
(STE form, which weight dX uses, discrete ids fixed and not differentiated, VJP collectives, saved
state) and checks that the master has not changed since the forward, so recompute and asynchronous
updates cannot mix weight versions.

- Never detach the inference output and train on it. Where a fused kernel has no backward, either the
  `autograd.Function` recomputes the same math in PyTorch inside `backward`, or the forward returns
  `visible + (graph − graph.detach())` with `graph` that same math (bytes unchanged except a −0.0 sign; same
  dtype on both terms; the surrogate graph stays in memory) — [training-side.md](training-side.md#backward-reuse-glue-or-new).
- Verify separately, none substituting for another: forward parity (G4); a finite gradient for every
  trainable parameter; the gradient contract against an independent reference that does not call the
  backward under test, same cotangent in the same dtype, tolerances fixed first, a planted scaled error
  must fail (G5). Operator-level VJP passes do not prove the whole-model gradient, and a finite gradient
  is not learning: check non-degenerate advantages, non-zero gradients, parameter and optimizer change.

## Changing the arithmetic baseline

When both sides' arithmetic changes on purpose to align them (new kernel, segment or tile constants,
another reduction tree): keep the old baseline, build a new frozen rollout under the new arithmetic,
state that old outputs are no longer guaranteed bitwise unchanged, and re-run quality and performance
— old success numbers do not transfer. Constants shared by both sides (attention segments, MoE tiles)
change together, are re-proved bitwise and are accepted on the end-to-end A/B only; microbenchmark
winners lost end to end twice in the reference.
