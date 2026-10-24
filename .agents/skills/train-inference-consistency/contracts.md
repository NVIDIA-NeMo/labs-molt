# Numerical contracts

What must be equal, operator by operator, for the trainer's recomputed logprob to be the bytes vLLM
returned. Model-independent: the C0 inventory fills each row with the kernels of the model at hand
([molt-integration.md](molt-integration.md#operator-table) has the H100 table); this file says what "equal"
means per row ([decomposition](#decomposition)) and which gate checks it
([workflow.md](workflow.md#gate-table)).

## Object of comparison

- What: for every valid response token `y[t]` (inside the response mask), `log p(y[t] | prompt, y[:t])` —
  the engine's selected-token logprob vs the trainer's recompute. The first response token comes from the
  logits at the last prompt position.
- Under which weights: the same weight version, token history and mask, before that version's update.
- As what bytes: the raw fp32 logprob compared in the original dtype (`view(int32)` / `view(int16)`). A
  dtype difference is a contract violation, never cast away.
- What fails: NaN/Inf at a valid position fails even with identical bytes; zero valid tokens is
  `inconclusive`; a ±1 shift is a comparison bug, not a tolerance.
- Processing: vLLM returns `processed_logprobs` (temperature applied in the engine) and the trainer divides
  by `--rollout.temperature` after `.float()` — identical processing, hence `--rollout.top_p 1.0` and no
  top-k; any truncation makes the kept token set a discrete decision compared before the renormalised
  logprob ([below](#discrete-decisions)). Metric names: [mismatch
  metrics](molt-integration.md#mismatch-metrics); int-view code:
  [debugging.md](debugging.md#compare-per-token-logprobs).

## Decomposition

Each term guards one root cause — a different order of operations or a different computation path — and one
gate checks it; `mismatch = 0` means both sides compute the same thing, not that training is good
([acceptance.md](acceptance.md#evaluation-contract) accepts quality separately).

| Term | Meaning | Checked by |
|---|---|---|
| Kernel determinism | same kernel, same input, same bytes in every run and process | [ladder](workflow.md#cheapest-failure-criterion-first) tier 2 (one op: repeat, cross-process, graph vs eager); C3 engine A/A |
| Batch independence | a request's bytes do not depend on batch-mates, slot or the engine's merge/split; trainer: micro-batch, packing, per-expert token counts | C3 alone vs with neighbours; C5 packing A/A then A/C |
| Phase consistency | one token gets the same bytes in full prefill, chunked prefill and decode | C3 prefill vs chunked prefill vs token-by-token decode; [boundaries](#shape-process-and-state-boundaries) |
| Cross-GPU consistency | reductions (all-reduce, reduce-scatter, expert combine) fixed in order and dtype; a degenerate layout (TP = 1) matches a multi-rank side only by simulating its tree | C3/C5 at the verified layout, [layout rules](molt-integration.md#layout-rules), [training-side.md](training-side.md#layout-length-and-world-size) |
| Same numerical contract | same accumulation dtype, rounding and kernel arithmetic ([table](#per-operator-contract)) | C0 inventory, C5 |
| Same discrete choices | top-k ties and order, routing, kept token set ([below](#discrete-decisions)) | C5 with R3; set-flip rate reported |
| One update per step | one optimizer step per rollout ([synchronous run](molt-integration.md#synchronous-run)); compare only within one weight version | C8, [two-step sequence](workflow.md#two-step-rl-sequence) |
| Correct backward | the forward terms say nothing about gradients ([boundary](#shared-forward-and-backward-boundary)) | C6–C7 |
| Consistent first step after restore | masters restore bitwise, first refit sends the pre-save bytes, first compare 0 | C9, [resume](acceptance.md#checkpoint-resume-and-memory) |

## Per-operator contract

"Same math" is not "same bytes": the same source executes differently through autotune, compile
specialisation, a different caller or GPU — verify the executed kernel and its arguments, not the Python
function. Semantics first: a feature inside the kernel on one side and outside or missing on the other
(sink, sliding window, `1 + w` norm, q/k-norm position, RoPE variant) never goes away with precision
settings; enable one feature at a time on the proxy.

| Operator class | Must match |
|---|---|
| Dense GEMM (q/k/v/o, gate/up/down, lm_head) | one kernel and one split-K / tile plan for every M; fp32 accumulation; epilogue |
| Attention | split constants, reduction order, scale and LSE, in-kernel features, KV page layout — [attention contract](#attention-contract) |
| Norm (RMSNorm, fused add + norm) | same kernel and block geometry at every M; fp32 variance; eps position; `1 + w` variant; the `(hidden, residual)` pair |
| Activation (+ quant) | `silu(gate) * up` in fp32, rounded once; `cat(gate, up)` order; same approximation |
| Router / top-k | fp32 gate GEMM at every M; fp32 softmax/sigmoid, bias, renormalisation, grouped top-k; stable tie-break |
| Expert GEMM | per-row result independent of layout and position; one tile config; no split-K along K |
| Combine | fp32 accumulation on the owning rank in top-k slot order, independent of EP degree |
| KV read path | same bytes through the same dequant in prefill, chunked prefill, decode and trainer |
| SSM scan / conv (state layers) | chunk grid, boundary-state dtype, gate position, conv window, `cu_seqlens` — [state-layer contract](#state-layer-contract) |
| lm_head + scorer | head GEMM as dense GEMM; logits dtype at hand-over; fp32 divide by temperature; two trainer scorers and one engine scorer inventoried |
| Embedding | table bytes as in the checkpoint; exact gather; any scale (`sqrt(d)`) in the same dtype and position |
| RoPE | cos/sin table device and dtype; neox vs interleaved; partial dims; position relative to q/k norm |

Details per row:

- Dense GEMM: no shape-dependent kernel switch (small-M branch, library heuristic); FMA usage; tile M
  independent of the K reduction; fixed cuBLAS workspace; bias, fused activation and the rounding point of
  the output.
- Norm: the stock `_C.rms_norm` switches block size at 256 tokens — pin it or test 255/256/257; weight
  dtype; the fused op returns the new residual; norm + activation + quant fusion only if identical on both
  sides.
- Activation: rounding `silu` first can differ by one ulp, which a later FP8 quant turns into a code flip;
  `erf` vs `tanh` GELU, exp implementation; if quantized: scale grouping, amax, rounding, clamp, aggregation
  order.
- Router: autocast off; tie-break by request-local index; output order fixed; invalid / padding index
  sentinel; routing weights re-gathered from the fp32 scores; a fused grouped top-k is called, not
  re-derived in torch ([operator details](molt-integration.md#operator-details)).
- Expert GEMM and combine: masked vs contiguous layout, capacity padding; same input rounding across
  dispatch modes; padding rows never alter valid rows; routing-weight dtype and multiplication order;
  shared-expert add position.
- KV read path: a dtype label is not alignment; page size and block-table semantics; prefix-cache hit =
  miss; cache reset after every refit.
- lm_head + scorer: no unconditional `.float()` on one side only; `.float()` then divide by temperature even
  at temperature 1; fp32 log-softmax and its reduction order; vocab-shard reduction order under TP; the
  engine's sampler logprob and the trainer's `gather − logsumexp` are different kernels.
- Embedding and RoPE: a tied head shares the table bytes; vision / audio embeddings shipped from the engine,
  not recomputed; CPU and GPU round cos/sin differently; scaling variant; positions identical under packing
  and CP; mRoPE streams.

### Attention contract

- Fixed split / segment constants over keys, independent of batch and length; a length ceiling errors, never
  re-splits silently.
- QK, online softmax, PV and the merge order of partials; softmax scale and where it is applied; LSE base
  (ln vs log2) and dtype; fp32 partial accumulators.
- Mask, sink, sliding window and GQA head mapping inside the kernel on both sides.
- KV page layout (`--vllm.block_size`; read the resolved value in the engine log).
- Algebraic rewrites (MLA weight absorption, a folded softmax scale) are different rounding paths, not the
  same kernel.

### State-layer contract

- Chunk grid: chunk size and chunk boundaries at sequence boundaries; the chunked Mamba2 scan (SSD) is
  split-invariant only at 128-aligned splits, so a shared chunk grid is a multiple of 128.
- Boundary-state dtype (`--vllm.mamba_ssm_cache_dtype`) and where it is rounded; gate applied inside the
  fp32 scan vs after rounding; cumsum tile; conv window alignment — the conv state is written tail-aligned
  and read over `[0, width-1)`, so both sides must share that offset.
- The full declared state list copied on a cache hit; every packed sequence consumes `cu_seqlens`.
- Chunked prefill vs per-token decode are two rounding paths ([below](#ssm-chunk-vs-per-token-recurrence)).

<a id="discrete-decisions"></a>
## Discrete decisions before last-bit error

- Router: after the logits compare the selected set, its order and the tie-break — the same set in a
  different order changes downstream gather and reduction order. The tie-break defines the index order for
  equal scores (request-local index, never the global batch offset) and the invalid / padding index.
- Top-k / top-p: a last-bit logit difference moves tokens across the cutoff and the processed-logprob
  denominator jumps with them; compare the kept token set before the processed logprob. Forcing the token
  set is legal only if the report says it masks the set difference and counts the set-flip rate separately.
- MoE: dispatch, expert capacity, grouped-GEMM geometry, quantization and combine are fixed or verified, not
  assumed. R3 (`--train.routing_replay`) is a declared contract that takes natural routing off the compared
  path — report the rate it masks ([reference measurements](molt-integration.md#reference-measurements)).
  Forcing ids ad hoc to pass a comparison isolates the downstream (a labelled diagnostic); it never fixes a
  router difference.

## Shape, process and state boundaries

For the same victim sequence build the cells of the [measurement
matrix](evidence.md#measurement-basis-and-matrix): batch 1 vs with neighbours; a different physical slot;
lengths just before and after every boundary (chunk, page, segment, capture bucket); a fresh process
(autotune and compile caches); prefill vs decode; graph vs eager. Launches differing when only M changes is
not a numerical finding — measure the output; no difference covers only the shapes that ran. An oracle whose
autotune picks its blocking by timing flakes an A/A — pin its config. Pinning graph, compile, autotune and
cache state: [debugging.md](debugging.md#execution-state).

### SSM chunk vs per-token recurrence

- Chunked prefill and token-by-token decode are two rounding paths today, so one token's bytes depend on how
  it was scheduled. The model is supportable only when both give the same bytes.
- Match the chunk grid, the boundary-state precision and the gate order on both sides (a kernel change in
  vLLM, not a determinism switch), then prove prefill == chunked prefill == decode in C3 — decode is a cell
  that must pass, not a tolerated exception.
- `--vllm.mamba_ssm_cache_dtype float32` and a prompt-sized `--vllm.max_num_batched_tokens` fix the engine's
  state dtype and chunking (a multiple of 128, [state-layer contract](#state-layer-contract)); the trainer's
  scan walks the same grid. Under prefix caching the page decides the split instead ([Mamba
  hybrid](molt-integration.md#mamba-hybrid)).
- Upstream PRs: exact decode replay vllm-project/vllm#55905, cumsum tile #55895, gate position #56244.

### History-dependent operators (conv, scan, attention over a cache)

- Localise over the full prompt batch: first differing token per layer, then the operators inside that layer
  in order ([bisect](debugging.md#bisect-and-first-divergence)).
- A single-position probe narrows candidates, but an upstream history difference looks like an error in the
  probed operator.
- Hooks in a compiled graph may hold stale warm-up data — check freshness with a sentinel (embedding,
  residual) first.

## Kernel changes: rules for an operator author

Rules for any kernel change (a new tile plan, a fused index op, a top-k, a split-K) that touches the
compared path. The default (BI=0) path stays unchanged; a change ships as a separately named candidate that
passes the full contract gate.

- Top-k and order: take the tie rule from the current baseline, never invent one; keep the output indices
  *and* their order (swapping equal-score indices changes the later reduction order); keep valid length,
  mask, invalid sentinel, K larger than the valid count, empty rows, padding, tail blocks. A final stable
  sort cannot restore equal-score candidates an earlier pass dropped.
- Workspaces: verify the binding and temporary-storage lifetime of any CUB or custom sort; a pre-built
  index/workspace must not be overwritten under streams, concurrent requests or graph replay; poison-fill a
  reused buffer to catch unwritten tails.
- Index and elementwise fusion: fuse only confirmed hotspots; per row write down input coordinates,
  half-open interval, output capacity, valid range and who writes the padding; a fusion that crosses dequant
  or norm keeps the original dtype conversions and rounding positions; integer indices are not automatically
  right (off-by-one, request offset, CP boundary, overflow).
- Split-K and grouped GEMM: one kernel and one split plan for every M; tile M never touches the K reduction;
  `expert_num_tokens.max().item()` is a host sync, not invariance, and cannot sit inside graph capture; a
  kernel shipped as precompiled cubins cannot be audited or pinned and is not a candidate; a fixed `n_split`
  is a candidate, not a proof of prefill = decode.
- Attention coverage: invariance covers QK, online softmax, PV and the merge of splits; tile scheduler, Q
  staging, cluster selection, page layout, mask/sink, score scale and accumulator can all depend on batch or
  length — cover batch size, position, long context, chunked prefill, decode, graph buckets, mixed batches,
  page boundaries and prefix hit/miss, and record the coverage.
- Measure, do not count lines: the gain on every real shape (a fusion can raise register pressure); whether
  the change also speeds BI=0 (then it is not BI-specific overhead); same-input bitwise, BI composition and
  prefill/decode gates; BI=0 unchanged. On failure save the minimal counterexample and withdraw from the
  default path. A correctness failure is never covered by a performance number, and a stable-sort patch is
  not removed to reproduce a long-length mismatch.
- Scope: stay inside the asked scope (a top-k/index overhead task is not whole-model tuning); switching DSL
  or library does not relax the contract; switching algorithms is allowed when the BI=0 kernel cannot meet
  the tie or reduction contract — state the cost. Fairness and the record per candidate:
  [performance](acceptance.md#performance), [ledger](evidence.md#candidate-evidence-ledger).

### Observation in graph mode

- A successful capture does not make graph bytes equal eager bytes; the graph-or-eager switch is "total
  tokens ≤ max capture size": each path is invariant on its own but not equal to the other.
- Python hooks run at capture, not at replay — a capture dump is not a replay. Observe replay through
  device-side static buffers and state whether the instrumentation changed the capture; a hook not firing is
  not a layer not executing.
- Localise in eager, re-check in the production graph mode, keep eager / compiled-without-graph / graph
  artifacts separately. Turning graphs off changes speed, hence batching, hence the path a request takes:
  fix tokens, neighbours, padding and cache state first, then run a 2×2 (mode × filler length).
- Record the resolved mode, capture sizes, step types (prefill/decode/mixed) and kernel-call evidence: a
  requested FULL can land as decode-only with mixed steps eager, and the switch reappears under FULL when
  the capture limit is below `max_num_seqs × decode_query_len`. A small proxy does not trigger the
  production mixed steps; for an intermittent failure keep every round and the trigger rate.

### Determinism only on the main thread

- `torch.use_deterministic_algorithms(True)` and the Inductor deterministic config are thread-local in
  recent PyTorch: autograd's CUDA thread still sees `deterministic=False`, a compiled VJP first compiled
  during backward autotunes its reduction block, and backward step 0 is not reproducible across runs
  (forward logprobs unaffected).
- Do both: `torch.compile(..., options={"deterministic": True})` at the compile site and
  `TORCHINDUCTOR_DETERMINISTIC=1` in the environment. A backward with atomics (SSM scan) needs its own
  deterministic mode.
- `--train.full_determinism_enable` is main-thread determinism, not batch invariance ([seed
  determinism](molt-integration.md#seed-determinism)).

## Weights and precision chain

BF16 route (molt default): both sides read the same bf16 checkpoint bytes; the weight contract is three
facts ([master weights and dtype](molt-integration.md#master-weights-and-dtype)).

| Fact | Rule |
|---|---|
| fp32-kept parameters | Parameters the trainer computes in fp32 (router gate via `gate_precision`, whatever the AutoModel class keeps fp32) are fp32 on the engine too or rounded identically before use. FSDP2's `MixedPrecisionPolicy(param_dtype=bf16)` downcasts them unless excluded, and a later `.float()` cannot restore them — measure dtype inside the forward |
| Refit dtype | The broadcast sends the fp32 master and vLLM rounds it to bf16 on `load_weights`, the same cast FSDP2 applies for the training forward — equal bytes for plainly copied parameters, not for derived ones; prove the arriving bytes as in [refit](molt-integration.md#refit) |
| Mapping | Checked by value and semantics, not shape — fused QKV and gate/up order, expert numbering, tied weights, buffers, derived state; a written reason for every missing or extra item ([training-side.md](training-side.md#call-kernels-do-not-port-the-scheduler)) |

Four-stage chain — `A master (fp32 + declared exceptions) → B weights the training forward sees → C exported
bytes → D engine bytes after load_weights and post-load` — one conversion owner per tensor family, converted
exactly once:

- A: dtype recorded per parameter and buffer.
- B: bf16 (under QAT `DQ(Q(W))`), equal to `dequant(C)` bitwise.
- C: computed on the full logical tensor (`full_tensor()` then convert, never per shard — block boundaries
  and fusion-group scales cross shards), idempotent on the same master.
- D: post-load only reorders; calibration scales (`input_scale`, `k_scale`, `v_scale`) pass through
  unchanged.

On the BF16 route B, C and D are the same bytes; in general deployed bytes = `requantize(master)` with the
exporter's own function. Dtype boundaries to record per model:

| Boundary | FSDP2 / AutoModel in molt | Must check |
|---|---|---|
| Source weights and conversion | HF loader, initialisation dtype, AutoModel state-dict adapter | Same revision; logical value, scale and layout match, not only the target dtype |
| Parameter storage | fp32 masters; class-level fp32 exceptions | An fp32 master does not mean an fp32 forward |
| Forward parameters | `MixedPrecisionPolicy(param_dtype=bf16 / fp16, reduce_dtype=fp32)`, `cast_forward_inputs` | Measure inside forward, under the real autocast |
| Operator computation and output | `BackendConfig` kernels, `gate_precision`, `rms_norm` compute dtype | An fp32 output does not prove an fp32 product; a bf16 output does not prove bf16 accumulation |
| Gradient and optimizer | `reduce_dtype=fp32`, gradient accumulation, GPU or offloaded AdamW | fp32 reduction does not restore values rounded in forward; CPU and GPU AdamW are not bit-identical ([resume](acceptance.md#checkpoint-resume-and-memory)) |
| Export and serving | DTensor `full_tensor()`, HF naming, dtype-faithful refit, engine cast on `load_weights`, DCP save | Compare values on the same logical layout; the first refit after a restore sends the pre-save bytes |

### FP8

molt `main` has no `--vllm.quantization` flag, so FP8 is out of scope until an engine exists; NVFP4-class
formats are out of scope. vLLM online quantization (`quantization="fp8"`) is receiver-owned — the trainer
exports bf16, the engine quantizes — and is outside the bitwise contract unless the trainer calls the same
FP8 GEMM on `requantize(master)` with the same scale rule. That route follows the QAT/STE rules below; the
full conversion-chain and QAT treatment is in the vLLM skill's precision guide
(vllm-project/vllm#61551, `guides/precision-quant.md`):

- The training forward sees `w + (DQ(Q(w)) − w).detach()` on the deployment grid — the exporter's own
  function (range, axis, block size, scale dtype and rounding, fusion-group scale); STE identity for the
  gradient, dX on the dequantized weight by default; `QDQ(master) == dequant(deployed bytes)` bitwise. QAT
  closes the weight-value gap only; the same kernels close the rest.
- FP8 is a format name, not a rule: block (`weight_scale_inv`, commonly 128×128) vs per-tensor
  (`weight_scale`) scale, scale dtype, rounding, static vs dynamic activation scale come from
  `quantization_config` / `hf_quant_config.json` layer by layer. Activation and KV scales (`input_scale`,
  `k_scale`, `v_scale`) are calibration values passed through unchanged.
- Write the STE explicitly (`q.detach() + (x − x.detach())`): an FP8 cast on the autograd path underflows
  small gradients to 0. bf16 training with FP8 rollout is not QAT, and a difference that survives a
  precision change was never quantization — rule out routing, mask, tokens and layout first.

## Shared forward and backward boundary

The visible forward calls the vLLM kernel for the value (bitwise with inference) and saves inputs without
building a graph; the native backward is the training stack's same-math implementation and gives the VJP
([reuse / STE then reuse / glue](training-side.md#backward-reuse-glue-or-new)). It is not bitwise; it
satisfies a written [gradient contract](training-side.md#gradient-contract-and-validation) (STE form, which
weight dX uses, discrete ids fixed and not differentiated, VJP collectives, saved state) and checks that the
master has not changed since the forward, so recompute and asynchronous updates cannot mix weight versions.

- Never detach the inference output and train on it; the two legal forms (an `autograd.Function` recomputing
  the same math in `backward`, or `visible + (graph − graph.detach())` in eager) are in
  [training-side.md](training-side.md#backward-reuse-glue-or-new).
- Forward parity (C5), a finite gradient for every trainable parameter (C6) and the gradient contract (C7)
  are verified separately ([gate table](workflow.md#gate-table)); operator-level VJP passes do not prove the
  whole-model gradient, and a finite gradient is not learning.

## Changing the arithmetic baseline

When both sides' arithmetic changes on purpose (new kernel, segment or tile constants, another reduction
tree): keep the old baseline, build a new frozen rollout under the new arithmetic, state that old outputs
are no longer guaranteed bitwise unchanged, and re-run quality and performance — old numbers do not
transfer. Shared constants change together, are re-proved bitwise and are accepted on the end-to-end A/B
only ([reference measurements](molt-integration.md#reference-measurements)).
