# Evidence and experiment rules

What counts as evidence and how experiments are recorded, in molt terms. [workflow.md](workflow.md#evidence)
is the short form; mechanisms live in [debugging.md](debugging.md), [contracts.md](contracts.md) and
[acceptance.md](acceptance.md). Only rules are here.

## Evidence labels

The labels are `pass / fail / inconclusive / not run` ([workflow.md](workflow.md#evidence)). `inconclusive`
always carries its reason: *invalid* (did not enter the target path: aligned kernel not dispatched, patched
class not instantiated, forward skipped, rollout logprobs never requested), *vacuous* (zero hits: no valid
tokens, no sequence crossed the chunk boundary, no token reached the expert under test), or a GPU/system
fault (Ray actor lost, NCCL timeout, OOM). The [ledger](#candidate-evidence-ledger) additionally marks rows
*superseded* (closure changed since) or *historical* (another candidate). A numerical comparison failure is
`fail`; `pass` only when every declared item of the scope completed; out of scope is `not run`, never
`pass`, and no acceptance item is added on our own. One superseded result makes the gate it covered stale. A
finished run, train/rollout consistency and a quality gain are three conclusions.

## Evidence that must not be confused

1. `VLLM_BATCH_INVARIANT=1` on the engine is not cross-stack alignment: the trainer runs its own kernels
   until `--train.batch_invariant` swaps them; IS correction or FP8 online quant only re-weights the gap.
2. The same text is not the same logprob: identical greedy answers say nothing about the bytes.
3. Zero KL, zero probability difference or identical printed numbers are not bitwise
   ([object of comparison](contracts.md#object-of-comparison)).
4. A fixed-input replay (`--train.rollout_replay_dir`) localizes; it is not the live loop, and statistics
   over old trajectories are not an unbiased KL under the current policy.
5. The same function name, `BackendConfig` key or `MOLT_*` value is not the same kernel: an auto
   `--vllm.attention_backend` runs FA3/FlashInfer and the aligned kernel never executes, without an error;
   record the executed kernel name (profiler), launch config and shared-library hash.
6. Diagnostic interventions (`--vllm.enforce_eager`, substitute attention, forced routing, graph off,
   packing bypass) are valid causal experiments that prove only the path they ran; label them.
7. A quality score is not per-problem identity; same-problem quality, bitwise logprob and fixed-workload
   performance are three cells. When an acceptance script fails, read the producer receipt; never resample.
8. Never change tokens, weights, geometry, precision or tolerance to make a comparison pass, nor slow the
   BI=0 arm or drop a correctness patch to meet a budget; a changed baseline is a new experiment.
9. Source-tree tests are not the delivered image: container digest and installed molt / vLLM /
   nemo-automodel must match the acceptance object; the evidence chain is rebuilt after repackaging.
10. A metric that is 0 by construction or absent is not 0 (`policy_kl`, missing `mismatch/*`, a trainer
    forward skipped on a zero-advantage batch): [mismatch metrics](molt-integration.md#mismatch-metrics).

## Inferences no longer carried forward

- Historical numbers stay with the run that produced them; a new task inherits no model, dataset, layout,
  statistics method or pass status.
- An old candidate's pass does not transfer after a backend, image, kernel or geometry change — the same
  source pin, container tag or earlier proxy says nothing about the current dispatch; verify branch, patch,
  wheel, `.so` and digest, then redo the entry gates.
- Small-model error magnitudes do not extrapolate
  ([reference measurements](molt-integration.md#reference-measurements)); a proxy pass verifies the
  execution path, not full-depth quality.
- Completion (finished steps, finite gradients) is not acceptance: record the instantiated class and its
  import source.
- A number in a summary is not evidence: find numerator and denominator for the same policy, workload and
  config, or retract.
- A pass on one path or split covers no other batch, phase (prefill/decode), first round after a resume or
  cross-node layout — each gets its own row.
- Merge status, recommended pins and best-performing configs are volatile: look them up in the target repo,
  never quote them from memory.

## Experiment contract

**Objects of equality first** ([contracts.md](contracts.md#object-of-comparison)): weight version (refit
id), tokenizer and chat template, raw token ids, position ids, attention and action mask (whether EOS is
a valid token, how truncated and padded rows are excluded), packing (`--fsdp.packing_samples`, THD
`cu_seqlens`), logits processing on each side (`--rollout.temperature` in vLLM processed logprobs vs the
fp32 divide in `log_probs_from_logits`; `top_p` 1, top-k off), raw or processed logprob returned. The
same seed does not give the same scheduling or RNG consumption: freeze the tokens; never rely on
resampling reproducing the text. Record these goals separately:

| Goal | Valid evidence | Cannot infer |
|---|---|---|
| Single-side repeatability | A/A, same input and state | Cross-batch or cross-process consistency |
| Batch/layout invariance | Same sample identical under other slots, neighbours, packing | Trainer equals engine |
| Prefill/decode parity | Two engine paths on the same forced tokens | Backward or the update is correct |
| Cross-stack forward parity | Real trainer forward against a frozen rollout | Full-vocabulary equality from sampled tokens |
| Effective training loop | Non-degenerate gradients, parameter change, post-update parity | Convergence, long-run stability |

A finite logprob is necessary: empty mask, all-zero placeholders, a missed capture, NaN or the wrong weight
version cannot pass; a legitimate 0 is told from a placeholder by hook count, score source and a negative
control. **Recipe and coverage record** (recover missing items from the existing config first):

| Item | Record |
|---|---|
| Workload | Training context; prompt/response length distribution; `--rollout.batch_size` × `--rollout.n_samples_per_prompt`; `--vllm.max_num_batched_tokens`; packing; training token batch |
| Hardware | GPU count, memory, interconnect; trainer `--fsdp.{tp,cp,ep}_size` and engine `--vllm.tensor_parallel_size` / `--vllm.enable_expert_parallel` separately |
| Numerics | fp32 masters, `--fsdp.param_dtype`, activation, KV and `--vllm.mamba_ssm_cache_dtype`, accumulation and gradient dtype; FP8 route if any |
| Scope | Sampled-token logprob or full vocabulary; graph/eager, cache, lengths and layouts covered |
| Budget | Throughput, peak memory, quality and gradient criteria, experiment count, stop condition; unknown threshold → report, no pass |

- Measure the config at hand first; the fixed-workload bench ([acceptance.md](acceptance.md#performance))
  never replaces the target context and natural lengths.
- The proxy records full-model feature → proxy position → test → uncovered, keeps the real
  hidden/head/expert geometry and is re-checked after any model, dtype or layout change.
- **Environment fingerprint:** the run manifest of
  [acceptance.md](acceptance.md#environment-and-provenance) plus both layouts, dtypes, resolved
  `BackendConfig`, `MOLT_*` env, untracked runtime files, engine backend and graph/compile/cache/autotune
  settings, read inside the Ray worker and engine process; a fixed weight is a content hash after
  `load_weights`, not a path; keep immutable known-good and known-bad artifacts.
- **Scope of conclusions:** per result the model scale, valid samples, independent tokens, lengths and
  boundaries, process count, layout and update count; four samples repeated to batch 32 are four samples;
  mixed scheduling a small model never triggers covers nothing on the full model; synthetic inputs are
  labelled as such, never as real activations; budget, maximum experiments and stop condition go into the
  record.
- **Experiment discipline:** write the nine rules into the record before comparing arms; an unmet rule
  leaves "pending re-check".

| Rule | How |
|---|---|
| One image per recipe | Same container, script and data for both arms; only `--train.batch_invariant` differs; overlay hash recorded |
| One seed per config | More seeds only when variance is the question, reported separately; most faults are systematic |
| Small-workload iteration | Small batch, short responses, few steps while iterating; the recipe workload only for acceptance |
| Freeze the criterion, plant an error | Window and tolerance in writing before the run; the planted error must read FAIL ([workflow.md](workflow.md#evidence)) |
| No single-run conclusion | Measure noise first; median of paired rounds ([acceptance.md](acceptance.md#pairing-and-computation)); an intermittent failure keeps every round and its trigger rate, not the passing rounds |
| Same fabric | Same NCCL transport (NVLink/P2P vs IB) for both arms, env in the record |
| Same node | Paired arms run sequentially on the same node(s); fragmentation, residents and topology differ elsewhere |
| No JIT/autotune pollution | No Triton/Inductor recompile or autotune inside the measured window; warm up first; a run with recompiles is void |
| Metrics by key name | Read `mismatch/*`, `vllm_kl`, grad norm from jsonl/W&B by full key; one case read a grad-norm column as entropy |

## Dual-path contract

Two independent receipts in C5, each against the saved rollout logprobs under the same θ, token history
and mask, before the update.

- (a) **Native no-grad old-logprob path**: `experience_maker.make_experience` under `torch.no_grad()`;
  keep its mode, never force autograd on; it proves only that this scoring path matches rollout. molt
  note: under `--train.force_on_policy` with `--algo.kl.init_coef 0` the path does not run (old = detach of
  the training forward) — run C5 without that combination or mark the row `not run`, never `pass`.
- (b) **Grad-enabled actor forward**: the branch the loss consumes, logprob captured before the loss
  transform, with the target packing, `--actor.gradient_checkpoint`, operators and layout, no diagnostic
  substitute; its pass does not replace C6–C7.
- Record for each: model mode, grad mode, output `requires_grad`, operator class / backend / source hash
  (resolved `BackendConfig`, patched class), dtype, valid tokens, mismatch. `forward_only` alone is not
  invalid; (a) passing never proves (b).
- For an injected patch also record class, source file and key config of the live object and assert the
  call site was hit.
- Several minibatch updates per round: compare within one weight version only
  ([workflow.md](workflow.md#two-step-rl-sequence)).

## Execution state

Graph vs eager (`--vllm.enforce_eager`), compile mode, prefix cache (`--vllm.enable_prefix_caching`),
autotune state and parallel layout are closure fields; results from different states are separate
evidence (mechanisms: [debugging.md](debugging.md#execution-state)). Record per comparison: resolved
graph mode and capture sizes; prefill, decode or mixed step; kernel-call evidence (a captured graph is
neither every step on the same graph nor eager numerics); cache hit or isolated (a pass with cache off does
not fix the native lifecycle); weight lifecycle phase — cold load, after refit, sleep/wake, graph replay,
new process — one row each; saved chunk-boundary state of stateful layers. A new process passing does not
fix the old process's cache or graph. "Fails only under X": X gets its own ledger row and the fix is pinned
on both sides.

## Candidate evidence ledger

Purpose: an old pass is never upgraded to a new pass. One row per execution closure, created before
deciding what to rerun when model scale, dispatch, attention, MoE, graph or weights change.

| Field | Bound to |
|---|---|
| Candidate id, object | Model geometry (real layer count, adjacency, hidden/head/expert), tokenizer, weights id, policy version; proxy splice points listed |
| Source / image, both sides | Container digest, source and overlay hashes, imported classes and files, resolved config, native/JIT artifacts |
| Path | train()/autograd or no-grad; attention/norm/linear/MoE dispatch; precision; graph/eager, cache state; `--fsdp.*` and `--vllm.*` layout |
| Workload | Token digest, mask, length distribution, packing, concurrency, advantages, accepted samples |
| Gate, criterion | Gate id, dtype, valid tokens, mismatch, first difference; gradient / update / exit as separate states |
| Label, artifact | `pass`/`fail`/`inconclusive`/`not run`, inputs, scores, rank receipts, command, time, parent row |

Weights id answers three questions separately — manifest version, weight content (hash after
`load_weights`), run artifact (rollout dump hash); an equal version string is not equal bytes.

**Entry gates, cheap to expensive:**

1. Module — real triggering input and layout, A/A, candidate forward against an independent reference,
   dX/dW with the same dY; covers only what was tested.
2. Proxy with real adjacency — C3, then both C5 paths; fixed tokens only unless asked.
3. Full-model zero-update — real workload on the current candidate, tokens and raw scores saved, replay
   before any update, first failing sample first, then the agreed count; unequal → stop, never route around
   with multi-step RL or a quality eval.
4. Per-version update — for each θk compare, then reward/advantage, backward, masters, optimizer step.
5. Independent acceptance — gradients, real RL, quality, performance, clean exit and final image in separate
   cells; another candidate's numbers stay historical.

An engine overlay without sleep/wake is a screening candidate only. A mid-phase failure is split into
numerical, control, environment and resource planes; sealed rollouts and producer hashes are kept, and a
broken acceptor is replaced by a versioned recovery checker
([acceptance.md](acceptance.md#pairing-and-computation)), never by editing the receipt.

| Change | Minimum rerun; what no longer carries |
|---|---|
| Proxy layer count/adjacency or hidden/head/expert geometry | Coverage table, C3, C5; the old proxy's 0 is no longer promotion evidence |
| Quant/dispatch/JIT or compiler | Module bytes on real inputs, proxy, full-model zero-update; a direct-call unit test does not cover the compiled module |
| Attention or MoE op, or its autograd wrapper | Class and source, same-input op, its VJP, proxy, zero-update; a substitute-backend pass is diagnostic |
| Graph/cache/packing/parallel layout | The matching C3 cells, replay, lifecycle; eager does not cover FULL graph |
| Optimizer or export/refit | Gradient and one-step update, refit hash, post-update compare; a θ0 pass is not enough |
| Image or native build | Closure check inside the worker, affected numerics and performance; dev-mount evidence does not transfer |

A proven non-dependency may be written into the ledger instead of a rerun; file names, PR text, a tag or
"looks like a cleanup" never prove it. **Separate cells for performance and quality** ([paired
acceptance](acceptance.md#paired-acceptance), [evaluation contract](acceptance.md#evaluation-contract)):
neither transfers to a candidate whose numerical path changed; a changed threshold keeps the old one and date.

## Measurement basis and matrix

Fix policy, token history, position, mask and tensor meaning, then record per item the original
dtype/shape, valid tokens (> 0), non-finite count, bitwise mismatch count, max |Δ| and the first failing
(sequence, position) ([compare](debugging.md#compare-per-token-logprobs); the live `mismatch/*` keys carry
no max — it would not reduce over ranks). Dtype, int-view and NaN/Inf rules:
[object of comparison](contracts.md#object-of-comparison); float `==` conflates ±0, so compare contiguous
bytes or an equal-width int view.

| Dimension | Comparison under fixed semantics |
|---|---|
| A/A | Same weights, input and state twice; separate processes if needed |
| BI | One request alone, other batch sizes, neighbours, permuted slot, packed vs unpacked |
| PD | Full prefill, chunked prefill, token-by-token decode on the same history; mixed scheduling |
| Length and branches | Real lengths, chunk and causal boundaries, inputs that trigger MoE routing, sparse or special modules |
| Execution state | Target graph capture/replay, cache hit/miss/reset, the actual parallel layout |

Keep cases and artifacts per item for the target scope and list unsupported / not run explicitly. A
bitwise task requires mismatch = 0 in the original dtype, no float cast or tolerance loosened after; an
error-budget task names metric and threshold in advance, measures each side's own noise and is reported in
its own row, never as a bitwise pass. Gradient tolerances are defined independently
([training-side.md](training-side.md#gradient-contract-and-validation)).

## Limited exceptions and fallback

- A BI=0 baseline on the full model is not approval for full-model BI debugging. Go to the full model
  early only when the fault needs real depth, capacity, cross-layer state or cross-node topology; write
  first why the proxy is insufficient, the hypotheses to separate, the minimal scale and the stop condition.
- Faster iteration: judge each change's scope of impact; start from the minimal counterexample, single
  operator or fixed replay, then the affected proxy gates, then the staged full model; do not repeat all
  gates per commit; shared arithmetic or gradient changes regress the affected forward and backward;
  delivery still runs the full agreed acceptance; every scale-up states the evidence in hand and the one
  question the new scale answers.
- On a full-model failure save the inputs before the first divergence, the weight version, metadata, cache
  and routing state (routed expert ids) and RNG state, then extract a module or few-layer reproducer and
  iterate on it only if the fault survives shrinking.
- If a fix changes the sampled trajectory or EOS, save the new history and replay the old-logprob on it at
  the same weight version; the old trajectory stays a causal diagnostic. Isolation packages localize;
  acceptance returns to the delivered code and default mode.

## Failure records

Append-only: new results are added, nothing overwritten or deleted. A record holds the source, patch and
image hashes, the exact commands, per-item counts (valid, mismatch, non-finite), the earliest difference,
the hypotheses excluded and how, and before/after on the original reproducer. Code cleaned up afterwards
needs its own re-verification or explicit equivalence evidence; old tests do not vouch for it. Passed items
are not erased by unfinished ones, but the conclusion is scoped to what ran. Use one jsonl/W&B project per
comparison and read it back to confirm no step is missing.
