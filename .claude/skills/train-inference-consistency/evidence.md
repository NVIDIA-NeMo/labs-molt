# Evidence and experiment rules

Open before declaring a pass or reusing an old result. What counts as evidence and how experiments are
recorded, in molt terms; the operational short form — planted error and false greens — is
[workflow.md](workflow.md#evidence). Mechanisms live in [debugging.md](debugging.md),
[contracts.md](contracts.md) and [acceptance.md](acceptance.md). Only rules are here.

## Evidence labels

| Label | Meaning | Reason code, always recorded |
|---|---|---|
| `pass` | every declared item of the scope completed and met its criterion | — |
| `fail` | a numerical comparison or a criterion failed | the first failing item |
| `inconclusive` | the check did not answer | *invalid*: the target path did not run (aligned kernel not dispatched, patched class not instantiated, forward skipped, rollout logprobs never reached the trainer); *vacuous*: zero hits (no valid tokens, no sequence crossed the boundary, no token reached the expert); *fault*: Ray actor lost, NCCL timeout, OOM |
| `not run` | out of scope or not executed; never `pass` | the scope decision |

The [ledger](#candidate-evidence-ledger) additionally marks rows *superseded* (closure changed since) or
*historical* (another candidate); one superseded result makes the gate it covered stale. No acceptance item
is added on our own. A finished run, train/rollout consistency and a quality gain are three conclusions.

## Evidence that must not be confused

The byte rule, the "do not change the inputs" rule and the "absent is not 0" rule are in
[SKILL.md](SKILL.md#rules-that-hold-everywhere). Beyond them:

1. `VLLM_BATCH_INVARIANT=1` on the engine is not cross-stack alignment: the trainer runs its own kernels
   until `--train.batch_invariant` swaps them; IS correction or FP8 online quantization only re-weights the
   gap.
2. The same text is not the same logprob: identical greedy answers say nothing about the bytes.
3. A fixed-input replay (`--train.rollout_replay_dir`) localizes; it is not the live loop, and statistics
   over old trajectories are not an unbiased KL under the current policy.
4. The same function name, `BackendConfig` key or `MOLT_*` value is not the same kernel: an auto
   `--vllm.attention_backend` runs FA3/FlashInfer and the aligned kernel never executes, without an error.
   Record the executed kernel name (profiler), launch config and shared-library hash.
5. Diagnostic interventions (`--vllm.enforce_eager`, substitute attention, forced routing, graph off,
   packing bypass) are valid causal experiments that prove only the path they ran; label them.
6. A quality score is not per-problem identity; same-problem quality, bitwise logprob and fixed-workload
   performance are three cells. When an acceptance script fails, read the producer receipt; never resample.
7. Source-tree tests are not the delivered image: container digest and installed molt / vLLM /
   nemo-automodel must match the acceptance object; the evidence chain is rebuilt after repackaging.

## Inferences no longer carried forward

- Historical numbers stay with the run that produced them; a new task inherits no model, dataset, layout,
  statistics method or pass status.
- An old candidate's pass does not transfer after a backend, image, kernel or geometry change — the same
  source pin, container tag or earlier proxy says nothing about the current dispatch; verify branch, patch,
  wheel, `.so` and digest, then redo the entry gates.
- Small-model error magnitudes do not extrapolate ([reference
  measurements](molt-integration.md#reference-measurements)); a proxy pass verifies the execution path, not
  full-depth quality.
- Completion (finished steps, finite gradients) is not acceptance: record the instantiated class and its
  import source. A number in a summary is not evidence: find numerator and denominator for the same policy,
  workload and config, or retract.
- A pass on one path or split covers no other batch, phase (prefill/decode), first round after a resume or
  cross-node layout — each gets its own row.
- Merge status, recommended pins and best-performing configs are volatile: look them up in the target repo,
  never quote them from memory.

## Experiment contract

Objects of equality first ([object of comparison](contracts.md#object-of-comparison)):

| Object | Fixed by |
|---|---|
| Weight version | the refit id; compare before that version's update |
| Tokenizer, chat template, raw token ids, position ids | the saved rollout, never a re-tokenization |
| Attention and action mask | whether EOS is a valid token; how truncated and padded rows are excluded |
| Packing | `--fsdp.packing_samples`, THD `cu_seqlens` |
| Logits processing on each side | `--rollout.temperature` applied in vLLM processed logprobs vs the fp32 divide in `log_probs_from_logits`; `--rollout.top_p 1.0`, no top-k |
| Logprob kind | raw or processed, as returned |

The same seed does not give the same scheduling or RNG consumption: freeze the tokens; never rely on
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
control. Recipe and coverage record (recover missing items from the existing config first):

| Item | Record |
|---|---|
| Workload | Training context; prompt/response length distribution; `--rollout.batch_size` × `--rollout.n_samples_per_prompt`; `--vllm.max_num_batched_tokens`; packing; training token batch |
| Hardware | GPU count, memory, interconnect; trainer `--fsdp.{tp,cp,ep}_size` and engine `--vllm.tensor_parallel_size` / `--vllm.enable_expert_parallel` separately |
| Numerics | fp32 masters, `--fsdp.param_dtype`, activation, KV and `--vllm.mamba_ssm_cache_dtype`, accumulation and gradient dtype; [FP8](contracts.md#fp8) route if any |
| Scope | Sampled-token logprob or full vocabulary; graph/eager, cache, lengths and layouts covered |
| Budget | Throughput (rollout deficit ≤ 20 % unless a human set another: [performance budget](acceptance.md#performance-budget)), peak memory, quality and gradient criteria, experiment count, stop condition; unknown threshold → report, no pass |

- Measure the config at hand first; the fixed-workload bench ([acceptance.md](acceptance.md#performance))
  never replaces the target context and natural lengths. The proxy records full-model feature → proxy
  position → test → uncovered, keeps the real hidden/head/expert geometry and is re-checked after any model,
  dtype or layout change.
- Environment fingerprint: the run manifest of [acceptance.md](acceptance.md#environment-and-provenance)
  plus both layouts, dtypes, resolved `BackendConfig`, `MOLT_*` env, untracked runtime files, engine backend
  and graph/compile/cache/autotune settings, read inside the Ray worker and engine process. A fixed weight
  is a content hash after `load_weights`, not a path; keep immutable known-good and known-bad artifacts.
- Scope of a conclusion: per result the model scale, valid samples, independent tokens, lengths and
  boundaries, process count, layout and update count. Four samples repeated to batch 32 are four samples;
  mixed scheduling a small model never triggers covers nothing on the full model; synthetic inputs are
  labelled as such. Budget, maximum experiments and stop condition go into the record.
- Write the rules in the table below into the record before comparing arms; an unmet rule leaves "pending
  re-check".

| Rule | How |
|---|---|
| One image per recipe | Same container, script and data for both arms; only `--train.batch_invariant` differs; overlay hash recorded |
| One seed per config | More seeds only when variance is the question, reported separately; most faults are systematic |
| Small-workload iteration | Small batch, short responses, few steps while iterating; the recipe workload only for acceptance |
| Freeze the criterion, plant an error | Window and tolerance in writing before the run; the planted error must read `fail` ([workflow.md](workflow.md#evidence)) |
| No single-run conclusion | Measure noise first; median of paired rounds ([acceptance.md](acceptance.md#pairing-and-computation)); an intermittent failure keeps every round and its trigger rate, not the passing rounds |
| Same fabric | Same NCCL transport (NVLink/P2P vs IB) for both arms, env in the record |
| Same node | Paired arms run sequentially on the same node(s); fragmentation, residents and topology differ elsewhere |
| No JIT/autotune pollution | No Triton/Inductor recompile or autotune inside the measured window; warm up first; a run with recompiles is void |
| Metrics by key name | Read `mismatch/*`, `vllm_kl`, grad norm from jsonl/W&B by full key; never read a column by position |

## Dual-path contract

Two independent receipts in C5, each against the saved rollout logprobs under the same θ, token history and
mask, before the update.

- (a) Native no-grad old-logprob path: `experience_maker.make_experience` under `torch.no_grad()`; keep its
  mode, never force autograd on; it proves only that this scoring path matches rollout. Under
  `--train.force_on_policy` with `--algo.kl.init_coef 0` molt skips this forward (`skip_actor_old` in
  `experience_maker.py`): run (a) on the proxy with both flags off. If the frozen recipe uses them, the
  acceptance table marks (a) `not run`, never `pass`, and C8 relies on path (b).
- (b) Grad-enabled actor forward: the branch the loss consumes, logprob captured before the loss transform,
  with the target packing, `--actor.gradient_checkpoint`, operators and layout, no diagnostic substitute;
  its pass does not replace C6–C7.
- Record for each: model mode, grad mode, output `requires_grad`, operator class / backend / source hash
  (resolved `BackendConfig`, patched class), dtype, valid tokens, mismatch. `forward_only` alone is not
  invalid; (a) passing never proves (b). For an injected patch also record class, source file and key config
  of the live object and assert the call site was hit. Several minibatch updates per round: compare within
  one weight version only ([workflow.md](workflow.md#two-step-rl-sequence)).

## Execution state

Graph vs eager (`--vllm.enforce_eager`), compile mode, prefix cache (`--vllm.enable_prefix_caching`),
autotune state and parallel layout are closure fields; results from different states are separate evidence
(mechanisms: [debugging.md](debugging.md#execution-state)). Record per comparison: resolved graph mode and
capture sizes; prefill, decode or mixed step; kernel-call evidence (a captured graph is neither every step
on the same graph nor eager numerics); cache hit or isolated (a pass with cache off does not fix the native
lifecycle); weight lifecycle phase — cold load, after refit, sleep/wake, graph replay, new process — one row
each; saved chunk-boundary state of stateful layers. A new process passing does not fix the old process's
cache or graph. "Fails only under X": X gets its own ledger row and the fix is pinned on both sides.

## Candidate evidence ledger

Purpose: an old pass is never upgraded to a new pass ([inferences no longer carried
forward](#inferences-no-longer-carried-forward)). One row per execution closure, created before deciding
what to rerun when model scale, dispatch, attention, MoE, graph or weights change.

| Field | Bound to |
|---|---|
| Candidate id, object | Model geometry (real layer count, adjacency, hidden/head/expert), tokenizer, weights id, policy version; proxy splice points listed |
| Source / image, both sides | Container digest, source and overlay hashes, imported classes and files, resolved config, native/JIT artifacts |
| Path | train()/autograd or no-grad; attention/norm/linear/MoE dispatch; precision; graph/eager, cache state; `--fsdp.*` and `--vllm.*` layout |
| Workload | Token digest, mask, length distribution, packing, concurrency, advantages, accepted samples |
| Gate, criterion | Gate id, dtype, valid tokens, mismatch, first difference; gradient / update / exit as separate states |
| Label, artifact | `pass`/`fail`/`inconclusive`/`not run`, inputs, scores, rank receipts, command, time, parent row |

Weights id answers three questions separately — manifest version, weight content (hash after
`load_weights`), run artifact (rollout dump hash); an equal version string is not equal bytes. An engine
overlay without sleep/wake is a screening candidate only.

| Change | Minimum rerun; what no longer carries |
|---|---|
| Proxy layer count/adjacency or hidden/head/expert geometry | Coverage table, C3, C5; the old proxy's 0 is no longer promotion evidence |
| Kernel, op, tile or segment constant (always both sides together) | Same-input op test on both sides, module VJP, C5 A/A and A/C, the boundary rows of [training-side.md](training-side.md#layout-length-and-world-size) |
| Quant/dispatch/JIT or compiler | Module bytes on real inputs, proxy, full-model zero-update; a direct-call unit test does not cover the compiled module |
| Attention or MoE op, or its autograd wrapper | Class and source, same-input op, its VJP, proxy, zero-update; a substitute-backend pass is diagnostic |
| Graph/cache/packing | The matching C3 cells, replay, lifecycle, packing A/C; eager does not cover FULL graph |
| Parallel layout (trainer TP/EP/CP, engine TP/EP, engine count) | Layout rules 1–5 of [training-side.md](training-side.md#layout-length-and-world-size), then C5, then C8 |
| Microbatch split, recompute mode or offload | Gradient A/A, one-step update |
| Optimizer or export/refit | Gradient and one-step update, refit hash, post-update compare; a θ0 pass is not enough |
| Image, vLLM or AutoModel version, native build | Closure check inside the worker, everything in the closure; dev-mount evidence does not transfer |

A proven non-dependency may be written into the ledger instead of a rerun; file names, PR text, a tag or
"looks like a cleanup" never prove it. Performance and quality are separate cells ([paired
acceptance](acceptance.md#paired-acceptance), [evaluation contract](acceptance.md#evaluation-contract)):
neither transfers to a candidate whose numerical path changed; a changed threshold keeps the old one and its
date.

## Measurement basis and matrix

Fix policy, token history, position, mask and tensor meaning; then record per item the original dtype/shape,
valid tokens (> 0), non-finite count, bitwise mismatch count, max |Δ| and the first failing (sequence,
position) ([compare](debugging.md#compare-per-token-logprobs); the live `mismatch/*` keys carry no max — it
would not reduce over ranks). Dtype, int-view and NaN/Inf rules: [object of
comparison](contracts.md#object-of-comparison).

| Dimension | Comparison under fixed semantics |
|---|---|
| A/A | Same weights, input and state twice; separate processes if needed |
| BI | One request alone, other batch sizes, neighbours, permuted slot, packed vs unpacked |
| PD | Full prefill, chunked prefill, token-by-token decode on the same history; mixed scheduling |
| Length and branches | Real lengths, chunk and causal boundaries, inputs that trigger MoE routing, sparse or special modules |
| Execution state | Target graph capture/replay, cache hit/miss/reset, the actual parallel layout |

Keep cases and artifacts per item for the target scope and list unsupported / not run explicitly. A bitwise
task requires mismatch = 0 in the original dtype, no float cast or tolerance loosened after. An error-budget
task names metric and threshold in advance, measures each side's own noise and is reported in its own row,
never as a bitwise pass. Gradient tolerances are defined independently
([training-side.md](training-side.md#gradient-contract-and-validation)).

## Limited exceptions and fallback

- A BI=0 baseline on the full model is not approval for full-model BI debugging. Go to the full model early
  only when the fault needs real depth, capacity, cross-layer state or cross-node topology; write first why
  the proxy is insufficient, the hypotheses to separate, the minimal scale and the stop condition.
- Faster iteration: judge each change's scope of impact. Start from the minimal counterexample, single
  operator or fixed replay, then the affected proxy gates, then the staged full model; do not repeat all
  gates per commit. Shared arithmetic or gradient changes regress the affected forward and backward;
  delivery still runs the full agreed acceptance; every scale-up states the evidence in hand and the one
  question the new scale answers.
- On a full-model failure save the inputs before the first divergence, the weight version, metadata, cache
  and routing state (routed expert ids) and RNG state; then extract a module or proxy reproducer and iterate
  on it only if the fault survives shrinking.
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
