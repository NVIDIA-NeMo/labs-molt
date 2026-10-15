# Workflow: gates C0–C10

Turn "rollout and trainer disagree" into a repeatable experiment with fixed weights and fixed
tokens, find the first divergence, fix it, then return to the real training loop. Iterate at the
cheapest level that still reproduces the fault (single operator → frozen replay → proxy model →
full model); do not rerun every gate for every change, but never skip the two human stop points.
Recover missing inputs from the source and the resolved config; ask only for what cannot be
recovered and changes the approach. Forward parity (the two-step loop), the gradient contract and
quality (the three-arm SFT), and performance are accepted separately, none standing in for another.
An inference-only task runs C0–C3, the engine part of C4 and C10, and states that the training loop
is not accepted. The trainer-side order of work and its stop conditions:
[training-side.md](training-side.md#call-kernels-do-not-port-the-scheduler).

## Gate table

### C0 Inventory

- Input: checkpoint, vLLM model file, molt model path (AutoModel or HF).
- Do: one row per module with the columns of the
  [module inventory](training-side.md#module-inventory).
- Pass when: every module on the logprob path (lm_head, norms, embeddings, encoders and both scorers
  included) has a row with every column filled and a forward verdict: already equal / swap trainer
  op / needs vLLM change.
- Keep: `inventory.md`.

### C1 Recipe + BI=0 baseline

- Input: model, context length, GPUs.
- Do: fix the recipe (data, lengths, batch, parallelism, precision, optimizer, offload); agree the
  scoring scope, consistency standard and performance budget; state whether θ0 bytes equal the
  released checkpoint; tune the BI=0 arm honestly over its real knobs
  ([acceptance.md](acceptance.md#paired-acceptance)) and measure several rounds.
- Pass when: the baseline is the recipe's own BI=0 config (routing replay, prefix caching, memory
  split as shipped), not a serving-only optimum; resolved config and throughput recorded; **stop:
  human freezes the baseline** ([audit materials](#two-tracks-and-the-human-stop-points)).
- Keep: launch script, resolved args, image digest, tuning table (every config tried and why).

### C2 Proxy model

- Input: full config.
- Do: cut a 2–6 layer model that keeps every layer type, real hidden/head/expert geometry, the
  divisibility the parallel layout needs, the state paths (KV, SSM state) and the layer adjacency
  that matters; export it as HF weights; it is the daily development environment.
- Pass when: a coverage table (feature → proxy layer → test) lists what the proxy does not exercise.
- Keep: proxy checkpoint, coverage table.

### C3 Engine invariance

- Input: proxy, fixed prompts.
- Do: vLLM alone with the aligned settings (a standalone script must reproduce molt's engine
  environment, [standalone engine](molt-integration.md#engine-environment-and-worker-hook)): same
  request alone vs. with neighbours
  (batch invariance); full prefill vs. chunked prefill vs. token-by-token decode (prefill/decode
  invariance); prefix cache hit vs. miss; a mixed prefill + decode step; CUDA graph vs. eager. Judge
  each engine kernel's fit for long prefill and training token counts before the trainer adopts it.
- Pass when: byte-identical logprobs in every cell; proves consistency inside inference only.
- Keep: per-cell mismatch counts, minimal failing replay.

### C4 Structural performance

- Input: proxy on real operator shapes; the aligned engine first, then the aligned training forward
  once C5 has wired it — C4 starts on the engine and re-runs after C5 (a forward reference, not a
  cycle).
- Do: profile prefill, decode and the training forward: launches vs tokens, host syncs and
  synchronous readbacks, repeated copies, quantization or allocation, abnormal workspace, whether
  the target CUDA-graph mode captures and replays; remove the structural defects.
- Pass when: no per-token launch or synchronous readback in prefill or the training forward
  (decode's one step per token does not count), the graph captures, a cost estimate within the
  agreed budget, and the consistency cells still pass after each fix.
- Keep: profile, defect list, cost estimate.

### C5 Training forward ↔ rollout

- Input: proxy, fixed weights, a saved rollout ([dumps and replay](molt-integration.md#dumps-and-replay)).
- Do: the trainer recomputes the saved tokens on both paths of the
  [dual-path contract](evidence.md#dual-path-contract): (a) the no-grad old-logprob path
  (`experience_maker.make_experience`), (b) the grad-enabled actor forward used in the loss (the
  live `mismatch/*` metrics measure this path); both against the saved rollout logprobs;
  same-packing A/A (the same input twice), then different-packing A/C (the same sequences with
  other neighbours or order).
- Pass when: `mismatch/bitwise_frac = 0` on both paths; valid tokens > 0; the executed
  backend/class/dtype recorded. C3, a single-operator test or a diagnostic substitute path never
  stands in for this gate.
- Keep: compare report, dump files.

### C6 Backward contract

- Input: C5 forward.
- Do: reuse the native backward with one reuse / glue / new verdict per module
  ([training-side.md](training-side.md#backward-reuse-glue-or-new)); write the gradient contract
  per operator (which weight dX uses, STE where rounding exists, discrete ids fixed, VJP
  collectives, saved state).
- Pass when: every module has a verdict and a written contract; every trainable parameter has a
  finite gradient. A forward 0 diff or finite gradients do not stand in for C7.
- Keep: backward inventory, gradient contract.

### C7 Gradient validation

- Input: C6 contract.
- Do: compare dX/dW and one optimizer step against a reference that does not call the backward
  under test, tolerances fixed first, bitwise and tolerance reported apart; plant a scaled error
  and confirm the check fails ([planted error](#evidence)); when quality is in scope run the
  three-arm proxy SFT ([training-side.md](training-side.md#gradient-contract-and-validation)).
- Pass when: the independent comparison passes within the fixed tolerances and the planted error
  fails; without SFT the record says `quality not covered` (a falling loss is not quality).
- Keep: gradient report, SFT curves.

### C8 Two-step RL on the proxy

- Input: proxy, real `train_rl_ray.py` entry.
- Do: θ0 rollout → compare → update → θ1 rollout → compare → update → θ2 rollout → compare
  ([sequence](#two-step-rl-sequence)), strictly synchronous, non-degenerate advantages, master
  weights actually change.
- Pass when: all three compares `mismatch = 0`, two effective updates, clean exit. A proxy scoped
  to forward parity marks this row `not run`, never `pass`.
- Keep: step log, metrics.

### C9 Recipe review → full model

- Input: launch script, image digest, commits, resolved config, differences from the shipped
  recipe with reasons, proxy evidence.
- Do: **stop: human reviews the recipe** ([materials](#two-tracks-and-the-human-stop-points));
  fill the [entry-gate table](#entry-gate-to-the-full-model); then grow one axis at a time (depth,
  length, concurrency, parallelism, offload) and rerun C3/C5/C8 at each step; check the first step
  after a resume. A fault that needs real depth or topology may be chased on the full model early
  only with a written hypothesis, minimal scale and stop condition.
- Pass when: no escalation while C5 is `fail`, `inconclusive` or without artifact; the full model
  repeats C8 at the target length and layout; a full-model θ0 failure stops updates and shrinks to
  a replay.
- Keep: entry-gate table, review record.

### C10 Performance + final E2E

- Input: aligned full model.
- Do: profile the aligned arm, rank hotspots by time share and optimise them (E2E profile → micro
  bench → paired E2E, [acceptance.md](acceptance.md#performance)); regress bitwise and the affected
  gradients after each change.
- Pass when: final image/commit: θ0/θ1/θ2 bitwise through the real entry; paired quality level;
  performance within the agreed budget; the run stops there — no extra steps or gates.
- Keep: report.

## Entry gate to the full model

Before any full-model correctness or training run the evidence below is in hand for the current
candidate ([ledger](evidence.md#candidate-evidence-ledger)); missing or not executed is not passed,
and repeated full-model launches do not substitute. The gate escalates, it does not complete: two
updates prove no convergence and the proxy no depth, length or topology. Before any multi-step
full-model RL a zero-update θ0 compare on the real workload comes first.

| Evidence | Minimum |
|---|---|
| Proxy coverage (C2) | Operator types, adjacency, state and parallel paths listed; uncovered items named |
| Engine invariance (C3) | Every cell bitwise on the proxy; proves inference only — a C3 pass says nothing about C5 |
| Structural performance (C4) | Proxy profile on real shapes: no structural defect left; cost estimate within the budget |
| Training forward (C5) | Both paths on the same proxy, weights and fixed tokens; per path valid tokens, mismatch, executed backend |
| Backward and quality (C6–C7) | Verdict and contract per module; independent dX/dW comparison with tolerances and raw results; SFT per scope, or `quality not covered` |
| Real update (C8) | θ0/θ1/θ2 at 0, two effective updates, clean exit |

## Cheapest failure criterion first

1. Host metadata and shape arithmetic (pure functions, boundary cases).
2. One operator: same input, real dtype and layout, repeat, cross-process, graph vs eager.
3. Frozen rollout replay: the original failing A/C and an A/A control, all valid tokens.
4. Engine: batch composition, prefill/decode, chunk boundaries, cache, graph, parallel layout.
5. Real training: non-degenerate advantages, finite non-zero gradients, optimizer step, parameter
   change, alignment after the update.
6. Production: full model, agreed steps, resume, quality, paired performance.

When a rerun adds no localization information, change the probe. A fault that disappears when the
reproducer is shrunk is not fixed; keep the full-size reproduction. Not every task runs every tier:
report the tiers completed, and a diagnosis-only task starts no training. A backward fault is not
localized by replaying the forward — save checkpoint, batch, RNG and state and build a minimal
backward experiment.

## Two-step RL sequence

```
θ0 → rollout0 → trainer(θ0) recompute, compare bytes
   → backward / optimizer step → θ1
θ1 → rollout1 → trainer(θ1) recompute, compare bytes
   → backward / optimizer step → θ2
θ2 → rollout2 → trainer(θ2) recompute, compare bytes
```

Compare only within one weight version (after a refit, before the update). With several minibatch
updates per round, do not demand 0 between the updated actor and the old-θ rollout, and do not alter
the algorithm to manufacture equality. Per stage record the weight version, token/mask counts and the
byte comparison; per update record finite loss, gradient coverage, master update norm, optimizer step.
The θ2 compare needs no third update. A small fp32 master step may not cross a bf16 rounding
boundary, so the deployed bytes need not change every step: judge an update by master update norm
and optimizer step, never by step count or byte change.

Acceptance runs this sequence through `molt/cli/train_rl_ray.py` on the delivered commit and image
([acceptance.md](acceptance.md#two-step-acceptance-through-the-real-entry-point)), synchronous as in
the [synchronous run](molt-integration.md#synchronous-run) of molt-integration.md: commits, image
digest and resolved config frozen first; the formal recipe shortened only in the number of updates —
no smaller data, length, `--rollout.n_samples_per_prompt`, batch, optimizer, parallelism or backend; a
fixed-weight replay does not count. After the θ2 compare and the agreed performance pass the run
stops: no extra steps, long-run comparisons or new gates, and no extrapolation to long-run quality.

## Two tracks and the human stop points

The BI=0 baseline (C1) and the consistency work (C2–C8 on the proxy) run in parallel: do not wait
for the audit to start the proxy, and do not let parallel progress open the full model before C9.
The tracks share model, precision, hardware and image; code, environment and artifacts stay apart,
alignment patches stay out of the BI=0 arm, and no formal benchmark runs while the other track uses
the node's GPUs, CPU or I/O. The final gap is accepted against the human-confirmed baseline only.

- **After C1 — baseline audit.** Re-audit after a model, precision, hardware or image change; release
  nodes while waiting. Hand over:
  - model and weight version; GPUs and nodes; image digest; molt / vLLM / nemo-automodel commits;
  - launch and bench commands; precision, parallel layout, scheduling, graph/cache state and the
    executed kernels; proof that alignment patches leave BI=0 bytes unchanged;
  - the tuning table (configs tried and why, warmup, repeats, throughput and its variation, failed
    and uncovered items);
  - the config proposed for freezing with the pairing statistic, repeats, per-`c` criterion and
    budget; open questions.
- **Before C9 — recipe review.** Hand over: launch script, image digest, commits, full resolved
  config; every difference from the shipped recipe with its reason (data and reward, lengths,
  `--rollout.n_samples_per_prompt`, batch and microbatch, parallelism, KV dtype, optimizer); proxy
  results and uncovered items; the acceptance scope — zero-update compare, two updates, and which
  quality, resume and long-run checks are required or out of scope. The reference recipe's model,
  data and evaluation set are not defaults; `--data.prompt_dataset` and `--eval.dataset` stay
  distinct. A change to the workload or to numerical/update semantics is reviewed again; a path or
  log fix is only recorded. The review blocks neither authorized builds nor proxy work.

## Evidence

Labels: `pass`, `fail`, `inconclusive` (did not enter the target path, zero hits, GPU fault),
`not run`. A result belongs to one execution closure: model geometry, weights, source/image,
precision, parallel layout, graph/eager, cache state. Changing any of these invalidates the row;
old passes stay in the ledger as history and are not copied into the current table.

Freeze the criterion before the run — window, tolerance, denominator, pass line, its hash recorded —
and plant a known error: a flipped bit in a byte comparison, one module's gradients scaled ×1.25 and
×1.05, a known-wrong scale rule. The check must read `fail`; a check that cannot resolve ×1.05 is
useless, and no threshold moves after the result is seen.

False greens to check before reporting a pass:

- zero valid tokens; mask all false; the comparison never executed; a rank or sample missing;
- a KL, probability or log-ppl of 0 without a byte comparison
  ([object of comparison](contracts.md#object-of-comparison)); `mismatch/*` absent is not 0
  ([metrics](molt-integration.md#mismatch-metrics));
- the trainer skipped the forward (zero-advantage batch) and reported 0;
- gradient zero or all rewards equal: no effective update happened;
- small model or short output passes, full depth/length never run;
- many identical siblings in a rollout group (a sampling-seed defect, not consistency);
- the job exited 0 but the recipe's own exit code or skipped-test count says otherwise; a success
  marker written before the process died — a finished numerical experiment, a clean exit and
  long-run stability are three cells;
- a proxy's mismatch magnitude read as the full model's
  ([reference measurements](molt-integration.md#reference-measurements));
- a diagnostic switch (eager, substitute attention, graph off) was on: it proves that path only.

Full rules: [evidence.md](evidence.md).

## Report

Use the template in [acceptance.md](acceptance.md#report-template).
