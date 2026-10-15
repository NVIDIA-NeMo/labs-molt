# Workflow: gates G0–G8

Turn "rollout and trainer disagree" into a repeatable experiment with fixed weights and fixed
tokens, find the first divergence, fix it, then return to the real training loop. Iterate at the
cheapest level that still reproduces the fault (single operator → frozen replay → proxy model →
full model); do not rerun every gate for every change, but never skip the two human stop points.

## Gate table

| Gate | Input | Do | Pass when | Keep |
|---|---|---|---|---|
| **G0 Inventory** | checkpoint, vLLM model file, molt model path (AutoModel or HF) | For every operator class: the vLLM kernel in prefill and in decode, the trainer kernel, accumulation dtype, discrete choices (top-k, routing, tie-break), whether a same-math backward exists; the two scorers (vLLM sampler logprob vs molt `log_probs_from_logits`) | Table covers all modules incl. lm_head, norms, embeddings, encoders; every row has a verdict: already equal / swap trainer op / needs vLLM change | `inventory.md` |
| **G1 Recipe + BI=0 baseline** | model, context length, GPUs | Fix the recipe (data, lengths, batch, parallelism, precision, optimizer, offload); tune the BI=0 arm honestly (`--vllm.attention_backend`, `--vllm.enforce_eager`, `--vllm.max_num_batched_tokens`, `--vllm.gpu_memory_utilization`, `--train.async_queue_size`); measure several rounds | Resolved config and throughput recorded; **stop: human freezes the baseline** | launch script, resolved args, timings |
| **G2 Proxy model** | full config | Cut a 2–6 layer model that keeps every layer type, real hidden/head/expert geometry, the state paths (KV, SSM state) and the layer adjacency that matters; export it as HF weights | Coverage table lists what the proxy does not exercise | proxy checkpoint, coverage table |
| **G3 Engine invariance** | proxy, fixed prompts | vLLM alone with the aligned settings (a standalone script must reproduce molt's engine environment: `_configure_vllm_env` and the worker hook): same request alone vs. with neighbours (batch invariance), full prefill vs. chunked prefill vs. token-by-token decode (prefill/decode invariance), prefix cache hit vs. miss, CUDA graph vs. eager | Byte-identical logprobs in every cell; proves consistency inside inference only | per-cell mismatch counts |
| **G4 Training forward ↔ rollout** | proxy, fixed weights, saved rollout | The trainer recomputes the saved tokens: (a) the no-grad old-logprob path (`experience_maker.make_experience`; skipped when `--train.force_on_policy` is combined with `--algo.kl.init_coef 0` — then run G4 without `force_on_policy` or mark (a) `not run`), (b) the grad-enabled actor forward actually used in the loss (the live `mismatch/*` metrics measure this path); both against the saved rollout logprobs; same-packing A/A (the same input twice), then different-packing A/C (the same sequences with other neighbours or order) | `mismatch/bitwise_frac = 0` on both paths; valid tokens > 0; the executed backend/class/dtype recorded | compare report, dump files |
| **G5 Backward** | G4 forward | Reuse native backward; write the gradient contract (which weight dX uses, STE where rounding exists, discrete ids fixed); compare dX/dW and one optimizer step against a reference that does not call the backward under test, with tolerances fixed first; plant a scaled error and confirm the check fails | Independent comparison passes; every trainable parameter has a finite gradient | gradient report |
| **G6 Two-step RL on the proxy** | proxy, real `train_rl_ray.py` entry | θ0 rollout → compare → update → θ1 rollout → compare → update → θ2 rollout → compare, strictly synchronous, non-degenerate advantages, master weights actually change | all three compares `mismatch = 0`, two effective updates, clean exit | step log, metrics |
| **G7 Recipe review → full model** | launch script, image digest, commits, resolved config, proxy evidence | **stop: human reviews the recipe**; then grow one axis at a time (depth, length, concurrency, parallelism, offload) and rerun G3/G4/G6 at each step; check the first step after a resume. A fault that needs real depth or topology may be chased on the full model early only with a written hypothesis, minimal scale and stop condition | Full model repeats G6 at the target length and layout | entry-gate table |
| **G8 Performance + final E2E** | aligned full model | Profile the aligned arm, rank hotspots by time share, fix the structural ones (per-token launches, unbatched loops, repeated copies), regress bitwise after each change; run the paired protocol in [acceptance.md](acceptance.md) | Final image/commit: θ0/θ1/θ2 bitwise through the real entry; paired quality level; performance within the agreed budget | report |

## Cheapest failure criterion first

1. Host metadata and shape arithmetic (pure functions, boundary cases).
2. One operator: same input, real dtype and layout, repeat, cross-process, graph vs eager.
3. Frozen rollout replay: the original failing A/C and an A/A control, all valid tokens.
4. Engine: batch composition, prefill/decode, chunk boundaries, cache, graph, parallel layout.
5. Real training: non-degenerate advantages, finite non-zero gradients, optimizer step, parameter
   change, alignment after the update.
6. Production: full model, agreed steps, resume, quality, paired performance.

When a rerun adds no localization information, change the probe. A fault that disappears when the
reproducer is shrunk is not fixed; keep the full-size reproduction.

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

## Evidence

Labels: `pass`, `fail`, `inconclusive` (did not enter the target path, zero hits, GPU fault),
`not run`. A result belongs to one execution closure: model geometry, weights, source/image,
precision, parallel layout, graph/eager, cache state. Changing any of these invalidates the row;
old passes stay in the ledger as history and are not copied into the current table.

False greens to check before reporting a pass:

- zero valid tokens; mask all false; the comparison never executed; a rank or sample missing;
- probability, KL or log-ppl is zero but the bytes were not compared; `mismatch/*` missing from the
  log (rollout logprobs not requested) is not 0; `vllm_kl` ≈ 0 is not bitwise;
- the trainer skipped the forward (zero-advantage batch) and reported 0;
- gradient zero or all rewards equal: no effective update happened;
- small model or short output passes, full depth/length never run;
- many identical siblings in a rollout group (a sampling-seed defect, not consistency);
- the job exited 0 but the recipe's own exit code or skipped-test count says otherwise;
- a diagnostic switch (eager, substitute attention, graph off) was on: it proves that path only.

Full rules — what must not be confused, the experiment and dual-path contracts, the candidate
ledger — are in [evidence.md](evidence.md).

## Report

Use the template in [acceptance.md](acceptance.md#report-template).
