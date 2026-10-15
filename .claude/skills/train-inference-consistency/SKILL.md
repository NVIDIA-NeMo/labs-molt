---
name: train-inference-consistency
description: Make molt's training forward produce bitwise-identical per-token logprobs to vLLM's rollout (true on-policy RL) for a given model, validate it with the gate workflow (C0–C10), diagnose train/rollout mismatch, and recover throughput. Invoke for any "mismatch/bitwise_frac is not 0", "vllm_kl is not 0", "batch-invariant", "aligned kernels", or new-model consistency task.
---

# Train–inference consistency

Goal: for every valid response token, the logprob the trainer recomputes under the same weights
equals the logprob vLLM returned when it sampled the token — same bytes in the original dtype,
`mismatch/bitwise_frac = 0`, at every RL step. Then bring throughput back toward the BI=0 baseline.

Method, in one sentence: the training forward calls vLLM's own kernels for every operator that
produces a value (inference defines the numbers), and the backward reuses the native math of the
training stack (the gradient only has to satisfy a written contract, not be bitwise). Inspired by the
vLLM blog post [Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven][post]
(verl + Megatron), rebuilt for molt's stack (FSDP2 + AutoModel + vLLM); its gates C0–C10 are used
here one-to-one.

[post]: https://vllm-project.github.io/2026/10/14/train-inference-consistency.html

Where the code is: `main` has the measurement side — the `mismatch/*` metrics, rollout dump and
replay, the engine and trainer flags in molt-integration.md. The aligned kernels behind
`--train.batch_invariant` exist only on branch `jianh/batch-invariant` (vLLM 0.28, Qwen2/Qwen3). On
`main` the task is therefore a port along training-side.md, never a flag flip; when that branch's
vLLM pin differs from the installed one, say so before porting anything. Scope: consistency only —
day-0 model support, per-model case records and cluster notes are not part of this skill.

## Read in this order

| File | Use it for |
|---|---|
| [workflow.md](workflow.md) | The gates C0–C10 and their pass criteria, the two-step RL sequence, evidence labels and false greens |
| [contracts.md](contracts.md) | What "equal" means per operator, discrete decisions, kernel-level contracts, the weight/precision chain, the forward/backward boundary |
| [molt-integration.md](molt-integration.md) | Where consistency lives in molt (flags, metrics, engine settings, patch points), the reference aligned mode and its H100 operator table, adding an architecture |
| [training-side.md](training-side.md) | Building the aligned trainer forward (module inventory, call vLLM kernels, layout and world size) and its backward (reuse / glue / new, gradient contract, validation layers) |
| [debugging.md](debugging.md) | Symptom → first step, isolate the side, save/compare/replay a fixed rollout, bisect to the first diverging operator, classify, engine pitfalls |
| [acceptance.md](acceptance.md) | One recipe with BI off and on, evaluation contract, paired quality and performance statistics, checkpoint/resume/memory, provenance, delivery, report template |
| [evidence.md](evidence.md) | Evidence labels, what must not be confused, experiment and dual-path contracts, the candidate ledger, measurement matrix, exceptions, failure records |

Entry points: a new model starts at C0 with the module inventory in training-side.md; a live run
whose `mismatch/bitwise_frac` is not 0 starts at the [triage](debugging.md#triage) in debugging.md; a
delivery starts in acceptance.md.

Find by problem (live-mismatch symptoms: the first table of [debugging.md](debugging.md)):

| Problem | Read first |
|---|---|
| Setting the target or the evidence basis | [evidence.md](evidence.md#experiment-contract) |
| Fails only under graph, compile or cache | [debugging.md](debugging.md#execution-state), [evidence.md](evidence.md#execution-state) |
| Results change with packing, TP/CP/EP | [training-side.md](training-side.md#layout-length-and-world-size) |
| Wiring a backward, comparing gradients, SFT | [training-side.md](training-side.md#backward-reuse-glue-or-new) |
| Candidate, backend or image changed: does old evidence hold | [evidence.md](evidence.md#candidate-evidence-ledger) |
| Going to the full model, shrinking a failure, declaring completion | [workflow.md](workflow.md#entry-gate-to-the-full-model), [evidence.md](evidence.md#limited-exceptions-and-fallback) |
| Aligned; recovering performance, profiling, ablation | [acceptance.md](acceptance.md#performance) |
| Telling quality, fixed-workload speed and clean exit apart | [acceptance.md](acceptance.md#evaluation-contract) |
| Image, source, wheel or overlay versions confused | [acceptance.md](acceptance.md#environment-and-provenance) |
| Checkpoint OOM, first resume misaligned, host memory rising | [acceptance.md](acceptance.md#checkpoint-resume-and-memory) |
| BI=0 arm and the two modes in one recipe | [acceptance.md](acceptance.md#one-recipe-one-switch) |
| Many repeated samples in a rollout group | [debugging.md](debugging.md#sampling-seed-collision) |
| A difference that shows once in many steps | [debugging.md](debugging.md#replay-matrix) |
| Prefix cache hit ≠ miss, state layers under caching | [debugging.md](debugging.md#prefix-cache-semantics) |
| Writing or reviewing a kernel candidate (top-k, fusion, split-K, graph mode, BI cost) | [contracts.md](contracts.md#kernel-level-contracts) |
| Porting the aligned forward to a new model | [molt-integration.md](molt-integration.md#adding-an-architecture) |
| PRs, pins, image, delivery | [acceptance.md](acceptance.md#delivery-and-prs) |

## Rules that hold everywhere

- Compare logprobs as bytes; the live signal is `mismatch/bitwise_frac`. No KL, probability
  difference or printed number proves bitwise ([object of comparison](contracts.md#object-of-comparison)).
- Measure alignment synchronously with one optimizer step per rollout (`--train.batch_size` =
  `--rollout.batch_size × --rollout.n_samples_per_prompt`, `--train.max_epochs 1`, no partial rollout,
  [synchronous run](molt-integration.md#synchronous-run)); any lag ≥ 1 reads the bf16 floor, never 0
  ([layout rules](molt-integration.md#layout-rules)).
- A metric that is absent or 0 by construction is not evidence; check the producer before trusting
  a 0 ([mismatch metrics](molt-integration.md#mismatch-metrics)).
- Keep the user's model, precision, parallel layout and target; the scope of a conclusion is the
  scope that was executed (geometry, length, batch composition, graph/eager, parallelism).
- Never change tokens, weights, geometry, precision or tolerances to make a comparison pass.
- Inference defines the value. Do not detach an inference output and train on it, and do not
  rewrite arithmetic to imitate a kernel; call the kernel.
- Human stop points: freeze the BI=0 baseline (after C1) and review the full-model recipe
  (before C9). The skill grants no extra GPUs, job stops, image publishing or writes to other repos.
- Code changes follow `simplicity-first`: fewest lines, no new abstractions for one call site,
  knobs default off keep the BI=0 path byte-identical to before.

## Terms

| Term | Meaning |
|---|---|
| BI | batch invariance: a request's bytes do not depend on batch-mates; BI=0 / BI=1 = the switch off / on |
| IS | importance sampling: the per-token ratio π_old/μ the loss multiplies (`--algo.advantage.is_correction_*`) |
| R3 | rollout routing replay: the trainer replays the engine's expert ids (`--train.routing_replay`) |
| STE | straight-through estimator: a rounding passes its gradient through as identity |
| VJP | vector-Jacobian product: what a backward computes |
| THD | packed token layout (total tokens, heads, dim) with `cu_seqlens` boundaries |
| LSE | log-sum-exp, the softmax denominator attention saves for its backward |
| QAT | quantization-aware training: the forward sees `DQ(Q(w))` |
| AC | activation checkpointing (`--actor.gradient_checkpoint`) |
| DCP | torch distributed checkpoint, the format molt saves under `--ckpt.path` |
| PSS / RSS | proportional / resident set size (host memory; shared pages counted once vs per process) |
| GDN | gated delta net, a linear-attention state layer |
| MTP | multi-token prediction draft for speculative decoding (`--vllm.mtp_num_speculative_tokens`) |
| k3 | the `exp(d) − 1 − d` KL estimator behind `mismatch/k3_kl` |
| SSD | state-space duality: the chunked Mamba2 scan |
| TTFT / TPOT | time to first token / time per output token |
