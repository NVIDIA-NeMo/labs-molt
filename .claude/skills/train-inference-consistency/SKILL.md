---
name: train-inference-consistency
description: Make molt's training forward produce bitwise-identical per-token logprobs to vLLM's rollout (true on-policy RL) for a given model, validate it with the gate workflow, diagnose train/rollout mismatch, and recover throughput. Invoke for any "mismatch/bitwise_frac is not 0", "vllm_kl is not 0", "batch-invariant", "aligned kernels", or new-model consistency task.
---

# Train–inference consistency

Goal: for every valid response token, the logprob the trainer recomputes under the same weights
equals the logprob vLLM returned when it sampled the token — same bytes in the original dtype,
`mismatch/bitwise_frac = 0`, at every RL step. Then bring throughput back toward the BI=0 baseline.

Method, in one sentence: the training forward calls vLLM's own kernels for every operator that
produces a value (inference defines the numbers), and the backward reuses the native math of the
training stack (the gradient only has to satisfy a written contract, not be bitwise). This is Molt's
take on the idea, inspired by the vLLM blog post [Finish RL Training Faster: Bitwise-Consistent,
High-Performance, Agent-Driven](https://vllm-project.github.io/2026/10/14/train-inference-consistency.html) (verl + Megatron) and rebuilt for molt's stack (FSDP2 +
AutoModel + vLLM); its gates C0–C10 correspond roughly to G0–G8 here.

Where the code is: `main` has the measurement side — the `mismatch/*` metrics, rollout dump and
replay, the engine and trainer flags in molt-integration.md. The aligned kernels behind
`--train.batch_invariant` exist only on branch `jianh/batch-invariant` (vLLM 0.28, Qwen2/Qwen3). On
`main` the task is therefore a port along training-side.md, never a flag flip; when that branch's
vLLM pin differs from the installed one, say so before porting anything.

## Read in this order

| File | Use it for |
|---|---|
| [workflow.md](workflow.md) | The gates G0–G8 and their pass criteria, the two-step RL sequence, evidence labels and false greens |
| [contracts.md](contracts.md) | What "equal" means per operator, discrete decisions, kernel-level contracts, the weight/precision chain, the forward/backward boundary |
| [molt-integration.md](molt-integration.md) | Where consistency lives in molt (flags, metrics, engine settings, patch points), the reference aligned mode and its H100 operator table, adding an architecture |
| [training-side.md](training-side.md) | Building the aligned trainer forward (module inventory, call vLLM kernels, layout and world size) and its backward (reuse / glue / new, gradient contract, validation layers) |
| [debugging.md](debugging.md) | Symptom → first step, isolate the side, save/compare/replay a fixed rollout, bisect to the first diverging operator, classify, engine pitfalls |
| [acceptance.md](acceptance.md) | One recipe with BI off and on, evaluation contract, paired quality and performance statistics, checkpoint/resume/memory, provenance, delivery, report template |
| [evidence.md](evidence.md) | Evidence labels, what must not be confused, experiment and dual-path contracts, the candidate ledger, measurement matrix, exceptions, failure records |

Entry points: a new model starts at G0 with the module inventory in training-side.md; a live run
whose `mismatch/bitwise_frac` is not 0 starts in debugging.md; a delivery starts in acceptance.md.

## Rules that hold everywhere

- Compare logprobs as bytes (`bf16.view(int16)`, `fp32.view(int32)`): the live signal is
  `mismatch/bitwise_frac`; `vllm_kl`, any KL, a probability difference or identical printed numbers
  prove nothing. A dtype difference between the two sides
  is a contract violation, not something to cast away. A NaN/Inf at a valid position fails even
  when both sides agree; a ±1 position shift is a bug in the comparison, never a tolerance.
- Measure alignment synchronously: `--train.force_sync_mode`, queue 1, one epoch, no partial rollout
  (the synchronous-run row of molt-integration.md). Under any lag ≥ 1 the rollout came from older
  weights and `mismatch/bitwise_frac` reads ≈ 1 whatever the kernels; an asynchronous recipe is
  accepted on quality and performance and reads the lag floor, never 0.
- Zero valid tokens, a skipped forward, a path not executed, or a metric that is 0 by construction
  (`policy_kl` under `--train.force_on_policy` with `--algo.kl.init_coef 0`) are not evidence. Check the
  producer before trusting a 0.
- Keep the user's model, precision, parallel layout and target; the scope of a conclusion is the
  scope that was executed (geometry, length, batch composition, graph/eager, parallelism).
- Never change tokens, weights, geometry, precision or tolerances to make a comparison pass.
- Inference defines the value. Do not detach an inference output and train on it, and do not
  rewrite arithmetic to imitate a kernel; call the kernel.
- Human stop points: freeze the BI=0 baseline (after G1) and review the full-model recipe
  (before G7). Do not escalate GPU count or scope on your own.
- Code changes follow `simplicity-first`: fewest lines, no new abstractions for one call site,
  knobs default off keep the BI=0 path byte-identical to before.
