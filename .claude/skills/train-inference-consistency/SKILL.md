---
name: train-inference-consistency
description: Make molt's training forward produce bitwise-identical per-token logprobs to vLLM's rollout (true on-policy RL) for a given model, validate it with the gate workflow (C0–C10), diagnose train/rollout mismatch, and recover throughput to within the 20 % rollout budget. Invoke for any "mismatch/bitwise_frac is not 0", "vllm_kl is not 0", "batch-invariant", "aligned kernels", or new-model consistency task.
---

# Train–inference consistency

Goal: for every valid response token, the logprob the trainer recomputes under the same weights equals the
logprob vLLM returned when it sampled the token — the same bytes in the original dtype,
`mismatch/bitwise_frac = 0`, at every RL step. Then bring rollout throughput back to within the
[performance budget](acceptance.md#performance-budget): the aligned arm loses at most 20 % of the BI=0
arm's rollout tokens/s.

Where the code is: `main` has the measurement side — the `mismatch/*` metrics, rollout dump and replay, the
engine and trainer flags in molt-integration.md. The aligned kernels behind `--train.batch_invariant` exist
only on branch `jianh/batch-invariant`, the reference implementation (vLLM 0.28; Qwen2/Qwen3 dense and
Qwen3-MoE). On `main` the task is therefore a port along training-side.md, never a flag flip. Read the
installed vLLM inside a worker ([which vLLM am I on](molt-integration.md#which-vllm-am-i-on)); when it
differs from the reference implementation's pin, say so before porting anything. Engine facts that depend
on the installed version sit in one table with their re-verify checks: [facts bound to the pinned
vLLM](molt-integration.md#facts-bound-to-the-pinned-vllm).

Method: the training forward calls vLLM's own kernels for every operator that produces a value, so inference
defines the numbers. The backward reuses the native math of the training stack under a written gradient
contract; it is not bitwise. Molt's take on the idea, inspired by the vLLM blog post [Finish RL Training
Faster: Bitwise-Consistent, High-Performance, Agent-Driven][post] (verl + Megatron): the gates C0–C10 match
it one-to-one so the two can be read together, and every rule here is written for FSDP2 + AutoModel + vLLM.
Scope: consistency only — day-0 model support, per-model case records and cluster notes are not part of this
skill.

[post]: https://vllm-project.github.io/2026/10/14/train-inference-consistency.html

## Start by task

Read the sections of your route in order; the rest of the skill is lookup material.

| Task | Read in order |
|---|---|
| New model, or `main` without the aligned mode (a port) | 1. [module inventory](training-side.md#module-inventory) (C0) 2. [adding an architecture](molt-integration.md#adding-an-architecture) 3. [call kernels, do not port the scheduler](training-side.md#call-kernels-do-not-port-the-scheduler) 4. [gate table](workflow.md#gate-table) C2–C8 |
| Live run whose `mismatch/bitwise_frac` is not 0 | 1. [triage](debugging.md#triage) 2. [symptom → first step](debugging.md#symptom--first-step) 3. [isolate the side first](debugging.md#isolate-the-side-first) 4. [save and compare protocol](debugging.md#save-and-compare-protocol) |
| Delivery or acceptance of an aligned model | 1. [one recipe, one switch](acceptance.md#one-recipe-one-switch) and its [recipe deltas](acceptance.md#acceptance-recipe-as-deltas) 2. [synchronous run](molt-integration.md#synchronous-run) 3. [two-step RL sequence](workflow.md#two-step-rl-sequence) and [false greens](workflow.md#evidence) 4. [report template](acceptance.md#report-template) |

An inference-only or forward-parity task runs a subset of the gates: [task scope](workflow.md#task-scope).

## Files

| File | Open when | Jump to |
|---|---|---|
| [workflow.md](workflow.md) | Running or scoping gates C0–C10 | [task scope](workflow.md#task-scope), [entry gate to the full model](workflow.md#entry-gate-to-the-full-model), [verification ladder](workflow.md#cheapest-failure-criterion-first), [two tracks and stop points](workflow.md#two-tracks-and-the-human-stop-points) |
| [contracts.md](contracts.md) | Deciding what "equal" means for an operator, a weight or a backward | [per-operator contract](contracts.md#per-operator-contract), [discrete decisions](contracts.md#discrete-decisions), [kernel changes](contracts.md#kernel-changes-rules-for-an-operator-author), [weights and precision chain](contracts.md#weights-and-precision-chain), [forward/backward boundary](contracts.md#shared-forward-and-backward-boundary) |
| [molt-integration.md](molt-integration.md) | Looking for a flag, metric, engine setting or patch point | [where consistency lives](molt-integration.md#where-consistency-lives-in-molt-main), [the aligned mode](molt-integration.md#the-aligned-mode), [operator table](molt-integration.md#operator-table), [pinned vLLM facts](molt-integration.md#facts-bound-to-the-pinned-vllm), [reference measurements](molt-integration.md#reference-measurements) |
| [training-side.md](training-side.md) | Building the aligned forward or its backward | [layout, length and world size](training-side.md#layout-length-and-world-size), [backward: reuse, glue or new](training-side.md#backward-reuse-glue-or-new), [gradient contract and validation](training-side.md#gradient-contract-and-validation) |
| [debugging.md](debugging.md) | A mismatch, a stall, a seed collision, a cache- or graph-only fault | [bisect and first divergence](debugging.md#bisect-and-first-divergence), [classification](debugging.md#classification-and-handling-order), [execution state](debugging.md#execution-state), [rollout engine pitfalls](debugging.md#rollout-engine-pitfalls), [replay matrix](debugging.md#replay-matrix) |
| [acceptance.md](acceptance.md) | C1, C9–C10, performance, provenance, PRs, the final report | [evaluation contract](acceptance.md#evaluation-contract), [paired acceptance](acceptance.md#paired-acceptance), [performance](acceptance.md#performance), [checkpoint, resume and memory](acceptance.md#checkpoint-resume-and-memory), [delivery and PRs](acceptance.md#delivery-and-prs) |
| [evidence.md](evidence.md) | Declaring a pass or reusing an old result | [evidence labels](evidence.md#evidence-labels), [experiment contract](evidence.md#experiment-contract), [dual-path contract](evidence.md#dual-path-contract), [candidate evidence ledger](evidence.md#candidate-evidence-ledger), [limited exceptions](evidence.md#limited-exceptions-and-fallback) |

## Rules that hold everywhere

- Compare logprobs as bytes; the live signal is `mismatch/bitwise_frac`. No KL, probability difference or
  printed number proves bitwise ([object of comparison](contracts.md#object-of-comparison)).
- Measure alignment synchronously, one optimizer step per rollout ([synchronous
  run](molt-integration.md#synchronous-run)); any lag ≥ 1 reads the lag floor, never 0 ([layout
  rules](molt-integration.md#layout-rules)).
- A metric that is absent or 0 by construction is not evidence; check the producer before trusting a 0
  ([mismatch metrics](molt-integration.md#mismatch-metrics)).
- Keep the user's model, precision, parallel layout and target; a conclusion's scope is the executed closure
  (geometry, length, batch composition, graph/eager, parallelism).
- Never change tokens, weights, geometry, precision or tolerances to make a comparison pass.
- Inference defines the value: never detach an inference output and train on it, and never rewrite
  arithmetic to imitate a kernel — call the kernel.
- Performance budget: the BI=1 arm's rollout deficit `1 − T_BI1/T_BI0` is at most 20 % at every formal `c`
  ([performance budget](acceptance.md#performance-budget)); over budget is `fail`, and only a human
  tightens or waives it, in writing.
- Human stop points: freeze the BI=0 baseline (after C1) and review the full-model recipe (before C9). Ask
  before requesting GPUs, stopping jobs, publishing images or writing to other repos.
- Code changes follow `simplicity-first`; with the switch off, the BI=0 path stays byte-identical to before.

## Terms

One sense per term, used in every file.

| Term | Meaning |
|---|---|
| A/A | the same input scored twice on one side: run-to-run determinism |
| A/C | the same sequences scored with other neighbours, order or packing: batch and layout invariance |
| AC | activation checkpointing (`--actor.gradient_checkpoint`) |
| aligned mode | the `--train.batch_invariant` path: invariant engine kernels plus the aligned trainer forward; the BI=1 arm runs it |
| arm | one side of a paired run: the BI=0 arm and the BI=1 arm of the recipe; the three SFT arms of C7 |
| BI | batch invariance: a request's bytes do not depend on batch-mates; BI=0 / BI=1 = the switch off / on |
| BI=0 baseline | the human-frozen C1 measurement of the BI=0 arm (config, throughput, mismatch) |
| `c` | concurrent requests per engine in the fixed-workload bench |
| candidate | the build under evaluation — code, image, resolved config; one ledger row per closure |
| closure | execution closure: the fields that bound a result — GPU architecture, geometry, weights, source/image, precision, layout, graph/eager, cache state |
| DCP | torch distributed checkpoint, the format molt saves under `--ckpt.path` |
| deficit | the paired rollout throughput loss `1 − T_BI1/T_BI0` at one `c`; the budget caps it at 20 % |
| false green | a `pass` that is not evidence: nothing executed, 0 by construction, the wrong path ran |
| IS | importance sampling: the per-token ratio π_old/μ the loss multiplies (`--algo.advantage.is_correction_*`) |
| k3 | the `exp(d) − 1 − d` KL estimator behind `mismatch/k3_kl` |
| lag floor | the mismatch of a rollout sampled under older weights (`bitwise_frac` ≈ 1); no kernel removes it |
| ledger | the candidate evidence ledger: one row per closure; old passes stay as history |
| LSE | log-sum-exp, the softmax denominator attention saves for its backward |
| MTP | multi-token prediction draft for speculative decoding (`--vllm.mtp_num_speculative_tokens`) |
| oracle | an independent forward or gradient reference that does not call the code under test |
| planted error | a known error injected to prove that a check reads `fail` |
| proxy | the C2 model: 2–6 real layers of the target with every layer type kept; the daily development environment |
| QAT | quantization-aware training: the forward sees `DQ(Q(w))` |
| R3 | rollout routing replay: the trainer replays the engine's expert ids (`--train.routing_replay`); not a rollout replay, not a decode kernel |
| receipt | the per-worker or per-θ record of what executed: counts, hashes, exit code |
| reference implementation | branch `jianh/batch-invariant`, the aligned mode this skill was derived from |
| refit | molt's weight broadcast trainer → engine after an update (`broadcast_to_vllm`) |
| replay | rollout replay: training on a saved rollout without sampling (`--train.rollout_replay_dir`) |
| scorer | the kernel that turns logits into the selected-token logprob: two on the trainer, one in the engine |
| STE | straight-through estimator: a rounding passes its gradient through as identity |
| THD | packed token layout (total tokens, heads, dim) with `cu_seqlens` boundaries |
| VJP | vector-Jacobian product: what a backward computes |
