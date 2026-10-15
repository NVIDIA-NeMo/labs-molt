# Recipe, arms and paired acceptance

One recipe with the aligned mode off and on, one evaluation basis, paired statistics, performance on a fixed
workload, resume proven, environment pinned, a report of what was and was not verified. Gates and false
greens: [workflow.md](workflow.md); flags and patch points: [molt-integration.md](molt-integration.md).

## One recipe, one switch

Both arms share launch script, image digest, data, lengths, batch, parallel layout, optimizer and seeds;
the only difference is `--train.batch_invariant` and what it implies (engine kernels, replay, correction).

| Arm | Engine | Trainer | Correction | Expected mismatch |
|---|---|---|---|---|
| BI=0 (baseline) | default kernels, graphs on | default `BackendConfig` | the recipe's IS / gate / replay as shipped | the recipe's measured `mismatch/bitwise_frac` (≈ 1 under default kernels) and `abs_diff_mean` — record them as the BI=0 baseline |
| BI=1 (aligned) | invariant kernels, `TRITON_ATTN`, replay on for MoE | aligned forward | `--algo.advantage.is_correction_level token --algo.advantage.is_correction_threshold 0 1e9`, kept on so rollout logprobs and the `mismatch/*` metrics exist (its ratio weight is exactly 1 once bitwise) | `mismatch/bitwise_frac` 0 at every step of the synchronous run; the lag floor when asynchronous |

- Separate code paths: BI=0 is the default molt path with nothing of the aligned implementation injected;
  "the BI=1 actor with default kernels swapped in" is neither baseline nor true on-policy. Consistency
  patches never change BI=0 bytes: replay one saved rollout before and after the patch with the switch
  off — identical int-view logprobs and throughput.
- Resolved command byte-identical except the switch: emit the parsed arguments (not shell variables) for
  both arms and `diff`; BI=1 differs from the shipped recipe only in the switch, BI=0 only in names, paths
  and new explicit switches. Same image digest, same `--train.seed`, same data order.
- Separate `--ckpt.path` and `--ckpt.output_dir` per arm: `--ckpt.load_enable` resumes from
  `<--ckpt.path>/_actor/latest`, so a shared path resumes onto the other arm. The run name plays no part in the resume.
- The recipe header states only what was measured to hold; known engine-config differences between the
  arms go into the comparison report; untested switches are deleted.

BI=0 integration pitfalls, re-checked per model: a default that silently changes with the switch (attention
backend, `enforce_eager`, prefix caching, `logprobs_mode`, compile mode) — resolve every default explicitly
so the diff shows only the switch; a cache or graph setting that differs (CUDA-graph mode, prefix cache,
compile cache) — same in both arms or a quantified confounder; a metric that exists in one arm only
(`mismatch/*` need rollout logprobs: `is_correction_level` ≠ off or score centering; a blank cell gets
read as 0) — request rollout logprobs in both arms every step; on `main` any request path applies the per-token IS
weight to the BI=0 arm too, so declare it as a confounder or use the recipe's own correction in both arms; a packing path that differs (`cu_seqlens`, THD style, HybridEP
padding) — a stateful layer scans the pack as one sequence; training-only model defaults (router bias
update, MTP head, frozen modules) — pinned for both arms, trainable set listed.

Smoke, both arms at proxy scale (a few updates): finite loss and gradients, non-degenerate advantages, refit
bytes equal (same-tensor hash both sides after `load_weights`), exit code 0 from the recipe, not the
launcher. BI=1: `mismatch/bitwise_frac` 0 over all valid tokens every step. BI=0: `bitwise_frac` near 1 and
`abs_diff_mean` at the kernel-noise magnitude, grad norm finite. Grad norm exactly 0 means same-reward
groups before numerics. Skipped forward or zero valid tokens ⇒
`inconclusive`.

## Evaluation contract

One artifact answers one question: bitwise logprobs on fixed tokens prove scoring of one token history,
natural generation answers quality, fixed workloads answer throughput. Six items; any difference means a
different score:

1. Same prompts and prompt format: dataset version and `--eval.split`, chat template, thinking mode.
2. Same decoding settings: `--eval.temperature`, `--eval.top_p`, `--eval.max_new_tokens`,
   `--eval.n_samples_per_prompt`, stop tokens; greedy n=1, sampled mean and pass@k are different scores;
   pass@1 at n=8 is not greedy accuracy; training cap and eval cap may differ.
3. Per-problem identity under greedy: the aligned arm reproduces the engine's answer tokens for the same
   prompt and weight version; report the number of differing problems, not only the score.
4. Sampled scores paired: same problem ids, seed, n, scorer, no retries; per problem record
   `both_correct / only_bi0 / only_bi1 / both_wrong`, summing to the problem count.
5. Statistical variation reported: seeds, repeats, noise floor, agreed threshold.
6. Unfinished parts listed: not evaluated is `not run`, never implied.

Under greedy the score is a multiple of 1/N problems (30 problems: 3.33 points each); less than one problem
is invisible. Rising training reward, mismatch 0 and a held-out gain are three conclusions; a close score is
not per-problem identity; quality passing does not make logprobs bitwise; a relaxed token budget is not a
training gain. Report model and weight version, problem count, truncation ratio next to accuracy; charts
from raw metrics with the step basis stated, no fabricated step 0. When changing a scorer, run both offline
on the same responses and inspect only the disagreements.

<a id="paired-acceptance"></a>
## Paired acceptance and pairing statistics

Freeze first: checkpoint content, image digest, nodes, parallel layout, precision, actual attention / GEMM /
MoE backends, graph and cache resolved config, what is compared, how it is computed, the pass line and the
noise floor; plant a known error and confirm the criterion returns `fail`. Tune BI=0 honestly, human-audited
(G1); never downgrade it to narrow the gap. Table every difference between the arms before comparing; each
is aligned or quantified separately — a delta where several items change together is not attributed to the
switch.

### Pairing and computation

- **A human freezes before the run:** repeats, pairing order, aggregation formula, budget per concurrency
  point `c`, invalid-leg rule.
- **Statistic:** median of per-round paired loss `median_r(1 − T_BI1[c,r] / T_BI0[c,r])`.
  `1 − median_r(T_BI1) / median_r(T_BI0)` is a different statistic ("ratio of median throughputs"), must be
  labelled; neither is switched after seeing results.
- **Fixed workload:** frozen prompt set, fixed `max_new_tokens`, EOS ignored, agreed `c` such as 1/8/32/64,
  request count and lengths recorded, natural-EOS listed separately.
- **Same node, same container:** no other GPU or heavy I/O work.
- **Crossed order (AB/BA/AB):** each side warmed up per round, never each side's fastest run across node
  pools.
- **Per leg on every worker:** image digest, actual kernel/backend, resolved config, graph state, cache hits,
  request and token statistics, client and server exit checked separately.
- **Invalid leg:** a leg whose workload or exit receipt differs is `inconclusive` (no loss from unequal
  request volumes).
- **Report:** raw throughput and `1 − T_BI1/T_BI0` per round, then the formal result per `c` — an average
  across `c` hides no single `c` over budget.

- **Screening** may rerun only the `c` that failed, deciding only whether to continue.
- **A candidate at the budget line** cannot pass against the old baseline's historical median: re-pair both
  sides on the same node, image and time window.
- **Fast/slow tiers in one config:** check tuning receipts, CPU/GPU state, graph, cache and actual operator
  selection before claiming a gain.
- **Unexplained fluctuation** ⇒ `inconclusive`; a formal point over budget ⇒ `fail`; no cherry-picking;
  profile outside the formal timing.
- **Invalid pair:** tell producer, numerics, scorer and parser apart. Sealed worker receipts may be re-read by
  a separately versioned recovery checker (hash recorded) that verifies workers, engine, package identity and
  workload, keeps the original failure status and fabricates nothing.
- **Failure records:** keep them as they are; append.

The final acceptance table lists separately: train-inference forward, module backward, real update, quality,
performance, clean exit, final immutable artifacts; each `pass`/`fail`/`inconclusive`/`not run`. Example
scope: "three steps bitwise, quality pairing passes, c=32/64 over budget".

## Performance

- **Three bases, never compared across each other:**
  - **rollout tokens/s at fixed workload**: completed tokens of all engines / one common window; no
    per-engine sums, no window from unsynchronised clocks;
  - **trainer forward+backward time** per update at fixed tokens;
  - **RL-step makespan**: first request in to last out plus update — what RL cares about, moved by tails and
    scheduling.
- **Translate the rollout target:** `--rollout.batch_size` × `--rollout.n_samples_per_prompt` = requests;
  `--vllm.max_num_batched_tokens` is a per-pass token budget, not a request count. Record per-engine requests,
  prefill chunking, decode batch over time, cache state, EOS vs ignore-EOS, length percentiles, completed
  requests.
- **The replay bench** replays one real rollout's length table ignore-EOS in both arms, so tokens, requests
  and tail are identical. Measure same-config repeat noise first; a short profile is not long decode.
- **Metrics are cumulative values:** vLLM `/metrics` counters count since process start — every stage is
  `after − before` (the first stage is right by accident), reconciled once against client-side token counts.

- **Fair A/B.** Two kinds, never mixed:
  - the causal comparison of the switch (same software, scheduling, cache; only the switch changes; the
    kernel it selects recorded);
  - each side's best valid config (BI=0 the strong default, BI=1 the fastest bitwise path, every difference
    disclosed).
- **Do not force BI=0** onto a slower attention, MoE backend or eager; a BI=1 run with switches locally
  disabled is not "native". Common optimizations go to both sides, common gain and BI-overhead reduction
  reported separately; BI=1 faster than BI=0 triggers an audit.
- **Fix** weights, input, tokens/lengths, cap, concurrency, nodes, software, warmup, timing boundaries;
  fresh-process first rollout apart from warm state; same transport (NCCL IB vs NVLink/P2P). Zero JIT compile
  lines during the bench; the profile shows BI=0 did not run the aligned kernel.
- **Speed switches and numerical switches** are reported separately: a switch that changes bytes belongs to
  the consistency evidence.
- **Deficit and overhead:** `deficit = 1 − T_BI1/T_BI0`, `overhead = t_BI1/t_BI0 − 1` (matched work). Latency
  reduction ≠ throughput gain; gains of A and B do not add; skipping the old-logprob forward
  (`--train.force_on_policy` with `--algo.kl.init_coef 0`) is an algorithm choice, not a BI gain.

Breakdown and ablation. Two stages: after proxy consistency, remove structural defects (per-token launches,
synchronous readbacks, repeated copies, abnormal workspace, graph not capturable), measuring the training
step too; after full consistency, E2E profile → micro bench → paired E2E.

**Breakdown:**

- **Break down the overhead first by phase** (load/JIT, prefill, decode, old-logprob, backward/update,
  refit); kernel times overlap and do not add to wall time.
- **One profile with enough attribution:** verified serving config and I/O statistics, off-by-default NVTX at
  module boundaries, nsys with CUDA-graph node tracing, calls / GPU time / launch / sync / allocation / shape
  per path, ncu on a few kernels, final timing without profiler on the same code.
- **A kernel name is not a module identity:** who launched it is proven by class and import source at init,
  shape and dispatch at the callsite, launch correlation in the profiler.
- **Pin "where is it slow"** to a process and call stack (py-spy on engine, worker, frontend) before
  optimizing; GPU utilization is not evidence.
- **Single-operator replay and synthetic micro bench complement:** the shape matrix proves boundaries, real
  replay gives distributions, ties, masks, layouts and call frequency. Time with and without workspace
  allocation; candidate bitwise against baseline; cover batch 1/4/32/128, neighbour compositions,
  prefill/decode, graph; list uncovered shapes.

**Ablation:**

- **One variable at a time;** a single 0-mismatch round does not count for an intermittent leak.
- **"Turning off X fixes it"** must check whether X's switch moves another branch; a switch that changes speed
  changes batch composition (pin it offline).
- **The target path must be hit** — zero hits is `inconclusive`.
- **Interactions:** identify with `base / A / B / A+B`; the round after restoring step N is N+1; stop cleanly
  after rollout when only rollout is measured.
- **Segment-by-segment isolation:** arms enabled up to each boundary (all off / capture only / worker data
  plane / IPC / return path only / all on) whose differences add back up to all on − all off, else
  interaction or dirty cut.
- **Priorities:** idle and repeated work → host sync and launches, graph restored → fused boundaries keeping
  rounding, order and sentinels → only then algorithms, chunking or collectives (new equivalence evidence).
- **Gates per candidate:** same-input kernel comparison (real dtype/shape, masked layout, empty experts, ties,
  dirty buffers, graph replay) → fixed-token engine gates → real training. Regress bitwise after each change;
  rerun timings on the final code; no gain ⇒ withdrawn from the default path.
- **Prefer vLLM's existing invariant kernels;** a new kernel only when the profile names it, and it goes
  upstream so both sides share one implementation.

## Checkpoint, resume and memory

Three independent contracts: **file** (serialization and publish — DCP under `--ckpt.path`; `extra_state.pt`
written last by rank 0 after the all-rank barrier, `latest` by atomic replace; a step dir without
`extra_state.pt` is not loadable), **memory** (transient save/load peaks: killed by Ray after the write,
cgroup OOM on load), **derived state** (files complete, the first step after restore mismatches, the second
recovers). Decide which failed before saying "the checkpoint is broken".

Exact resume needs: fp32 masters (strict load, no silent missing/unexpected keys); optimizer state at the same
step incl. CPU-offloaded moments (`--fsdp.offload optimizer`); the LR scheduler's actual value; RNG streams
where the recipe depends on them (molt's client state holds episode, consumed prompts, dataloader and
rollout-generator state, not RNG — a resumed trajectory is not bitwise against an uninterrupted run and need
not be); the weight version id; a run contract (code, data, recipe hashes) checked item by item. Rebuilt
after restore, not stored: refit caches, export caches, routing/replay caches, graph-captured weight
addresses — every parameter-derived cache is invalidated in the load lifecycle that owns it, not by waiting
one step. CPU AdamW (`--fsdp.offload optimizer`) is documented as equivalent to the GPU step but not bit-identical (sqrt
and division rounding): never swapped across a resume.

Prove resume: (1) restore in a new process (PIDs changed; resolved step confirmed before the first training
log line — `--ckpt.load_enable` alone is not evidence); (2) first refit bytes identical to the refit before
the save (same-tensor hash); (3) first bitwise compare 0; (4) one effective update (non-zero gradient, master
update norm); (5) compare again at 0; (6) clean exit. Also: step-0 cold load, new code reading an old
checkpoint.

Commit protocol: completion marker last, after all ranks confirmed; uncommit (delete) an old step dir before
overwriting; all ranks check completeness together and fail together; shard-less ranks still join the
collective; a requested but missing file raises, never falls back to a fresh optimizer. Capture the first
exception (errno, rank, file, offset), not the last message; reproduce with a save-only loop; a save fix is
accepted by N consecutive saves + kill → resume → save, run before the first long run of any new layout.
Publish node-local → temp name on the target FS → same-FS rename; read with the image that wrote. Rotation
(`--ckpt.max_num`, `--ckpt.dcp_max_num`, `--ckpt.max_mem`) evicts the oldest directories by mtime without a
completeness check, so an interrupted save can outlive a good one; keep one checkpoint outside rotation in
long runs.

Memory:

- **Host:** the caching host allocator rounds pinned blocks to a power of 2, so optimizer offload holds tens
  of percent above nominal — allocate long-lived pinned blocks at exact size; D2H staging stays in the pinned
  pool.
- **Host budget:** cgroup `memory.max` × Ray threshold; `memory.current` includes page cache and Ray kills on
  it; sum PSS, not RSS; the save-step peak is closest to the limit.
- **GPU:** `--vllm.gpu_memory_utilization` (default 0.95) per recipe and mode — the aligned attention's fp32
  partial buffers (sized by the context ceiling, never freed) and autotune workspace cut the KV budget; read
  the resolved KV budget in the vLLM log, not the shell variable.
- **Trainer peaks:** all-gathered module + activations + workspace, aligned-forward partial buffers,
  `full_tensor()` of the largest parameter plus flush buffers at refit, fp32 logprob chunks.
- **Locate GPU OOM by lifecycle** — startup / FSDP wrap, first forward/backward, refit, eval/generation, save
  — recording phase, rank, requested / allocated / reserved / free and the last completed phase. A failure
  that did not reach update is not an update OOM; a lifecycle error is fixed at its boundary, a legitimate
  peak changes the topology and records the support boundary.
- **Reserved vs allocated:** PyTorch's free cached blocks are not available to another allocator (vLLM's CuMem
  KV pool, NCCL). `empty_cache()` frees reclaimable cache only, `gc.collect()` helps only when it drops a last
  owner — compare allocated/reserved/free before and after and the same allocation's result; high reserved
  alone proves nothing.

## Environment and provenance

Code existing does not mean it is executing. One run manifest per run, audited inside the Ray worker
(interpreter, import path, mount namespace), not the login shell — "which version is running now":

| Layer | Evidence | Cannot substitute |
|---|---|---|
| Source | remote, branch, HEAD, dirty patch hash, source hash of every patched file | `latest`, directory name |
| Installed package | molt, vLLM, nemo-automodel, transformers, torch versions as imported in the worker | `pip show` in another venv |
| Native library | path and hash of the `.so` actually loaded, build inputs, arch | Python HEAD, file name |
| JIT kernel | Triton/Inductor source hash, compiler version, cache origin | `.so` unchanged |
| Container | OCI digest, parent image, build record | mutable tag, creation time |
| Run | resolved args, worker env, weight digest, executed kernel names | an export in the head-node shell |

- **`unknown`** for unknown items; generic switches (`CUDA_DEVICE_MAX_CONNECTIONS`, NCCL pins) recorded, never
  quietly added and attributed to BI.
- **"Latest" comes with a branch:** `git rev-list --left-right --count A...B`, `git merge-base A B`,
  `git diff --stat A B` answer how far and what; fetch is not checkout.
- **Overlay vs native wheel:** same-commit worktree, per-file hash-checked sync into a container not yet
  running the workload; never mount the whole repo over the installed package; recompile native/binding
  changes; overlays are diagnosis, never delivery.
- **Compile/autotune caches** on node-local disk mounted explicitly (not `noexec`, not per-container tmpfs),
  keyed by image, commit and GPU arch, one copy per rank; an existing cache dir is not a hit; none is
  committed.
- **Delivery gate = the final immutable image:** clean commits, pinned base digest and dependencies; imports
  and hashes checked in a new container with no source mounts and no inherited JIT cache; the agreed gates
  run there; the same artifact published and its registry digest verified (one tag may point to different
  digests on different days).
- **Trainer and engine run the same digest and Python/Torch set:** two environments inside one image cannot
  dodge a compatibility pin (vLLM ↔ torch ↔ nemo-automodel ↔ transformers).

## Delivery and PRs

- **Change map before coding:** feature → existing module → proposed location → reason, covering model
  construction (`BackendConfig`), forward/backward, refit, checkpoint, CLI, tests.
- **Implement in the formal execution path:** BI=0 keeps the reviewed native path, BI=1 selects explicitly
  and errors early when a capability is missing.
- **No runtime hijacking in delivery** (`inspect.getsource`, AST rewriting, `exec`, `sitecustomize`,
  `PYTHONPATH` tricks, methods replaced at import); the only opaque boundary is a `torch.autograd.Function`
  whose forward calls the vLLM op.
- **Kernel, binding and build registration ship together.**
- **Parameter registration, state_dict, reduction and recompute contracts stay;** optimizer parameter
  identity is preserved — masters are never replaced by an inference cache.
- **A generic interface change** needs call-chain evidence, the proposed interface, impact scope and human
  review before implementation.
- **PR hygiene:** one problem, one minimal diff (`simplicity-first`); tests extend `tests/unit` and
  `tests/e2e` and test behaviour (bytes, dispatch, config validation), not source strings. Cleaned-up code is
  re-verified — it does not inherit the prototype's pass; the description records purpose, base and patch
  stack, related PRs, test commands and results, model impact, rollback.

Delivery split, pinned versions, build order kernel/runtime → integration → recipe/pins → image: a **vLLM
PR** (upstream, or a pinned fork SHA until merged) for kernel or engine behaviour that had to change; a
**nemo-automodel PR or pin** when the model definition changes, molt pinning that commit; the **molt PR**
with switch, guards, worker hook, patched forwards, tests, docs page (operator table, known limits), recipe
and Dockerfile pins, default path unchanged. Reference versions are not proof of passing: distinguish the
head viewed, the version a PR body claims and the version measured.

## Two-step acceptance through the real entry point

Run the formal recipe shortened only in the number of updates ([sequence](workflow.md#two-step-rl-sequence))
and made synchronous (the synchronous-run row of molt-integration.md): the bitwise proof needs identical
weights on both sides. Keep the formal learning-rate schedule rather than compressing it into two steps;
the asynchronous production recipe is accepted on quality and performance, reading the lag floor. The final evidence is
produced on the delivered commit/image through `molt/cli/train_rl_ray.py`, not a standalone replay.

## Report template

workflow.md keeps only a pointer here. Every cell is `pass`/`fail`/`inconclusive`/`not run`; out of scope is
`not run`, never `pass`; finishing the short run completes no missing required check.

```
Outcome and scope      verdict for the requested scope; model/layers, samples, valid tokens, dtype, layout,
                       graph/cache path, update count, untested conditions; human-approved checklist link
Immutable experiment   molt / vLLM / nemo-automodel / transformers / torch as imported in the worker, patch
                       hashes, image digest, GPU, checkpoint and input digests, resolved args, commands,
                       budgets fixed before testing; measured runtime == delivered image, else the gap
Delivery manifest      per deliverable (vLLM PR / nemo-automodel PR or pin / molt PR / image / approved
                       recipe / final validation): URL, base SHA, head SHA, build or acceptance evidence
Proxy coverage and     feature → proxy layer → test; per operator: visible forward, trainables, rounding/
gradient contract      STE rule, fixed discrete ids, VJP collectives, saved state; independent reference
                       (why it does not call the tested backward), dY coverage, thresholds, one-step deltas
Symptom and            per-request/token diffs, A/A and A/C, side isolation, last-equal / first-unequal
localization           boundary, same-input op compare, competing explanations
Cause and change       violated contract, smallest trigger, repair vs diagnostic, files/kernels/flags,
                       regression kept, what BI=0 users see (nothing); "unknown" if unproven
Verification           table: gate | pass/fail/inconclusive/not run | command | artifact | closure; rows:
                       original failing replay; operator regression; engine layouts (G3); native
                       old-logprob and grad-enabled forward (G4, two rows); module VJP + one optimizer
                       step (G5); θ0/θ1/θ2 (G6) with per-θ receipt (mask identity, valid tokens,
                       `bitwise_frac`, update norm); first post-resume refit/compare; quality; clean
                       shutdown; final immutable image without overlays
SFT controls and       the three arms of training-side.md (BI=0 native / aligned + native backward /
quality                aligned + reference backward); controlled variables; raw loss, grad norm, update RMS, clipping and
                       non-finite counts; paired quality cells and threshold
Performance            basis (rollout tok/s, fwd+bwd, makespan), matched work, warm/cold, node and order,
                       repeats, raw pairs, exclusions, formula and aggregate per c, speed vs numerical
                       switches, causal vs each-side-best, lengths and completed tokens
Evidence corrections   superseded claims with the correcting artifact; retained bundles and budget
Remaining work         gaps and the smallest next experiment; production hooks/config restored?
```

```yaml
purpose: "frozen-policy old-logprob replay"          # each variant changes one variable
target: "bitwise selected-token fp32 logprob"
rollout: "<--train.rollout_dump_dir output, rollout_step{N}.pt>"
checkpoint: "content id + pre-update path"
environment: "image digest, molt/vLLM/nemo-automodel commits + patch hashes, GPU"
topology: {trainer: "tp/cp/ep/dp", engine: "tp/ep/dp"}
state: "RNG, graph/eager, cache and reset policy"
budget: {max_runs: 3, timeout_seconds: 600}
variants:
  - {name: A, change: "none (A/A)", cmd: "", output: "", exit: null, result: ""}
  - {name: C, change: "packing reversed", cmd: "", output: "", exit: null, result: ""}
```
