# Recipe, arms and paired acceptance

Gates and false greens: [workflow.md](workflow.md); flags and patch points:
[molt-integration.md](molt-integration.md).

## One recipe, one switch

Both arms share launch script, image digest, data, lengths, batch, parallel layout, optimizer and seeds;
the only difference is `--train.batch_invariant` and what it implies (engine kernels, replay, correction).

| Arm | Engine | Trainer | Correction | Expected mismatch |
|---|---|---|---|---|
| BI=0 (baseline) | default kernels, graphs on | default `BackendConfig` | identical to BI=1 (one rule, below) | the recipe's measured `mismatch/bitwise_frac` (≈ 1 under default kernels) and `abs_diff_mean` — record them as the BI=0 baseline |
| BI=1 (aligned) | invariant kernels, `TRITON_ATTN`, replay on for MoE | aligned forward | the recipe's own IS correction, else the non-binding `--algo.advantage.is_correction_level token --algo.advantage.is_correction_threshold 0 1e9` | `mismatch/bitwise_frac` 0 at every step of the synchronous run; the lag floor when asynchronous |

- One correction rule: both arms run identical correction settings — the recipe's IS correction if it has
  one, else the non-binding token IS above — so rollout logprobs and `mismatch/*` exist in both arms; the
  report declares the IS-weight confounder (the ratio weight is exactly 1 only when bitwise,
  [rollout logprobs](molt-integration.md#rollout-logprobs)).
- Preconditions and supported layout: `--fsdp.cp_size 1` (the reference guards it at startup; no CP path
  exists); an EP/TP pair is supported only with the [control-matrix](training-side.md#control-matrix)
  proof — verified today on the reference branch: trainer EP 4 / TP 1 against engine TP 4; `--fsdp.offload`
  identical in both arms.
- Separate code paths: BI=0 is the default molt path with nothing of the aligned implementation injected;
  "the BI=1 actor with default kernels swapped in" is neither baseline nor true on-policy. Consistency
  patches never change BI=0 bytes: replay one saved rollout before and after the patch with the switch
  off — identical int-view logprobs and throughput.
- Resolved command byte-identical except the switch: molt has no dry-run — take `args` from each driver
  log (`strategy.print(printed_args)` in `train_rl_ray.py`) and `diff` the two; BI=1 differs from the
  shipped recipe only in the switch, BI=0 only in names, paths and new explicit switches. Same image digest,
  same `--train.seed`, same data order.
- Separate `--ckpt.path` and `--ckpt.output_dir` per arm: `--ckpt.load_enable` resumes from
  `<--ckpt.path>/_actor/latest`, so a shared path resumes onto the other arm. The run name plays no part in
  the resume.
- The recipe header states only what was measured to hold; known engine-config differences between the
  arms go into the comparison report; untested switches are deleted. Every deviation from the reference
  recipe is tabulated (item, reference value, ours, why, which conclusion it touches) with who approved it.

BI=0 integration pitfalls, re-checked per model:

| Pitfall | Rule |
|---|---|
| A default that silently changes with the switch (attention backend, `enforce_eager`, prefix caching, `logprobs_mode`, compile mode) | Resolve every default explicitly so the diff shows only the switch |
| A cache or graph setting that differs (CUDA-graph mode, prefix cache, compile cache) | Same in both arms, or a quantified confounder |
| A metric that exists in one arm only (`mismatch/*` need rollout logprobs; a blank cell reads as 0) | The one correction rule above: both arms request them every step |
| A packing path that differs (`cu_seqlens`, THD style, HybridEP padding) | Same path in both arms; a stateful layer scans the pack as one sequence |
| Training-only model defaults (router training vs `--actor.freeze_moe_router`, MTP head, frozen modules) | Pinned for both arms; trainable set listed |

Smoke, both arms at proxy scale (a few updates): finite loss and gradients, non-degenerate advantages, refit
bytes equal ([refit](molt-integration.md#refit)), exit code 0 from the recipe, not the launcher, the count of
patched modules printed per rank and identical on every rank, the expected mismatch of the arms table. Grad
norm exactly 0 means same-reward groups before numerics.

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
is invisible. Rising training reward, mismatch 0 and a held-out gain are three conclusions
(`--algo.dynamic_filtering_enable` changes the sample composition behind the training reward); a relaxed
token budget is not a training gain. Report model and weight version, problem count, truncation ratio next
to accuracy; charts from raw metrics with the step basis stated, no fabricated step 0: molt logs `train/*`
against `train/global_step` and eval at its own `eval/global_step`, and a resumed job opens a new W&B run
under the same `--logger.wandb.run_name` — align segments on `global_step`, not on the run. When changing a
scorer, run both offline on the same responses and inspect only the disagreements.

<a id="paired-acceptance"></a>
## Paired acceptance and pairing statistics

Freeze first: checkpoint content, image digest, nodes, parallel layout, precision, actual attention / GEMM /
MoE backends, graph and cache resolved config, what is compared, how it is computed, the pass line and the
noise floor; plant a known error and confirm the criterion returns `fail`
([workflow.md](workflow.md#evidence)). Tune BI=0 honestly over its real knobs (`--vllm.attention_backend`,
`--vllm.enforce_eager`, `--vllm.max_num_batched_tokens`, `--vllm.gpu_memory_utilization`,
`--train.async_queue_size`; for MoE the `MOLT_MOE_EXPERTS` / `MOLT_MOE_DISPATCHER` combinations),
human-audited (C1); never downgrade it to narrow the gap or force it onto a slower attention, MoE backend or
eager.

### Pairing and computation

- **A human freezes before the run:** repeats, pairing order, aggregation formula, budget per concurrency
  point `c`, invalid-leg rule.
- **Statistic:** median of per-round paired loss `median_r(1 − T_BI1[c,r] / T_BI0[c,r])`.
  `1 − median_r(T_BI1) / median_r(T_BI0)` is a different statistic ("ratio of median throughputs"), must be
  labelled; neither is switched after seeing results. Without a reviewed basis report measured values only,
  no budget verdict; historical results keep their formula.
- **Fixed workload:** frozen prompt set, fixed `max_new_tokens`, EOS ignored, agreed `c` = concurrent
  requests per engine (1/8/32/64 is an example only), request count and lengths recorded, TTFT/TPOT and
  error count per `c`, natural-EOS listed separately.
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

The final acceptance table lists separately: train-inference forward, module backward, real update, quality,
performance, clean exit, final immutable artifacts; each `pass`/`fail`/`inconclusive`/`not run`. Example
scope: "three steps bitwise, quality pairing passes, c=32/64 over budget". On an FP8 route the table adds:
training / export / rollout dtype and conversion owner per tensor family, a zero-update re-export equal to
θ0 bytes, every family quantized once and closed by name / shape / dtype / count / bytes
([precision chain](contracts.md#weights-and-precision-chain)).

## Performance

- **Three bases, never compared across each other** (per-request tok/s = 1/TPOT is a latency, kept apart):
  - **rollout tokens/s at fixed workload**: completed tokens of all engines / one common window; no
    per-engine sums, no window from unsynchronised clocks;
  - **trainer forward+backward time** per update at fixed tokens;
  - **RL-step makespan**: first request in to last out plus update — what RL cares about, moved by tails and
    scheduling.
- **Translate the rollout target:** `--rollout.batch_size` × `--rollout.n_samples_per_prompt` = requests;
  `--vllm.max_num_batched_tokens` is a per-pass token budget, not a request count. Record per-engine requests,
  prefill chunking, decode batch over time, cache state, EOS vs ignore-EOS, length percentiles, completed
  requests.
- **The replay bench:** no molt tool on `main` yet — replay a saved rollout's length table ignore-EOS
  through a [standalone engine](molt-integration.md#engine-environment-and-worker-hook) in both arms, so
  tokens, requests and tail are identical. Measure same-config repeat noise first; a short profile is not
  long decode — the short profile, the full-length bench and the training E2E get separate experiment ids.
- **Metrics are cumulative values:** vLLM `/metrics` counters count since process start — every stage is
  `after − before` (the first stage is right by accident), reconciled once against client-side token counts;
  a new script's first numbers are reconciled the same way (a regex that never matches looks plausible).

- **Fair A/B.** Two kinds, never mixed:
  - the causal comparison of the switch (same software, scheduling, cache; only the switch changes; the
    kernel it selects recorded);
  - each side's best valid config (BI=0 the strong default, BI=1 the fastest bitwise path, every difference
    disclosed).
- **No downgraded BI=0** ([above](#paired-acceptance)); a BI=1 run with switches locally disabled is not
  "native" — a replaced path is a separately named candidate that passes the full contract gate. Common
  optimizations go to both sides, common gain and BI-overhead reduction reported separately; BI=1 faster than
  BI=0 triggers an audit; a run that failed a backend guard never enters the denominator.
- **Fix** weights, input, tokens/lengths, cap, concurrency, nodes, software, warmup, timing boundaries;
  fresh-process first rollout apart from warm state; same transport (NCCL IB vs NVLink/P2P). Zero JIT compile
  lines during the bench; the profile shows BI=0 did not run the aligned kernel.
- **Speed switches and numerical switches** are reported separately: a switch that changes bytes belongs to
  the consistency evidence.
- **Deficit and overhead:** `deficit = 1 − T_BI1/T_BI0`, `overhead = t_BI1/t_BI0 − 1` (matched work). Latency
  reduction ≠ throughput gain; gains of A and B do not add; skipping the old-logprob forward
  (`--train.force_on_policy` with `--algo.kl.init_coef 0`) is an algorithm choice, not a BI gain.

Breakdown and ablation, two stages ([gates](workflow.md#gate-table)): C4 on the proxy removes structural
defects (per-token launches, synchronous readbacks, repeated copies, abnormal workspace, graph not
capturable), measuring the training step too; C10 on the full model runs E2E profile → micro bench → paired
E2E. Kernel candidates follow the [kernel-level contracts](contracts.md#kernel-level-contracts) and their
[cost breakdown](contracts.md#cost-breakdown-bi0--bi1).

**Breakdown:**

- **Break down the overhead first by phase** (load/JIT, prefill, decode, old-logprob, backward/update,
  refit); kernel times overlap and do not add to wall time. molt logs per step `timing/generation`,
  `timing/make_experience`, `timing/policy_train`, `timing/broadcast` (= `broadcast_lock_wait` +
  `broadcast_transfer`), `timing/vllm_idle_wait`, `timing/actor_idle_wait`, `timing/step_total`; molt has
  no cold-start key: time load, JIT and first capture yourself and keep them out of the training phases.
- **One profile with enough attribution:** verified serving config and I/O statistics, off-by-default NVTX at
  module boundaries, nsys with CUDA-graph node tracing, calls / GPU time / launch / sync / allocation / shape
  per path, ncu on a few kernels, final timing without profiler on the same code.
- **A kernel name is not a module identity:** who launched it is proven by class and import source at init,
  shape and dispatch at the callsite, launch correlation in the profiler. Graph replay hides Python `cpu_op`
  events, so a zero count does not prove a call site never ran; a roofline bound uses the actual dtype's
  peak and the memory level studied.
- **Pin "where is it slow"** to a process and call stack (py-spy on engine, worker, frontend; asyncio loop
  lag for the frontend) before optimizing; GPU utilization is not evidence.
- **Single-operator replay and synthetic micro bench complement:** the shape matrix proves boundaries, real
  replay gives distributions, ties, masks, layouts and call frequency. Time with and without workspace
  allocation; candidate bitwise against baseline; cover batch 1/4/32/128, neighbour compositions,
  prefill/decode, graph; list uncovered shapes.

**Ablation:**

- **One variable at a time;** a single 0-mismatch round does not count for an intermittent leak.
- **"Turning off X fixes it"** must check whether X's switch moves another branch; a switch that changes speed
  changes batch composition (pin it offline). A knob is credited only after the resolved config and the
  step log show it acting: `--train.max_tokens_per_gpu` splits microbatches only under
  `--train.dynamic_batch_enable` (which needs `--fsdp.packing_samples`).
- **Interactions:** identify with `base / A / B / A+B`; the round after restoring step N is N+1; stop cleanly
  after rollout when only rollout is measured.
- **Segment-by-segment isolation:** arms enabled up to each boundary (all off / capture only / worker data
  plane / IPC / return path only / all on) whose differences add back up to all on − all off, else
  interaction or dirty cut.
- **Priorities:** idle and repeated work → host sync and launches, graph restored → fused boundaries keeping
  rounding, order and sentinels → only then algorithms, chunking or collectives (new equivalence evidence).
- **Gates per candidate:** same-input kernel comparison (real dtype/shape, masked layout, empty experts, ties,
  dirty buffers, graph replay) → fixed-token engine gates → real training. Regress bitwise after each change;
  rerun timings on the final code; no gain ⇒ withdrawn from the default path. Each delivered item states
  whether quality and long-run stability were run; a kernel speedup is not an end-to-end conclusion.
- **Prefer vLLM's existing invariant kernels;** a new kernel only when the profile names it, and it goes
  upstream so both sides share one implementation.

## Checkpoint, resume and memory

Three independent contracts: **file** (serialization and publish — DCP under `--ckpt.path`; `extra_state.pt`
written last by rank 0 after the all-rank barrier, `latest` by atomic replace; a step dir without
`extra_state.pt` is not loadable), **memory** (transient save/load peaks: killed by Ray after the write,
cgroup OOM on load), **derived state** (files complete, the first step after restore mismatches, the second
recovers). Decide which failed before saying "the checkpoint is broken".

Exact resume needs:

- fp32 masters (strict load, no silent missing/unexpected keys);
- optimizer state at the same step incl. CPU-offloaded moments (`--fsdp.offload optimizer`) — molt's
  optimizer DCP load tolerates partial state (`allow_partial_load=True`), so count the restored optimizer
  keys yourself; the LR scheduler's actual value;
- RNG streams where the recipe depends on them (molt's client state holds episode, consumed prompts,
  dataloader and rollout-generator state, not RNG — a resumed trajectory is not bitwise against an
  uninterrupted run and need not be);
- the weight version id; a run contract (code, data, recipe hashes) checked item by item;
- rebuilt after restore, not stored: refit caches, export caches, routing/replay caches, graph-captured
  weight addresses — every parameter-derived cache is invalidated in the load lifecycle that owns it, not by
  waiting one step;
- CPU AdamW (`--fsdp.offload optimizer`) is documented as equivalent to the GPU step but not bit-identical
  (sqrt and division rounding): never swap `--fsdp.offload` across a resume or inside one comparison.

Prove resume: (1) restore in a new process (PIDs changed; resolved step confirmed before the first training
log line — `--ckpt.load_enable` alone is not evidence); (2) first refit bytes identical to the refit before
the save (same-tensor hash; derived engine state ([refit](molt-integration.md#refit)) comes from the restored
weights after this refit — hash it then); (3) first bitwise compare 0; (4) one effective update
(non-zero gradient, master update norm); (5) compare again at 0; (6) clean exit. Also: step-0 cold load, new
code reading an old checkpoint.

molt specifics:

- `--ckpt.load_enable` resumes only "if present": a missing `<--ckpt.path>/_actor` starts at step 0
  silently; an `_actor` without a complete step dir starts at 0 and logs `checkpoint_states: {}`; a stale
  `latest` falls back to the mtime-newest complete dir with a warning — read `global_step` in the
  `checkpoint_states:` line before the first step.
- A missing `optim/` raises unless `LOAD_MODEL_ONLY=1` (weights only: a new optimizer, not a resume).
- The CLI optimizer hyperparameters are re-applied on load (lr rescaled to the schedule position), so a
  changed flag silently changes the resumed run — diff the resolved args of both segments.
- `--ckpt.warm_resume_rollouts` restores untrained groups generated before the save under whatever policy
  produced them: off for the post-resume bitwise proof.
- Prove the DCP round trip bitwise per rank on the real `--fsdp.ep_size` layout before the first long run —
  hash one expert shard per rank before the save and after the restore: the HF-keyed shards split grouped
  experts per expert, and an expert restored onto the wrong rank (shards that share a key read another
  rank's experts) is silent.

Commit protocol:

- Completion marker last, after all ranks confirmed; uncommit (delete) an old step dir before overwriting;
  all ranks check completeness together and fail together; shard-less ranks still join the collective.
- Capture the first exception (errno, rank, file, offset), not the last message; reproduce with a save-only
  loop; a save fix is accepted by N consecutive saves + kill → resume → save, run before the first long run
  of any new layout.
- Discriminate in order: target FS vs node-local disk with the same serializer, then concurrency and pinned
  vs pageable, then thread/FD growth, only then filesystem client code.
- A written checkpoint is verified by shard set, non-empty sizes, no leftover temp files and one real
  restore — a save digest is not a restore test; never kill a job mid-write.
- Publish node-local → temp name on the target FS → same-FS rename; read with the image that wrote.
- Rotation (`--ckpt.max_num`, `--ckpt.dcp_max_num`, `--ckpt.max_mem`) evicts the oldest directories by
  mtime without a completeness check, so an interrupted save can outlive a good one; keep one checkpoint
  outside rotation in long runs (a step wanted for eval may be gone, and another chain's checkpoint is no
  substitute).
- Resuming on other nodes: copy checkpoint and caches first, change only the node selection, write a
  transfer record.

Memory:

- **Host:** the caching host allocator rounds pinned blocks to a power of 2, so optimizer offload holds tens
  of percent above nominal — allocate long-lived pinned blocks at exact size; D2H staging stays in the pinned
  pool.
- **Offload and object store:** `--fsdp.offload` is a nested progression (`none` → `optimizer` → `full`
  streams the params too): change one level at a time and measure the phase peak before and after. Ray
  object-store mappings (molt parks rollout tensors in the runners' object stores) count into each
  process's PSS as shared pages.
- **Host budget:** cgroup `memory.max` × Ray threshold; `memory.current` includes page cache and Ray kills on
  it; sum PSS, not RSS; the save-step peak is closest to the limit, and raising the Ray threshold or
  shrinking the microbatch buys no headroom. Tell the killers apart: a cgroup OOM is an `oom_kill` event
  and SIGKILL with no Python stack, a Ray kill logs `exceeded threshold`, a CUDA allocator retry warning is
  pressure, not a kill. `free`/`psutil` see the node, `ru_maxrss`/Slurm MaxRSS are cumulative peaks, pinned
  memory shows as shmem in `memory.stat`.
- **Host growth per step:** per-process PSS floor per step from `smaps_rollup`, find the process that rises
  monotonically, reproduce CPU-only before changing code; verify per save path that the pinned staging is
  released after the save.
- **Load peak:** accept a load with model + optimizer + extra state at real scale, not one shard — DCP may
  stage the full optimizer in anonymous memory and CPU offload a second pinned copy.
- **GPU:** `--vllm.gpu_memory_utilization` (default 0.95; it assumes the engine owns its GPUs) per recipe and
  mode — the aligned attention's fp32
  partial buffers ([operator table](molt-integration.md#operator-table)) and autotune workspace cut the KV
  budget; read the resolved KV budget in the vLLM log, not the shell variable.
- **Trainer peaks:** all-gathered module + activations + workspace, aligned-forward partial buffers,
  `full_tensor()` of the largest parameter plus flush buffers at refit, fp32 logprob chunks.
- **Locate GPU OOM by lifecycle** — startup / FSDP wrap, first forward/backward, refit, eval/generation, save
  — recording phase, rank, requested / allocated / reserved / free and the last completed phase. A failure
  that did not reach update is not an update OOM; a lifecycle error is fixed at its boundary, a legitimate
  peak changes the topology and records the support boundary. Never add a resident full-tensor dump to
  chase an OOM.
- **Reserved vs allocated:** PyTorch's free cached blocks are not available to another allocator (vLLM's CuMem
  KV pool, NCCL). `empty_cache()` frees reclaimable cache only, `gc.collect()` helps only when it drops a last
  owner — compare allocated/reserved/free before and after and the same allocation's result; high reserved
  alone proves nothing.

## Environment and provenance

Code existing does not mean it is executing. One run manifest per run, audited inside the Ray worker
(interpreter, import path, mount namespace), not the login shell — "which version is running now":

| Layer | Evidence | Cannot substitute |
|---|---|---|
| Source | remote, branch, HEAD, dirty patch hash, untracked runtime files, source hash of every patched file | `latest`, directory name |
| Installed package | molt, vLLM, nemo-automodel, transformers, torch versions as imported in the worker | `pip show` in another venv |
| Native library | path and hash of the `.so` actually loaded, build inputs, arch | Python HEAD, file name |
| JIT kernel | Triton/Inductor source hash, compiler version, cache origin | `.so` unchanged |
| Container | OCI digest, parent image, build record | mutable tag, creation time |
| Run | resolved args, worker env, weight digest, executed kernel names | an export in the head-node shell |

- **`unknown`** for unknown items; generic switches (`CUDA_DEVICE_MAX_CONNECTIONS`, NCCL pins) recorded, never
  quietly added and attributed to BI. An image tag, a PyPI version or an upstream PR's merge status never
  replace one minimal inference run inside the image; a venv package of the same name imports before the
  image's — check install plan and import path together; a suspect dynamic link is read from the process's
  mappings, not from `ldd`.
- **"Latest" comes with a branch:** `git rev-list --left-right --count A...B`, `git merge-base A B`,
  `git diff --stat A B` answer how far and what; fetch is not checkout; diverged branches have no
  before/after, commit time and hash size decide no ancestry, a shallow clone fetches history first.
- **Overlay vs native wheel:** same-commit worktree, per-file hash-checked sync into a container not yet
  running the workload; never mount the whole repo over the installed package; recompile native/binding
  changes; overlays are diagnosis, never delivery. An artifact already in the image is verified and used,
  not rebuilt; changing only a source pin changes no `.so`; an extension load failure is never fixed by
  unsetting a variable; observability overlays stay separate and off by default.
- **Compile/autotune caches** on node-local disk mounted explicitly (not `noexec`, not per-container tmpfs),
  keyed by image, commit and GPU arch, one copy per rank; an existing cache dir is not a hit; none is
  committed.
- **Delivery gate = the final immutable image:** clean commits, pinned base digest and dependencies; imports
  and hashes checked in a new container with no source mounts and no inherited JIT cache; the agreed gates
  run there; the same artifact published and its registry digest verified (one tag may point to different
  digests on different days). The receipt states how the image was produced — Dockerfile rebuild from a
  clean commit, or a container commit exported as `.sqsh` (development only) — the registry, a tag carrying
  the molt/vLLM/nemo-automodel SHAs, and both the `.sqsh` SHA256 and the registry digest; a new source layer
  on an old base brings in no other image's changes, and a deleted base file needs a whiteout check.
- **Trainer and engine run the same digest and Python/Torch set:** two environments inside one image cannot
  dodge a compatibility pin (vLLM ↔ torch ↔ nemo-automodel ↔ transformers); compare packages and dynamic
  dependencies before and after any upgrade.

## Delivery and PRs

- **Change map before coding:** feature → existing module → proposed location → reason, covering model
  construction (`BackendConfig`), forward/backward, refit, checkpoint, CLI, tests. Fix the engine-side
  locations at C3 and the trainer-side ones at C5; build inside the real trainer path from day one — a
  standalone engine `train_rl_ray.py` cannot call is not progress.
- **Implement in the formal execution path:** BI=0 keeps the reviewed native path, BI=1 selects explicitly
  and errors early when a capability is missing; where only a model, layout or backend is aligned the
  startup guard says so — nothing implies other models are covered.
- **A fix lives with its owner:** engine backend or selector faults in vLLM, model definitions in
  nemo-automodel (an architecture it lacks runs on the HF fallback until then), the rest in molt; never a
  trainer-side monkeypatch for a vLLM fault, never code hidden in a recipe script to keep a PR small.
  Reuse `mismatch_metrics`, `log_probs_from_logits` and the existing flags over synonymous metrics or
  compatibility switches.
- **No runtime hijacking in delivery** (`inspect.getsource`, AST rewriting, `exec`, `sitecustomize`,
  `PYTHONPATH` tricks, methods replaced at import); the only opaque boundary is a `torch.autograd.Function`
  whose forward calls the vLLM op.
- **Kernel, binding and build registration ship together;** a Python call that depends on an uncommitted
  `.so`, source file or build script in the experiment container is a delivery gap.
- **Parameter registration, state_dict, reduction and recompute contracts stay;** optimizer parameter
  identity is preserved — masters are never replaced by an inference cache.
- **A generic interface change** needs call-chain evidence, the proposed interface, impact scope and human
  review before implementation; the minimal fix lands in the owning repo.
- **Complete diffs:** each PR against the agreed base (owner, branch, base SHA and head recorded; merge-base
  and the actual diff checked), not stacked on an unmerged branch, no history drift or unrelated deletions,
  cross-repo dependencies pinned by commit; after a rebase the diff is re-reviewed and old-head evidence
  does not transfer. Every runtime file outside the PRs is a pinned dependency.
- **PR hygiene:** one problem, one minimal diff (`simplicity-first`); tests extend `tests/unit` and
  `tests/e2e` and test behaviour (bytes, dispatch, config validation), not source strings; existing tests
  are never deleted and a slimmed PR still runs them; diagnostics, draft docs and experiment artifacts stay
  out. Cleaned-up code is re-verified — it does not inherit the prototype's pass. The description records
  purpose, base and patch stack, related PRs and why this is not a duplicate, RL-only content, test
  commands and results, model impact, rollback, AI assistance. Split a large PR into reviewable pieces and
  prove equivalence with the tree hash; open PRs from your own fork; resolve conflicts on the PR's head
  branch, never in a look-alike new PR.
- **One launch entry:** change config along the entry script the user named so temporary launchers do not
  drift; deliver one script that runs with defaults, overrides and pins traceable — exports that exist only
  in chat are not delivery.
- **Re-accept from a new container** ([delivery gate](#environment-and-provenance)): worker interpreter,
  imports and libraries checked, then proxy, zero-update and the real loop, then quality and performance;
  on a difference shrink to a fixed failing replay, do not relaunch full training to locate it; a passing
  environment check proves preconditions, not numerics or speed. After merge and a new image: archive the
  worktree, delete the overlay, keep PR, commit, manifest and report.
- **Before declaring completion:** every goal of the user covered; every result from the final code and
  image; no wording that hides an untested item; an insufficient result is written as the gap, never as a
  percentage or a success marker.

Delivery split, pinned versions, build order kernel/runtime → integration → recipe/pins → image: a **vLLM
PR** (upstream, or a pinned fork SHA until merged) for kernel or engine behaviour that had to change; a
**nemo-automodel PR or pin** when the model definition changes, molt pinning that commit; the **molt PR**
with switch, guards, worker hook, patched forwards, tests, docs page (operator table, known limits), recipe
and Dockerfile pins, default path unchanged. Reference versions are not proof of passing: distinguish the
head viewed, the version a PR body claims and the version measured.

## Two-step acceptance through the real entry point

The [two-step sequence](workflow.md#two-step-rl-sequence) on the delivered commit/image through
`molt/cli/train_rl_ray.py`, synchronous ([synchronous run](molt-integration.md#synchronous-run)), the formal
recipe shortened only in the number of updates — keep the formal learning-rate schedule rather than
compressing it into two steps; the asynchronous production recipe reads the
[lag floor](molt-integration.md#layout-rules).

## Report template

Every cell is `pass`/`fail`/`inconclusive`/`not run`; out of scope is `not run`, never `pass`; finishing
the short run completes no missing required check.

```
Outcome and scope      verdict for the requested scope; model/layers, samples, valid tokens, dtype, layout,
                       graph/cache path, update count, untested conditions; human-approved checklist link
                       (required vs out-of-scope checks, recipe revision, stop condition)
Immutable experiment   molt / vLLM / nemo-automodel / transformers / torch as imported in the worker, patch
                       hashes, loaded `.so` and JIT hashes, image digest, GPU, checkpoint and input
                       digests, score semantics (processed vs raw logprob, dtype), resolved args,
                       commands; recipe workload (context, length distribution, concurrency, packing,
                       dtypes) and budgets fixed before testing — an unknown budget is not accepted;
                       measured runtime == delivered image, else the gap
Delivery manifest      per deliverable (vLLM PR / nemo-automodel PR or pin / molt PR / image / approved
                       recipe / final validation): URL, base SHA, head SHA, build or acceptance evidence;
                       PRs complete against the agreed base, not stacked; image: base and final digests,
                       pins, worker imports, no source mounts; recipe: script SHA, resolved config,
                       approval, differences from the shipped recipe; validation: commands and artifact
                       ids tied to that image, proxy / zero-update / θ0–θ2 / quality / performance apart
Proxy coverage and     feature → proxy layer → test, omitted depth/geometry/topology named; per operator:
gradient contract      visible forward, trainables, rounding/STE rule, fixed discrete ids, VJP collectives,
                       saved state and its memory; independent reference (why it does not call the tested
                       backward), dY coverage (random, zero, sparse, captured), per item missing /
                       non-finite, max abs, relative L2, cosine; thresholds with reasons, one-step deltas
Symptom and            per-request/token diffs, A/A and A/C, side isolation, last-equal / first-unequal
localization           boundary, same-input op compare, competing explanations
Cause and change       violated contract, smallest trigger, repair vs diagnostic, files/kernels/flags,
                       regression kept, what BI=0 users see (nothing); "unknown" if unproven
Verification           entry-gate table first (workflow.md) with command, artifact, valid tokens and
                       escalation decision per row; inference-only and cross-stack parity never merged
                       into one row. Then: gate | status | command | artifact | closure | scope
                       limitation; rows: original failing replay; operator regression; engine layouts
                       (C3); native old-logprob and grad-enabled forward (C5, two rows); backward
                       inventory and contract (C6); module VJP + one optimizer step and three-arm SFT
                       (C7); real backward and update; θ0/θ1/θ2 (C8) with per-θ receipt (mask identity,
                       valid tokens, `bitwise_frac`, trainer forward executed, update norm); first
                       post-resume refit/compare; quality; clean shutdown (worker exit checked at final
                       shutdown, not per θ); final immutable image without overlays
SFT controls and       the three arms of training-side.md (BI=0 native / aligned + native backward /
quality                aligned + reference backward); controlled variables; raw train, held-out and
                       fixed-probe loss, grad norm, update RMS, clipping, non-finite and skipped-step
                       counts on one axis (optimizer steps or tokens); window and threshold fixed first;
                       "smoke only" when it is; paired quality cells and threshold
Performance            basis (rollout tok/s, fwd+bwd, makespan), matched work, warm/cold, node and order,
                       repeats, raw pairs, exclusions and the invalid-leg rule, formula and aggregate per
                       c, speed vs numerical switches, causal vs each-side-best, prompts vs expanded
                       requests, lengths and completed tokens, wall-time denominator, peak memory and
                       launch counts; measured, rejected and estimated apart; no summed overlapping GPU
                       ranges or ablation percentages
Evidence corrections   superseded claims with the correcting artifact; retained bundles and budget;
                       missing eval steps; a curve is never filled from another policy lineage
Remaining work         gaps and the smallest next experiment; production hooks/config restored?
```

```yaml
purpose: "frozen-policy old-logprob replay"          # each variant changes one variable
target: "bitwise selected-token fp32 logprob"
rollout: "<--train.rollout_dump_dir output, rollout_step{N}.pt>"   # the saved batch, not exported scores
checkpoint: "content id + pre-update path"
environment: "image digest, molt/vLLM/nemo-automodel commits + patch hashes, GPU"
topology: {trainer: "tp/cp/ep/dp", engine: "tp/ep/dp"}
state: "RNG, graph/eager, cache and reset policy"
budget: {max_runs: 3, timeout_seconds: 600}
variants:
  - {name: A, change: "none (A/A)", cmd: "", output: "", exit: null, result: ""}
  - {name: B, change: "none (A/A repeat)", cmd: "", output: "", exit: null, result: ""}
  - {name: C, change: "packing reversed", cmd: "", output: "", exit: null, result: ""}
```
