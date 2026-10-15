# Debugging a mismatch

Find the earliest divergence, not the largest; isolate the side before bisecting; classify before touching
precision. One variable per experiment; a path that did not execute is `inconclusive` ([workflow.md](workflow.md)).

## Symptom → first step

| Symptom | Do first |
|---|---|
| Rollout and old-logprob differ at θ0 | [Isolate the side](#isolate-the-side-first): whichever two controls agree rule that side out |
| One stable failing sample | Same-prefix sibling; save that rollout, replay it alone and with its original neighbours, then bisect by layer |
| Identical operator inputs still differ | Kernel or arguments differ: executed kernel name, accumulation dtype, splits, tile, epilogue, fusion boundary; operator table in [molt-integration.md](molt-integration.md) |
| Only under graph / compile / cache | [Execution state](#execution-state) |
| Fails after changing packing / CP / TP / EP | Trainer packing A/C and the [layout rules](molt-integration.md#layout-rules); a C failure may be a page, tile, segment or expert boundary rather than the packing code |
| Many identical samples in a group | [Sampling seed collision](#sampling-seed-collision): a sampling defect the mismatch metric cannot see |
| Differs only after resume | [Resume](molt-integration.md#resume): masters restore bitwise, first refit bytes equal the pre-save refit, derived state reset |

## Isolate the side first

μ ≠ π (rollout vs trainer logprob) does not say which side moved. Same weights, same tokens, four controls:

| Control | How | Tells you |
|---|---|---|
| Engine A/A | Same request twice, separate processes when in doubt | Run-to-run nondeterminism |
| Engine alone vs neighbours and position | One request vs inside a full batch, and moved to another slot | Batch invariance |
| Engine prefill / chunked / decode | Score one history by full prefill (`SamplingParams(prompt_logprobs=0, max_tokens=1)` in a standalone script), chunked prefill and token-by-token decode | Prefill/decode invariance (attention splits, state-layer chunk grid) |
| Trainer A/A and packing A/C | Recompute twice; alone vs packed with others, permuted | Trainer determinism, packing invariance |

- **Verdict.** Trainer A/C fails, engine stable → trainer layout first. Both stable alone → weights, operator
  semantics, scorer (run both scorers on one identical logits tensor). `prompt_logprobs=0` = no extra top-k, not
  off; the first response token is predicted at the last prompt position — never shift. Same-prefix siblings
  (`n_samples_per_prompt` > 1) must agree over the shared prefix only; μ equal, π differ → trainer layout/state.
- **Per-sequence pattern** (valid/mismatching tokens, first failing position, longest run, lengths, group, packing
  slot, rank): from token 0 → prefill, weights, scorer or a shift (fix that token first with a prompt-only
  recompute); after a boundary (KV segment, chunk, page, packing edge) → state layer or split policy; growing →
  propagation, only the first position is independent; sparse spikes → discrete flip or a last bit across a cast;
  all drift → weights (lag) or scorer dtype; only after an update → refit. Check shift, mask and EOS first;
  ±1/±2 offsets are diagnosis only (the dump's own comment names `actor[i] ≈ vllm[i+1]`).
- **Cheapest control.** Save a small batch of real prompt+response ids, prompt lengths and mask; only score from
  then on (never regenerate or re-tokenize); score one version repeatedly per side for its noise floor, then
  early vs late weights when the gap grows with training (any lag ≥ 1 reads the bf16 floor).
- **Measurement basis.** Per valid token, the fp32 logprob of the sampled token (temperature applied on both
  sides), vLLM's vs the trainer's, as bytes — never a batch mean: `vllm_kl` is a signed masked mean that cannot
  tell a last bit from 0, and rollout logprobs exist only when `is_correction_level` ≠ off or score centering
  is on; `MOLT_DUMP_ROLLOUT_LOGPROBS` is one sequence, rank 0, once, 6 decimals. Like with like (`processed_logprobs`
  vs the scorer's fp32 temperature divide, `top_p` = 1). Report the `mismatch/*` set; k3 on an old trajectory is
  a diagnostic, not the policy KL; relative L2 / cosine are for hidden tensors only.
- **Validity of controls.** Cache off may run a prefill that never ran; eager changes scheduling: count executions
  of the target branch, one round proves nothing, a switch that also changes scheduling is not attributable.
- **CUDA async errors** surface later than the faulting kernel: `CUDA_LAUNCH_BLOCKING=1` or scoped
  `torch.cuda.synchronize()`, one process per variant, keep every rank's log; a corrupted context yields no
  numerical evidence; swapping the attention backend is not a fix.

## Save and compare protocol

- **Archive by step.** `steps/INDEX.md`; `steps/<NN>-<gate>/<run-id>/{config,scripts,results,summary.md}`; large
  tensors in `results/<run-id>/`, not in git. Per run: launch command, resolved args, image digest, molt / vLLM /
  nemo-automodel commits plus dirty patch, evidence of the executed backend (a flag is not execution), exit code.
  New run = new id; invalidated conclusions are marked, not deleted.
- **Replayable raw data.** `--train.rollout_dump_dir` writes `rollout_step{N}.pt`: the sample list after
  generation, before the old-logprob forward (tokens, masks, rollout logprobs, routed expert ids under replay).
  Add the weight version (step N's rollout comes from the policy before update N; the post-update checkpoint
  does not reproduce it), sampling params, resolved engine config, packing map, parallel layout. Under partial
  rollout or async lag, segments of one sequence come from different policies and molt stores no per-segment
  version: dump with partial rollout off and `--train.force_sync_mode`. `--train.rollout_replay_dir` trains on
  the dump without generating (a full step: old-logprob forward, update, refit) — verify the save fired,
  counts match the log and the file reads back.

### Compare per-token logprobs

Each side exports `{request_id: (logprobs, response_mask)}` as original-dtype tensors via `torch.save` — no
JSON, no printed logs. Rollout side: the `rollout_step{N}.pt` dump; trainer side: save `action_log_probs` and
`action_mask` next to the `mismatch_metrics` call in `policy_actor.py`, keyed by the same request ids. Align by request id, not row; request sets must match (error, not intersection); masks
equal; ≥ 1 valid token per sequence; hash the input tokens first. Positions after the first mismatch are not
independent for history-dependent layers: count them, localize on the first. Pass = `bitwise_count` 0 (the offline count; the live key is
`mismatch/bitwise_frac`) with `valid_tokens` > 0 and `nonfinite` 0; equal sampled-token logprobs do not prove equal logits.

```python
def compare_logprobs(a, b):  # per side {uid: (logprobs, mask)}, original dtype, same uid set
    assert a.keys() == b.keys(), "request sets differ"
    n = bad = nonfinite = 0; first = None; max_abs = sum_abs = 0.0
    for uid in sorted(a):
        (x, mx), (y, my) = a[uid], b[uid]
        assert x.dtype == y.dtype and torch.equal(mx, my) and mx.any(), f"{uid}: dtype, mask or no valid token"
        x, y = x[mx], y[my]
        itype = torch.int32 if x.dtype == torch.float32 else torch.int16
        diff = (x.view(itype) != y.view(itype)).nonzero()
        if len(diff) and first is None: first = (uid, int(diff[0]))
        d = (x.double() - y.double()).abs(); max_abs = max(max_abs, d.max().item()); sum_abs += d.sum().item()
        n += len(x); bad += len(diff); nonfinite += int((~torch.isfinite(x) | ~torch.isfinite(y)).sum())
    return {"dtype": str(x.dtype), "valid_tokens": n, "bitwise_count": bad, "nonfinite": nonfinite,
            "abs_diff_max": max_abs, "abs_diff_mean": sum_abs / n, "first_fail": first}
```

- **Per-layer trace.** Same semantic boundaries on both sides under the same packing (embedding, each layer's
  attention/scan output, router ids, residual, final norm, logits), keyed `(request_id, boundary, logical slice)`
  after stitching TP/EP shards by global coordinates and stripping padding. Hash first —
  `hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()` plus shape, dtype, execution
  index — after `clone()` (engine workspaces and graph output buffers are overwritten by the next call); raw
  tensors only at the first differing boundary, which is not the cause until its input hash is shown equal.
- **Replay controls.** One process per variant through a synchronous entry (no fire-and-forget `sbatch`), a new
  directory per variant with command, env, exit code, time. Prove frozen: weight fingerprint before and after,
  no generation, no optimizer step, the packing actually in effect. Variants touching compile, libraries,
  autotune or init env start a fresh process; any cache fingerprint binds weights, input and config.

## Bisect and first divergence

- **Search axis**, one at a time: version (good/bad on the same reproducer; save the full dirty patch, a SHA is
  not a program), config (paired single factor, 2×2 when coupled), data trigger (subset, length, neighbours,
  boundary remainder), compute path (layer → attention/MoE/norm → operator → metadata, launch order).
- **Single-request prefill first.** (1) Existing logs: metric definition, weight version, sample source. (2)
  Fixed-input scoring, same weights both ends, repeated for the noise floor. (3) Minimal forward: one request,
  one prefill, one output token, no cache hit, warmup/draft/decode calls told apart; eager does not stand for
  graph. (4) Earliest divergence: embedding, block outputs, logits at the last valid prompt position. (5)
  Same-input operator replay. (6) Back to the original failing path at full depth, decode and graph.
- **Dump order inside the suspect layer:** norm in → norm out → q/k/v projections (combine a fused QKV before
  comparing with separate q/k/v) → position encoding (q/k after RoPE, q/k norm in vLLM's order) → attention
  core / SSM scan / conv → router logits, ids, weights → expert outputs → combine → output projection → residual
  add (the fused add+norm returns the new residual). Core and output projection get separate boundaries: a
  fused capture once held a `7.8e-3` difference through four configs until the split put it all in o_proj.
- **Align probes by semantics:** tensor meaning, dtype, global token coordinate, valid mask — not module names or
  tuple indices (FSDP wrapper names, `(hidden, residual)` pairs, THD vs padded, LSE ln vs log2, before/after
  norm or residual). EP dispatch reorders, duplicates and pads tokens: align expert tensors by (token, expert,
  slot), then compare reduction, shared+routed combine and post-compile rounding separately.
- **Credible observations:** a hook never replaces an output; under FSDP sample boundaries already materialized
  in this forward; count forward vs activation-recompute vs graph-replay so the recompute does not overwrite the
  forward; refuse comparisons with missing boundaries or incompatible shapes; non-finite fails explicitly.
- **Recommended log line** (reuse existing logs; terminal prints the summary only; unknown stays unknown):
  `[mismatch] run= step= weights=<digest> input=<sha> phase=<prefill|decode> rank= repeat= scope=<fixed|live>
  intervention=<none|name> metric= value= self_repeat=<noise> report=<path> class=<semantic|…|unknown>`.
- **Limits of bisect.** A difference can vanish downstream in rounding, masking or routing, so "layer k differs"
  is not monotonic in depth: coarse scan → suspect interval → fine scan → re-verify the boundaries before it. A
  cut point carries full state (residual pair, KV and block table, SSM/conv state, positions, routing ids,
  `cu_seqlens`). Only a same-input A/B separates propagation from a local error; injected reference activations,
  forced routing or a swapped kernel prove causality but are diagnostics — accept the fix on the restored path.
- **Reviewable root cause:** minimal input, last-equal/first-unequal boundary, same-input control, mechanism,
  trigger, minimal change; a config difference is a lead, not a cause. **Stop:** three rounds of only "final
  score differs" → add probes or re-check the sample mapping, do not rerun; stop when this round's goal is met,
  no config changes along the way; output replacement is diagnosis, not a fix; ablation gains do not add up.

## Classification and handling order

| Class | Signature | Action |
|---|---|---|
| Semantic | One side lacks or applies a feature (sinks, sliding window, q/k-norm position, `1 + w` norm, RoPE variant) | Port the feature; no numerical work until both sides agree |
| Discrete | Same scores, different selection, order or tie-break | Fix the ordering contract; replay routing (R3) if the engine's choice must be reproduced |
| Reduction order | Same inputs, last bits differ, grows with batch or length | Make the kernel invariant (fixed splits/tiles) or call the same kernel from the trainer |
| Dtype / cast position | Appears at a cast or fp32 island | Align accumulation dtype and where the cast happens |
| State | Only after a chunk boundary, cache hit or decode step | Chunk grid, boundary-state dtype, cache reset, prefill/decode path |
| Execution state | Only under graph / compile / specific shape | Compile boundary, capture shape, autotune: pin the kernel on both sides |
| Weights | Bytes differ between trainer and engine | Refit dtype, fp32-kept params, requantization; hash the same tensor on both sides after `load_weights` (`--train.check_weight_update_equal` only lists parameters the broadcast never assigned) |

Order: semantic → discrete → weights → reduction/dtype → execution state; do not move on until the previous
class is ruled out; `unknown` is a valid label, "just rounding" needs same-input evidence. **Delivery:** re-verify
on the original reproducer at full depth, decode and graph, counts before and after, production path restored;
only then shrink. Nothing here authorizes a dtype change, GPU escalation or recipe change.

## Execution state

| State | Rule |
|---|---|
| Graph vs eager | Python hooks run at capture, not at replay; replay reads captured buffers (refit copies weights in place). Dump the engine in eager and prove graph == eager bytes separately (G3). Piecewise graphs fall back to eager above the capture size — each path invariant, the two not equal — hence `CompilationConfig(mode=NONE, cudagraph_mode=FULL)` in aligned mode; `--vllm.enforce_eager` proves eager only |
| Compile boundaries | Inductor fusions round differently; a same-named module may run a compiled native forward on one side and a CUDA op on the other — check the executed kernel (profiler, nsys), not the Python name |
| Autotune, determinism | Autotune is per process and keyed by the first shapes seen (Triton matmul, `fused_moe`): pin configs on both sides, rerun a possibly re-tuned variant in a fresh process. Determinism switches are scoped: `--train.full_determinism_enable` is seeds, torch deterministic algorithms and a single-process engine; thread-local settings can leave a worker thread non-reproducible while the main thread passes |
| Cache state | Prefix hit == miss == off, bitwise; molt resets the cache after a refit only when caching is on; every cache or fingerprint binds weights, input and config |
| Parallel layout | Trainer TP vocab-shard reduction, engine TP all-reduce, EP combine order, CP concatenation are the contract; bitwise holds for the verified layout only — record it per row |

## Rollout engine pitfalls

### Sampling seed collision

Identically seeded engines sample identical tokens for their n-th request; molt seeds `seed + i` per engine, but
in-engine DP ranks and any engine you add must differ too. Accept on the dump: distinct samples per group and
identical sibling pairs — the healthy baseline is in the [reference measurements](molt-integration.md#reference-measurements);
more is RNG correlation (zero-advantage groups come from task difficulty). Greedy n repeats are identical by
construction and scores move in steps of 1/problems: a greedy A/B cannot see a sampling fix. Never give one
prompt's siblings the same seed.

### Prefix cache semantics

Premise: one request is bitwise identical on hit, miss and cache off. Pure attention gets this from KV block
reuse; state layers (Mamba2, GDN) depend on which states the engine copies — verify. molt: off by default; when
on, `reset_prefix_cache` after every refit (keep-mode pause clears nothing reliably, so a request straddling the
refit can hold old-policy KV — keep partial rollout off so no rollout spans a refit). A hit changes the step's
token count and may change the graph bucket. Diagnose with cache off; hit, preemption and recompute paths get
their own gate; two fixes passing alone may fail stacked.

- **Chunked prefill.**
  - vLLM enables it by default with a version-dependent token budget, and a Mamba hybrid with prefix caching
    cannot turn it off (mamba cache mode `align`).
  - Segmented attention is chunk-invariant; state layers hand state across chunks in another rounding path:
    `--vllm.max_num_batched_tokens` ≥ `max_model_len` keeps a prompt in one chunk unless the step is already
    full, `--vllm.mamba_ssm_cache_dtype float32`.
  - Test 2047/2048/2049 and the segment edges 1023/1024/1025, 16383/16384/16385.
- **Speculative decoding / MTP.**
  - `--vllm.mtp_num_speculative_tokens` > 0: the verification logprob must be bitwise with the full forward —
    unproven, so off for bitwise.
  - Both arms or neither; molt rejects it with routing replay and with prefix caching.
- **Routing replay (R3).**
  - `--train.routing_replay` → `enable_return_routed_experts`, ids per token per MoE layer; 0 is a valid id —
    check range and distinctness, not non-zero.
  - Incompatible with partial rollout (preemption drops routing) and MTP.
  - Tokens the engine did not generate (tool outputs, multi-turn prompts) have no record and fall back to
    natural routing ([reference measurements](molt-integration.md#reference-measurements)): scope MoE claims
    to single turn.
  - vLLM 0.31 returns the ids through its AuxOutput store keyed by prefix-cache block hashes, so prefix
    caching must be on (MoE model, Model Runner V2).
  - Per-step return costs frontend time — profile before blaming kernels; report the disagreement rate replay
    masks.
- **Lifecycle, memory, metrics.**
  - molt refits under the vLLM lock: `pause_generation(mode="keep")` → broadcast → `reset_prefix_cache`
    (if on) → `resume_generation`; in-flight must be 0 before pause, the queue empty after resume.
  - Requests queued during pause run on the new weights and belong to no version.
  - Sleep levels, if added: 1 drops KV and prefix cache and backs weights on host, 2 leaves GPU weights
    uninitialized until reloaded.
  - `--vllm.gpu_memory_utilization` (default 0.95) assumes the engine owns its GPUs — read the resolved value
    and KV budget in the vLLM log.
  - `vllm:*_total` counters are cumulative: use after − before per phase.

### Replay matrix

Use when a difference shows once in many steps — rerunning changes tokens, scheduling, lengths, cache and
packing at once. Save after generation, before the old-logprob forward; replay with `--train.rollout_replay_dir`
from the same checkpoint, no generation (molt's replay still updates and refits, so compare the first replayed
step before its update and restart every variant from the checkpoint); A/A first (unstable A/A → RNG, buffers, streams, state).

| Variant | Change | Answers |
|---|---|---|
| A | Original packing | Does the frozen input reproduce it |
| B | Exact repeat of A | Run-to-run nondeterminism |
| C | Sequence permutation / packing | Layout dependence (or a page, tile, segment, expert boundary) |
| D | Micro-batch size or target position | Call shape / neighbour dependence |
| E | Fresh process or equivalent node | Initialization / device state |

Run only the variants that separate live hypotheses; compare by request id and position after reordering. Shrink
keeping the trigger: target sample and neighbours first, then length and count, rerunning the criterion after
every cut (graph fillers and boundary remainders are often the trigger). Frozen rollout and weights do not
freeze dropout, RNG, replay ids, caches or workspaces; verify A/A between a fresh and a reused process. Leaving
replay: back to generation, real backward, next rollout — training-only success covers the frozen forward only.

## Communication and transport

Both arms on the same transport. In aligned mode trainer and engine workers share the weight-sync NCCL group, so
the NCCL algo/proto/channel pins (vLLM `override_envs_for_invariance()`) must be set in both processes and
whitelisted into the Ray runtime env. World-size sanity: write trainer (DP × TP × CP × EP) and engine (engines ×
TP) groups separately, count GPUs from the runtime, not the config string (two DP4/EP4 groups are not EP8). Run
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` once per arm (`NCCL_DEBUG` propagates through the Ray env whitelist;
`NCCL_DEBUG_SUBSYS` must go into the recipe's node env) and confirm IB vs P2P/NVLink per group. Never compare
arms on different fabrics or nodes; `export VAR=` in a recipe overrides the container's `-e`, write `${VAR:-default}`.

## Evidence hygiene

Beyond [workflow.md](workflow.md#evidence): a fix that changes the sampled trajectory needs a new saved history
(the old one stays a causal diagnostic); diagnostic interventions (eager, cache off, forced routing, output
replacement) are labelled in every row; write `side → first operator → cause → fix → re-verified on the original
reproducer` with counts before and after; keep failing samples and ruled-out hypotheses for the next round.
