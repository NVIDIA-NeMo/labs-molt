# Training side: the aligned forward and its backward

How to build a trainer forward that computes vLLM's math for a new model, and the backward that trains it
(gates C0, C5, C6 and C7 of [workflow.md](workflow.md#gate-table)). Flags, engine settings, patch points and
the per-operator kernel map live in [molt-integration.md](molt-integration.md#the-aligned-mode) and its
[operator table](molt-integration.md#operator-table). One route: an FSDP2-sharded model definition
(AutoModel custom class, else HF) whose value-producing operators are swapped for vLLM ops inside
`torch.autograd.Function`s whose backward is the native training math.

## Module inventory

C0, day one: one row per module, every column filled, `unknown` in blanks, never "should exist".

| Column | Question to answer |
|---|---|
| Module | One row per fusion boundary as vLLM executes it (residual-add + norm is one op), not per HF class |
| Inference kernel, prefill / decode | The function vLLM actually calls, not the class name: attention backend, custom op vs compiled path, the branch under `VLLM_BATCH_INVARIANT=1`; two entries when prefill and decode differ (attention splits, SSM chunked scan vs per-token recurrence) |
| Same-math training implementation | Model-agnostic code with autograd: the AutoModel native module (TE, `grouped_mm` experts, flash-attn), torch, mamba_ssm / causal_conv1d. Only-rounding differences are same math; a different clamp, norm variant (`1 + w`) or router formula is not. Same-model code that appeared after release does not count |
| Quantization rounding | FP8 rounding in the forward, weights and activations separately (BF16 route: none) |
| Discrete choice | top-k expert ids, kept token set, tie-break, ordering; ids are fixed and not differentiated, the continuous score needs a derivative |
| Backward exists | In which library, and whether its saved intermediates (LSE, rstd, token permutation) line up with what the vLLM forward saves |
| Verdict | Forward: already equal / swap trainer op / needs vLLM change. Backward: reuse / glue / new, by the rules in [Backward](#backward-reuse-glue-or-new) |

A module is anything producing a value on the logprob path: embedding, every norm (q/k and final), position
encoding, attention, SSM/conv, router, routed and shared experts, activation, lm_head, every encoder, both
trainer scorers (`log_probs_from_logits` and its TP variant) and the engine's sampler logprob. Parts that do
not take part in training (MTP head, draft model, redundant experts, a tower under
`--actor.freeze_visual_encoder`) get a row with mapping and fact: load and refit skip them by explicit list
and the log says "skipped", not "no error". A part the engine uses (MTP on) is used by both arms ([engine
knobs](molt-integration.md#engine-knobs)); a frozen encoder that both sides run is on the value path —
identical kernels on both sides or embeddings shipped from the engine; freezing does not make bytes equal.

Read vLLM's model file, not the HF modeling file, in this order:

1. Layer types and order; residual and norm positions (`1 + w` variant?); position encoding (neox / mRoPE /
   none); attention visibility (sliding window, sinks, GQA); router (softmax or sigmoid, bias,
   renormalisation, grouped top-k, tie-break); activation; shared experts; state layers (Mamba, conv);
   special embeddings and encoders.
2. The kernel each layer calls and its quantization variant; attention backend and KV dtype.
3. Weight loader: name mapping, fused order, derived scales.
4. Branches under the BI switch; parameters and buffers confirmed from checkpoint and config.

Model-specific math is written from the inference source, not from the paper or the HF modeling file. Sizes
sharing one vLLM model file share one port; a multimodal variant is validated per task.

<a id="call-kernels-do-not-port-the-scheduler"></a>
## Reproduce the forward: call kernels, do not port the scheduler

- Teacher-forced full-sequence forward, boundaries in `cu_seqlens`; borrow the kernel only — no KV paging
  logic, decode scheduling or CUDA-graph management from vLLM.
- Same branch as inference: compile config (`CompilationConfig` mode, custom-op boundaries), attention
  backend and BI-switch kernel selection reproduced in the trainer process (the reference implementation's
  `setup_process()`: env before the first cuBLAS handle, same kernel patches) and in every test wrapper.
- Op-level calls (`torch.ops._C.*`, `fused_topk`, `fused_experts`, the attention op), not vLLM module
  objects (`FusedMoE`, parallel linears), which need vLLM TP/EP process groups the trainer would have to
  create.
- Keep the training parameter container: parameters stay the FSDP2 DTensors behind the fp32 master; inside
  the forward use the full parameter FSDP2 gathered (`full_tensor()` for an explicitly sharded one, as the
  refit's `gather_full_param` does); no parameters or copies the optimizer does not know. Keep the native
  entry point (`model(**batch)`): patch class forwards, do not re-route the call chain. The patch lives in
  source behind a flag and is proved from a fresh process through the real loader; a REPL-only patch or
  monkeypatch is not evidence.
- Reuse order: autograd-capable native module → a short torch re-implementation of vLLM semantics →
  ready-made op with a backward → custom autograd/kernel only when necessary. Throughput is not a goal at
  this stage. What is trained, frozen or carries an auxiliary loss follows the model definition: no
  detaching to get it running, no `--actor.aux_loss_coef` the recipe does not have.
- fp32 where vLLM is fp32 (RoPE angles, router logits and bias, norm statistics) is set per operator
  (`gate_precision`, `MOLT_RMS_NORM=torch_fp32`, [trainer kernel
  choice](molt-integration.md#trainer-kernel-choice)), never by a forward-level `torch.autocast` — molt runs
  none, and one would downcast the fp32-kept gate; casting matmul inputs alone does not stop a downgrade, so
  measure dtypes inside the forward.

Three rules for "the same kernel" that still differs:

1. Direct and compiled paths of one kernel are two numerics: call it through the same opaque custom-op
   boundary on both sides, or with compile off on both.
2. Attribute only after a same-input op test: a parameter that differs between the sides (a reduction block
   size) is a lead, not a cause, until the identical captured kernel on the same input differs.
3. Reproduce the executed cast position: keep the real bf16 head output and share the scoring; never
   truncate an fp32 projection to manufacture equality.

Reproduce the math from the same checkpoint, config and inference code: layer order, parameters and buffers,
residual and norm positions, position encoding, visibility range, router, special embeddings.

- A frozen base still propagates input gradients: `W.requires_grad_(False)`, never `no_grad()` around the
  block (it cuts adapters and encoders upstream; a frozen table lookup is the one exception). A
  zero-initialized adapter shows zero gradient legitimately: judge with a non-zero adapter and a real step,
  not a finite loss.
- Metric-only quantities (`mismatch/*`, `vllm_kl`, entropy) are computed under `no_grad` on detached copies,
  never in place on a tensor the loss needs. molt's LoRA (`--actor.lora_dim`) is off the bitwise contract by
  construction ([refit](molt-integration.md#refit)).
- Mapping checks cover values and semantics, not only shape: per weight class, source → training → engine
  name, axis transforms, sharing (tied embeddings); fused QKV, gate/up order, expert numbering, extra
  norm/bias, non-persistent buffers, derived state; fused parameters overwritten whole; every missing or
  extra item with a written reason. In molt the refit broadcast is that mapping;
  `--train.check_weight_update_equal` lists unassigned names ([refit](molt-integration.md#refit)).
- Stateful modules (conv, SSM scan, recurrence) must consume `cu_seqlens`; a correct attention mask proves
  nothing about them. The engine starts its chunk grid at each sequence's position 0, so the packed forward
  restarts it at every `cu_seqlens[i]`, not at the row offset, and decode must reproduce the chunked
  reduction order. Test unequal lengths alone and merged; sentinel-fill outputs with a different value per
  side to catch missed writes (zero-filling both sides reads as a false green).
- Recompute: under `--actor.gradient_checkpoint full|selective` the recompute must read the same saved state
  (routing ids, RNG) and be bitwise — a deterministic forward is; under EP > 1 molt keeps the router
  projection saved under AC (`ignore_router_for_ac`); on the EP = 1 path verify it yourself — a near-tie
  token must not re-route in the recompute.
- Proxy: the smallest layer set covering every non-equivalent path (usually 2–6 layers): keep mixed
  attention, cross-layer sharing and lookups that live in specific layers; record the kept layer indices and
  config transform; it proves the covered paths only.
- Proof: C5 on vLLM-captured trajectories, both paths of the [dual-path
  contract](evidence.md#dual-path-contract), `mismatch/bitwise_frac` 0 and replay count 0
  ([debugging.md](debugging.md#compare-per-token-logprobs)). Under default kernels (BI=0) there is no 0-diff
  target: judge each side against an oracle, then attribute what remains to precision.

Order of work: inventory → module comparison (same weights and inputs; forward bytes, gradients vs an
oracle) → proxy training (non-zero loss, finite gradients, the intended parameters change) → save and
restore in a fresh process. Every step ends with a short conclusion, the source version, a reproducible
command and an artifact; distribute only after the first two pass, scale up after the proxy two-step loop
passes ([ladder](workflow.md#cheapest-failure-criterion-first)). Cost is judged per blocker (an operator
without a backward or clear semantics is its own item), never as a uniform figure.

## Layout, length and world size

Record trainer and engine topologies separately. The trainer mesh is nemo_automodel's
`_create_device_meshes(ParallelismSizes(tp, pp, cp, ep), world_size)` (`molt/trainer/fsdp/strategy.py`): `dp
= world / (tp × cp × pp)` is inferred; FSDP shards and reduces over `dp_cp = dp × cp` (CP ranks share
samples and split sequence work; gradient accumulation counts `dp` only); EP is a separate `moe_mesh` whose
`ep_size` divides `dp × cp × tp` — it shards expert state over those ranks and does not multiply into the
world. The startup line `[FSDP] world= dp= cp= tp= ep= dp_cp= grad_accum=` states the resolved layout;
record it verbatim next to the engine's `--vllm.tensor_parallel_size`, `--vllm.enable_expert_parallel` and
engine count.

Budget the aligned forward's memory before asking for GPUs: fp32 attention partials sized by the context
ceiling ([operator table](molt-integration.md#operator-table)), one paged KV copy per packed sequence, the
fp32 recompute in backward, chunked fp32 logits in the scorer, the refit's per-parameter `full_tensor()`
gather. `--fsdp.offload optimizer` changes the optimizer's arithmetic, not the forward; `full` also streams
parameters through CPU (slower forward; the flag's help lists MoE breakage).

Packing: `--fsdp.packing_samples` flattens the microbatch to one row with `cu_seqlens` (`cu_seq_lens_q` on
the HF path) and per-sequence positions; the aligned attention builds one block table per packed sequence so
bytes do not depend on neighbours — prove it (alone vs packed with 1/3/7/15 others, bitwise). In molt the
packed path runs only at `cp_size == 1` (`base.py`); under CP the padded `[B, S]` batch is sharded by the
model-owned `ContextParallelSharder`.

CP > 1 (the reference implementation guards `cp_size == 1`; a CP claim needs its own C3/C5 proof):

- Attention is exact only with an all-gather of K/V and the local query rows at their global positions;
  AutoModel's TE path selects `cp_comm_type="all_gather"` for sliding-window layers and `p2p` (ring
  exchange, another softmax merge order) for full attention — inventory which runs.
- Mamba2 is exact when the layer runs whole sequences of a head slice: AutoModel's hidden-parallel
  `MambaContextParallel` all-to-alls to `[B, L_global, D / cp]` and back.
- The pad to a multiple of `2 × cp_size` before sharding is TE's THD layout (two chunks per rank for load
  balance); labels, positions and R3 ids are padded with it (`−1` keeps the live routing).
- Prove rows at global positions: alone vs CP-sharded, bitwise, before and after the shard boundary.

EP and collectives: keep the AutoModel dispatcher (`MOLT_MOE_DISPATCHER`) for moving tokens; the aligned
expert forward runs on one rank's local tokens with the engine's fused kernel and tile config; the combine
(fp32, top-k slot order) is part of the contract. Record expert mapping, tokens per rank, padding capacity,
route order; an empty rank still joins the collective; reused buffers get sentinel fills. NCCL pins remove
reduction differences at a throughput cost — a verified choice that must match across the weight-sync group
(transport checks: [debugging.md](debugging.md#communication-and-transport)).

Bitwise identity after changing the parallel layout. Addition is not associative: the layout decides what is
summed in which order and where it rounds, so "bitwise" holds for the verified layout only.

1. The reduction tree is part of the contract — per level the dtype, order and rounding points; every legal
   engine layout and the trainer walk the same tree (engine TP all-reduce, EP combine, trainer TP
   vocab-shard `log_probs_from_vocab_parallel_logits`: all-reduce max, then sum).
2. A degenerate layout (TP=1, EP=1) is not automatically equal to a multi-rank one; it simulates the tree:
   group by rank, partial sums, round at the same places, merge in the same order.
3. After a layout change compare raw bytes on fixed inputs first, then train; never revert to the old layout
   to get around a non-zero mismatch.
4. Locate by layer-wise dump: the first differing layer whose inputs (router logits included) are bitwise
   has the fault in its reduction ([debugging.md](debugging.md#bisect-and-first-divergence)).
5. Simulating an order costs time; measure it and report it apart from the layout change's gain.

The reference implementation verified trainer EP=4 / TP=1 against engine TP=4; any other pair needs its own
proof. Asynchronous communication lifetime: check producer/consumer streams, events, buffer owners and reuse
time (dispatcher, refit broadcast); passing after a global synchronize only proves a timing fault — find the
missing dependency locally and re-accept on the same path.

| Long-sequence boundary | Fact | Test |
|---|---|---|
| Segment ceiling | The aligned attention's context ceiling ([operator table](molt-integration.md#operator-table); 16k in the reference implementation) is a correctness bound shared by both sides, not a performance knob | Exceeding it raises, never truncates; two real lengths below it; decode == prefill at segment edges (1023/1024/1025 …) |
| Chunk boundaries | Engine prefill chunking (`--vllm.max_num_batched_tokens`) and the SSM chunk grid are independent numerical boundaries | Chunked == full prefill == token-by-token decode at the chunk edges (C3); one sequence crossing an SSM chunk edge |
| KV block | The trainer's paged KV copy has block edges (16 tokens in the reference implementation) | Sparse dY across a block edge |
| Packing prepend | Prepending a short sequence shifts every row offset of the target | Two prepend controls vs none, bitwise |
| Microbatch fits the longest row | `--train.max_tokens_per_gpu` / `--train.micro_batch_size` must hold the longest packed row at the formal length | One OOM probe at the formal length; no extrapolation from 8k |
| Embedding backward | CUDA `embedding_dense_backward` takes another accumulation path for small index counts | Tolerance tiered by length |

Microbatch split and gradient difference: the same batch under another split gives a measurably different
gradient (loss normalization, accumulation order, the embedding backward above; magnitude in the [reference
measurements](molt-integration.md#reference-measurements)). Not a bug, but the criterion denominator is
`max(bf16 rounding floor, implementation noise floor)`, never the fp32 ideal; both arms use the same split;
a split change means A/A first; `grad_norm` is not comparable across microbatch counts.

<a id="control-matrix"></a>
Control matrix — what to re-run when an axis changes: the [change → rerun
table](evidence.md#candidate-evidence-ledger). With policy and tokens fixed also run the original layout,
reversed order, target first/last, a changed microbatch and one sample crossing each boundary; a fix passes
on the original triggering layout and its neighbours — lowering the trigger probability is not a fix.

## Backward: reuse, glue or new

Backward is determined by the math, not the forward kernel: `y = f(x, W)` needs the formula and the saved
inputs (`dX = G·W`, `dW = Gᵀ·x`). A vLLM forward kernel does not call for a new backward, and "no backward
at the same precision" is no reason to write one — the backward runs in bf16/fp32 anyway.

Wiring: `visible_forward(visible, native, *inputs)` is one `torch.autograd.Function` per value-producing op.
Its forward calls the vLLM op without a graph and saves inputs plus the intermediates the native backward
consumes (LSE, routing ids); its backward recomputes `native` (flash-attn varlen, `grouped_mm`, fp32 torch)
from them and differentiates it. Never return `visible.detach()` as the trainable value. Where a fused
kernel has no backward at all, `visible + (graph − graph.detach())` with `graph` the same math in PyTorch
keeps the kernel's bytes: eager only (a compiler may rewrite the add/subtract), never as the oracle bridge
of layer 1, never modified in place downstream; it holds the surrogate graph in memory, which the `Function`
form avoids; `x − x` is exactly +0 and only a −0.0 result flips sign, so keep dtypes equal.

| Case | Signature | Examples | Do |
|---|---|---|---|
| A: only rounding differs | Same math; summation order or chunking differs; last-bit difference | Linear, RMSNorm, attention, SiLU·up, RoPE, SSM scan, conv1d | Reuse the native backward; write no math |
| B: forward has rounding | FP8 quantization rounds; the true derivative is 0 almost everywhere | FP8 GEMM, FP8 Q/K/V ([off the BF16 route](contracts.md#fp8)) | Fix the STE contract (rounding passes as 1; dX uses the dequantized value by default), then reuse the bf16 GEMM backward; quantify in [layer 2](#layer-2-gradient-contract-comparison) |
| C: no same-math implementation for the segment | Composite of several ops, no library backward for the whole | Routed experts: sort → grouped GEMM → activation → restore → top-k weighted sum; model-specific scores | Glue over ready-made ops (`grouped_mm`, analytic activation derivative, differentiable combine), compared against an oracle |

Decision flow: rounding in the forward (top-k id selection does not count)? no → A; yes → fix the STE →
packaged backward for the whole segment? yes → B, no → C. Look in the AutoModel native module and torch
autograd first, then TE / flash-attn / mamba_ssm. What forces glue (none of it "missing same-precision
backward"): a library backward bound to intermediates of its own forward (rstd, token permutation) that the
vLLM forward lacks — recompute them; model-specific math in no library; discrete selection that must copy
the forward (a top-k recomputed in backward can pick another expert on a tie — store ids, differentiate only
the continuous score); generic autograd recompute too costly in memory or time (a pure-torch SSM scan is an
oracle, not a production backward) — use a closed form or the library kernel.

Three rules when reusing:

1. Intermediate definitions match: attention backward takes the forward's O and LSE — ln vs log2, softmax
   scale folded or not, per-head vs per-token layout; GQA dK/dV summed within the head group in a fixed
   order (an A/A needs it). If they do not match, recompute the forward with the training kernel inside the
   backward (what the reference implementation does); O, LSE and P come from one forward — the kernel's O
   inside `D = rowsum(dO⊙O)` against a recomputed P leaves `rowsum(dS) ≠ 0` and biases dQ (systematic once P
   is FP8-rounded).
2. Backward is not required to be bitwise: recomputed intermediates differ in the last bf16 bit, gradients
   by the same amount; the contract must hold and the error sit near the noise floor. Report bitwise and
   tolerance apart.
3. Reused code is verified too: an fp32 oracle that does not call the backward under test, the same dY, the
   dX / dW error in multiples of the noise floor.

False differences (test errors, never a train/inference difference):

- FSDP2 unsharded copy after a forward-only pass: the root module is not resharded (and with
  `MOLT_MOE_RESHARD_AFTER_FWD=0` experts stay gathered), so a parameter fetched by name is the bf16 compute
  copy, not the fp32 master; perturbing it or reading its `.grad` misleads. Reshard or finish a backward
  first.
- The patched path was never hit: an op instantiated directly misses the production context (compile config,
  env before the first handle, forced attention backend), or an import without GPU visibility makes vLLM
  pick a non-CUDA platform; the native path is compared with itself. Assert the target callsite was hit
  (kernel name in the profiler, a counter in the patch); use the production config.
- dY fp32 on one side and bf16-rounded on the other; dumps taken at different pipeline points (before vs
  after the FSDP reduce, EP scaling, loss normalization) read as a multiplicative gap; "frozen" by zeroed
  grad — AdamW weight decay still moves it; a real freeze is `grad = None` or removal from the optimizer.

## Gradient contract and validation

Forward 0 diff proves the same logprob, not correct gradients. Three layers, run 1 → 2 → 3: a layer-1
failure is fixed before anything else; a layer-3 anomaly sends you back to 1 and 2; none proves RL
stability.

| Layer | Question | Method | Cannot replace |
|---|---|---|---|
| [1. Module-level VJP](#layer-1-module-level-vjp-comparison) | Does the implementation equal the contract (missing scale, wrong transpose, broken chain) | dX, dW, other params and one optimizer step per module vs an oracle | 2, 3 |
| [2. Gradient contract](#layer-2-gradient-contract-comparison) | Is the contract itself sound — does an approximation bias training | Same batch, gradients under both conventions per class and per layer | 1 |
| [3. Proxy SFT three-arm](#layer-3-proxy-sft-three-arm-control) | Does the whole chain carry a systematic bias | Three SFT arms, same data and hyperparameters | 1, 2 |

Write the contract first. A weight has three tiers and the gradient must return to the first:

| Tier | In molt | What it is |
|---|---|---|
| master | fp32 FSDP2 shard | Optimizer state; updates accumulate here |
| training-forward | bf16 via `MixedPrecisionPolicy(param_dtype)`; fp32-kept params (gate, norms per class) | The value the aligned kernel multiplies |
| deployed | The bytes vLLM holds after refit (`load_weights` casts to the engine dtype) | BF16 route: the training-forward bytes; FP8 route: `requantize(master)` |

Keep the optimizer's parameter identity; the deployed value is not trainable. Per module class write down:
the visible forward (op, I/O dtype and layout, cast positions, accumulation precision; FP8: quant positions,
scale granularity, rounding); the trainable variables (input, master weight, bias, norm weight, router; what
is frozen); the surrogate (dX from master or dequantized value; identity or clipped STE and the value at the
threshold — BF16 route: none); the discrete state (ids from R3, kept token sets, ordering; continuous
combine weights still get gradients); the backward (reused or glue; cast, accumulation and reduction
precision, collectives, normalization); save/recompute (inputs, LSE, ids, RNG; the surrogate never switches
on recompute); the oracle, shapes and paths, metrics and tolerances fixed.

Which weight dX uses: `Y = X·Wᵀ + b`, upstream `G`: `dW = Gᵀ·X_ref`, `db = Σ_tokens G`, `dX = G·W_ref`. BF16
route: `W_ref` is the bf16 training-forward weight the kernel used, i.e. the deployed byte — no choice. FP8:
dequantized (default; exact derivative of the executed forward) or master (reuses the plain GEMM backward,
ignores the quantization error in dX); the master choice makes [layer
2](#layer-2-gradient-contract-comparison) mandatory. State `X_ref` (original or quantized activation)
likewise. Before reusing a linear or grouped backward check flatten/transpose, padding and valid tokens,
bias reduction, saved inputs, scaling, accumulation dtype; an equal output shape proves nothing.

### Layer 1: module-level VJP comparison

Oracle: torch ops and autograd, never the backward under test; where a discrete op is involved implement the
agreed surrogate explicitly (do not differentiate bare rounding and call the zero an oracle). Oracle path:
visible value from the inference op under `no_grad`; surrogate graph from the same-math torch code in normal
grad mode; a bridge Function whose forward asserts shape/device/dtype and returns the visible value and
whose backward passes the gradient to the surrogate. Steps:

1. Fix the state (inputs, weights, mask, routing ids), independent copies, cleared grads; check the visible
   forward equals the inference bytes, then the saved state; at integration confirm the real forward
   produces it.
2. Same upstream gradient, same dtype, representable values: fixed random, zero, sparse (non-zero at chosen
   positions), then gradients captured from real training (synthetic inputs are labelled as such).
   `y.sum().backward()` alone misses transposes, permutations and broadcasts.
3. Compare per item — dX, dW, every bias/norm/router/scale: missing, shape, finiteness, non-zero where
   expected; unused differentiable inputs listed, not ignored. Report max abs error, relative L2 `‖g −
   g_ref‖₂ / max(‖g_ref‖₂, eps)`, norm ratio and cosine (undefined at zero norm → absolute error).
4. Tolerances fixed in advance from dtype, reduction length, oracle precision and this model's own noise
   floor and baseline (never another model's number), per module, reasoning recorded; never relaxed after a
   failure, never with "STE is approximate anyway" for a same-semantics kernel.
5. Coverage: small M (decode-like) and large M, several lengths, unaligned tails, padding/packing/mask,
   zeros and extremes; MoE: empty experts, repeated routing, shared experts, top-k ties; shared parameters:
   gradient accumulation across use sites; under the target dtype and layout, activation checkpointing,
   gradient accumulation and the real parallel reduction (fp32 reduce, EP) — local first, then integrated.
6. One-step update from the same parameters and optimizer state; compare the master change. Gradients close
   but updates apart → scaling, clipping, optimizer state, parameter binding.

Smooth modules can be cross-checked with small high-precision gradcheck / finite differences (discrete
branch fixed, away from non-smooth points); never accept an STE with finite differences of a quantized
forward.

| Module | Reuse first | Semantics to check |
|---|---|---|
| Linear / QKV / lm_head | Native linear VJP; gradient returns to master | dX weight convention, cast, bias, tied-weight accumulation |
| Norm / RoPE | Native VJP, fp32 statistics | eps, `1 + w` variant, position offset, neox layout, where cos/sin were built |
| Activation | Analytic derivative composed as the forward fused it | Saturation, fusion order (`silu_and_mul` on `cat(gate, up)`) |
| Attention | Training backward with the same mask/index semantics | Causal boundary, softmax scale, LSE definition, GQA group sum, O/LSE/P from one forward, per-sequence block table / `cu_seqlens` |
| MoE experts | Grouped VJP plus a differentiable combine | Inverse dispatch permutation, repeated tokens, empty experts, shared experts, top-k weight normalization, weight before or after the down GEMM |
| Router | Ids fixed, derivative of the continuous score | No gradient for ids; the router output is never detached whole; the fp32 gate stays fp32 |
| SSM / linear attention | Training backward with the same chunk arithmetic | Chunk boundary, initial state, state dtype, conv state |
| Selected logprob | log-softmax / gather VJP matching the definition | Vocab-shard reduction, temperature divided in fp32, token mask, loss normalization |

### Layer 2: gradient-contract comparison

When the contract holds an approximation (dX from master, any STE), fix checkpoint and batch, compute dX
under both conventions for every affected layer, and report cosine and norm ratio per class and per layer
with outliers named, per precision combination. On the BF16 route there is no approximation and layer 2 is
the written statement of that fact; it becomes mandatory the moment FP8 enters. A result holds for the
precision, learning rate and run length measured — re-measure when any of them grows.

### Layer 3: proxy SFT three-arm control

| Arm | Forward | Backward | Purpose |
|---|---|---|---|
| BI=0 arm | Default kernels (`BackendConfig`) | Native autograd | The recipe's baseline |
| Aligned + native backward | Aligned forward | Native backward (production) | The deliverable |
| Reference arm | Aligned forward, same snapshot | Oracle surrogate (the layer-1 bridge) | Isolates backward implementation error |

Do not start SFT until layer 1 passes; build the BI=0 arm's model before the process-wide patch.

- Order: first compare gradients and the one-step update of deliverable vs Reference arm on a fixed
  checkpoint and batch, then run the curves; after the trajectories diverge later agreement is not assumed;
  the deliverable-vs-BI=0 gap includes forward-path differences and is not all backward; when the budget
  forbids the full Reference-arm curve, compare representative batches and state the missing range.
- Controlled: initial weights and optimizer state, data and order, labels/mask, packing, loss normalization,
  global batch and accumulation, learning rate and schedule, clipping, precision, seed, recompute policy.
  Recorded: train loss, NLL on a fixed validation set, gradient norm, update RMS, clip ratio, non-finite
  count, throughput, on an x-axis of optimizer steps or cumulative tokens; window and acceptable deviation
  agreed before the run.
- Localization: loss differs at the initial weights → forward, data, mask; initial loss equal, one-step
  update differs → backward, reduction, scale, clipping, optimizer; divergence after several steps → save
  the first abnormal checkpoint, check recompute, learning rate, RNG, then a same-weight replay. Unexplained
  → `inconclusive`; forward 0 diff does not cover it.

Freeze the criterion; prove the update lands on the master:

- The criterion (tolerance, denominator, thresholds) is frozen before the run, its hash recorded, never
  edited. Planted error first ([workflow.md](workflow.md#evidence)). Several snapshots (e.g. steps 0 / 50 /
  150 / 250): a step-0 pass does not carry. On failure, bisect by injecting oracle gradients block by block.
- The gradient export path gets its own A/A (‖g‖ bitwise, local shards equal to the dump, replica diff 0).
  Cross-process bitwise backward is not required, but a sporadically non-bitwise first step has a cause
  (compile autotune by timing, a thread-local deterministic config) and is found, not tolerated.
- The optimizer consumes the clipped gradient, the update lands on the fp32 master, and the deployed bytes
  after the next refit equal the cast of master (FP8: `requantize(master)`); every parameter-group type is
  covered (weight-decay-0 groups, fp32-kept params), not one step of one group. Failures split into a
  criterion modeling gap (reason written — e.g. the first Adam step moves every coordinate by the same
  magnitude, which voids ratio-type checks) and an implementation error (fixed). `--fsdp.offload` stays the
  recipe's ([CPU AdamW](acceptance.md#checkpoint-resume-and-memory)).

Report per module class: the contract (dX choice, STE type, scale handling), oracle, dY construction, shapes
and paths covered, metrics and tolerances with reasoning; layer 2 cosine / norm ratio per class and layer
with outliers (or "none: BF16 route"); layer 3 config diff of the three arms, curves, window conclusion; the
planted-error result and snapshots; missing coverage listed separately, never "the rest is analogous". It
feeds the Gates and Scope sections of the [report template](acceptance.md#report-template).
