<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AutoModel Engine integration

SFT execution is Engine-only on this branch. Molt no longer has a second SFT
implementation for global token normalization, backward, FSDP synchronization,
gradient clipping, optimizer updates, or gradient clearing.

## Execution boundary

`SFTDataset.__getitem__` emits one AutoModel `Datum` per sample. The dataloader
only groups those objects into a list, so AutoModel owns padding, packing,
position IDs, loss-side-channel collation, pinning, and CP preparation.

| Field | Layout |
| --- | --- |
| text `input_ids` | one shifted 1-D sequence per Datum |
| VLM processor outputs | one unbatched, unshifted processor mapping per Datum |
| `labels` | `PER_TOKEN` targets; unsupervised positions are `-100` |
| `weights` | `PER_TOKEN` boolean supervision mask |

Text uses AutoModel's canonical `collate_datums`, including its THD packing
mode. VLM uses AutoModel's `collate_vlm_datums`, which delegates processor
padding and shifting to `pad_collate_fn`, preserves additional processor
tensors such as Nemotron-Omni's `image_flags` and `imgs_sizes`, and uses the
canonical packed-VLM THD materializer when packing is enabled. The complete
accumulation window maps to one `forward_backward([datum0, datum1, ...])` call
and one `optim_step()`.

Engine calls AutoModel's `MaskedCrossEntropy(reduction="sum")` through a small
output-normalization callback because HF models return `.logits` while native
AutoModel models may return the logits tensor directly. Engine owns the global
weight denominator, model-parallel loss reductions, gradient synchronization,
clipping, optimizer update, and `zero_grad` lifecycle.

Molt advances its existing Transformers scheduler immediately after a
successful Engine optimizer step. It is intentionally not passed as an Engine
scheduler: AutoModel's scheduler uses `step(1)` as an increment, while a
Transformers scheduler interprets that argument as an absolute epoch.

Evaluation uses `Engine.forward`, accumulates its loss and weight sums over the
whole validation dataset, then performs one final data-parallel reduction. It
does not average batch means or communicate once per validation batch.

Molt still constructs and distributes `Actor.model` and retains its checkpoint
cadence, format, retention policy, and consumed-sample counter. Moving model
construction and checkpoint policy into another backend would also replace
Molt's generic model fallback and RL-shared strategy setup, so it is outside
this SFT execution integration.

## Current fail-fast boundary

There is no legacy SFT fallback. Padded text and VLM input, text THD packing,
VLM THD packing on supported native models, TP, CP, EP, and sequence
parallelism all stay on the same Engine path. Unsupported combinations fail
before the first training batch:

- optimizer or full CPU offload;
- PP, because Molt's shared strategy does not yet construct an `AutoPipeline`;
- THD packing on a Hugging Face fallback model; Molt's old FA2 packing adapter
  has been removed and model construction fails before loading that path;
- packed VLM CP when the active model/backend does not declare packed-CP
  support;
- multi-axis mRoPE with packed THD CP, which AutoModel currently rejects
  because aligned document padding and CP token reordering do not yet preserve
  its three position axes;
- explicit auxiliary-loss reporting.

Muon and `MOLT_DEFER_GRAD_SYNC=0` use the same Engine path: Engine steps the
already-built optimizer generically and accepts the FSDP synchronization toggle
directly.

## PPO critic

Critic optimization is also Engine-only. Molt converts each replay-buffer
microbatch into one prebatched Datum and provides only the clipped value-loss
callback. Engine returns the detached token values in their original CP/THD
coordinates for Molt's epoch-level metrics. The value projection is installed
through AutoModel's `pre_fsdp_hook`, before parameter discovery and FSDP wrap;
the old replicated-head broadcast and manual DP gradient all-reduce have been
deleted.

The hook returns AutoModel's managed-task-module declaration. AutoModel keeps the
value head replicated across TP and EP, gives it an fp32 FSDP unit over DP and CP,
excludes it from PEFT and lower-precision transforms, and includes it in training
checkpoints. Molt therefore uses the same critic path with TP, CP, EP, and sequence
parallelism; PP remains unsupported. Molt does not expose critic PEFT,
quantization, FP8, or QAT options, so those AutoModel capabilities are not Molt
feature claims. Critic CPU optimizer/full offload, Hugging Face fallback THD
packing, and RL VLM packing also fail explicitly. The existing Transformers
scheduler remains Molt-owned for the same `step()` versus `step(1)` protocol
reason as SFT.

## RL policy actor

Policy optimization is Engine-only as well. Each already-collated replay-buffer
microbatch becomes one prebatched Datum. Its shifted target tokens, action
weights, old/base/rollout log-probabilities, advantages, and optional rollout
routes are explicit loss-side channels. A complete optimizer window is one
`forward_backward` call followed by one `optim_step`; Molt's callback contains
only PPO/GSPO/CISPO, KL, and entropy numerators.

Typed per-token callback outputs let Engine restore action log-probabilities and
entropy from CP or packed THD order before Molt computes dense replay-buffer
metrics. For GSPO and sequence/geometric IS correction, Molt supplies sequence
IDs as a `PER_TOKEN` side channel and reduces detached per-sequence statistics
over the CP group. This preserves the dense per-sequence objective after THD
packing and CP sharding without gathering differentiable log-probabilities.

AutoModel's `RouterReplayAdapter` consumes rollout routes only after Engine has
applied packing and CP layout, and its context covers forward, activation-
checkpoint recomputation, and backward. Native AutoModel MoE gates keep their
`MoEAuxLossAutoScaler` path; adding the surfaced scalar aux loss in the callback
would count that gradient twice.

Padded text and VLM, native text THD packing, TP, CP, EP, sequence parallelism,
R3, entropy regularization, and PPO/GSPO/CISPO all use this path. RL VLM packing,
HF-fallback THD packing and every HF-fallback MoE model fail during model
construction. Molt retains neither the HF varlen-attention packing kwargs nor an
HF MoE training or scalar auxiliary-loss optimization branch.
The policy's Transformers scheduler remains Molt-owned and advances once after a
successful Engine optimizer update.

The legacy `Actor.forward` input-layout code remains only for collection-time
old/reference log-probability inference. It no longer owns policy backward,
gradient synchronization, clipping, or optimizer mutation.

With SFT, critic, and policy updates all on Engine, `FsdpStrategy` no longer
contains its duplicate `backward`, accumulation/sync, clipping,
`optimizer_step`, or grad-norm cache. It retains topology, optimizer/scheduler
construction, collectives, checkpointing, refit support, and optional gradient
debugging.

## Remaining blockers and retained boundaries

| Boundary | Current behavior | Missing contract |
| --- | --- | --- |
| CPU optimizer/full offload | SFT and RL training fail before the first update | Engine invokes a standard `Optimizer.step`; Molt's `CpuOptimizerOffloader.step(optimizer, params)` needs a standard optimizer wrapper or an Engine optimizer-mutation adapter |
| Transformers scheduler | Supported through one explicit Molt `step()` after `optim_step()` | Engine schedulers use incremental `step(1)`, while HF `LambdaLR` interprets the argument as absolute epoch 1 |
| Pipeline parallelism | Molt CLI fails fast at `pp_size > 1` | Engine and R3 now support per-inner-microbatch contexts, but `FsdpStrategy` still constructs an eager model rather than `AutoPipeline` |
| RL VLM packing | Actor and critic fail fast; padded VLM remains supported | AutoModel's current VLM Datum collater owns SFT `labels`/`weights`, but does not collate arbitrary PPO side channels such as old values/log-probabilities, advantages, and replay routes |
| HF fallback packing | Model construction fails fast | The old FlashAttention varlen packing implementation was deleted; packed training requires AutoModel's native THD Datum contract |
| HF fallback MoE | Model construction always fails fast, independent of auxiliary-loss settings or EP size | All MoE training requires an AutoModel-native implementation; the old scalar auxiliary-loss branch was deleted |
| Multi-axis mRoPE + packed THD CP | Intentionally unsupported and fail-fast | The agreed scope excludes this combination; AutoModel also rejects 3-D packed position IDs when CP/PP reorders or splits the token stream |

## Dependency and validation status

Source and Docker installs pin AutoModel revision `174d7c175`, which contains
the current Datum Engine, processor-ready recursive Datum pinning, padded and
packed VLM Datum collation, pipeline batch contexts, model-scoped routing
replay across local pipeline parts, and managed pre-FSDP task modules used by
the critic value head.
Molt's PyPI build still replaces source pins with `nemo-automodel>=0.5.0`; no
released version floor currently guarantees this API.

CPU tests use the real Engine and cover unequal binary masks, left/right
padding, shifted targets and position IDs, typed token-output restoration,
critic and policy backward/optimizer mutation, packed GSPO and sequence-level
IS parity, and a live two-rank gloo CP reduction of sequence statistics. A
two-GPU FSDP2 SFT parity smoke with rank-asymmetric data and two accumulated
microbatches matched the single-model reference loss, full gradients, and
updated parameters.

Critic H100 smokes with a real Qwen3 checkpoint passed Engine backward and
optimizer update for DP=2, TP=2 with sequence parallelism, CP=2, and TP=2+CP=2;
the managed value head was FSDP-wrapped, received a finite nonzero global
gradient, updated, and remained replica-consistent. EP/R3 and critic
checkpoint-resume smokes remain to be run.

The remaining external distribution blocker is that AutoModel commit
`174d7c175` is one local commit ahead of its remote integration branch, so the
source pin cannot be installed elsewhere until it is pushed. Local validation
uses that source checkout; Molt does not add a workaround for an unavailable
dependency revision.
