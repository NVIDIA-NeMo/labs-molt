<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AutoModel Engine integration

SFT execution is Engine-only on this branch. Molt no longer has a second SFT
implementation for global token normalization, backward, FSDP synchronization,
gradient clipping, optimizer updates, or gradient clearing.

## Execution boundary

`SFTDataset.collate_fn` emits one prebatched AutoModel `Datum` per dataloader
batch. No trainer-side batch conversion remains.

| Field | Layout |
| --- | --- |
| `input_ids`, `attention_mask`, `position_ids` | shifted model inputs `[batch, sequence - 1]` |
| `labels` | `PER_TOKEN` next-token targets; unsupervised positions are `-100` |
| `weights` | `PER_TOKEN` boolean supervision mask |

Engine uses `collate_prebatched` and calls Molt's thin
`MaskedCrossEntropy(reduction="sum")` callback. The complete accumulation
window maps to one `forward_backward([datum0, datum1, ...])` call and one
`optim_step()`. Engine owns the global weight denominator, model-parallel loss
reductions, gradient synchronization, clipping, optimizer update, and
`zero_grad` lifecycle.

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

There is no legacy SFT fallback. The CLI rejects these configurations before
model construction:

- VLM/media preparation and visual-encoder configuration;
- packed samples;
- optimizer or full CPU offload;
- TP, CP, EP, PP, or sequence parallelism;
- explicit auxiliary-loss reporting.

Muon and `MOLT_DEFER_GRAD_SYNC=0` use the same Engine path: Engine steps the
already-built optimizer generically and accepts the FSDP synchronization toggle
directly.

The shared `FsdpStrategy` execution methods remain because policy and critic
training still call them, the critic still synchronizes a replicated value
head, and Molt still owns model construction and checkpoints. Removing those
methods as SFT cleanup would break RL rather than simplify this integration.

## Dependency and validation status

Source and Docker installs pin AutoModel revision `f864aadbe`, which contains
the current Datum Engine, `forward` evaluation, complete-window accumulation,
optimizer ownership, prepared-batch contexts, and the latest main-line
context-parallel implementation.
Molt's PyPI build still replaces source pins with `nemo-automodel>=0.5.0`; no
released version floor currently guarantees this API.

CPU tests use the real Engine and cover unequal binary masks, variable right
padding, shifted labels and position IDs, tensor and model-output logits,
pre-step gradients, parameter updates, scheduler ordering, validation
aggregation, and zero-supervision validation. A two-GPU FSDP2 parity smoke with
rank-asymmetric data and two accumulated microbatches matched the single-model
reference loss, full gradients, and updated parameters. Checkpoint-resume smoke
remains required before production adoption.
