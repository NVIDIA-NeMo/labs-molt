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
- THD packing on a Hugging Face fallback model, whose packing adapter is not
  the AutoModel Engine THD contract;
- packed VLM CP when the active model/backend does not declare packed-CP
  support;
- multi-axis mRoPE with packed THD CP, which AutoModel currently rejects
  because aligned document padding and CP token reordering do not yet preserve
  its three position axes;
- explicit auxiliary-loss reporting.

Muon and `MOLT_DEFER_GRAD_SYNC=0` use the same Engine path: Engine steps the
already-built optimizer generically and accepts the FSDP synchronization toggle
directly.

The shared `FsdpStrategy` execution methods remain because policy and critic
training still call them, the critic still synchronizes a replicated value
head, and Molt still owns model construction and checkpoints. Removing those
methods as SFT cleanup would break RL rather than simplify this integration.

## Dependency and validation status

Source and Docker installs pin AutoModel revision `5420b30fd`, which contains
the current Datum Engine, processor-ready recursive Datum pinning, padded and
packed VLM Datum collation, pipeline batch contexts, model-scoped routing
replay across local pipeline parts, the pre-FSDP structure hook used by the
critic value head, and the latest main-line context-parallel implementation.
Molt's PyPI build still replaces source pins with `nemo-automodel>=0.5.0`; no
released version floor currently guarantees this API.

CPU tests use the real Engine and cover unequal binary masks, variable right
padding, shifted labels and position IDs, tensor and model-output logits,
pre-step gradients, parameter updates, scheduler ordering, validation
aggregation, and zero-supervision validation. A two-GPU FSDP2 parity smoke with
rank-asymmetric data and two accumulated microbatches matched the single-model
reference loss, full gradients, and updated parameters. Checkpoint-resume smoke
remains required before production adoption.
