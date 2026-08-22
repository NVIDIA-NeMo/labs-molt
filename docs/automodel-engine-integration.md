{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# AutoModel Engine integration

This branch exercises AutoModel's Datum Engine on the dense text SFT path. It
is deliberately a vertical slice: unsupported configurations continue through
Molt's existing trainer and print the reason at startup.

## Execution boundary

Molt passes the distributed backbone (`Actor.model`) to `Engine`, not the
outer `Actor`. The outer wrapper already owns packing and context-parallel
sharding; passing it to Engine would repeat those transformations and would
hide model post-step capabilities such as MoE gate updates.

Each existing dataloader microbatch becomes one prebatched Datum:

| Field | Layout |
| --- | --- |
| `input_ids`, `attention_mask`, `position_ids` | model input `[batch, sequence]` |
| `target_tokens` | `PER_TOKEN [batch, sequence]` |
| `weights` | `PER_TOKEN [batch, sequence]` |

`target_tokens` is `input_ids` rolled by one position. `weights` is the SFT
reply mask with the final position forced to zero, so the rolled wraparound
never contributes. The loss callback returns masked per-token negative
log-probabilities; Engine alone owns the DP denominator, backward scaling,
FSDP synchronization, clipping, optimizer step, and gradient clearing.

An SFT accumulation window maps directly to one
`forward_backward([datum0, datum1, ...])` call and one `optim_step`.

This keeps Molt's microbatch memory behavior while removing its duplicate
global-token and gradient-lifecycle math from the integrated path.
Evaluation continues through the existing Actor path in this prototype.

The CPU test uses the real Engine and checks unequal/fractional token weights,
the reported loss, gradient ownership, and the updated parameters against a
single-window reference. Distributed GPU and checkpoint-resume parity have not
yet been run for this prototype; model-parallel configurations stay on the
legacy path until they have that coverage.

## Current eligibility

The Engine path is selected for dense text, non-packed Adam SFT with no CPU
offload, no model/sequence parallelism, deferred FSDP gradient sync, and no
explicit auxiliary-loss metric. Other configurations retain the existing
trainer. This conservative gate keeps shipped VLM, packed, Muon, offload, and
model-parallel behavior unchanged until each has parity coverage.

## Gaps found during integration

- Molt uses a Transformers scheduler whose `step()` advances once. AutoModel's
  scheduler contract uses `step(1)` as an increment. Passing the Molt scheduler
  into Engine would repeatedly select epoch 1, so this slice advances it once
  immediately after a successful `optim_step`.
- Source installs are pinned to an AutoModel revision containing Datum Engine.
  Molt's PyPI build replaces git pins with `nemo-automodel>=0.5.0`, which does
  not prove that API is present. The lazy import therefore falls back to the
  legacy trainer until an AutoModel release provides an appropriate version
  floor.
- Packed text needs one collater that transforms model inputs, targets, and
  weights to the same THD layout. `collate_prebatched` cannot silently fulfill
  the existing packing knob.
- VLM integration must retain processor media routing and model-specific token
  type preparation; a text Datum is not a valid substitute.
- Policy training is not a mechanical SFT replacement. GSPO/geo objectives
  need full-sequence statistics under CP, while routing replay must cover the
  CP-prepared forward and activation-checkpoint backward.
- The critic value head is added after FSDP wrapping and requires Molt's
  explicit replicated-gradient reduction. Engine must not step it until that
  head is moved inside the distributed model boundary.
- Optimizer CPU offload uses `CpuOptimizerOffloader.step`, which Engine does
  not currently expose as an optimizer mutation hook.
