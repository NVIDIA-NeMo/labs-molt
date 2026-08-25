<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AutoModel training integration

Molt follows the same eager-training boundary as OpenRLHF on DeepSpeed. The RL
trainer shows the algorithm in its natural order:

```python
model_output = actor(sequences, action_mask, ...)
loss = policy_loss(model_output.action_log_probs, old_log_probs, advantages, ...)
actor.model.backward(loss)
actor.model.step()
```

`actor.model` is an AutoModel `Engine`. It is a small `nn.Module` wrapper
around an already-distributed model, not an RL batch runner. Its public eager
API is:

- `engine(*args, **kwargs)`: ordinary model forward;
- `engine.backward(loss)`: backward for a caller-computed scalar loss;
- `engine.step()`: advance one accumulation microstep and update at the boundary;
- `engine.zero_grad()` and `engine.get_global_grad_norm()`.

There is no Datum, loss callback, output envelope, or packing protocol at this
boundary. AutoModel does not know PPO, advantages, action masks, or value loss.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| Molt dataset/replay buffer | logical samples, tokens, media, masks, old/reference log-probabilities, advantages, returns, rollout routes |
| Molt `Actor` / `Critic` | convert a logical batch to the model's padded or packed inputs, enter CP/R3 contexts, run the model, and restore dense token outputs |
| Molt trainer | PPO/GSPO/CISPO, KL, entropy and value objectives; global token normalization; metrics and checkpoint cadence |
| AutoModel model/distributed components | TP/CP/EP/FSDP model execution, vocab-parallel token log-probability and entropy primitives, router replay |
| AutoModel `Engine` | deferred FSDP synchronization, backward, distributed gradient finalization, clipping, optimizer update, gradient clearing, scheduler advancement |

This keeps the policy worker readable while keeping model-layout mechanics out
of the RL algorithm. Physical layout handling is concentrated in `BaseModel`,
which is shared by policy, reference, critic, and SFT paths.

## Policy and critic

`PolicyTrainer.training_step` is deliberately explicit: actor forward, policy
loss, optional KL and entropy terms, backward, step, then metrics. `PolicyLoss`
owns the RL formula. The trainer computes one data-parallel global action-token
denominator for the complete optimizer window, so unequal dynamic microbatches
are normalized as one update.

`CriticTrainer.training_step` has the same shape: critic forward, clipped value
loss, backward, and step. The fp32 scalar value head is a molt-owned module
installed after AutoModel wraps the backbone; it is replicated across ranks
(rank-0 broadcast at init, gradient all-reduce via `sync_replicated_grads`
before the optimizer step) and optimized together with the backbone. Critic routing replay requires the critic to use the actor
checkpoint because captured actor routes have no semantic meaning for an
unrelated MoE topology.

Collection is also direct. Policy and reference actors return action
log-probabilities, and the critic returns action values. The model wrappers
restore packing and CP layouts before returning, so workers only see dense
`[batch, sequence - 1]` tensors and never manipulate physical-token indices.

## SFT

`SFTDataset` returns ordinary dictionaries. Its collater right-pads token
tensors and retains one processor result per VLM sample. `SFTTrainer` then
runs Actor forward, computes masked next-token cross entropy from the returned
log-probabilities, and calls Engine backward/step. Evaluation is a direct
no-grad forward and a token-weighted data-parallel reduction.

Native custom-MoE auxiliary loss remains an AutoModel autograd path through
`MoEAuxLossAutoScaler`. Molt configures its coefficient but does not add the
same differentiable scalar again. The reported `sft_loss` is cross entropy;
detached auxiliary-loss observability is not wired yet.

## Packing, VLM, CP, and routing replay

Molt accepts normal dense batches at the trainer boundary. `BaseModel` owns
the model-specific physical conversion:

- native packed text uses THD metadata;
- dense Hugging Face FA2 packing uses an indexed document mask;
- packed VLM batches reuse AutoModel's VLM packing/collation primitives;
- padded and packed outputs are scattered back to the original dense token
  coordinates before `Actor.forward` or `Critic.forward` returns;
- `ContextParallelSharder` prepares CP-local model inputs and restores token
  outputs;
- rollout routes are rearranged into the same physical token order and passed
  to `RouterReplayAdapter.replay`.

CP and R3 contexts cover both model forward and backward so activation-
checkpoint recomputation sees the same layout and routes. These contexts are
implemented inside the model wrapper and entered through an `ExitStack`
supplied by the trainer; PPO/value-loss code does not inspect them.

Dynamic replay batching is unchanged. Each replay batch is forwarded directly;
the Engine accumulation size is set to the actual number of microbatches in the
current optimizer window.

## Parallelism boundary

The eager Engine intentionally rejects `AutoPipeline`. Like DeepSpeed,
pipeline parallelism has a different execution contract because only the last
stage can compute the loss. AutoModel recipes retain their separate
`AutoPipeline` schedule. Molt still fails fast for `pp_size > 1` because its
trainers do not yet build that pipeline-specific path.

Other retained boundaries are:

- multi-axis mRoPE with packed THD context parallelism is intentionally
  unsupported;
- Hugging Face fallback MoE is unsupported; MoE training requires an
  AutoModel-native implementation;
- Hugging Face indexed-mask packing requires FA2 and CP1/PP1/EP1;
- full parameter CPU offload is unsupported; ``--fsdp.offload optimizer`` runs
  the AdamW step on CPU through Molt's own ``CpuOptimizerOffloader``, which
  wraps the optimizer with the ``step()``/``zero_grad()`` surface Engine drives.

## Strategy and checkpoints

`FsdpStrategy` still owns topology construction, model loading, optimizer and
scheduler construction, collectives, checkpoint policy, and vLLM refit. It no
longer implements a second backward/optimizer loop. During `prepare`, it wraps
the already-distributed model in Engine; checkpoint and refit code unwrap
`engine.module` to reach the model state.

The source dependency must be pinned to the AutoModel commit containing this
Engine API. Molt's PyPI metadata still uses a released-version floor, so source
and Molt commits must be published together until that API is in an AutoModel
release.

## Validation

Unit coverage exercises direct Actor and Critic outputs, unequal accumulation
microbatches, THD and indexed-mask round trips, routing replay, SFT training and
evaluation, and the Engine backward/step boundary. Distributed validation must
continue to cover DP-global token normalization, TP vocab scoring, CP and R3
activation-checkpoint replay, HybridEP/custom-MoE gradients, packed VLM, and
CPU optimizer offload.
