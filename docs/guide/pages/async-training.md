# Asynchronous training

Agent episodes differ widely in length. In a synchronous step the trainer waits for the slowest
episode, and the rollout GPUs idle during the update. Asynchronous training lets generation and
optimization overlap. This page covers the switches and how to keep stale samples usable.

> Source: [`molt/trainer/rl_trainer.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/trainer/rl_trainer.py) · [`molt/trainer/rollout/samples_generator.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/trainer/rollout/samples_generator.py) · [`molt/trainer/workers/policy_actor.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/trainer/workers/policy_actor.py)

## Switches

| Flag | Default | Effect |
|---|---|---|
| `--train.async_queue_size` | `1` | Size of the queue between the sampler and the trainer. Raise it to `2` to decouple them. |
| `--train.partial_rollout_enable` | off | Pauses and resumes vLLM during weight sync, so generation overlaps with training. |
| `--train.force_sync_mode` | off | Frees the rollout slot only after the trainer has refit vLLM, so the next batch is generated with the weights the trainer recomputes it under. |
| `--train.force_on_policy` | off | Accumulates gradients over the whole rollout batch and takes a single optimizer step. |

The queue size bounds how much work is buffered, and so how many updates a sample can trail the
trainer by.

## Partial rollout

With `--train.partial_rollout_enable`, a request that is in progress during a weight sync keeps
its generated prefix and continues under the new weights. One trajectory can then mix policy
versions. Molt records how many leading tokens of each action were generated under the old
weights and drops them from the action mask, so they receive no gradient.

Partial rollout cannot be combined with `--train.routing_replay`, because vLLM frees the routing
on preemption.

## Stale samples

Every action token keeps the log-probability it was sampled with. That is what makes a sample
from an older policy usable: [IS correction](is-correction.md) reweights or filters tokens by the
ratio of the current to the recorded probability. That page also has the two asynchronous
recipes, FlashREINFORCE and REINFORCE Zero.

## Measure the mismatch

| Metric | Meaning |
|---|---|
| `vllm_kl` | Signed mean gap between rollout and trainer log-probs |
| `mismatch/bitwise_frac` | Fraction of response tokens whose fp32 log-prob bytes differ |
| `mismatch/abs_diff_mean` | Mean absolute log-prob difference |
| `mismatch/k3_kl` | k3 estimate of the KL |
| `is_filter_ratio` | Fraction of sequences removed by the IS-correction gate |

`bitwise_frac` is only meaningful in a strictly synchronous run with `--train.force_sync_mode`.
Under any lag the rollout came from older weights, and the fraction reads about 1.
