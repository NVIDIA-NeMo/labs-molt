# Algorithms

The learning algorithm is chosen by name with `--algo.advantage.estimator`. This page lists the
estimators that ship, the optional terms around them, and how to add your own.

> Source: [`molt/trainer/algorithm/advantage.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/trainer/algorithm/advantage.py) · [`molt/models/loss.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/models/loss.py)

## Estimators

An estimator turns one clipped reward per rollout into per-token advantages.

| Name | Baseline | Notes |
|---|---|---|
| `reinforce` | None | REINFORCE++. The reward sits on the last response token, discounted returns are accumulated, and advantages are whitened across the batch. |
| `reinforce_baseline` | Group mean | Advantages are whitened across the batch. |
| `grpo` | Group mean and std | `(r - mean) / (std + eps)`. |
| `dr_grpo` | Group mean | No std and no whitening. |
| `rloo` | Leave-one-out | The mean of the other samples in the group. |
| `flash_reinforce` | Batch mean | For one rollout per prompt, where there is no group to average. No whitening. |
| `gae` | Learned value model | PPO. Adds a critic; see below. |
| `on_policy_distill` | None | The signal is the reverse KL to a frozen teacher; see below. |

A group is the set of rollouts that share a prompt. The group estimators need more than one
sample per prompt, so pair them with `--rollout.n_samples_per_prompt 8`.

## KL regularization

KL is off by default. Turn it on with a coefficient:

```bash
--algo.kl.init_coef 0.3 --algo.kl.use_loss --algo.kl.estimator k3
```

| `--algo.kl.ref` | The policy is compared to | Cost |
|---|---|---|
| `rollout`, the default | The policy that generated the samples, read from the log-probs vLLM returns | No reference workers and no extra memory. Needs `--rollout.top_p 1.0`. |
| `model` | A frozen reference model | Reference workers, placed with `--ref.num_nodes` and `--ref.num_gpus_per_node` or colocated with `--train.colocate_fsdp_models`. `--ref.model_name_or_path` defaults to the actor checkpoint. |

## Loss aggregation

`--actor.loss_agg_mode` decides what weighs the same in the policy loss.

| Value | Every ... weighs the same |
|---|---|
| `token-mean`, the default | token |
| `seq-mean-token-mean` | rollout |
| `prompt-mean-token-mean` | prompt |

## PPO critic

`--algo.advantage.estimator gae` adds a value model. It runs as its own Ray group, colocated on
the actor's GPUs by default, and it uses the same TP, EP, and CP path as the actor.

| Flag | Meaning |
|---|---|
| `--algo.advantage.lam` | GAE lambda. `1.0` is the Monte-Carlo return minus the value baseline. `length_adaptive` sets a per-sample lambda from the number of generated tokens. |
| `--algo.advantage.length_adaptive_alpha` | The alpha of `length_adaptive`. Default `0.5`. |
| `--critic.value_clip` | Clip range of the value loss. Default `0.2`. |
| `--critic.model_name_or_path` | Critic initialization. Defaults to the actor checkpoint. |
| `--critic.adam.lr` | The critic has its own optimizer and learning rate. |

## On-policy distillation

One switch turns the reference model into a teacher:

```bash
python3 -m molt.cli.train_rl_ray \
  --actor.model_name_or_path /path/to/student \
  --ref.model_name_or_path /path/to/teacher \
  --algo.advantage.estimator on_policy_distill \
  --data.prompt_dataset /path/to/prompts.jsonl \
  --data.input_key input
```

The advantage becomes `-kl_coef * (log pi_student - log pi_teacher)` on the student's own samples.
There is no scalar reward, no group baseline, and no whitening.

- The teacher must share the student's tokenizer and processor, so the per-token log-probs align.
  It loads inference-only, colocated with `--train.colocate_fsdp_models` or on its own
  `--ref.num_nodes`.
- Without `--train.agent_path`, Molt uses the built-in single-turn generator in
  `molt/agents/distill_agent.py`. To distill a multi-turn tool-use distribution, point
  `--train.agent_path` at the task's real agent. Its reward is ignored.
- Selecting the estimator sets `--algo.kl.estimator k1`, turns `--algo.kl.use_loss` off, and
  defaults `--algo.kl.init_coef` to `1.0`.
- Watch `kl` and `logprobs_diff` fall toward zero.

`examples/scripts/slurm/rl_distill_omni3_30b.sh` is a complete example.

## Reward shaping

Rewards are clipped to `--reward.clip_range`. Two length penalties can shape the reward before
the clip. Both are off by default, and the CLI rejects enabling both.

| Flag | Effect |
|---|---|
| `--reward.overlong_buffer_len B` | A DAPO-style penalty that ramps up to `--reward.overlong_penalty_factor` over the last `B` tokens of `--data.max_len`. |
| `--reward.stop_properly_penalty_coef C` | Scales a truncated rollout's reward by `C` in `[0, 1]`, or replaces it with `C` when `C` is negative. |

Both measure a rollout by its largest segment, because context compaction resets the window. The
mean correction is logged as `overlong_penalty` or `stop_properly_penalty`.

`--algo.dynamic_filtering_enable` with `--algo.dynamic_filtering_range 0.0 1.0` filters samples by
the agent's `score`.

## Add an estimator

An estimator is one registered function. This one reproduces `dr_grpo`:

```python
from molt.trainer.algorithm.advantage import broadcast_advantages, register_advantage_estimator


@register_advantage_estimator("group_mean")
def group_mean(rewards, groups, ctx):
    advantages = rewards.clone()
    for group in groups:
        advantages[group] = rewards[group] - rewards[group].mean()
    returns = broadcast_advantages(advantages, ctx)
    return [ret.clone() for ret in returns], returns
```

| Argument | Shape | Holds |
|---|---|---|
| `rewards` | `(R,)` | One clipped reward per rollout |
| `groups` | list of index lists | The rollouts that share a prompt |
| `ctx` | `AdvantageContext` | Masks, KL tensors, gamma, and flags |

It returns per-token advantages and returns, one `(B, L)` tensor per experience. Put the function
in `molt/trainer/algorithm/advantage.py` and add its name to the `choices` of
`--algo.advantage.estimator` in `molt/cli/train_rl_ray.py`.
