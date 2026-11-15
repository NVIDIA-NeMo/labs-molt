# Evaluate and debug

Evaluation runs through the same vLLM engines as rollout. This page covers scoring during and
without training, and the tools for reproducing a training step.

> Source: [`molt/cli/train_rl_ray.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/cli/train_rl_ray.py) · [`molt/trainer/rl_trainer.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/trainer/rl_trainer.py)

## Evaluate during training

```bash
--eval.dataset /path/to/eval.jsonl --eval.steps 50
```

`--eval.steps` is the interval in training steps. The default `-1` disables evaluation.

Eval sampling falls back to the rollout settings. Override any of them:

| Flag | Falls back to |
|---|---|
| `--eval.temperature` | `--rollout.temperature` |
| `--eval.top_p` | `--rollout.top_p` |
| `--eval.max_new_tokens` | `--rollout.max_new_tokens` |
| `--eval.n_samples_per_prompt` | `--rollout.n_samples_per_prompt` |

`--ckpt.best_metric_key` names the eval metric that selects the best checkpoint. Left empty, the
first pass@1 metric is used. Set it to `none` to disable.

## Evaluate a checkpoint only

```bash
--eval.eval_only --eval.dataset /path/to/eval.jsonl
```

The run scores the dataset once and exits. vLLM already holds the weights, so the trainer is
never built and its GPUs go to the evaluation.

## Replay a rollout batch

To debug the training side without regenerating, dump a batch and train on it again:

```bash
--train.rollout_dump_dir /path/to/dump
```

```bash
--train.rollout_replay_dir /path/to/dump
```

The first run saves each batch as `rollout_step{N}.pt`. The second loads those files instead of
generating.

## Check weight sync

`--train.check_weight_update_equal` warns, after each broadcast, which parameters vLLM did not
refresh. Use it when rollout and trainer log-probs drift apart for no clear reason.

## Metrics to watch

| Metric | Tells you |
|---|---|
| `vllm_kl` | How far rollout and trainer log-probs are apart |
| `is_filter_ratio` | How many sequences the IS-correction gate removed |
| `rollout/truncated_rate` | How often rollouts hit a length or turn limit |

[Asynchronous training](async-training.md) lists the mismatch metrics in detail.
