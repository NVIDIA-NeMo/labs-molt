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

`--train.check_weight_update_equal` proves the weight sync before the first update: it zeroes the
vLLM weights, syncs them from the actor, and compares the SHA-256 of every parameter on every vLLM
worker with the loaded checkpoint. The run stops if any tensor was not written or differs. Use it
when rollout and trainer log-probs drift apart for no clear reason, or for a new model. It is
skipped on a resumed run, where the actor no longer holds the checkpoint the engines loaded.

The check reports a false difference when vLLM pads MoE weights beyond the checkpoint shape (DeepEP
hidden size, MXFP4): the padding is not rewritten after the zeroing. See the limitations in vLLM's
[weight checker](https://docs.vllm.ai/en/latest/training/weight_checker/).

`--vllm.reload_mode` selects how vLLM applies a sync. The default, `layerwise`, leaves a weight the
trainer does not send (a module it never builds) at its checkpoint value. `auto` reloads directly,
which is faster and fails any sync that does not carry every weight the engine loaded
(`direct reload: the update did not load N parameters`).

## Metrics to watch

| Metric | Tells you |
|---|---|
| `vllm_kl` | How far rollout and trainer log-probs are apart |
| `is_filter_ratio` | How many sequences the IS-correction gate removed |
| `rollout/truncated_rate` | How often rollouts hit a length or turn limit |

[Asynchronous training](async-training.md) lists the mismatch metrics in detail.
