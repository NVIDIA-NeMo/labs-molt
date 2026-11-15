# Quick start

One command trains an agent with RL. This page shows that command, what each input is, and the
switches you are most likely to reach for next. Run everything from the repository root, inside
the container.

> Source: [`molt/cli/train_rl_ray.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/cli/train_rl_ray.py) · [`molt/cli/train_sft.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/cli/train_sft.py) · [`examples/python/agents/math.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/examples/python/agents/math.py)

## What a run needs

| Input | Flag | Notes |
|---|---|---|
| A Hugging Face checkpoint | `--actor.model_name_or_path` | The trainer and the vLLM engines load the same directory. |
| A prompt dataset | `--data.prompt_dataset` | `--data.input_key` names the prompt column, `--data.label_key` the ground truth used for scoring. |
| An agent module | `--train.agent_path` | A Python file that exports `AgentRunner`. See [Agent contract](agents.md). |
| An advantage estimator | `--algo.advantage.estimator` | See [Algorithms](algorithms.md). |

## Run RL

```bash
python3 -m molt.cli.train_rl_ray \
  --actor.model_name_or_path /path/to/automodel \
  --data.prompt_dataset /path/to/prompts.jsonl \
  --data.input_key input \
  --data.label_key answer \
  --train.agent_path examples/python/agents/math.py \
  --vllm.num_engines 2 \
  --vllm.tensor_parallel_size 2 \
  --rollout.batch_size 128 \
  --train.batch_size 128 \
  --train.micro_batch_size 1 \
  --algo.advantage.estimator reinforce \
  --fsdp.attn_implementation te \
  --ckpt.output_dir ./ckpt/rl
```

`math.py` is a single-turn grader: the model answers, the agent scores the boxed answer as 0 or 1.
The final Hugging Face export lands in `--ckpt.output_dir`. Every flag is described in the
[RL flags](cli-rl.md) reference.

To try it on one node without writing the command yourself:

```bash
MODEL_PATH=/path/to/Qwen3-4B bash examples/scripts/quick_start/rl_qwen3_4b.sh
```

## Run SFT

SFT uses the same AutoModel and FSDP2 model-loading path as RL.

```bash
torchrun --standalone --nproc_per_node=8 -m molt.cli.train_sft \
  --model.model_name_or_path /path/to/automodel \
  --data.dataset /path/to/sft.jsonl \
  --data.input_key input \
  --data.output_key output \
  --ckpt.output_dir ./ckpt/sft \
  --fsdp.attn_implementation te
```

## Common switches

| Goal | Flags | More |
|---|---|---|
| Compare several samples per prompt | `--rollout.n_samples_per_prompt 8` with `reinforce_baseline`, `rloo`, `grpo`, or `dr_grpo` | [Algorithms](algorithms.md) |
| Decouple rollout from training | `--train.async_queue_size 2` | [Asynchronous training](async-training.md) |
| Keep generating during weight sync | `--train.partial_rollout_enable` | [Asynchronous training](async-training.md) |
| Correct stale rollout log-probs | `--algo.advantage.is_correction_level geo` | [IS correction](is-correction.md) |
| Add KL regularization | `--algo.kl.init_coef 0.3 --algo.kl.use_loss --algo.kl.estimator k3` | [Algorithms](algorithms.md) |
| Score a checkpoint without training | `--eval.eval_only --eval.dataset /path/to/eval.jsonl` | [Evaluate and debug](evaluate-debug.md) |
| Scale to a larger model | `--fsdp.ep_size`, `--fsdp.cp_size`, `--fsdp.tp_size` | [Scaling](scaling.md) |

## Next

- [Agent contract](agents.md): write the reward for your own task.
- [Recipes](recipes.md): the launch scripts that ship with the repository.
