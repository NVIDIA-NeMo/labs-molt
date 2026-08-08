# Molt CPU RL Walkthrough

[Overview](./README.md) | [中文教程](./WALKTHROUGH_CN.md)

## Motivation

Molt is designed to make RL experimentation easy, but the production quick starts necessarily bring up Ray, vLLM, AutoModel, FSDP, and a GPU cluster before a new contributor can step through the reward-to-update logic.

This tutorial provides a small CPU learning harness for that logic. It reuses Molt's original dataset, agent, environment, trajectory, experience, advantage, replay-buffer, and loss implementations, while replacing only the distributed scheduler and GPU model-execution boundaries. The result is intentionally not a second training backend: it is a readable, single-process path for learning, debugging, and validating lightweight changes before moving them to a production recipe.

The same structure also helps coding agents trace a change from input row to prompt, rollout, reward, experience, loss, gradient, and updated policy without having to emulate a cluster.

## What stays real

- The Qwen3 tokenizer and chat template loaded through Molt's `get_tokenizer()`.
- Hugging Face `Dataset`, `PromptDataset`, and `StatefulDataLoader`.
- `StepEnvRunner`, the math environment and grader, `Result`, and `Trajectory`.
- Group completeness checks, dynamic filtering, and `Trajectory -> Experience` conversion.
- Old-policy and frozen-reference forwards through `RemoteExperienceMaker`.
- `reinforce_baseline`, replay-buffer collation, `PolicyLoss`, KL-as-loss, autograd, gradient clipping, AdamW, and the learning-rate scheduler.

The local policy is a real CPU `Embedding -> Linear` model rather than a random shape stub. The fake generation transport samples tokens from its softmax, records the corresponding behavior-policy log probabilities, and shares the same model object with training, so an optimizer update is immediately visible to the next forward.

## What is replaced

- Ray actors, queues, placement, and `remote/wait` scheduling.
- vLLM serving and transport.
- AutoModel/FSDP/CUDA/NCCL model execution.
- The `PolicyTrainer`/`FsdpStrategy` shell that is coupled to those distributed runtimes. The walkthrough spells out its corresponding CPU forward, loss, backward, clipping, optimizer, and scheduler operations.
- FSDP-to-vLLM weight broadcast. In one process, rollout and training share one model object.

Every replacement is visible near the top of the single walkthrough file. Ray queue scheduling and vLLM `SamplingParams` fail immediately if the tutorial accidentally enters those production paths.

## Files and source correspondence

- `cpu_rl_walkthrough.py` is the only executable entry point. It is deliberately linear so it can be debugged from the first line.
- `requirements-cpu.txt` contains the pinned CPU dependencies and omits Ray and GPU runtimes.

No Molt implementation is copied. The import boundary follows the eager imports in `molt/models/__init__.py`, `samples_generator.py`, and `experience_maker.py`. `TinyPolicy` and the fake engine implement the interfaces consumed by `Actor.forward` and `StepEnvRunner`. The explicit update block corresponds to `PolicyTrainer.training_step` and the CPU-meaningful parts of `FsdpStrategy.backward/optimizer_step`.

The existing `molt/`, `tests/`, and production examples are unchanged; all adaptation code lives in this directory.

## Run

The tutorial requirements are intentionally separate from the full repository requirements:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r examples/tutorial/requirements-cpu.txt
.venv/bin/python examples/tutorial/cpu_rl_walkthrough.py
```

The script adds the repository root to `sys.path`, so it does not need `pip install -e . --no-deps` and does not create incomplete `molt` package metadata. Direct dependencies are pinned; Linux also queries the official PyTorch CPU wheel index.

The tokenizer is pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca` of `Qwen/Qwen3-0.6B`. `snapshot_download()` allows only config, tokenizer, vocabulary, and merge files. The first run downloads about 15 MB and no model weights.

After the cache is warm, the tutorial can run explicitly offline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python examples/tutorial/cpu_rl_walkthrough.py
```

## Executed path

```text
Hugging Face Dataset.from_list (one local row)
  -> PromptDataset.__getitem__ / collate_fn
  -> StatefulDataLoader(batch_size=1)
  -> _collect_prompt_batch
  -X Ray dispatch / vLLM transport (direct runner + TinyVllmEngine call)
  -> examples/python/agents/math.py
  -> StepEnvRunner.execute -> MathEnv.step -> Result -> Trajectory
  -> SamplesGenerator._filter_group (complete-group check + dynamic filtering)
  -> SamplesGenerator._process_response_into_experience
  -> RemoteExperienceMaker.build_experiences
  -> TinyPolicy old-policy forward + frozen-reference forward
  -> reinforce_baseline advantage
  -> balance_experiences
  -> NaiveReplayBuffer -> torch DataLoader
  -X PolicyTrainer / AutoModel / FSDP Actor (explicit CPU update + TinyPolicy)
  -> log_probs_from_logits / PolicyLoss / compute_approx_kl / agg_loss
  -> loss.backward -> clip_grad_norm_ -> AdamW -> scheduler
  -X FSDP-to-vLLM broadcast (one shared model object)
```

`-X` marks every cut boundary. Local `ray.get` only unwraps synchronous local results; real `wait`/queue scheduling and vLLM `SamplingParams` remain disabled. `Result`, `Trajectory`, and `Experience` are the original Molt records, not tutorial dictionaries.

## Deliberate small-scale changes

- The dataset is one in-memory Hugging Face row, with the same `prompt` and `reward_model` fields as the Qwen3 math quick start.
- The production recipe's eight samples per prompt are reduced to two. Per-token inverse-CDF draws deterministically produce one `\boxed{4}` and one `\boxed{5}` while preserving the autoregressive policy/log-probability contract.
- The two rollouts share a group ID, so the real `reinforce_baseline` estimator produces positive and negative advantages.
- The quick-start settings for dynamic filtering, old-policy/reference forwards, KL-as-loss (`k2`, coefficient `0.001`), PPO clipping, and importance-sampling correction remain active. Actor and reference are identical on the first step, so the KL value is zero even though the complete branch executes.
- DP, TP, and CP are one; dynamic batching is disabled. These settings remove throughput and memory scheduling without changing the single-rank tensor semantics being studied.
- Pinned memory is disabled on CPU. The prompt loader is still `StatefulDataLoader`, and training still uses `NaiveReplayBuffer + DataLoader`.
- AdamW uses the quick-start betas, epsilon, zero weight decay, constant scheduler, and `max_norm=1.0`. Only the learning rate is increased from `1e-6` to `0.01` so one tutorial step produces a visible update.

## Suggested breakpoint order

1. `PromptDataset.__getitem__`: prompt rendering before tokenization.
2. `_collect_prompt_batch`: the five DataLoader columns.
3. `StepEnvRunner.execute`: tokenizer, fake generation transport, and real environment step.
4. `SamplesGenerator._filter_group`: rollout completeness and the `0.5` group score passing dynamic filtering.
5. `SamplesGenerator._process_response_into_experience`: token axis `T` becoming next-token axis `T-1`.
6. `RemoteExperienceMaker.make_experience`: local old-policy/reference forwards and log-probability fields.
7. `RemoteExperienceMaker.compute_advantages_and_returns`: rewards becoming positive and negative advantages.
8. `NaiveReplayBuffer.append/collate_fn`: rollout batches becoming training microbatches.
9. `PolicyLoss.forward`, then `compute_approx_kl/agg_loss`: PPO, importance correction, and KL-as-loss.
10. `total_loss.backward()`: gradients followed by clipping, AdamW, scheduler, and zero-grad.

## Validation

The walkthrough has been verified on macOS arm64 with Python 3.12 in a fresh environment containing no installed `molt`, Ray, or vLLM package. `pip check`, online tokenizer bootstrap, cached offline execution, and the end-to-end assertions all pass. Linux resolution selects the PyTorch `2.11.0+cpu` wheel; a Linux runtime smoke test remains desirable in CI.

The full Molt test environment still requires the production vLLM dependency. This tutorial requirements file is only for the CPU walkthrough, not a replacement development environment for the entire repository.

## Roadmap

A planned follow-up is a coding-agent skill built around this CPU path. The intended workflow is:

1. make and inspect a lightweight change against the CPU walkthrough;
2. trace the affected Molt records, tensors, branches, and metrics locally;
3. transfer the change to the corresponding GPU/distributed path;
4. validate only the remaining distributed and hardware-specific behavior on the server.

Keeping the CPU path structurally close to the production code should make that transfer easier for both users and coding agents.
