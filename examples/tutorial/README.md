# CPU-First RL Learning Path: Design and Scope

[中文](./README_CN.md) | [Detailed walkthrough](./WALKTHROUGH.md)

## Summary

This directory adds a single-process, CPU-only path through Molt's core RL
logic. It is designed for contributors who want to understand or make small
changes to the reward-to-update path on a laptop before moving the same change
to a GPU server.

The tutorial reuses Molt's production dataset, agent, environment, trajectory,
experience, advantage, replay-buffer, and loss code. It replaces only the
boundaries that require distributed scheduling or GPU model execution.

This is a learning and debugging harness, not a second production training
backend.

## Motivation

Molt makes RL experiments convenient, but its production path necessarily
combines the learning algorithm with Ray, vLLM, AutoModel, FSDP, CUDA, and the
communication between them. That is the right architecture for real training,
but it makes the underlying control flow difficult to inspect without a GPU
cluster.

A CPU path gives contributors a fast way to:

- follow one example from dataset row to prompt, rollout, reward, experience,
  loss, gradient, and optimizer update;
- use ordinary breakpoints in one process;
- validate lightweight algorithm or data-flow changes locally;
- identify the exact remaining distributed and hardware-specific behavior that
  must be tested on a server.

The same explicit path also helps coding agents trace a change from its input
field or flag to the executed branch, tensor, metric, and test.

## Design principles

1. **Keep Molt's source authoritative.** Import and execute the existing
   implementation instead of copying it into tutorial code.
2. **Cut only infrastructure boundaries.** Replace Ray scheduling, vLLM serving,
   distributed model execution, and weight broadcast; keep the RL records and
   calculations real.
3. **Keep the path linear.** One script exposes the complete sequence without a
   parallel orchestration layer, so both people and tools can trace it directly.
4. **Make substitutions visible.** Local shims fail fast if the walkthrough
   accidentally enters Ray queue scheduling or vLLM sampling configuration.
5. **Leave production code unchanged.** All CPU-specific adaptation stays under
   `examples/tutorial/`.

## What is included

- `cpu_rl_walkthrough.py`: the executable CPU learning path.
- `requirements-cpu.txt`: a small, pinned environment without Ray, vLLM, or GPU
  runtimes.
- `WALKTHROUGH.md` and `WALKTHROUGH_CN.md`: setup, source correspondence,
  execution flow, deliberate small-scale settings, and suggested breakpoints.

The walkthrough uses the real Qwen3 tokenizer and chat template, but it never
downloads model weights. Its local model is a real CPU
`Embedding -> Linear` policy whose sampled tokens, behavior log probabilities,
backward pass, and optimizer update remain internally consistent.

## Reused and replaced boundaries

| Area | Behavior in this tutorial |
| --- | --- |
| Dataset and loader | Real Hugging Face `Dataset`, `PromptDataset`, and `StatefulDataLoader` |
| Agent, environment, reward | Real math agent, `StepEnvRunner`, environment, grader, and `Result` |
| RL records | Real `Trajectory` and `Experience` |
| Sampling pipeline | Real group checks, dynamic filtering, and trajectory conversion |
| Experience building | Real `RemoteExperienceMaker` logic with synchronous local result unwrapping |
| Algorithm | Real `reinforce_baseline`, replay collation, PPO policy loss, importance correction, and KL-as-loss |
| Optimization | Real PyTorch autograd, gradient clipping, AdamW, and scheduler |
| Ray | Scheduling, actors, queues, placement, and waiting are removed |
| vLLM | Replaced by a small local generation transport |
| GPU model stack | AutoModel, FSDP, CUDA/NCCL, and weight broadcast are replaced by a CPU actor shared by rollout/training plus a frozen reference copy |

## Non-goals

- Providing a supported CPU backend for production training.
- Simulating model quality, throughput, memory pressure, or distributed timing.
- Validating multi-rank, vLLM, FSDP, CUDA, or communication correctness.
- Replacing Molt's full development environment or test dependencies.

Those behaviors still belong on the normal GPU/distributed path. The tutorial
only makes the algorithmic handoff to that path smaller and easier to inspect.

## Verification

The walkthrough has been exercised end to end on macOS arm64 with Python 3.12
in a fresh environment containing no installed `molt`, Ray, or vLLM package.
The online tokenizer bootstrap, cached offline run, `pip check`, and internal
shape/data-flow assertions pass without downloading model weights.

Repository `compileall` and the CPU-safe unit tests also pass. Collecting the
complete production test suite still requires vLLM; the tutorial environment
does not replace those development dependencies. Linux dependency resolution
selects the official PyTorch CPU wheel; a Linux runtime smoke test is a useful
CI follow-up.

## Follow-up

The planned next step is a companion skill for users and coding agents:

1. make and inspect a lightweight change against the CPU path;
2. trace the affected records, tensors, branches, and metrics locally;
3. map the change to the corresponding GPU/distributed path;
4. use the server only for the remaining hardware and distributed validation.

See the [detailed walkthrough](./WALKTHROUGH.md) for installation, the exact
executed path, source mapping, and breakpoint order.
