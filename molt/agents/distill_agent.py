# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in generation agent for on-policy distillation.

On-policy distillation needs no reward function and no task-specific agent: the dense
training signal is the per-token reverse KL to the teacher (the reference model),
supplied by the ``on_policy_distill`` advantage estimator. This env just samples one
on-policy completion per prompt and returns reward 0.0 — present only because the
rollout contract requires a scalar (``make_experience`` rejects ``None``). The estimator
ignores it.

Because it is generic (no grader, no tools), it ships in the package and is selected
automatically when ``--algo.advantage.estimator on_policy_distill`` is set without an
explicit ``--train.agent_path`` (mirrors slime shipping ``slime.rollout.on_policy_distillation``
as a built-in). So pure distillation is just::

    --algo.advantage.estimator on_policy_distill
    --ref.model_name_or_path <teacher checkpoint>   # same processor/tokenizer as the student

It runs on the ``StepEnv`` path: ``StepEnvRunner`` owns the generation loop, so the
token trace (ids + rollout logprobs) comes straight from vLLM's request output and VLM
prompts go through the exact same ``process_prompt_with_images`` tokenization the trainer
uses. No HTTP capture/stitch server, no chat re-templating drift — the single-turn,
no-tool distillation case has nothing to gain from the chat-agent machinery.

This is single-turn. For distilling a multi-turn tool-use distribution (matching how the
student is actually deployed), point ``--train.agent_path`` at the task's real agent
instead (e.g. chat_geo3k.py) — its reward is simply ignored by the estimator.
"""

from __future__ import annotations

from molt.agents.base import Env, Result, StepEnvRunner


class DistillationEnv(Env):
    async def step(self, state) -> Result:
        # One on-policy completion, no grading: the per-token reverse KL to the teacher
        # is the whole signal (on_policy_distill estimator), so this reward is a
        # placeholder the estimator ignores. terminated defaults to True → single turn.
        return Result(reward=0.0)


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(DistillationEnv)
