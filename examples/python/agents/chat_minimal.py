# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hello-world chat agent — single turn, boxed-answer grading.

Models the rollout as a true black box: hand `ctx.base_url` + `ctx.api_key`
+ the prompt to any stock OpenAI client (or a remote harness like
opencode / claude code) and the server captures the token trace for RL.
No `extra_body`, no `logprobs=True`, no `session_id` plumbing in agent
code. Useful as a starting template for tool-using chat agents
(OSWorld, AgentScope, etc.).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from openai import AsyncOpenAI

from molt.agents import ChatAgent, ChatAgentRunner, ChatContext, Result

_GRADER_SPEC = importlib.util.spec_from_file_location(
    "math_grader",
    Path(__file__).resolve().parent.parent / "utils" / "math_grader.py",
)
_GRADER = importlib.util.module_from_spec(_GRADER_SPEC)
_GRADER_SPEC.loader.exec_module(_GRADER)


class MathAgent(ChatAgent):
    async def run(self, ctx: ChatContext) -> Result:
        client = AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)
        resp = await client.chat.completions.create(
            model=ctx.model_name,
            messages=list(ctx.messages),  # the dataset row, ready to send
            max_tokens=ctx.sampling_params.max_tokens,
            temperature=ctx.sampling_params.temperature,
        )
        text = resp.choices[0].message.content or ""
        graded = _GRADER.score_response(text, "", ctx.label or "")
        reward = torch.tensor(float(graded.get("reward", 0.0)), dtype=torch.float32)
        return Result(reward=reward, info={"math_exact": reward})


class AgentRunner(ChatAgentRunner):
    def __init__(self):
        super().__init__(MathAgent)
