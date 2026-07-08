# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Molt agents — Gymnasium-aligned public surface.

User writes ONE of:
  - `class MyEnv(Env)`         with async step() / reset() returning `Result`
  - `class MyAgent(ChatAgent)`  with async run(ctx) returning `Result`

User binds it via a Runner subclass and exports as `AgentRunner`:
  - `class AgentRunner(StepEnvRunner): super().__init__(MyEnv)`
  - `class AgentRunner(ChatAgentRunner): super().__init__(MyAgent)`

For the chat-agent path, build your own client from the session URL — either
`openai.AsyncOpenAI(base_url=ctx.base_url)` or
`anthropic.AsyncAnthropic(base_url=ctx.session_url)`. We don't wrap the SDKs —
use whichever directly; the server speaks both wires against the same engine.
"""

from molt.agents.base import Env, Result, Runner, StepEnvRunner, Trajectory
from molt.agents.chat_agent import ChatAgent, ChatAgentRunner, ChatContext

__all__ = [
    "Env",
    "ChatAgent",
    "ChatAgentRunner",
    "ChatContext",
    "Result",
    "Runner",
    "StepEnvRunner",
    "Trajectory",
]
