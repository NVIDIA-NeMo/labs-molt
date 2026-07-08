# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-turn math env for DAPO / AIME / GSM8K-style text RL.

Model produces a chain-of-thought response containing `\\boxed{...}`; we grade
it with `utils/math_grader.py` and emit reward 0 or 1. No tools, no extra
turns — for that see `geo3k.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from molt.agents import Env, Result, StepEnvRunner

_GRADER_SPEC = importlib.util.spec_from_file_location(
    "math_grader",
    Path(__file__).resolve().parent.parent / "utils" / "math_grader.py",
)
_GRADER = importlib.util.module_from_spec(_GRADER_SPEC)
_GRADER_SPEC.loader.exec_module(_GRADER)


class MathEnv(Env):
    async def step(self, state) -> Result:
        prompt, action, label = state.get("observation_text", ""), state["action_text"], state.get("label") or ""
        result = _GRADER.score_response(prompt + action, prompt, label)
        reward = torch.tensor(float(result.get("reward", 0.0)), dtype=torch.float32)
        return Result(
            reward=reward,
            info={
                "math_exact": reward,
                "missing_answer": torch.tensor(float(result.get("missing_answer", 0.0))),
            },
        )


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(MathEnv)
