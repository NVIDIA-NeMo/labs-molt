# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-turn geo3k VLM env — Python tool-call helper + `<answer>` final answer.

Each turn the model can emit a `<tool_call>` invoking `python_executor(code=...)`.
The env runs the snippet in a sandboxed subprocess and feeds the captured
stdout back as a `<tool_response>` user turn so the model can continue
reasoning. The loop runs up to `MAX_AGENT_TURNS`. At any point the model
may emit `<answer>ANSWER</answer>` (or `\\boxed{ANSWER}` for legacy
training distributions); the final value found at trajectory end is graded
against the ground truth — that grade becomes the trajectory reward.

Tool-call parsing uses vLLM's qwen3 XML tool parser, imported by dotted class
path with a version fallback (Qwen3XMLToolParser on vLLM <=0.23,
Qwen3EngineToolParser on >=0.24); override via `VLLM_TOOL_PARSER_CLS`.

Environment variables:
  MAX_AGENT_TURNS         (default 5)       — caps tool_call iterations.
  VLLM_TOOL_PARSER_CLS    — full dotted class path of a vLLM ToolParser.
  PYTHON_EXECUTOR_TIMEOUT (default 10)      — wall-clock per tool call (s).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

from molt.agents import Env, Result, StepEnvRunner

logger = logging.getLogger(__name__)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PROJECT_DIR = Path(__file__).resolve().parent.parent
_GRADER = _load_module("math_grader", _PROJECT_DIR / "utils" / "math_grader.py")
_PYTHON_EXECUTOR = _load_module("python_executor", _PROJECT_DIR / "tools" / "python_executor.py").TOOL
_TOOLS = {_PYTHON_EXECUTOR.schema["function"]["name"]: _PYTHON_EXECUTOR}

_MAX_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "5"))
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
# Default candidates cover the vLLM rename: <=0.23 ships Qwen3XMLToolParser,
# >=0.24 replaces it with the Rust-backed Qwen3EngineToolParser (same interface).
_PARSER_CLS_PATHS = [
    os.environ.get("VLLM_TOOL_PARSER_CLS") or "vllm.tool_parsers.qwen3xml_tool_parser.Qwen3XMLToolParser",
    "vllm.tool_parsers.qwen3_engine_tool_parser.Qwen3EngineToolParser",
]
_PARSER = None


def _load_parser():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"], trust_remote_code=True)
    for path in _PARSER_CLS_PATHS:
        module_path, _, cls_name = path.rpartition(".")
        try:
            cls = getattr(__import__(module_path, fromlist=[cls_name]), cls_name)
        except (ImportError, AttributeError):
            continue
        return cls(tok)
    raise ImportError(f"no vLLM qwen3 tool parser found among {_PARSER_CLS_PATHS}")


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    global _PARSER
    if _PARSER is None:
        _PARSER = _load_parser()
    result = _PARSER.extract_tool_calls(text, request=None)
    if not result.tools_called or not result.tool_calls:
        return None
    tc = result.tool_calls[0]
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return {"name": tc.function.name, "arguments": args}


def _final_answer(text: str) -> str:
    """Return the model's final answer, including boxed expressions with nested braces."""
    answers = _ANSWER_RE.findall(text)
    if answers:
        return answers[-1].strip()
    boxed = _GRADER._last_braced_command(text, r"\boxed") or _GRADER._last_braced_command(text, r"\fbox")
    return boxed.strip() if boxed else ""


# `label` is the raw dataset reward_model field — typically a dict like
# {"ground_truth": "3", "style": "rule"}. math_grader.score_response unwraps
# it via _ground_truth_from_label; pass through verbatim (stringifying would
# freeze the dict's repr into the grader's target and silently zero rewards).
# Accept `<answer>` (Nemotron Omni convention) and `\\boxed{}` (Qwen / DeepSeek-Math)
# so a single grader works across either prompt distribution — the dataset prep
# script (`prepare_geo3k.py --answer-format`) decides which wrapper the model is
# asked to emit, and this grader catches whichever the model actually produces.
def _grade_answer(text: str, label) -> tuple[float, str]:
    if not label:
        return 0.0, ""
    answer = _final_answer(text)
    if not answer:
        return 0.0, ""
    try:
        result = _GRADER.score_response(f"\\boxed{{{answer}}}", "", label)
        return float(result.get("reward", 0.0)), answer
    except Exception as exc:
        logger.debug("grader failed: %s", exc)
        return 0.0, answer


# Qwen3.5/3.6 chat-template wrapping: tool responses live inside
# <|im_start|>user<tool_response>...</tool_response><|im_end|>. ``close`` ends
# the assistant turn only when its generated token ids did not already do so.
def _tool_observation(close: str, content: str) -> str:
    return (
        f"{close}\n<|im_start|>user\n"
        f"<tool_response>\n{content}\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def _final_observation(close: str, status: str) -> str:
    return f"{close}\n<|im_start|>user\n{status}<|im_end|>\n"


class GeoEnv(Env):
    def __init__(self):
        self.turn = 0
        self.assistant_history: list[str] = []
        self.tool_call_count = 0

    async def reset(self, state):
        self.turn = 0
        self.assistant_history = []
        self.tool_call_count = 0
        return {"observation": state["observation"]}

    def _final_reward(self, label) -> tuple[torch.Tensor, str]:
        joined = "\n".join(self.assistant_history)
        score, parsed = _grade_answer(joined, label)
        return torch.tensor(score, dtype=torch.float32), parsed

    async def step(self, state) -> Result:
        action = state["action_text"]
        label = state.get("label")  # may be a dict like {"ground_truth": ...}
        self.turn += 1
        self.assistant_history.append(action)
        is_last_turn = self.turn >= _MAX_TURNS
        # vLLM keeps a matched stop token in token_ids even when excluding it from
        # text. Add the marker only when the generated turn does not already carry it.
        close = "" if action.endswith("<|im_end|>") else "<|im_end|>"

        tool_call = _extract_tool_call(action)

        # Terminate once the model commits a final answer (`<answer>` / `\boxed`),
        # even if it co-emits a tool_call, or when it stops calling tools. Grading
        # the committed answer prevents post-answer verification loops that inflate
        # length/turns with no reward gain (a length-hacking failure mode).
        committed_answer = bool(_final_answer(action))
        if committed_answer or tool_call is None:
            reward, parsed = self._final_reward(label)
            status = "Correct." if reward.item() >= 1.0 else f"Done. Final answer: {parsed or 'none'}"
            return Result(
                reward=reward,
                observation=_final_observation(close, status),
                terminated=True,
                info=self._info(reward),
            )

        # Dispatch tool_call; tools handle their own argument validation.
        self.tool_call_count += 1
        name = tool_call["name"]
        tool = _TOOLS.get(name)
        obs_text = (
            tool.execute(tool_call.get("arguments") or {})
            if tool
            else (f"Tool `{name}` is not supported. Available: {list(_TOOLS)}")
        )

        reward, _ = self._final_reward(label) if is_last_turn else (torch.tensor(0.0), "")
        feedback = _final_observation(close, obs_text) if is_last_turn else _tool_observation(close, obs_text)
        return Result(
            reward=reward,
            observation=feedback,
            terminated=False,
            truncated=is_last_turn,
            info=self._info(reward),
        )

    def _info(self, reward) -> dict:
        return {
            "geo3k_tool_call_total": torch.tensor(float(self.tool_call_count), dtype=torch.float32),
            "geo3k_correct": reward,
            "turn_index": torch.tensor(float(self.turn), dtype=torch.float32),
        }


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(GeoEnv)
