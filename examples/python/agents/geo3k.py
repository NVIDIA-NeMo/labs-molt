# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
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
    answers = _ANSWER_RE.findall(text)
    if answers:
        answer = answers[-1].strip()
    else:
        # Balanced \boxed/\fbox extraction — handles nested braces (\boxed{\frac{1}{2}})
        # that the flat _BOXED_RE truncates to no-match (=0 reward). chat_geo3k parity.
        boxed = _GRADER._last_braced_command(text, r"\boxed") or _GRADER._last_braced_command(text, r"\fbox")
        answer = boxed.strip() if boxed else ""
    if not answer:
        return 0.0, ""
    try:
        result = _GRADER.score_response(f"\\boxed{{{answer}}}", "", label)
        return float(result.get("reward", 0.0)), answer
    except Exception as exc:
        logger.debug("grader failed: %s", exc)
        return 0.0, answer


# Qwen3.5/3.6 chat-template wrapping: tool responses live inside
# <|im_start|>user<tool_response>...</tool_response><|im_end|>. The leading
# <|im_end|> CLOSES the model's assistant turn: vLLM excludes the stop token from
# the generated ids (no include_stop_str_in_output), so the action carries no
# trailing <|im_end|>; supplying it here matches the chat-template boundary exactly
# (...</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>...) — chat_geo3k parity.
def _tool_observation(content: str) -> str:
    return (
        "<|im_end|>\n<|im_start|>user\n"
        f"<tool_response>\n{content}\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def _final_observation(status: str) -> str:
    return f"<|im_end|>\n<|im_start|>user\n{status}<|im_end|>\n"


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

        tool_call = _extract_tool_call(action)

        # Terminate once the model commits a final answer (`<answer>` / `\boxed`),
        # even if it co-emits a tool_call, or when it stops calling tools. Grading
        # the committed answer prevents post-answer verification loops that inflate
        # length/turns with no reward gain (a length-hacking failure mode).
        committed_answer = bool(_ANSWER_RE.search(action) or _BOXED_RE.search(action))
        if committed_answer or tool_call is None:
            reward, parsed = self._final_reward(label)
            status = "Correct." if reward.item() >= 1.0 else f"Done. Final answer: {parsed or 'none'}"
            return Result(
                reward=reward,
                observation=_final_observation(status),
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
        feedback = _final_observation(obs_text) if is_last_turn else _tool_observation(obs_text)
        return Result(
            reward=reward,
            observation=feedback,
            terminated=is_last_turn,
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
