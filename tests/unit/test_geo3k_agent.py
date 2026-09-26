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

import asyncio
import importlib.util
from pathlib import Path

_AGENT_PATH = Path(__file__).resolve().parents[2] / "examples" / "python" / "agents" / "geo3k.py"


def _load_geo3k():
    spec = importlib.util.spec_from_file_location("geo3k_agent", _AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


geo3k = _load_geo3k()


def test_step_terminates_on_answer_even_with_co_emitted_tool_call(monkeypatch):
    """A turn that commits a final answer must terminate, even if it also emits a
    tool_call — otherwise the rollout keeps tool-calling and inflates length."""
    env = geo3k.GeoEnv()
    monkeypatch.setattr(
        geo3k, "_extract_tool_call", lambda text: {"name": "python_executor", "arguments": {"code": "print(1)"}}
    )
    monkeypatch.setattr(geo3k, "_grade_answer", lambda text, label: (1.0, "5"))

    result = asyncio.run(
        env.step(
            {
                "action_text": "<answer>5</answer> let me double-check <tool_call>x</tool_call>",
                "label": {"ground_truth": "5"},
            }
        )
    )

    assert result.terminated is True
    assert result.reward.item() == 1.0
    assert env.tool_call_count == 0  # did NOT run a tool after the answer was committed
    assert result.observation.startswith("<|im_end|>\n<|im_start|>user")


def test_step_terminates_on_nested_boxed_answer_with_tool_call(monkeypatch):
    env = geo3k.GeoEnv()
    monkeypatch.setattr(
        geo3k, "_extract_tool_call", lambda text: {"name": "python_executor", "arguments": {"code": "print(1)"}}
    )

    result = asyncio.run(
        env.step(
            {
                "action_text": r"\boxed{\frac{1}{2}} <tool_call>x</tool_call>",
                "label": {"ground_truth": r"\frac{1}{2}"},
            }
        )
    )

    assert result.terminated is True
    assert result.reward.item() == 1.0
    assert env.tool_call_count == 0


def test_step_continues_on_tool_call_without_answer(monkeypatch):
    """No committed answer + a tool_call → keep going (mid-trajectory, not terminal)."""
    env = geo3k.GeoEnv()
    monkeypatch.setattr(
        geo3k, "_extract_tool_call", lambda text: {"name": "python_executor", "arguments": {"code": "print(1)"}}
    )

    result = asyncio.run(
        env.step({"action_text": "let me compute <tool_call>x</tool_call>", "label": {"ground_truth": "5"}})
    )

    assert result.terminated is False
    assert env.tool_call_count == 1
    assert result.observation.startswith("<|im_end|>\n<|im_start|>user")


def test_step_marks_last_tool_call_turn_truncated(monkeypatch):
    env = geo3k.GeoEnv()
    monkeypatch.setattr(geo3k, "_MAX_TURNS", 1)
    monkeypatch.setattr(
        geo3k, "_extract_tool_call", lambda text: {"name": "python_executor", "arguments": {"code": "print(1)"}}
    )

    result = asyncio.run(
        env.step({"action_text": "let me compute <tool_call>x</tool_call>", "label": {"ground_truth": "5"}})
    )

    assert result.terminated is False
    assert result.truncated is True
    assert env.tool_call_count == 1


def test_step_reuses_generated_turn_end_for_feedback(monkeypatch):
    env = geo3k.GeoEnv()
    monkeypatch.setattr(geo3k, "_extract_tool_call", lambda text: None)
    monkeypatch.setattr(geo3k, "_grade_answer", lambda text, label: (1.0, "5"))
    final = asyncio.run(env.step({"action_text": "<answer>5</answer><|im_end|>", "label": {"ground_truth": "5"}}))

    env = geo3k.GeoEnv()
    monkeypatch.setattr(
        geo3k, "_extract_tool_call", lambda text: {"name": "python_executor", "arguments": {"code": "print(1)"}}
    )
    tool = asyncio.run(env.step({"action_text": "<tool_call>x</tool_call><|im_end|>", "label": {"ground_truth": "5"}}))

    assert final.observation.startswith("\n<|im_start|>user")
    assert tool.observation.startswith("\n<|im_start|>user")
