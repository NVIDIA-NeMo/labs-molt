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

"""Unit tests for chat-agent prompt helpers."""

import asyncio
import threading
from types import SimpleNamespace

import pytest


def _import_helpers():
    try:
        from molt.agents.chat_agent import _extract_prompt_text
    except ImportError as exc:
        pytest.skip(f"chat_agent dependencies not available: {exc}")
    return _extract_prompt_text


def test_extract_prompt_text_uses_last_string_user_turn():
    _extract_prompt_text = _import_helpers()
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "last"},
    ]
    assert _extract_prompt_text(prompt) == "last"


def test_extract_prompt_text_handles_scalar_prompt():
    _extract_prompt_text = _import_helpers()
    assert _extract_prompt_text("plain prompt") == "plain prompt"


def test_extract_prompt_text_extracts_text_from_structured_content():
    _extract_prompt_text = _import_helpers()
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    assert _extract_prompt_text(prompt) == "hello world"


def test_extract_prompt_text_returns_empty_string_when_no_user_turn():
    _extract_prompt_text = _import_helpers()
    assert _extract_prompt_text([{"role": "system", "content": "system only"}]) == ""


def test_extract_prompt_text_follows_last_user_turn_over_earlier_string():
    """The scalar view is the LAST user turn's text, even when an earlier user turn was a
    plain string and the final one carries structured (text + image) content."""
    _extract_prompt_text = _import_helpers()
    prompt = [
        {"role": "user", "content": "earlier"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "final"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
    ]
    assert _extract_prompt_text(prompt) == "final"


def test_runner_prepares_images_without_blocking_event_loop(monkeypatch):
    from molt.agents import chat_agent

    class Agent(chat_agent.ChatAgent):
        async def run(self, ctx):
            return chat_agent.Result(reward=1.0)

    started = threading.Event()
    release = threading.Event()
    timed_out = threading.Event()

    def slow_wire_messages(prompt, images):
        started.set()
        if not release.wait(timeout=1):
            timed_out.set()
        return [{"role": "user", "content": "prompt"}]

    monkeypatch.setattr(chat_agent, "_wire_messages", slow_wire_messages)
    monkeypatch.setattr(chat_agent, "stitch_session", lambda *args: [])
    runner = chat_agent.ChatAgentRunner(Agent)
    runner._state = SimpleNamespace(model_name="policy", open=lambda *args: None, discard=lambda *args: None)
    runner._server_root = "http://localhost"

    async def run_execute():
        task = asyncio.create_task(runner.execute("prompt", "label", SimpleNamespace(), 128, None, None, images=["u"]))
        while not started.is_set():
            await asyncio.sleep(0)
        loop_remained_responsive = not timed_out.is_set()
        release.set()
        return await task, loop_remained_responsive

    result, loop_remained_responsive = asyncio.run(run_execute())

    assert result == []
    assert loop_remained_responsive
