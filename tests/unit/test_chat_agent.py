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
