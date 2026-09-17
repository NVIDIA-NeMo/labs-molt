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

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "python" / "utils" / "prepare_math.py"
pytest.importorskip("datasets")


def _load():
    spec = importlib.util.spec_from_file_location("prepare_math", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_format_row_appends_boxed_answer_when_solution_lacks_it():
    row = _load()._format_row({"problem": " 1+1? ", "answer": "2", "solution": "Add them."}, "src")
    assert row["prompt"] == [{"role": "user", "content": "1+1?" + _load()._INSTRUCTION}]
    assert row["reward_model"] == {"ground_truth": "2", "style": "rule"}
    assert row["response"][0]["content"] == "Add them.\n\n\\boxed{2}"


def test_format_row_keeps_solution_that_already_boxes():
    row = _load()._format_row({"problem": "p", "answer": "2", "solution": "So \\boxed{2}."}, "src")
    assert row["response"][0]["content"] == "So \\boxed{2}."
    assert _load()._format_row({"problem": "p", "answer": "2", "solution": None}, "src")["response"][0]["content"] == "\\boxed{2}"
