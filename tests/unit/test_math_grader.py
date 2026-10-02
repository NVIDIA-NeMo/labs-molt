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

"""Unit tests for the geo3k math-answer grader.

Covers the two-tier equivalence (string normalization + symbolic) and guards
against over-counting. Run: python3 tests/unit/test_math_grader.py
"""

import asyncio
import importlib.util
import threading
from pathlib import Path

import pytest

_MG = Path(__file__).resolve().parents[2] / "examples" / "python" / "utils" / "math_grader.py"
_spec = importlib.util.spec_from_file_location("math_grader", _MG)
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)

try:
    import sympy  # noqa: F401

    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

# Must be judged EQUAL (resolvable by string/normalization, no sympy).
MATCH = [
    ("36", r"36^\circ"),
    (r"36^{\circ}", "36"),
    ("36°", "36"),  # unicode degree
    ("12", "12 cm"),
    ("12cm", "12"),
    ("36 degrees", "36"),
    ("0.5", r"\frac{1}{2}"),
    (r"\frac{1}{2}", "0.5"),
    ("0.5", "1/2"),
    ("30", "x=30"),  # short leading assignment
    ("100", "100"),
    ("3.0", "3"),
]

# Must be judged EQUAL but only via sympy (skipped if sympy absent).
MATCH_SYMPY = [
    (r"\sqrt{12}", r"2\sqrt{3}"),
    (r"2\sqrt{3}", r"\sqrt{12}"),
    ("1/2", "0.50"),
]

# Must be judged DIFFERENT — over-count guard.
NOMATCH = [
    ("36", "37"),
    ("0.5", "0.6"),
    ("12 cm", "13 cm"),
    (r"2\sqrt{3}", r"3\sqrt{2}"),
    ("", "5"),
    ("5", ""),
    ("36", "360"),
    ("1/3", "2/6"),  # unreduced fractions must match exactly
]


@pytest.mark.parametrize("prompt", ["What is 1 * 7?", r"Check \boxed{7}."])
@pytest.mark.parametrize("response", ["", "I do not know."])
def test_score_response_does_not_extract_an_answer_from_the_prompt(prompt, response):
    result = mg.score_response(prompt + response, prompt, {"ground_truth": "7"})

    assert result["reward"] == 0.0
    assert result["prediction"] == ""
    assert result["missing_answer"] == 1.0


@pytest.mark.parametrize("prompt", ["", "What is 1 * 7?"])
@pytest.mark.parametrize(
    "response,target,reward",
    [(r"\boxed{7}", "7", 1.0), (r"\boxed{8}", "7", 0.0), (r"\boxed{\frac{1}{2}}", "0.5", 1.0), ("7", "7", 1.0)],
)
def test_score_response_grades_the_response_with_or_without_a_prompt(prompt, response, target, reward):
    result = mg.score_response(prompt + response, prompt, {"ground_truth": target})

    assert result["reward"] == reward
    assert result["missing_answer"] == 0.0


def test_math_env_grading_keeps_event_loop_responsive(monkeypatch):
    agent_path = Path(__file__).resolve().parents[2] / "examples" / "python" / "agents" / "math.py"
    spec = importlib.util.spec_from_file_location("math_agent", agent_path)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)

    started = threading.Event()
    release = threading.Event()
    timed_out = threading.Event()

    def slow_grader(*args):
        started.set()
        if not release.wait(timeout=1):
            timed_out.set()
        return {"reward": 1.0, "missing_answer": 0.0}

    monkeypatch.setattr(agent._GRADER, "score_response", slow_grader)

    async def run_step():
        task = asyncio.create_task(agent.MathEnv().step({"action_text": "answer", "label": "1"}))
        while not started.is_set():
            await asyncio.sleep(0)
        loop_remained_responsive = not timed_out.is_set()
        release.set()
        return await task, loop_remained_responsive

    result, loop_remained_responsive = asyncio.run(run_step())

    assert result.reward.item() == 1.0
    assert loop_remained_responsive


def main() -> None:
    fails = []
    for a, b in MATCH:
        if not mg.answers_match(a, b):
            fails.append(f"EXPECTED MATCH but differ: {a!r} vs {b!r}")
    for a, b in NOMATCH:
        if mg.answers_match(a, b):
            fails.append(f"EXPECTED DIFFER but match: {a!r} vs {b!r}")

    # Balanced \boxed extraction through the public scoring entry point.
    r = mg.score_response(r"\boxed{36^\circ}", "", {"ground_truth": "36"})
    if r["reward"] != 1.0:
        fails.append(f"score_response boxed-degree reward={r['reward']} (expected 1.0)")

    if HAVE_SYMPY:
        for a, b in MATCH_SYMPY:
            if not mg.answers_match(a, b):
                fails.append(f"EXPECTED MATCH (sympy) but differ: {a!r} vs {b!r}")
        rs = mg.score_response(r"\boxed{\frac{1}{2}}", "", {"ground_truth": "0.5"})
        if rs["reward"] != 1.0:
            fails.append(f"score_response nested-boxed reward={rs['reward']} (expected 1.0)")
        note = "sympy cases RAN"
    else:
        note = "sympy cases SKIPPED"

    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"ALL GRADER TESTS PASSED. {note}")


if __name__ == "__main__":
    main()
