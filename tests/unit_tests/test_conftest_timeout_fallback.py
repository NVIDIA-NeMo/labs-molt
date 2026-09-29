# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Behavioral checks for the unit-test runtime policy in ``conftest.py``.

Each test copies the real conftest into an isolated pytester project. Git commits
model the base and pull-request revisions so the checks exercise the same
changed-test discovery used in CI.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.timeout(60)

_CONFTEST_SOURCE = Path(__file__).with_name("conftest.py").read_text()
_POLICY_ARGS = ("--unit-test-runtime-budget=0.05", "--unit-test-hard-timeout=0.5")
_TEST_MODULE = """
import time
from pathlib import Path

import pytest

{module_marker}
{function_marker}def test_sleep():
{body}
"""


def _git(pytester: pytest.Pytester, *args: str) -> None:
    subprocess.run(["git", *args], cwd=pytester.path, check=True, capture_output=True, text=True)


def _initialize_repository(pytester: pytest.Pytester) -> None:
    _git(pytester, "init", "-q")
    _git(pytester, "config", "user.email", "runtime-policy@example.com")
    _git(pytester, "config", "user.name", "Runtime Policy Test")
    _git(pytester, "add", ".")
    _git(pytester, "commit", "-q", "-m", "baseline")


def _module_source(*, body: str, module_marker: str = "", function_marker: str = "") -> str:
    indented_body = "\n".join(f"    {line}" for line in body.splitlines())
    return _TEST_MODULE.format(
        module_marker=module_marker,
        function_marker=function_marker,
        body=indented_body,
    )


def _run_sleeper(
    pytester: pytest.Pytester,
    seconds: float,
    *args: str,
    module_marker: str = "",
    function_marker: str = "",
    changed: bool = True,
    run_in_subprocess: bool = False,
) -> tuple[pytest.RunResult, Path]:
    """Run one sleeping test under a copy of the unit-test conftest."""
    pytester.makeconftest(_CONFTEST_SOURCE)
    baseline_body = (
        'Path("completed").write_text("yes")'
        if changed
        else f'time.sleep({seconds})\nPath("completed").write_text("yes")'
    )
    test_path = pytester.makepyfile(
        test_sleep=_module_source(
            body=baseline_body,
            module_marker=module_marker,
            function_marker=function_marker,
        )
    )
    _initialize_repository(pytester)

    if changed:
        test_path.write_text(
            _module_source(
                body=f'time.sleep({seconds})\nPath("completed").write_text("yes")',
                module_marker=module_marker,
                function_marker=function_marker,
            )
        )
        _git(pytester, "add", str(test_path.name))
        _git(pytester, "commit", "-q", "-m", "change test")

    if run_in_subprocess:
        # pytester relocates HOME, which can hide user-site packages such as torch
        # from the fresh interpreter used by local development environments.
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in sys.path if path))
            result = pytester.runpytest_subprocess("-p", "no:cacheprovider", *_POLICY_ARGS, *args)
    else:
        result = pytester.runpytest("-p", "no:cacheprovider", *_POLICY_ARGS, *args)
    return result, pytester.path / "completed"


def test_changed_test_fails_soft_budget_after_completing(pytester: pytest.Pytester):
    result, completed = _run_sleeper(pytester, 0.12)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*took *s in setup+call, exceeding its 0.05s runtime budget*"])
    assert completed.read_text() == "yes"
    assert "Timeout (" not in result.stdout.str()


def test_inherited_timeout_does_not_exempt_changed_test(pytester: pytest.Pytester):
    result, completed = _run_sleeper(
        pytester,
        0.12,
        module_marker="pytestmark = pytest.mark.timeout(0.4)",
    )

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*exceeding its 0.05s runtime budget*"])
    assert completed.exists()


def test_unchanged_test_preserves_existing_timeout_marker(pytester: pytest.Pytester):
    result, _ = _run_sleeper(
        pytester,
        0.12,
        module_marker="pytestmark = pytest.mark.timeout(0.4)",
        changed=False,
    )

    result.assert_outcomes(passed=1)


def test_exact_runtime_budget_allows_documented_slow_test(pytester: pytest.Pytester):
    result, completed = _run_sleeper(
        pytester,
        0.12,
        function_marker="@pytest.mark.runtime_budget(0.2, hard_timeout=0.5)\n",
    )

    result.assert_outcomes(passed=1)
    assert completed.exists()


def test_module_runtime_budget_is_rejected(pytester: pytest.Pytester):
    result, _ = _run_sleeper(
        pytester,
        0,
        module_marker="pytestmark = pytest.mark.runtime_budget(0.2, hard_timeout=0.5)",
        changed=False,
    )

    assert result.ret != 0
    result.stderr.fnmatch_lines(["*runtime_budget must be applied directly to a test*"])


def test_cli_timeout_zero_disables_policy(pytester: pytest.Pytester):
    result, completed = _run_sleeper(pytester, 0.12, "--timeout=0")

    result.assert_outcomes(passed=1)
    assert completed.exists()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="starts a fresh pytest subprocess to isolate the intentional timeout",
)
def test_cli_timeout_overrides_policy(pytester: pytest.Pytester):
    result, completed = _run_sleeper(pytester, 0.12, "--timeout=0.05", run_in_subprocess=True)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.05s) from pytest-timeout*"])
    assert not completed.exists()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="starts a fresh pytest subprocess to isolate the intentional timeout",
)
def test_env_timeout_overrides_policy(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PYTEST_TIMEOUT", "0.05")
    result, completed = _run_sleeper(pytester, 0.12, run_in_subprocess=True)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.05s) from pytest-timeout*"])
    assert not completed.exists()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="starts a fresh pytest subprocess to isolate the intentional timeout",
)
def test_ini_timeout_overrides_policy(pytester: pytest.Pytester):
    pytester.makeini("[pytest]\ntimeout = 0.05\n")
    result, completed = _run_sleeper(pytester, 0.12, run_in_subprocess=True)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Timeout (>0.05s) from pytest-timeout*"])
    assert not completed.exists()


def test_hard_watchdog_is_scoped_to_the_conftest_tree(pytester: pytest.Pytester):
    pytester.makepyfile(
        **{
            "unit_tests/conftest.py": _CONFTEST_SOURCE,
            "unit_tests/test_inside.py": "def test_inside():\n    pass\n",
            "other/test_outside.py": "def test_outside():\n    pass\n",
        }
    )
    items, _ = pytester.inline_genitems("unit_tests", "other")
    markers = {item.name: item.get_closest_marker("timeout") for item in items}

    assert markers["test_inside"].args == (60.0,)
    assert markers["test_outside"] is None
