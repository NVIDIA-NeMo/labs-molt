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
import time
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[2] / "examples" / "python" / "tools" / "python_executor.py"
_spec = importlib.util.spec_from_file_location("python_executor", _TOOL_PATH)
python_executor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(python_executor)
run_python = python_executor.run_python


def test_run_python_returns_stdout():
    assert run_python("print(6 * 7)").strip() == "42"


def test_run_python_reports_error_on_nonzero_exit():
    out = run_python("1 / 0")
    assert "ZeroDivisionError" in out


def test_run_python_isolates_file_writes_to_tempdir(tmp_path, monkeypatch):
    # cwd into a clean temp dir so a regression litters here (auto-cleaned by
    # pytest), not the repo root — and so we can assert nothing leaked into cwd.
    monkeypatch.chdir(tmp_path)
    sentinel = "py_exec_sentinel.txt"
    out = run_python(f"open({sentinel!r}, 'w').write('x'); print('done')")
    assert "done" in out
    # The snippet ran in its own throwaway cwd, so nothing lands in our cwd.
    assert not (tmp_path / sentinel).exists()
    assert list(tmp_path.iterdir()) == []


def test_run_python_timeout_terminates_spawned_children(tmp_path):
    ready = tmp_path / "ready.txt"
    survivor = tmp_path / "survived.txt"
    child = f"import time; open({str(ready)!r}, 'w').close(); time.sleep(1); open({str(survivor)!r}, 'w').close()"
    code = (
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"while not os.path.exists({str(ready)!r}): time.sleep(0.01)\n"
        "time.sleep(30)\n"
    )

    out = run_python(code, timeout_seconds=0.5)
    assert ready.exists()
    time.sleep(1.1)

    assert "timed out" in out
    assert not survivor.exists()


def test_run_python_timeout_does_not_wait_for_escaped_child(tmp_path):
    ready = tmp_path / "ready.txt"
    release = tmp_path / "release.txt"
    child = (
        "import time\n"
        "from pathlib import Path\n"
        f"Path({str(ready)!r}).touch()\n"
        "deadline = time.monotonic() + 5\n"
        f"while time.monotonic() < deadline and not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
    )
    code = (
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True)\n"
        f"while not os.path.exists({str(ready)!r}): time.sleep(0.01)\n"
        "time.sleep(30)\n"
    )

    started = time.monotonic()
    try:
        out = run_python(code, timeout_seconds=0.5)
    finally:
        release.touch()
    elapsed = time.monotonic() - started

    assert ready.exists()
    assert "timed out" in out
    assert elapsed < 2
