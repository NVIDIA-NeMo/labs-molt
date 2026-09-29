# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import signal
import subprocess
import sys


def run_test_script(folder, test_filename):
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Ensure functional test scripts run from the repo root.
    # Many scripts rely on relative paths like `examples/...` and also do `$(pwd)`.
    # If pytest is invoked from a different (or problematic) cwd (e.g., stale NFS/autofs),
    # inheriting that cwd can make `pwd`/getcwd() block and the test appears to "hang".
    repo_root = os.path.dirname(dir_path)
    test_file_path = os.path.join(dir_path, "functional_tests", folder, test_filename)

    # Check if -s flag was passed to pytest and propagate it to the bash script
    env = os.environ.copy()
    if "-s" in sys.argv or "--capture=no" in sys.argv:
        env["PYTEST_PROPAGATE_S"] = "1"

    p = subprocess.Popen(
        ["bash", test_file_path],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    output_lines = []
    try:
        for line in p.stdout:
            decoded = line.decode("utf-8", errors="replace")
            output_lines.append(decoded)
            sys.stdout.write(decoded)
            sys.stdout.flush()
        p.wait()
        if p.returncode != 0:
            tail = "".join(output_lines[-50:])
            raise AssertionError(
                f"Script {test_filename} failed with exit code {p.returncode}.\n--- last 50 lines of output ---\n{tail}"
            )
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
