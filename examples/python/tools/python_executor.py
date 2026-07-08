# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sandboxed Python executor used as a model-callable tool during RL rollouts.

The model emits `<tool_call>` payloads like

    <tool_call><function=python_executor>
    <parameter=code>
    import sympy as sp
    x = sp.symbols('x'); print(sp.solve(x**2 - 4, x))
    </parameter>
    </function></tool_call>

and we run the `code` string in a subprocess with a hard wall-clock timeout
plus a memory cap so a runaway tool call cannot wedge the rollout. Stdout
(and stderr on error) is truncated to a small budget and returned as the
tool's response — the model uses it to continue reasoning.

Security: we run with `-I` (isolated mode, no user site / env), drop access
to disk / network via os.chroot is *not* feasible from userland, so we rely
on the timeout + mem-cap + subprocess isolation. Each call runs in a throwaway
temp directory (its cwd) that is deleted afterwards, so a snippet that writes
files (e.g. `open('x_value.txt', 'w')`) cannot pollute the rollout actor's
working directory. Acceptable for an on-cluster rollout actor; do NOT expose
this endpoint over the internet.
"""

from __future__ import annotations

import os
import resource
import subprocess
import tempfile

# How long we let a single tool call run before SIGKILL.
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("PYTHON_EXECUTOR_TIMEOUT", "10"))
# Per-call soft cap on resident memory (bytes). The hard cap is +50%.
DEFAULT_MEM_LIMIT_BYTES = int(os.environ.get("PYTHON_EXECUTOR_MEM_BYTES", str(1024 * 1024 * 1024)))
# Truncate captured output so a runaway print doesn't blow up the next prompt.
DEFAULT_OUTPUT_CHARS = int(os.environ.get("PYTHON_EXECUTOR_OUTPUT_CHARS", "2048"))

# Preamble: only the cheap `math` stdlib (sympy/numpy cold-start is ~3s each
# and would exhaust the 5s wall-clock budget on every call). The model is told
# in the tool description to import sympy/numpy itself if it needs them.
_PREAMBLE = "import math\n"


def _set_limits():
    soft = DEFAULT_MEM_LIMIT_BYTES
    hard = int(soft * 1.5)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    except (ValueError, OSError):
        # Some environments forbid setrlimit; rely on timeout alone.
        pass


def _truncate(text: str, cap: int = DEFAULT_OUTPUT_CHARS) -> str:
    if len(text) <= cap:
        return text
    head = text[: cap - 64]
    return f"{head}\n... [{len(text) - len(head)} more chars truncated]"


def run_python(
    code: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    output_chars: int = DEFAULT_OUTPUT_CHARS,
) -> str:
    """Run `code` and return the captured stdout/stderr.

    Returns a single string. On timeout / non-zero exit, the error is included
    in the returned string so the model sees what went wrong. Never raises.
    """
    if not isinstance(code, str) or not code.strip():
        return "Error: empty code argument."
    script = f"{_PREAMBLE}\n{code}\n"
    try:
        # Run in a throwaway working directory so files the snippet writes land
        # in temp and are removed on exit, instead of polluting the actor's cwd.
        with tempfile.TemporaryDirectory(prefix="py_executor_") as workdir:
            proc = subprocess.run(
                ["python3", "-I", "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=_set_limits,
                cwd=workdir,
            )
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout_seconds:.1f}s."
    except Exception as exc:  # pragma: no cover - defensive
        return f"Error: failed to launch interpreter ({type(exc).__name__}: {exc})."
    stdout = _truncate(proc.stdout or "", output_chars)
    stderr = _truncate(proc.stderr or "", output_chars)
    if proc.returncode != 0:
        return f"Exit code {proc.returncode}.\nstdout:\n{stdout}\nstderr:\n{stderr}".strip()
    if not stdout and stderr:
        return f"(no stdout)\nstderr:\n{stderr}"
    return stdout if stdout else "(no output)"


class PythonExecutor:
    """Sandboxed Python interpreter the model can call for math/geometry math."""

    schema = {
        "type": "function",
        "function": {
            "name": "python_executor",
            "description": (
                "Run a Python snippet in a sandbox for math/geometry calculations. "
                "Returns captured stdout (capped). `math` is preloaded; import sympy/numpy "
                "yourself if needed. Use print() to read intermediate values. "
                "Call as many times as needed to verify reasoning steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source. Use print() for any value you want to read.",
                    },
                },
                "required": ["code"],
            },
        },
    }

    @classmethod
    def execute(cls, arguments):
        code = str((arguments or {}).get("code", "")).strip()
        if not code:
            return "Error: empty `code` argument."
        return run_python(code)


TOOL = PythonExecutor
