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
import ast
import importlib
import multiprocessing
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from shutil import rmtree

import psutil
import pytest
import torch

os.environ.setdefault("HF_CACHE", "/home/TestData/lite/hf_cache")
os.environ.setdefault("HF_HOME", "/home/TestData/HF_HOME")

# ---------------------------------------------------------------------------
# Shim: ``transformers.initialization`` was added in transformers >=4.48.
# Older versions keep ``no_init_weights`` in ``transformers.modeling_utils``.
# Provide a thin compatibility module so that
# ``from transformers.initialization import no_init_weights`` works regardless
# of the installed version.
# ---------------------------------------------------------------------------
if "transformers.initialization" not in sys.modules:
    try:
        importlib.import_module("transformers.initialization")
    except ModuleNotFoundError:
        from transformers.modeling_utils import no_init_weights

        _compat = types.ModuleType("transformers.initialization")
        _compat.no_init_weights = no_init_weights
        sys.modules["transformers.initialization"] = _compat

# Ensure tests import the in-repo sources (not an installed site-packages copy).
# This is important when `nemo_automodel` is also installed in the environment.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_DEFAULT_RUNTIME_BUDGET_SECONDS = 5.0
_DEFAULT_HARD_TIMEOUT_SECONDS = 60.0
_RUNTIME_BUDGET_ATTRIBUTE = "_automodel_runtime_budget_seconds"
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _run_git(repo_root: Path, *args: str) -> str | None:
    """Run a small, read-only git query, returning ``None`` outside a checkout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _runtime_budget_base(repo_root: Path) -> str | None:
    """Find the revision that represents main for this checkout.

    Pull-request CI tests a synthetic merge commit, whose first parent is the
    exact base revision. Local branches use their merge-base with
    ``origin/main`` when it is available. The environment override is useful
    for other CI entry points.
    """
    override = os.environ.get("AUTOMODEL_RUNTIME_BUDGET_BASE")
    if override:
        return override

    parents = _run_git(repo_root, "rev-list", "--parents", "-n", "1", "HEAD")
    if not parents:
        return None
    revisions = parents.split()
    if len(revisions) > 2:
        return revisions[1]

    origin_main = _run_git(repo_root, "rev-parse", "--verify", "origin/main")
    if origin_main:
        merge_base = _run_git(repo_root, "merge-base", "HEAD", origin_main)
        if merge_base:
            return merge_base

    return revisions[1] if len(revisions) == 2 else revisions[0]


def _changed_python_lines(repo_root: Path, unit_tests_root: Path) -> tuple[dict[Path, set[int]], set[Path]]:
    """Return added/modified lines and untracked Python files under unit tests."""
    try:
        relative_root = unit_tests_root.relative_to(repo_root)
    except ValueError:
        return {}, set()

    base = _runtime_budget_base(repo_root)
    if base is None:
        return {}, set()

    diff = _run_git(
        repo_root,
        "diff",
        "--unified=0",
        "--no-ext-diff",
        "--diff-filter=AM",
        base,
        "--",
        str(relative_root),
    )
    changed_lines: dict[Path, set[int]] = {}
    current_path: Path | None = None
    for line in (diff or "").splitlines():
        if line.startswith("+++ b/"):
            current_path = (repo_root / line[6:]).resolve()
            if current_path.suffix != ".py":
                current_path = None
            continue
        if current_path is None:
            continue
        match = _DIFF_HUNK_RE.match(line)
        if match is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or 1)
        # A deletion-only hunk has no added lines. Include its new location so
        # deleting work from the middle of a test still counts as a change.
        changed_lines.setdefault(current_path, set()).update(range(start, start + max(1, count)))

    untracked = _run_git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        str(relative_root),
    )
    new_files = {(repo_root / path).resolve() for path in (untracked or "").splitlines() if path.endswith(".py")}
    return changed_lines, new_files


def _test_definitions(path: Path) -> list[tuple[str, int, int]]:
    """Return qualified test names and source ranges from a Python module."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    definitions: list[tuple[str, int, int]] = []

    def visit(body: list[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, (*parents, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                decorator_lines = [decorator.lineno for decorator in node.decorator_list]
                start = min([node.lineno, *decorator_lines])
                definitions.append(("::".join((*parents, node.name)), start, node.end_lineno or node.lineno))

    visit(tree.body)
    return definitions


def _changed_test_definitions(unit_tests_root: Path) -> set[tuple[Path, str]]:
    """Map the current diff to exact pytest test functions."""
    repo_text = _run_git(unit_tests_root, "rev-parse", "--show-toplevel")
    if repo_text is None:
        return set()
    repo_root = Path(repo_text).resolve()
    changed_lines, new_files = _changed_python_lines(repo_root, unit_tests_root.resolve())

    changed_tests: set[tuple[Path, str]] = set()
    for path in changed_lines.keys() | new_files:
        lines = changed_lines.get(path, set())
        for name, start, end in _test_definitions(path):
            if path in new_files or any(start <= line <= end for line in lines):
                changed_tests.add((path, name))
    return changed_tests


def _item_definition(item: pytest.Item) -> tuple[Path, str]:
    qualified_name = "::".join(part.split("[", 1)[0] for part in item.nodeid.split("::")[1:])
    return item.path.resolve(), qualified_name


def _direct_marker(item: pytest.Item, name: str) -> pytest.Mark | None:
    """Return a marker only when it is attached to this exact test item."""
    for node, marker in item.iter_markers_with_node(name=name):
        if node is item:
            return marker
    return None


def _runtime_budget(marker: pytest.Mark, default_hard_timeout: float) -> tuple[float, float]:
    if len(marker.args) != 1 or set(marker.kwargs) - {"hard_timeout", "reason"}:
        raise pytest.UsageError(
            "runtime_budget requires one duration and accepts only hard_timeout= and reason= keyword arguments"
        )
    try:
        budget = float(marker.args[0])
        hard_timeout = float(marker.kwargs.get("hard_timeout", max(default_hard_timeout, budget * 3)))
    except (TypeError, ValueError) as error:
        raise pytest.UsageError("runtime_budget durations must be numbers") from error
    if budget <= 0 or hard_timeout <= budget:
        raise pytest.UsageError("runtime_budget requires a positive budget and a hard_timeout greater than it")
    if budget > _DEFAULT_RUNTIME_BUDGET_SECONDS and not str(marker.kwargs.get("reason", "")).strip():
        raise pytest.UsageError("runtime_budget exceptions above 5s require a non-empty reason")
    return budget, hard_timeout


def _global_timeout_was_configured(config: pytest.Config) -> bool:
    return config.getoption("timeout") is not None or "PYTEST_TIMEOUT" in os.environ or bool(config.getini("timeout"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply soft runtime budgets to changed tests and hard watchdogs to hangs.

    A runtime regression should fail only after the test has completed, without
    interrupting compiler or multiprocessing cleanup. The pytest-timeout marker
    is reserved for a much larger hard watchdog. Slow-test exceptions use the
    exact-test ``runtime_budget`` marker, so a module-wide timeout cannot silently
    exempt future tests added to that module.
    """
    if _global_timeout_was_configured(config):
        return

    runtime_budget = float(config.getoption("unit_test_runtime_budget"))
    hard_timeout = float(config.getoption("unit_test_hard_timeout"))
    if runtime_budget <= 0 or hard_timeout <= runtime_budget:
        raise pytest.UsageError(
            "--unit-test-runtime-budget must be positive and --unit-test-hard-timeout must be greater than it"
        )

    unit_tests_root = Path(__file__).parent
    changed_tests = _changed_test_definitions(unit_tests_root)
    for item in items:
        if unit_tests_root not in item.path.parents:
            continue

        runtime_markers = list(item.iter_markers_with_node(name="runtime_budget"))
        inherited_runtime_markers = [(node, marker) for node, marker in runtime_markers if node is not item]
        if inherited_runtime_markers:
            raise pytest.UsageError(
                f"{item.nodeid}: runtime_budget must be applied directly to a test, not to a module or class"
            )
        direct_runtime_marker = _direct_marker(item, "runtime_budget")
        direct_timeout_marker = _direct_marker(item, "timeout")

        if direct_runtime_marker is not None:
            if direct_timeout_marker is not None:
                raise pytest.UsageError(
                    f"{item.nodeid}: runtime_budget owns the hard watchdog; remove the direct timeout marker"
                )
            budget, item_hard_timeout = _runtime_budget(direct_runtime_marker, hard_timeout)
            setattr(item, _RUNTIME_BUDGET_ATTRIBUTE, budget)
            item.add_marker(pytest.mark.timeout(item_hard_timeout), append=False)
            continue

        is_changed_test = _item_definition(item) in changed_tests
        if is_changed_test:
            setattr(item, _RUNTIME_BUDGET_ATTRIBUTE, runtime_budget)

        # Exact timeout markers are intentional hard-watchdog tests. Inherited
        # markers are legacy slow-module allowances and must not exempt a newly
        # changed test from the default hard watchdog.
        if direct_timeout_marker is not None:
            continue
        if is_changed_test or item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(hard_timeout), append=False)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Turn a completed test's runtime-budget overage into a normal failure."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "setup":
        setattr(item, "_automodel_setup_duration", report.duration)
    if report.when == "call":
        budget = getattr(item, _RUNTIME_BUDGET_ATTRIBUTE, None)
        elapsed = getattr(item, "_automodel_setup_duration", 0.0) + report.duration
        if report.passed and budget is not None and elapsed > budget:
            report.outcome = "failed"
            report.longrepr = (
                f"{item.nodeid} took {elapsed:.2f}s in setup+call, exceeding its {budget:g}s runtime budget. "
                "Optimize the test or add an exact @pytest.mark.runtime_budget(..., reason=...) exception."
            )
        setattr(item, "_automodel_call_report", report)
    if report.failed and "Timeout (" in report.longreprtext:
        setattr(item, "_automodel_timed_out", True)


def _descendant_processes() -> list[psutil.Process]:
    try:
        return psutil.Process().children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _is_inductor_compile_worker(process: psutil.Process) -> bool:
    try:
        command = " ".join(process.cmdline()).replace("\\", "/")
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return "torch/_inductor/compile_worker" in command or "torch._inductor.compile_worker" in command


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Remember long-lived children so a timeout does not kill another test's pool."""
    setattr(item, "_automodel_child_pids_before", {process.pid for process in _descendant_processes()})


def pytest_addoption(parser):
    """Additional command-line arguments passed to pytest.
    For now:
        --cpu: use CPU during testing (DEFAULT: GPU)
        --use_local_test_data: use local test data/skip downloading from URL/GitHub (DEFAULT: False)
    """
    parser.addoption(
        "--cpu", action="store_true", help="pass that argument to use CPU during testing (DEFAULT: False = GPU)"
    )
    parser.addoption(
        "--with_downloads",
        action="store_true",
        help="pass this argument to active tests which download models from the cloud.",
    )
    parser.addoption(
        "--unit-test-runtime-budget",
        dest="unit_test_runtime_budget",
        type=float,
        default=_DEFAULT_RUNTIME_BUDGET_SECONDS,
        help="Soft call-time budget in seconds for newly added or modified unit tests.",
    )
    parser.addoption(
        "--unit-test-hard-timeout",
        dest="unit_test_hard_timeout",
        type=float,
        default=_DEFAULT_HARD_TIMEOUT_SECONDS,
        help="Hard pytest-timeout watchdog in seconds for unmarked unit tests.",
    )


@pytest.fixture
def device(request):
    """Simple fixture returning string denoting the device [CPU | GPU]"""
    if request.config.getoption("--cpu"):
        return "CPU"
    else:
        return "GPU"


@pytest.fixture(autouse=True)
def run_only_on_device_fixture(request, device):
    """Fixture to skip tests based on the device"""
    if request.node.get_closest_marker("run_only_on"):
        if request.node.get_closest_marker("run_only_on").args[0] != device:
            pytest.skip("skipped on this device: {}".format(device))


@pytest.fixture(autouse=True)
def downloads_weights(request, device):
    """Fixture to validate if the with_downloads flag is passed if necessary"""
    if request.node.get_closest_marker("with_downloads"):
        if not request.config.getoption("--with_downloads"):
            pytest.skip(
                "To run this test, pass --with_downloads option. It will download (and cache) models from cloud."
            )


@pytest.fixture(autouse=True)
def cleanup_local_folder():
    """Cleanup local experiments folder"""
    # Asserts in fixture are not recommended, but I'd rather stop users from deleting expensive training runs
    assert not Path("./NeMo_experiments").exists()
    assert not Path("./nemo_experiments").exists()

    yield

    if Path("./NeMo_experiments").exists():
        rmtree("./NeMo_experiments", ignore_errors=True)
    if Path("./nemo_experiments").exists():
        rmtree("./nemo_experiments", ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables"""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def enforce_torch_memory_limit(request):
    """Enforce opt-in per-test PyTorch allocator budgets."""
    marker = request.node.get_closest_marker("torch_memory_limit")
    if marker is None:
        yield
        return

    cpu_limit_mb = marker.kwargs.get("cpu_mb")
    cuda_limit_mb = marker.kwargs.get("cuda_mb")
    if cpu_limit_mb is None and cuda_limit_mb is None:
        pytest.fail("torch_memory_limit requires cpu_mb and/or cuda_mb")

    cuda_device = None
    cuda_allocated_before = 0
    if cuda_limit_mb is not None and torch.cuda.is_available():
        cuda_device = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_device)
        cuda_allocated_before = torch.cuda.memory_allocated(cuda_device)

    if cpu_limit_mb is None:
        yield
        profiler = None
    else:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            acc_events=True,
        ) as profiler:
            yield

    bytes_per_mb = 1024**2
    if profiler is not None:
        cpu_allocated_bytes = sum(max(0, event.self_cpu_memory_usage) for event in profiler.key_averages())
        assert cpu_allocated_bytes <= cpu_limit_mb * bytes_per_mb, (
            f"test allocated {cpu_allocated_bytes / bytes_per_mb:.1f} MiB through the PyTorch CPU allocator; "
            f"limit is {cpu_limit_mb} MiB"
        )

    if cuda_device is not None:
        cuda_peak_delta_bytes = torch.cuda.max_memory_allocated(cuda_device) - cuda_allocated_before
        assert cuda_peak_delta_bytes <= cuda_limit_mb * bytes_per_mb, (
            f"test allocated {cuda_peak_delta_bytes / bytes_per_mb:.1f} MiB through the PyTorch CUDA allocator; "
            f"limit is {cuda_limit_mb} MiB"
        )


@pytest.fixture(scope="session", autouse=True)
def _fail_on_leaked_process_group():
    """Fail the session when a test leaves a torch.distributed process group running.

    A leaked group is invisible in the test that created it and silently corrupts
    later ones: helpers such as ``get_world_size_safe`` stop reading ``WORLD_SIZE``
    from the environment and report the live group's size instead, and tests that
    skip when a group already exists stop running at all. Whether it bites depends
    on collection order, so it can pass in CI and fail locally.

    Tests that need a real group must tear it down, for example by initializing it
    inside a fixture that destroys it in a ``finally`` block.
    """
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        pytest.fail(
            "a test left a torch.distributed process group initialized; "
            "re-run with -p no:randomly and bisect by file to find the fixture that "
            "calls init_process_group without a matching destroy_process_group",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _kill_leaked_child_processes(request: pytest.FixtureRequest):
    """Kill worker processes a test left behind, so a hung worker cannot stall the session.

    pytest-timeout's signal method only interrupts the pytest process. ``mp.spawn(join=True)``
    and TorchInductor's compile pool leave child processes running when the timeout interrupts
    their wait. Python then waits for those workers at interpreter exit, so a single hung worker
    can hold the CI job until its workflow timeout.
    """
    yield

    multiprocessing_children = multiprocessing.active_children()
    leaked_pids = {process.pid for process in multiprocessing_children if process.pid is not None}
    if getattr(request.node, "_automodel_timed_out", False):
        child_pids_before = getattr(request.node, "_automodel_child_pids_before", set())
        leaked_pids.update(
            process.pid
            for process in _descendant_processes()
            if process.pid not in child_pids_before or _is_inductor_compile_worker(process)
        )
    if not leaked_pids:
        return

    leaked_processes: list[psutil.Process] = []
    for pid in sorted(leaked_pids):
        try:
            process = psutil.Process(pid)
            process.kill()
            leaked_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Let multiprocessing reap its own children first. If psutil consumes their
    # waitpid status, Process.join() cannot record an exit code and active_children()
    # keeps reporting the dead workers in every subsequent test.
    for process in multiprocessing_children:
        process.join(timeout=5)
    if leaked_processes:
        psutil.wait_procs(leaked_processes, timeout=5)
    pytest.fail(
        f"test left {len(leaked_pids)} child process(es) running (pids {sorted(leaked_pids)}); "
        "they were killed. Join or terminate spawned workers before the test returns",
        pytrace=False,
    )


def pytest_configure(config):
    """Initial configuration of conftest.
    The function checks if test_data.tar.gz is present in tests/.data.
    If so, compares its size with github's test_data.tar.gz.
    If file absent or sizes not equal, function downloads the archive from github and unpacks it.
    """
    config.addinivalue_line(
        "markers",
        "run_only_on(device): runs the test only on a given device [CPU | GPU]",
    )
    config.addinivalue_line(
        "markers",
        "with_downloads: runs the test using data present in tests/.data",
    )
    config.addinivalue_line(
        "markers",
        "torch_memory_limit(cpu_mb=None, cuda_mb=None): limits per-test PyTorch allocator usage",
    )
    config.addinivalue_line(
        "markers",
        "runtime_budget(seconds, hard_timeout=None, reason=None): exact-test soft runtime budget and hard watchdog",
    )
