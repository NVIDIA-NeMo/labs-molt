# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from nemo_automodel.components.checkpoint import _torch_backports


def test_async_checkpoint_patch_initializes_daemon_before_save(monkeypatch: pytest.MonkeyPatch) -> None:
    call_order: list[str] = []
    expected_process_group = object()
    save_future: Future[None] = Future()

    class FakeProcessGroupInitInfo:
        def __init__(self, received_process_group: object) -> None:
            assert received_process_group is expected_process_group
            call_order.append("process_group_info")

    class FakeAsyncCheckpointProcess:
        def __init__(self, *, pg_init_info: FakeProcessGroupInitInfo) -> None:
            assert isinstance(pg_init_info, FakeProcessGroupInitInfo)
            call_order.append("checkpoint_process")

    class FakeExecutor:
        @staticmethod
        def _execute_save_impl() -> None:
            return None

        def execute_save(self, *, process_group: object) -> Future[None]:
            assert process_group is expected_process_group
            assert async_process_executor._CHECKPOINT_PROCESS is not None
            call_order.append("execute_save")
            return save_future

    async_process_executor = SimpleNamespace(
        _CHECKPOINT_PROCESS=None,
        _ProcessGroupInitInfo=FakeProcessGroupInitInfo,
        _AsyncCheckpointProcess=FakeAsyncCheckpointProcess,
        _ProcessBasedAsyncCheckpointExecutor=FakeExecutor,
    )
    monkeypatch.setattr(
        _torch_backports.importlib,
        "import_module",
        lambda _: async_process_executor,
    )
    _torch_backports.apply_async_checkpoint_patch()

    assert FakeExecutor().execute_save(process_group=expected_process_group) is save_future
    assert not save_future.done()
    assert call_order == ["process_group_info", "checkpoint_process", "execute_save"]


def test_async_checkpoint_patch_propagates_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    initialization_error = RuntimeError("daemon initialization failed")
    original_called = False
    save_future: Future[None] = Future()

    class FakeProcessGroupInitInfo:
        def __init__(self, process_group: object | None) -> None:
            assert process_group is None

    class FakeAsyncCheckpointProcess:
        def __init__(self, *, pg_init_info: FakeProcessGroupInitInfo) -> None:
            raise initialization_error

    class FakeExecutor:
        @staticmethod
        def _execute_save_impl() -> None:
            return None

        def execute_save(self) -> Future[None]:
            nonlocal original_called
            original_called = True
            return save_future

    async_process_executor = SimpleNamespace(
        _CHECKPOINT_PROCESS=None,
        _ProcessGroupInitInfo=FakeProcessGroupInitInfo,
        _AsyncCheckpointProcess=FakeAsyncCheckpointProcess,
        _ProcessBasedAsyncCheckpointExecutor=FakeExecutor,
    )
    monkeypatch.setattr(
        _torch_backports.importlib,
        "import_module",
        lambda _: async_process_executor,
    )
    _torch_backports.apply_async_checkpoint_patch()

    with pytest.raises(RuntimeError, match="daemon initialization failed"):
        FakeExecutor().execute_save()
    assert not original_called


def test_async_checkpoint_patch_initializes_daemon_once(monkeypatch: pytest.MonkeyPatch) -> None:
    initialization_started = threading.Event()
    finish_initialization = threading.Event()
    save_future: Future[None] = Future()
    initialization_count = 0
    returned_futures: list[Future[None]] = []

    class FakeProcessGroupInitInfo:
        def __init__(self, process_group: object | None) -> None:
            assert process_group is None

    class FakeAsyncCheckpointProcess:
        def __init__(self, *, pg_init_info: FakeProcessGroupInitInfo) -> None:
            nonlocal initialization_count
            initialization_count += 1
            initialization_started.set()
            assert finish_initialization.wait(timeout=1)

    class FakeExecutor:
        @staticmethod
        def _execute_save_impl() -> None:
            return None

        def execute_save(self) -> Future[None]:
            return save_future

    async_process_executor = SimpleNamespace(
        _CHECKPOINT_PROCESS=None,
        _ProcessGroupInitInfo=FakeProcessGroupInitInfo,
        _AsyncCheckpointProcess=FakeAsyncCheckpointProcess,
        _ProcessBasedAsyncCheckpointExecutor=FakeExecutor,
    )
    monkeypatch.setattr(
        _torch_backports.importlib,
        "import_module",
        lambda _: async_process_executor,
    )
    _torch_backports.apply_async_checkpoint_patch()

    def call_patched_execute_save() -> None:
        returned_futures.append(FakeExecutor().execute_save())

    callers = [threading.Thread(target=call_patched_execute_save) for _ in range(2)]
    for caller in callers:
        caller.start()
    try:
        assert initialization_started.wait(timeout=1)
    finally:
        finish_initialization.set()
        for caller in callers:
            caller.join(timeout=1)

    assert initialization_count == 1
    assert returned_futures == [save_future, save_future]
    assert async_process_executor._CHECKPOINT_PROCESS is not None
