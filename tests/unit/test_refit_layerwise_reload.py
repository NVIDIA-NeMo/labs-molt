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

"""begin/end_weight_update bracket a refit with vLLM's layerwise reload.

vLLM's MoE kernels may rewrite expert weights after loading, so a refit made of plain
load_weights flushes has to defer that processing: initialize the reload on the engine's
model before the first flush and finalize it, with the engine's model config, after the
last one.
"""

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap


@pytest.fixture
def reload_calls(monkeypatch):
    """Stub the vLLM reload entry points and record what the worker hands them."""
    calls = []

    @contextmanager
    def set_current_vllm_config(vllm_config, **_):
        calls.append(("config", vllm_config))
        yield

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": types.ModuleType("vllm.model_executor.model_loader"),
        "vllm.model_executor.model_loader.reload": types.ModuleType("vllm.model_executor.model_loader.reload"),
    }
    modules["vllm.config"].set_current_vllm_config = set_current_vllm_config
    reload = modules["vllm.model_executor.model_loader.reload"]
    reload.initialize_layerwise_reload = lambda model: calls.append(("initialize", model))
    reload.finalize_layerwise_reload = lambda model, model_config: calls.append(("finalize", model, model_config))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def _worker():
    worker = WorkerWrap()
    worker.vllm_config = object()
    worker.model_config = object()
    worker.model_runner = SimpleNamespace(model=object())
    return worker


def test_begin_initializes_layerwise_reload_on_the_engine_model(reload_calls):
    worker = _worker()
    worker.begin_weight_update()
    assert reload_calls == [("config", worker.vllm_config), ("initialize", worker.model_runner.model)]


def test_end_finalizes_with_the_engine_model_config(reload_calls):
    worker = _worker()
    worker.end_weight_update()
    assert reload_calls == [
        ("config", worker.vllm_config),
        ("finalize", worker.model_runner.model, worker.model_config),
    ]
