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

"""begin/end_weight_update bracket a refit with vLLM's layerwise reload: initialize it on the engine's
model before the first flush, finalize it with the engine's model config after the last."""

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap


def test_bracket_drives_vllm_layerwise_reload(monkeypatch):
    calls = []

    @contextmanager
    def set_current_vllm_config(vllm_config, **_):
        calls.append(("config", vllm_config))
        yield

    reload = "vllm.model_executor.model_loader.reload"
    for name in ("vllm", "vllm.config", "vllm.model_executor", "vllm.model_executor.model_loader", reload):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.config"].set_current_vllm_config = set_current_vllm_config
    sys.modules[reload].initialize_layerwise_reload = lambda model: calls.append(("initialize", model))
    sys.modules[reload].finalize_layerwise_reload = lambda model, cfg: calls.append(("finalize", model, cfg))

    worker = WorkerWrap()
    worker.vllm_config, worker.model_config = object(), object()
    worker.model_runner = SimpleNamespace(model=object())

    worker.begin_weight_update()
    worker.end_weight_update()
    assert calls == [
        ("config", worker.vllm_config),
        ("initialize", worker.model_runner.model),
        ("config", worker.vllm_config),
        ("finalize", worker.model_runner.model, worker.model_config),
    ]
