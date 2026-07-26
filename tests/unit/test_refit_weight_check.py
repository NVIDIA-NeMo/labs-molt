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

"""`--train.check_weight_update_equal` verifies the refit by VALUE (vllm_worker_wrap).

Name-based coverage cannot work across models: which names ``load_weights`` reports is
model-specific, so a diff against them flags in-sync weights as stale. Diffing the params'
values before and after the broadcast asks the only question that matters — did the engine
actually receive the new weights.
"""

import types

import torch

from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap


def _worker(params):
    """A WorkerWrap stub whose model exposes `params` as its float parameters."""
    worker = WorkerWrap.__new__(WorkerWrap)
    model = types.SimpleNamespace(named_parameters=lambda: iter(list(params.items())))
    worker.model_runner = types.SimpleNamespace(model=model)
    return worker


def test_reports_only_the_params_the_broadcast_did_not_move():
    params = {"a": torch.zeros(2), "b": torch.zeros(2)}
    worker = _worker(params)
    worker.reset_weight_update_check()
    params["a"] = torch.ones(2)  # refreshed by the broadcast
    assert worker.weight_update_missing() == (["b"], 2)


def test_a_fully_dropped_broadcast_shows_every_param_unchanged():
    worker = _worker({"a": torch.zeros(2), "b": torch.zeros(2)})
    worker.reset_weight_update_check()
    assert worker.weight_update_missing() == (["a", "b"], 2)


def test_a_complete_broadcast_reports_nothing_unchanged():
    params = {"a": torch.zeros(2), "b": torch.zeros(2)}
    worker = _worker(params)
    worker.reset_weight_update_check()
    params["a"], params["b"] = torch.ones(2), torch.full((2,), 3.0)
    assert worker.weight_update_missing() == ([], 2)


def test_check_off_reports_nothing():
    assert _worker({"a": torch.zeros(2)}).weight_update_missing() is None
