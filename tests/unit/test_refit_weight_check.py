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

"""`--train.check_weight_update_equal` coverage bookkeeping (vllm_worker_wrap).

The check subtracts the names vLLM's ``load_weights`` says it assigned from the worker's
float params. Models that return ``None`` instead of a name set must report "unavailable",
not "every param is stale" — that false positive is indistinguishable from a real
name-format break.
"""

import types

import torch

from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap


def _worker(loaded_return, params=None):
    """A WorkerWrap stub whose model holds `params` (default: two zero float params)."""
    params = params if params is not None else {"layers.0.w": torch.zeros(2), "layers.1.w": torch.zeros(2)}
    worker = WorkerWrap.__new__(WorkerWrap)
    model = types.SimpleNamespace(
        load_weights=lambda weights: loaded_return,
        named_parameters=lambda: iter(list(params.items())),
    )
    worker.model_runner = types.SimpleNamespace(model=model)
    worker._params = params
    return worker


def _record(worker, loaded_return):
    """Replay what update_weights_packed does with load_weights' return value."""
    if getattr(worker, "_weight_update_loaded", None) is not None:
        if loaded_return is None:
            worker._weight_update_reported = False
        else:
            worker._weight_update_loaded.update(loaded_return)


def test_all_params_refreshed_reports_empty():
    worker = _worker({"layers.0.w", "layers.1.w"})
    worker.reset_weight_update_check()
    _record(worker, {"layers.0.w", "layers.1.w"})
    assert worker.weight_update_missing() == ([], ["layers.0.w", "layers.1.w"])


def test_partially_refreshed_reports_the_stale_names():
    worker = _worker({"layers.0.w"})
    worker.reset_weight_update_check()
    _record(worker, {"layers.0.w"})
    assert worker.weight_update_missing() == (["layers.1.w"], ["layers.0.w", "layers.1.w"])


def test_model_without_loaded_names_falls_back_to_value_diff():
    worker = _worker(None)
    worker.reset_weight_update_check()
    _record(worker, None)
    worker._params["layers.0.w"] = torch.ones(2)  # this one was written by the broadcast
    assert worker.weight_update_missing() == (None, ["layers.1.w"])


def test_check_off_reports_nothing():
    worker = _worker(None)
    assert worker.weight_update_missing() is None
