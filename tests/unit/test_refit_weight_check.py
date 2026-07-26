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


def test_counts_unchanged_per_layer():
    params = {"layers.0.w": torch.zeros(2), "layers.0.b": torch.zeros(2), "layers.1.w": torch.zeros(2)}
    worker = _worker(params)
    worker.reset_weight_update_check()
    params["layers.0.w"] = torch.ones(2)  # only layer 0 was refreshed
    assert worker.weight_update_missing() == {0: [1, 2], 1: [1, 1]}


def test_a_fully_dropped_broadcast_leaves_every_layer_untouched():
    worker = _worker({"layers.0.w": torch.zeros(2), "layers.1.w": torch.zeros(2)})
    worker.reset_weight_update_check()
    assert worker.weight_update_missing() == {0: [1, 1], 1: [1, 1]}


def test_params_outside_the_decoder_stack_group_under_minus_one():
    params = {"embed_tokens.weight": torch.zeros(2), "layers.3.w": torch.zeros(2)}
    worker = _worker(params)
    worker.reset_weight_update_check()
    params["embed_tokens.weight"] = torch.ones(2)
    assert worker.weight_update_missing() == {-1: [0, 1], 3: [1, 1]}


def test_a_sign_flipping_update_is_not_missed():
    """Sum would cancel here; sum of squares must not."""
    params = {"layers.0.w": torch.tensor([1.0, -1.0])}
    worker = _worker(params)
    worker.reset_weight_update_check()
    params["layers.0.w"] = torch.tensor([2.0, -2.0])
    assert worker.weight_update_missing() == {0: [0, 1]}


def test_check_off_reports_nothing():
    assert _worker({"a": torch.zeros(2)}).weight_update_missing() is None
