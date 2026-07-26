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

import re
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
    params = {"layers.0.w": torch.zeros(2), "layers.1.w": torch.zeros(2)}
    worker = _worker(params)
    worker.reset_weight_update_check()
    params["layers.0.w"] = torch.ones(2)
    assert worker.weight_update_missing() == (["layers.1.w"], 2)


def test_a_fully_dropped_broadcast_shows_every_param_unchanged():
    worker = _worker({"layers.0.w": torch.zeros(2), "layers.1.w": torch.zeros(2)})
    worker.reset_weight_update_check()
    assert worker.weight_update_missing() == (["layers.0.w", "layers.1.w"], 2)


def test_a_sign_flipping_update_is_not_missed():
    """Sum would cancel here; sum of squares must not."""
    params = {"layers.0.w": torch.tensor([1.0, -1.0])}
    worker = _worker(params)
    worker.reset_weight_update_check()
    params["layers.0.w"] = torch.tensor([2.0, -2.0])
    assert worker.weight_update_missing() == ([], 1)


def _never_refreshed(unchanged_per_broadcast):
    """The trainer's accumulation: kinds unchanged in EVERY broadcast so far."""
    residue = None
    for unchanged in unchanged_per_broadcast:
        kinds = {re.sub(r"\.\d+\.", ".*.", name) for name in unchanged}
        residue = kinds if residue is None else residue & kinds
    return residue


def test_a_weight_too_small_to_move_every_step_drops_out_of_the_residue():
    """An RL step can round to the same bf16 value, so one broadcast proves nothing."""
    attn = "model.layers.0.self_attn.qkv_proj.weight"
    expert = "model.layers.0.mlp.experts.w13_weight"
    # attn misses two broadcasts but moves on the third; the expert never moves.
    residue = _never_refreshed([[attn, expert], [attn, expert], [expert]])
    assert residue == {"model.layers.*.mlp.experts.w13_weight"}


def test_one_kind_missed_across_every_layer_collapses_to_one_entry():
    """The shape a name-format break makes: every layer's experts stale, the rest fine."""
    unchanged = [f"model.layers.{i}.mlp.experts.w13_weight" for i in range(40)]
    assert _never_refreshed([unchanged]) == {"model.layers.*.mlp.experts.w13_weight"}


def test_check_off_reports_nothing():
    assert _worker({"a": torch.zeros(2)}).weight_update_missing() is None
