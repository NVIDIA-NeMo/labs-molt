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

"""`--train.check_weight_update_equal` finds the weights a refit broadcast never landed on.

vLLM's ``load_weights`` returns the params it assigned; anything it holds and never assigned
kept its old value and makes the rollout stale. The one trap is the namespace: a vLLM model
may report the pre-mapping or the fused name rather than its own parameter's, and then the
set difference accuses the whole model. So the diff is reported only when the reported names
really are its parameters.
"""

import types

import torch

from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap


def _worker(params, loaded):
    """A WorkerWrap stub holding `params`, whose last broadcast assigned `loaded`."""
    worker = WorkerWrap.__new__(WorkerWrap)
    model = types.SimpleNamespace(named_parameters=lambda: iter(list(params.items())))
    worker.model_runner = types.SimpleNamespace(model=model)
    worker._refit_loaded = set(loaded)
    return worker


def test_lists_the_weights_the_broadcast_never_landed_on():
    params = {"model.layers.0.w": torch.zeros(2), "model.layers.0.experts": torch.zeros(2)}
    assert _worker(params, ["model.layers.0.w"]).refit_unaddressed_params() == ["model.layers.0.experts"]


def test_reports_nothing_when_every_weight_was_assigned():
    params = {"model.layers.0.w": torch.zeros(2)}
    assert _worker(params, ["model.layers.0.w"]).refit_unaddressed_params() == []


def test_cannot_verify_when_the_reported_names_are_from_another_namespace():
    """vLLM may report the pre-mapping or fused name; a diff would then accuse everything."""
    params = {"model.layers.0.w": torch.zeros(2)}
    assert _worker(params, ["layers.0.w"]).refit_unaddressed_params() is None


def test_cannot_verify_when_the_model_reports_nothing():
    assert _worker({"model.layers.0.w": torch.zeros(2)}, []).refit_unaddressed_params() is None


def test_non_float_params_are_not_expected_to_be_assigned():
    params = {"model.layers.0.w": torch.zeros(2), "model.layers.0.idx": torch.tensor([7])}
    assert _worker(params, ["model.layers.0.w"]).refit_unaddressed_params() == []
