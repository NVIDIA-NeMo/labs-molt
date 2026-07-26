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

"""`--train.check_weight_update_equal` asks whether vLLM holds the trainer's weights.

Names cannot be compared across the two sides — vLLM concatenates gate_proj|up_proj into
w13_weight, fuses q|k|v, rewrites prefixes and shards params over its workers. None of that
changes the multiset of values, so per-layer energy (sum of squares) is equal on both sides
exactly when the engine holds what was sent, whatever the update's magnitude.
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


def test_energy_is_summed_per_decoder_layer():
    worker = _worker(
        {
            "model.layers.0.self_attn.qkv_proj.weight": torch.tensor([3.0, 4.0]),  # 25
            "model.layers.0.mlp.experts.w13_weight": torch.tensor([1.0]),  # 1
            "model.layers.1.self_attn.qkv_proj.weight": torch.tensor([2.0]),  # 4
        }
    )
    assert worker.weight_energy_by_layer() == {0: 26.0, 1: 4.0}


def test_params_outside_the_decoder_stack_group_under_minus_one():
    worker = _worker({"model.embed_tokens.weight": torch.tensor([2.0]), "model.layers.3.w": torch.tensor([1.0])})
    assert worker.weight_energy_by_layer() == {-1: 4.0, 3: 1.0}


def test_fusing_two_weights_into_one_preserves_the_layer_energy():
    """gate_proj|up_proj -> w13_weight is a concatenation, so the sender and the engine agree."""
    sender = _worker(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.tensor([3.0]),
            "model.layers.0.mlp.up_proj.weight": torch.tensor([4.0]),
        }
    )
    engine = _worker({"model.layers.0.mlp.w13_weight": torch.tensor([3.0, 4.0])})
    assert sender.weight_energy_by_layer() == engine.weight_energy_by_layer()


def test_a_layer_the_engine_never_received_shows_up_as_energy_drift():
    """Even an update too small for the weight's dtype to move leaves the energies unequal."""
    sent = _worker({"model.layers.0.w": torch.tensor([1.001, 2.0])}).weight_energy_by_layer()
    held = _worker({"model.layers.0.w": torch.tensor([1.0, 2.0])}).weight_energy_by_layer()
    assert abs(sent[0] - held[0]) / held[0] > 1e-4


def test_non_float_params_are_skipped():
    worker = _worker({"model.layers.0.w": torch.tensor([2.0]), "model.layers.0.idx": torch.tensor([7])})
    assert worker.weight_energy_by_layer() == {0: 4.0}


def _armed_worker(params, loaded):
    """A worker whose last broadcast reported `loaded` as the names it assigned."""
    worker = _worker(params)
    worker._refit_loaded = set(loaded)
    return worker


def test_by_name_coverage_lists_the_params_the_broadcast_never_addressed():
    params = {"model.layers.0.w": torch.tensor([1.0]), "model.layers.0.experts": torch.tensor([1.0])}
    worker = _armed_worker(params, ["model.layers.0.w"])
    assert worker.refit_unaddressed_params() == ["model.layers.0.experts"]


def test_by_name_coverage_is_unavailable_when_the_namespaces_differ():
    """vLLM may report the pre-mapping or fused name; then a diff would flag the whole model."""
    params = {"model.layers.0.w": torch.tensor([1.0])}
    worker = _armed_worker(params, ["layers.0.w"])  # same weight, different namespace
    assert worker.refit_unaddressed_params() is None


def test_by_name_coverage_is_unavailable_when_the_model_reports_nothing():
    assert _armed_worker({"model.layers.0.w": torch.tensor([1.0])}, []).refit_unaddressed_params() is None
