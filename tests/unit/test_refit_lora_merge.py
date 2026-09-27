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

"""`_lora_merge_map` keys the vLLM refit merge off state-dict names, not module names.

The refit folds ``scale·B@A`` into each base weight before broadcasting. Activation
checkpointing (the RL default ``--actor.gradient_checkpoint=full``) wraps decoder layers in
``CheckpointWrapper``, which keeps ``_checkpoint_wrapped_module.`` in ``named_modules()`` but
strips it from ``state_dict()`` keys — a module-name-keyed map never matches, and the merge
silently no-ops, leaving vLLM on the frozen base for the whole run.
"""

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from molt.trainer.workers.policy_actor import _lora_merge_map


class _LinearLoRA(nn.Module):
    """Stand-in for AutoModel's LinearLoRA: ``lora_A``/``lora_B`` submodules plus a scale."""

    def __init__(self, out=4, inp=4, dim=2, scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out, inp))
        self.lora_A = nn.Linear(inp, dim, bias=False)
        self.lora_B = nn.Linear(dim, out, bias=False)
        self.scale = scale


def _toy_model() -> nn.Module:
    model = nn.Module()
    blocks = []
    for _ in range(2):
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.q_proj = _LinearLoRA(scale=2.0)
        blocks.append(block)
    model.layers = nn.ModuleList(blocks)
    return model


def test_merge_map_uses_state_dict_names_under_checkpoint_wrapper():
    model = _toy_model()
    model.layers[0] = checkpoint_wrapper(model.layers[0])

    named = {name for name, _ in model.named_modules()}
    assert "layers.0._checkpoint_wrapped_module.self_attn.q_proj" in named
    assert all("_checkpoint_wrapped_module" not in key for key in model.state_dict())

    merge = _lora_merge_map(model, model.state_dict())

    assert set(merge) == {"layers.0.self_attn.q_proj.weight", "layers.1.self_attn.q_proj.weight"}
    assert merge["layers.0.self_attn.q_proj.weight"] == (
        "layers.0.self_attn.q_proj.lora_A.weight",
        "layers.0.self_attn.q_proj.lora_B.weight",
        2.0,
    )


def test_merge_map_is_empty_without_adapters():
    model = nn.Module()
    model.proj = nn.Linear(4, 4)
    assert _lora_merge_map(model, model.state_dict()) == {}


def test_merge_map_rejects_an_adapter_without_its_module():
    model = _toy_model()
    sd = model.state_dict()
    sd["orphan.lora_A.weight"] = torch.zeros(2, 4)
    with pytest.raises(RuntimeError, match="orphan.lora_A.weight"):
        _lora_merge_map(model, sd)
