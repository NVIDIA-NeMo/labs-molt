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

"""Muon param classification — grouped MoE experts must reach Muon.

AutoModel stores MoE experts as one 3D grouped weight; _classify_params routes it
to Muon (not AdamW). The detection must survive FSDP2's class rename under multi-node
EP (GroupedExpertsDeepEP -> FSDPGroupedExpertsDeepEP), which a plain __name__ check missed.
"""

import torch
import torch.nn as nn

from molt.trainer.fsdp.muon import _classify_params


class GroupedExpertsDeepEP(nn.Module):
    """Mimics AutoModel's grouped-experts module: one (num_experts, in, out) weight."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 8, 8))


def _model(experts_module):
    m = nn.Module()
    m.experts = experts_module
    return m


def test_grouped_experts_go_to_muon():
    matrix, vector, _, _ = _classify_params(_model(GroupedExpertsDeepEP()), experts_to_muon=True)
    assert any(p.ndim == 3 for p in matrix)  # expert weight -> Muon
    assert not any(p.ndim == 3 for p in vector)


def test_fsdp_renamed_experts_still_go_to_muon():
    # FSDP2 fully_shard swaps the class to a dynamic subclass; the base name lives on
    # in the MRO, so MRO-based detection still routes the expert weight to Muon.
    renamed = type("FSDPGroupedExpertsDeepEP", (GroupedExpertsDeepEP,), {})
    matrix, vector, _, _ = _classify_params(_model(renamed()), experts_to_muon=True)
    assert any(p.ndim == 3 for p in matrix)  # still Muon despite the rename
    assert not any(p.ndim == 3 for p in vector)


def test_experts_to_muon_false_keeps_experts_on_adamw():
    # escape hatch: experts fall back to AdamW (3D -> vector) when disabled
    matrix, vector, _, _ = _classify_params(_model(GroupedExpertsDeepEP()), experts_to_muon=False)
    assert not any(p.ndim == 3 for p in matrix)
    assert any(p.ndim == 3 for p in vector)


def test_lora_adapters_go_to_adamw():
    # Muon's orthogonalized update is defined for a full-rank weight, not for a rank-r factor
    # pair. Both 2D adapters would otherwise land in the Muon group as plain linear weights.
    m = nn.Module()
    m.lora_A = nn.Linear(8, 2, bias=False)
    m.lora_B = nn.Linear(2, 8, bias=False)
    matrix, vector, _, _ = _classify_params(m, experts_to_muon=True)
    assert matrix == []
    assert len(vector) == 2
