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

"""The bf16 rounding --train.stochastic_rounding applies to the fp32 master at each refit.

Every result is one of the input's two bf16 neighbours, a value already on the bf16 grid stays
put, and the mean over many draws is the input: the property round-to-nearest lacks.
"""

import torch

from molt.trainer.workers.policy_actor import _stochastic_round_to_bf16_


def _rounded(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    _stochastic_round_to_bf16_(out)
    return out


def test_result_is_a_bf16_neighbour():
    torch.manual_seed(0)
    x = torch.randn(10_000) * torch.logspace(-20, 20, 10_000)
    toward_zero = x.view(torch.int32) & -(1 << 16)
    away_from_zero = toward_zero + (1 << 16)
    q = _rounded(x).view(torch.int32)
    assert ((q == toward_zero) | (q == away_from_zero)).all()
    assert (q == away_from_zero).float().mean().item() > 0.4  # both directions are taken


def test_bf16_values_do_not_move():
    torch.manual_seed(0)
    x = torch.cat([torch.randn(10_000).bfloat16().float(), torch.tensor([0.0, -0.0])])
    assert torch.equal(_rounded(x).view(torch.int32), x.view(torch.int32))


def test_unbiased():
    torch.manual_seed(0)
    x = torch.randn(1000)
    mean = torch.stack([_rounded(x) for _ in range(4096)]).double().mean(dim=0)
    step = 2.0 ** (torch.frexp(x).exponent - 8)  # bf16 keeps 8 significant bits
    assert ((mean - x).abs() < 0.05 * step).all()  # one draw is off by up to a step; 4096 average it out
    assert ((x.bfloat16().float() - x).abs() > 0.4 * step).any()  # nearest stays up to half a step off
