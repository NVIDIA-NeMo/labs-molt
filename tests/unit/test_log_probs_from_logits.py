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

import math

import torch

from molt.models.utils import log_probs_from_logits


def test_chunked_log_probs_match_log_softmax_in_value_and_gradient():
    # bf16 input takes the chunked path; 700 rows leaves a partial trailing chunk.
    torch.manual_seed(0)
    logits = torch.randn(2, 350, 1000, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 1000, (2, 350))

    log_probs = log_probs_from_logits(logits, labels, temperature=0.7)
    log_probs.sum().backward()

    ref_logits = logits.detach().float().clone().requires_grad_(True)
    reference = torch.log_softmax(ref_logits / 0.7, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    reference.sum().backward()

    torch.testing.assert_close(log_probs, reference.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits.grad.float(), ref_logits.grad.to(torch.bfloat16).float(), atol=1e-2, rtol=1e-2)


def test_log_probs_from_logits_gathers_several_targets_per_position():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 50, dtype=torch.bfloat16)
    ids = torch.randint(0, 50, (2, 5, 4))
    out = log_probs_from_logits(logits, ids, temperature=0.7)
    reference = torch.log_softmax(logits.float() / 0.7, dim=-1).gather(-1, ids)
    assert out.shape == (2, 5, 4)
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
    # The single-target call is the k == 1 case of the same path.
    torch.testing.assert_close(log_probs_from_logits(logits, ids[..., 0], temperature=0.7), out[..., 0])


def test_sampled_binary_kl_is_zero_when_matched_and_follows_the_bernoulli_formula():
    from molt.models.utils import sampled_binary_kl

    lp = torch.log(torch.tensor([[0.2, 0.9]]))
    torch.testing.assert_close(sampled_binary_kl(lp, lp), torch.zeros(1, 2))
    p, q = 0.2, 0.25
    expected = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    torch.testing.assert_close(
        sampled_binary_kl(torch.tensor([[math.log(p)]]), torch.tensor([[math.log(q)]])), torch.tensor([[expected]])
    )
