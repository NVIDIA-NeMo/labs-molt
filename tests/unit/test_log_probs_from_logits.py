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
