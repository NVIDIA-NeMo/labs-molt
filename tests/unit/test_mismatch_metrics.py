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

from molt.models.utils import mismatch_metrics


def test_bitwise_frac_counts_last_bit_differences_that_a_printed_value_hides():
    rollout = torch.tensor([[-0.5, -1.25, -2.0, -3.0]])
    train = rollout.clone()
    train[0, 1] = torch.nextafter(train[0, 1], torch.tensor(0.0))  # one ulp away: same 6-decimal print
    mask = torch.tensor([[True, True, True, False]])  # the masked token is never counted
    m = mismatch_metrics(rollout, train, mask)
    assert abs(m["mismatch/bitwise_frac"].item() - 1 / 3) < 1e-6
    assert m["mismatch/abs_diff_mean"].item() > 0
    assert m["mismatch/k3_kl"].item() >= 0


def test_identical_bytes_read_exactly_zero_everywhere():
    rollout = torch.randn(2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    m = mismatch_metrics(rollout, rollout.clone(), mask)
    assert all(v.item() == 0.0 for v in m.values())
