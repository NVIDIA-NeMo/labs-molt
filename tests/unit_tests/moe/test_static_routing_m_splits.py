# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavioral coverage for the static-routing m_splits cache used by GroupedExpertsTE."""

import torch

from nemo_automodel.components.moe.experts import _resolve_m_splits


def test_without_static_routing_every_call_materializes():
    first = _resolve_m_splits(torch.tensor([3, 1, 0, 4]), static_routing=False, cached_m_splits=None)
    assert first == [3, 1, 0, 4]
    # A later microbatch with different counts must be re-materialized, never cached.
    second = _resolve_m_splits(torch.tensor([2, 2, 2, 2]), static_routing=False, cached_m_splits=first)
    assert second == [2, 2, 2, 2]


def test_static_routing_reuses_first_microbatch_splits():
    counts = torch.tensor([4, 4, 4, 4])
    first = _resolve_m_splits(counts, static_routing=True, cached_m_splits=None)
    assert first == [4, 4, 4, 4]

    # Warm cache: the cached list object is returned as-is and the tensor is not
    # consulted (this is the device-to-host sync being skipped). Passing different
    # counts documents the contract: static routing asserts the counts are constant,
    # so the cache — not the tensor — is authoritative.
    later = _resolve_m_splits(torch.tensor([9, 9, 9, 9]), static_routing=True, cached_m_splits=first)
    assert later is first


def test_sequence_inputs_pass_through_as_lists():
    assert _resolve_m_splits((5, 6), static_routing=False, cached_m_splits=None) == [5, 6]
    assert _resolve_m_splits([7], static_routing=True, cached_m_splits=None) == [7]
