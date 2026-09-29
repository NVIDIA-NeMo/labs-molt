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
"""Tests for the DTensor-safe embedding-row zeroing used by weight initialization."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from nemo_automodel.shared import embedding_padding
from nemo_automodel.shared.embedding_padding import zero_embedding_row_


@pytest.fixture
def single_rank_pg():
    """Single-rank gloo process group, enough to build CPU DTensors."""
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    yield
    dist.destroy_process_group()


def test_plain_tensor_row_is_zeroed():
    weight = torch.ones(6, 4)
    assert zero_embedding_row_(weight, 2)
    assert torch.all(weight[2] == 0)
    assert weight.sum() == 20


def test_negative_row_and_out_of_range():
    weight = torch.ones(6, 4)
    assert zero_embedding_row_(weight, -1)
    assert torch.all(weight[5] == 0)
    with pytest.raises(IndexError):
        zero_embedding_row_(weight, 6)


@pytest.mark.parametrize(
    "placement", [Shard(0), Shard(1), Replicate()], ids=["shard_vocab", "shard_hidden", "replicate"]
)
def test_dtensor_row_is_zeroed_without_redistribute(single_rank_pg, placement):
    mesh = init_device_mesh("cpu", (1,))
    weight = nn.Parameter(distribute_tensor(torch.ones(6, 4), mesh, [placement]))
    assert zero_embedding_row_(weight, 5)
    full = weight.full_tensor()
    assert torch.all(full[5] == 0)
    assert full.sum() == 20


def test_unsupported_placement_warns_instead_of_raising(single_rank_pg, caplog):
    # L2_Parallelism_VLM_Gemma4_TP2_Parity built a from_pretrained Gemma4 whose embedding carried
    # _StridedShard(0, 2); a raise here aborted model construction. Unknown layouts must degrade.
    from torch.distributed.tensor import Partial

    mesh = init_device_mesh("cpu", (1,))
    weight = distribute_tensor(torch.ones(6, 4), mesh, [Partial()])
    with caplog.at_level("WARNING"):
        assert not zero_embedding_row_(weight, 1)
    assert "unsupported DTensor placements" in caplog.text
    assert weight.to_local().sum() == 24


def test_strided_shard_placement_is_accepted(single_rank_pg):
    from torch.distributed.tensor.placement_types import _StridedShard

    mesh = init_device_mesh("cpu", (1,))
    weight = DTensor.from_local(torch.ones(6, 4), mesh, [_StridedShard(0, split_factor=1)])
    assert zero_embedding_row_(weight, 2)
    assert torch.all(weight.to_local()[2] == 0) and weight.to_local().sum() == 20


def test_offset_failure_warns_instead_of_raising(single_rank_pg, monkeypatch, caplog):
    mesh = init_device_mesh("cpu", (1,))
    weight = distribute_tensor(torch.ones(6, 4), mesh, [Shard(0)])

    def boom(*_a, **_k):
        raise NotImplementedError("exotic layout")

    monkeypatch.setattr(embedding_padding, "compute_local_shape_and_global_offset", boom)
    with caplog.at_level("WARNING"):
        assert not zero_embedding_row_(weight, 1)
    assert "could not resolve the local shard" in caplog.text
    assert weight.to_local().sum() == 24


def test_row_owned_by_another_rank_is_left_alone(single_rank_pg, monkeypatch):
    mesh = init_device_mesh("cpu", (1,))
    weight = distribute_tensor(torch.ones(6, 4), mesh, [Shard(0)])
    # Pretend this rank's shard covers rows [3, 6): row 1 lives elsewhere.
    monkeypatch.setattr(embedding_padding, "compute_local_shape_and_global_offset", lambda *a, **k: ((3, 4), (3, 0)))
    assert not zero_embedding_row_(weight, 1)
    assert weight.full_tensor().sum() == 24
