# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""CPU unit test for the ndim-aware MoE expert FSDP shard placement."""

import torch
from torch.distributed.tensor import Shard

from nemo_automodel.components.moe.parallelizer import (
    _EMPTY_TENSOR_DETERMINISM_CHECK,
    _moe_checkpoint_metadata_fn,
    _moe_shard_placement,
    _register_moe_checkpoint_determinism_check,
)


class TestMoeShardPlacement:
    """1D expert bias -> Shard(0); >=2D expert weight -> Shard(1)."""

    def test_1d_bias_shards_on_dim0(self):
        bias = torch.zeros(5760)  # per-expert bias [out_features]
        assert bias.ndim == 1
        assert _moe_shard_placement(bias) == Shard(0)

    def test_2d_weight_shards_on_dim1(self):
        weight = torch.zeros(2048, 768)  # [out, in]
        assert weight.ndim == 2
        assert _moe_shard_placement(weight) == Shard(1)

    def test_3d_grouped_weight_shards_on_dim1(self):
        grouped = torch.zeros(128, 2048, 768)  # [n_experts, out, in]
        assert grouped.ndim == 3
        assert _moe_shard_placement(grouped) == Shard(1)


class TestMoeCheckpointMetadata:
    def test_empty_dtype_is_ignored_but_shape_and_device_are_retained(self):
        bf16 = _moe_checkpoint_metadata_fn(torch.empty(0, dtype=torch.bfloat16))
        fp32 = _moe_checkpoint_metadata_fn(torch.empty(0, dtype=torch.float32))

        assert bf16 == fp32
        assert bf16["shape"] == torch.Size([0])
        assert bf16["device"] == torch.device("cpu")
        assert bf16["dtype"] is None

    def test_nonempty_dtype_is_still_checked(self):
        bf16 = _moe_checkpoint_metadata_fn(torch.ones(1, dtype=torch.bfloat16))
        fp32 = _moe_checkpoint_metadata_fn(torch.ones(1, dtype=torch.float32))

        assert bf16["shape"] == fp32["shape"]
        assert bf16["device"] == fp32["device"]
        assert bf16["dtype"] == torch.bfloat16
        assert fp32["dtype"] == torch.float32
        assert bf16 != fp32

    def test_checker_registration_is_explicit(self):
        import torch.utils.checkpoint as torch_checkpoint

        name = _register_moe_checkpoint_determinism_check()

        assert name == _EMPTY_TENSOR_DETERMINISM_CHECK
        assert torch_checkpoint._allowed_determinism_checks_to_fns[name] is _moe_checkpoint_metadata_fn
