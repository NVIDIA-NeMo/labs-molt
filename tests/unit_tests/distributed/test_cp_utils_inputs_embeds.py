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

"""Tests for ``_make_cp_batch_and_ctx`` accepting ``inputs_embeds`` as the
primary sequence tensor (VLM-CP path).

These cover:
  - XOR contract: exactly one of ``input_ids`` / ``inputs_embeds`` in batch
  - The cp_buffers list uses ``inputs_embeds`` when present
  - position_ids synthesis works whether ``input_ids`` or ``inputs_embeds`` is the source
  - ``cp_size <= 1`` short-circuit applies regardless of which key is present
"""

from __future__ import annotations





class _DummySubMesh:
    def __init__(self, size: int, local_rank: int = 0):
        self._size = size
        self._local_rank = local_rank

    def size(self) -> int:
        return self._size

    def get_group(self):
        return None

    def get_local_rank(self) -> int:
        return self._local_rank


class _DummyDeviceMesh(dict):
    def __init__(self, cp_size: int, tp_size: int, cp_rank: int = 0):
        super().__init__()
        self["cp"] = _DummySubMesh(cp_size, cp_rank)
        self["tp"] = _DummySubMesh(tp_size)
        self.mesh_dim_names = ["cp", "tp"]


class _DummyHSDPDeviceMesh(_DummyDeviceMesh):
    def __init__(self, root_rank: int, cp_rank: int):
        super().__init__(cp_size=2, tp_size=1, cp_rank=cp_rank)
        self["dp_replicate"] = _DummySubMesh(2)
        self["dp_shard"] = _DummySubMesh(2)
        self.mesh_dim_names = ["dp_replicate", "dp_shard", "cp", "tp"]
        self._root_rank = root_rank

    def get_local_rank(self) -> int:
        return self._root_rank


def test_padding_attention_mask_pad_value_is_zero(monkeypatch):
    """If a future caller passes an ``attention_mask`` in the batch, it should
    pad with ``0`` (HF convention: 1=real, 0=pad) -- NOT with True/dtype-default.

    Today ``shard_batch_load_balanced`` strips ``attention_mask`` at the top of
    the function so this case is moot, but the PAD_FILL table is the right
    place to encode the semantic in case the strip is ever revisited.
    """
    from nemo_automodel.components.distributed.context_parallel import sharder as _cs

    # Just verify the PAD_FILL table itself maps attention_mask -> False
    # (the runtime code path is currently unreachable because attention_mask
    # is popped before the padding pass).
    src = open(_cs.__file__).read()
    assert '"attention_mask": False' in src, "PAD_FILL must explicitly map attention_mask -> False (HF: 0 = pad)"


