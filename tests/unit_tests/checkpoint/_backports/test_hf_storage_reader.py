# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, ReadItem

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader


class _RecordingLoadPlanner:
    def __init__(self, target: torch.Tensor) -> None:
        self.target = target
        self.committed: list[torch.Tensor] = []

    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        return self.target

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        self.committed.append(tensor.detach().cpu())


def test_hf_storage_reader_integrated_device_stages_narrowed_mmap_slice(tmp_path):
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({"weight": source}, tmp_path / "model.safetensors")

    reader = _HuggingFaceStorageReader(str(tmp_path))
    metadata = reader.read_metadata()
    reader.set_up_storage_reader(metadata, is_coordinator=True)
    storage_index = next(index for index in metadata.storage_data if index.fqn == "weight")
    storage_info = metadata.storage_data[storage_index]

    assert storage_info.offset % storage_info.dtype.itemsize == 0

    read_item = ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=MetadataIndex(fqn="weight", offset=[0, 0]),
        dest_offsets=torch.Size([0, 0]),
        storage_index=storage_index,
        storage_offsets=torch.Size([1, 1]),
        lengths=torch.Size([2, 2]),
    )
    target = torch.empty(2, 2, dtype=torch.float32)
    planner = _RecordingLoadPlanner(target)

    original_clone = torch.Tensor.clone
    cloned_shapes = []

    def record_clone(tensor, *args, **kwargs):
        cloned_shapes.append(tuple(tensor.shape))
        return original_clone(tensor, *args, **kwargs)

    with (
        patch(
            "nemo_automodel.components.checkpoint._backports.hf_storage._is_integrated_cuda_device",
            return_value=True,
        ) as is_integrated,
        patch.object(torch.Tensor, "clone", record_clone),
    ):
        reader.read_data(LoadPlan([read_item]), planner).wait()

    expected = source[1:3, 1:3]
    torch.testing.assert_close(target, expected)
    torch.testing.assert_close(planner.committed[0], expected)
    is_integrated.assert_called_once_with(target.device)
    assert cloned_shapes == [(2, 2)]
