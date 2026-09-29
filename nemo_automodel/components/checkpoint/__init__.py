# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from ._torch_backports import apply_async_checkpoint_patch as _nemo__apply_async_patch
from ._torch_backports import apply_patches as _nemo__apply_patches
from .config import CheckpointingConfig, _is_geq_torch_2_9, _is_leq_torch_2_7_1

__all__ = ["CheckpointingConfig"]

if _is_leq_torch_2_7_1():
    _nemo__apply_patches()

if _is_geq_torch_2_9():
    _nemo__apply_async_patch()
