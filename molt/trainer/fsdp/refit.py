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

"""vLLM weight refit for the FSDP2/AutoModel backend.

Owns *how* to materialize each pushed parameter (``gather_full_param``): under
FSDP2, params are ``DTensor`` instances whose ``.full_tensor()`` gathers the
unsharded tensor across both FSDP shard and TP shard dims in one call.

The sender (``trainer/workers/policy_actor.py``) pushes every ``state_dict``
entry — vLLM's ``load_weights`` matches by name and ignores what it doesn't have,
so the "which weights to accept" decision lives on the vLLM side.
"""

from typing import Optional, Tuple

import torch
from torch.distributed.tensor import DTensor


def gather_full_param(param: torch.Tensor, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Size]:
    """Materialize the full unsharded tensor for an FSDP2/TP-sharded parameter.

    Returns ``(full_tensor, full_shape)`` where ``full_tensor`` is on the local
    device with all mesh dims gathered. For non-DTensor params (e.g., the value
    head we don't shard, or buffers), returns ``(param.data, param.shape)``.

    Caller invokes this on each rank; ``full_tensor`` is replicated. Memory cost
    is the size of the full tensor on every participating rank — acceptable for
    weight refit (one-shot per training step). For very large models the async RL
    path uses per-tensor streaming with a ping-pong buffer to bound peak memory.
    """
    if isinstance(param, DTensor):
        full = param.full_tensor()
    else:
        full = param.data
    if dtype is not None and full.is_floating_point():
        full = full.to(dtype=dtype)
    return full, full.shape
