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

"""Tensor-parallel output helpers for the AutoModel backend."""

from typing import Any

import torch
from torch.distributed.tensor import DTensor


def unshard_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a DTensor to a plain unsharded tensor on every rank.

    Use after model forward to gather TP-sharded outputs (logits with vocab dim
    sharded by ``ColwiseParallel(output_layouts=Shard(-1))``, or hidden states
    with sequence dim sharded by ``SequenceParallel`` in SP mode) so downstream
    loss / log-prob / entropy computations can run on plain tensors.

    No-op when not under TP/SP (input is a regular ``torch.Tensor``).
    Memory cost under TP=k: each rank holds the full unsharded tensor (∝ k× the
    sharded one); ok for activation-side tensors at training resolution.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def is_automodel_custom_model(model: Any) -> bool:
    """Best-effort check for AutoModel's native implementations.

    ``NeMoAutoModel*`` may return either a HF model (when ``force_hf`` is used or
    no native implementation exists) or a class under
    ``nemo_automodel.components.models``. Only the latter accepts THD packing
    kwargs directly.

    FSDP2's ``fully_shard`` swaps ``model.__class__`` for a dynamic subclass
    whose ``__module__`` no longer starts with ``nemo_automodel``; walk the MRO
    so the check survives that wrap.
    """
    for cls in type(model).__mro__:
        if getattr(cls, "_molt_automodel_custom", False):
            return True
        if issubclass(cls, torch.nn.Module) and cls.__module__.startswith("nemo_automodel.components.models"):
            return True
    return False
