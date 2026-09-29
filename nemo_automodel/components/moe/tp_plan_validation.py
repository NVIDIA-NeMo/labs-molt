# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Routed-expert ownership validation for custom-MoE tensor-parallel plans.

Both the dedicated MoE parallelizer (``components.moe.parallelizer``) and the
generic dense parallelizer (``components.distributed.parallelizer``, for the
EP=1 path) must reject TP plans that shard routed-expert or router parameters.
The validator lives in this dependency-free module so the two parallelizers can
share it without importing each other.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.distributed.tensor.parallel import ParallelStyle


def _is_unsafe_moe_tp_module_path(module_path: str) -> bool:
    path_segments = set(module_path.split("."))
    leaf = module_path.rsplit(".", 1)[-1]
    return (
        "experts" in path_segments
        or "gate_and_up_projs" in path_segments
        or "down_projs" in path_segments
        or "router" in path_segments
        or "shared_expert_gate" in path_segments
        or "gate" in path_segments
        or leaf in {"mlp", "moe"}
    )


def _validate_moe_tp_plan(plan: object, model: nn.Module | None = None) -> dict[str, ParallelStyle]:
    """Validate that a TP plan cannot shard routed-expert parameters.

    Routed experts are owned by expert parallelism and, when configured, the
    independent ``ep_shard`` FSDP mesh. Applying tensor parallelism to the same
    parameters would compose unrelated DTensor placements and either corrupt
    ownership or fail during the first grouped GEMM. Shared experts are regular
    dense MLPs and are intentionally allowed.
    """
    if not isinstance(plan, dict) or not plan:
        raise ValueError("The custom MoE tensor-parallel plan must be a non-empty dictionary.")

    unsafe_paths: list[str] = []
    unmatched_paths: list[str] = []
    module_names = tuple(name for name, _ in model.named_modules()) if hasattr(model, "named_modules") else ()
    for module_path in plan:
        if not isinstance(module_path, str):
            raise ValueError("Every custom MoE tensor-parallel plan key must be a module path string.")
        if _is_unsafe_moe_tp_module_path(module_path):
            unsafe_paths.append(module_path)
            continue
        if module_names:
            matched_names = [name for name in module_names if fnmatchcase(name, module_path)]
            if not matched_names:
                unmatched_paths.append(module_path)
            elif any(_is_unsafe_moe_tp_module_path(name) for name in matched_names):
                unsafe_paths.append(module_path)

    if unsafe_paths:
        formatted = ", ".join(sorted(unsafe_paths))
        raise ValueError(
            "Tensor parallelism must not target routed-expert parameters or router outputs because they are "
            "EP-owned or require global expert logits. "
            f"Remove these paths from tp_shard_plan: {formatted}. Shared-expert leaf modules remain supported."
        )
    if unmatched_paths:
        formatted = ", ".join(sorted(unmatched_paths))
        raise ValueError(
            "Custom MoE tensor-parallel plan patterns must each match at least one concrete module; "
            f"unmatched paths: {formatted}."
        )
    return plan
