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

"""Dense Qwen3-VL distributed-parallelization registration."""

from __future__ import annotations


def register_qwen3_vl_parallel_strategy() -> None:
    """Register the dense Qwen3-VL FSDP2 strategy once."""
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
        register_parallel_strategy,
    )

    name = "Qwen3VLForConditionalGeneration"
    if name in PARALLELIZATION_STRATEGIES:
        return

    @register_parallel_strategy(name=name)
    class Qwen3VLParallelizationStrategy(DefaultParallelizationStrategy):
        """Install the CP submesh used by Qwen3-VL's model-owned forward."""

        def parallelize(self, model, device_mesh, **kwargs):
            result = super().parallelize(model, device_mesh, **kwargs)
            cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
            model.cp_mesh = cp_mesh if cp_mesh is not None and cp_mesh.size() > 1 else None
            return result
