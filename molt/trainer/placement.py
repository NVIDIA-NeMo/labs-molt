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

import os


# Address https://github.com/ray-project/ray/issues/51117
# This function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
def get_bundle_indices(placement_group, index, length):
    import ray

    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = {}
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles.setdefault(node_id, []).append(bundle)

    sorted_bundle_indices = sum(node_id_to_bundles.values(), [])
    return sorted_bundle_indices[index * length : (index + 1) * length]


def model_placement_strategy() -> str:
    """PACK the actor's GPUs node-contiguously (rank i takes the i-th bundle in
    node-sorted order, see get_bundle_indices). SPREAD interleaves bundles
    across nodes, which scatters the EP/CP group across nodes and makes deep_ep's
    intra-node NVLink dispatch hit a cross-node peer (illegal memory access on the
    >1-node actor). PACK keeps EP=8/CP=8 within one node, matching torchrun/SFT."""
    return "PACK"


def ray_noset_visible_devices(env_vars=os.environ):
    # CUDA-only: refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    return bool(env_vars.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"))
