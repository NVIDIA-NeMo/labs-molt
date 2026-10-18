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
"""The per-group join must not keep a step's offloaded heavy tensors alive after the step consumed them."""

import gc
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

ray = pytest.importorskip("ray")

from molt.trainer.rollout.samples_generator import _gather_group  # noqa: E402

HEAVY = 4 * 1024 * 1024  # bytes of offloaded tensors per rollout
LIGHT = 200_000  # light fields large enough that results travel through the object store, as real batches do


class _Sample:
    def __init__(self, heavy_ref):
        self.heavy_ref = heavy_ref
        self.mask = np.zeros(LIGHT, dtype=np.uint8)


@ray.remote
class _Runner:
    def run_group(self):
        return [(_Sample(ray.put(np.ones(HEAVY, dtype=np.uint8))), None)]


def _plasma_mib():
    # The raylet debits the object_store_memory resource as objects are created and credits it as they are freed.
    total = ray.cluster_resources()["object_store_memory"]
    return (total - ray.available_resources()["object_store_memory"]) / 2**20


def test_a_consumed_group_releases_its_heavy_objects():
    # Workers import molt from this checkout and receive this module's classes by value.
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    root = str(Path(__file__).resolve().parents[2])
    pythonpath = os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
    ray.init(num_cpus=2, object_store_memory=256 * 1024 * 1024, include_dashboard=False, logging_level="ERROR",
             runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
    try:
        runners = [_Runner.remote() for _ in range(2)]
        for _ in range(6):  # six steps of two groups of four rollouts: 192 MiB if nothing were released
            groups = [_gather_group.remote(*[runners[i % 2].run_group.remote() for i in range(4)]) for _ in range(2)]
            samples = [sample for group in ray.get(groups) for sample, _ in group]
            ray.get([sample.heavy_ref for sample in samples])  # the training ranks' reload()
            del groups, samples
            gc.collect()
        time.sleep(1)  # frees reach the object store asynchronously
        assert _plasma_mib() < 32
    finally:
        ray.shutdown()
