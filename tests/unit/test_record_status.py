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

"""Cross-rank metric reduction of the policy trainer: keyed by name, weighted per key.

A microbatch's ``experience.info`` keys depend on its samples (a VLM dataset sets
``image_tokens`` only on rows with media; an env reports a failure code on failures only),
so DP ranks can hold different metric keys in different orders. Reducing them as one
collective per dict entry paired values by position and deadlocked on a key-count mismatch.
"""

import os
import socket
import types
from types import SimpleNamespace

import pytest
import torch

from molt.trainer.fsdp.strategy import FsdpStrategy
from molt.trainer.workers.policy_actor import PolicyTrainer


class _TwoRankStrategy:
    """Stands in for the collective: merges this rank's dict with a second rank's."""

    def __init__(self, other_rank):
        self.other = other_rank

    def all_reduce_dict(self, data):
        total = dict(data)
        for k, v in self.other.items():
            if k not in total:
                total[k] = v
            elif isinstance(v, tuple):
                total[k] = tuple(a + b for a, b in zip(total[k], v))
            else:
                total[k] += v
        return total


def _record(strategy, metrics, weights, n_tokens, n_samples):
    trainer = object.__new__(PolicyTrainer)
    trainer.strategy = strategy
    status_list = []
    trainer._record_status(
        {"metrics": metrics, "weights": weights, "num_action_tokens": n_tokens, "num_samples": n_samples},
        status_list,
        SimpleNamespace(set_postfix=lambda *_: None),
    )
    return status_list[0]


def test_metrics_are_paired_by_name_and_a_key_missing_on_one_rank_averages_over_its_reporters():
    # This rank: 2 samples / 10 tokens, kl (token-weighted) 0.1, image_tokens on both samples;
    # the other rank: 2 samples / 30 tokens, kl 0.3, no image_tokens key at all, keys in another order.
    other = {"_num_samples": 2.0, "_num_action_tokens": 30.0, "policy_loss": (0.2 * 30, 30.0), "kl": (0.3 * 30, 30.0)}
    merged = _record(
        _TwoRankStrategy(other),
        {
            "policy_loss": torch.tensor(0.1),
            "kl": torch.tensor(0.1),
            "image_tokens": torch.tensor([100.0, 200.0]),
            "actor_lr": 1e-6,
        },
        {"policy_loss": "token", "kl": "token", "image_tokens": "sample", "actor_lr": None},
        n_tokens=10.0,
        n_samples=2.0,
    )
    assert merged["kl"] == pytest.approx((0.1 * 10 + 0.3 * 30) / 40)  # token-weighted over both ranks
    assert merged["image_tokens"] == pytest.approx(150.0)  # averaged over the 2 samples that report it, not 4
    assert merged["actor_lr"] == 1e-6 and merged["_num_samples"] == 4.0 and merged["_num_action_tokens"] == 40.0


def test_sample_metric_reported_by_a_subset_of_samples_uses_the_reporting_count():
    # make_experience_batch leaves None on samples without the key; the trainer keeps the values
    # that exist, so a 1-D tensor shorter than the microbatch must be weighted by its own length.
    merged = _record(
        _TwoRankStrategy(
            {"_num_samples": 4.0, "_num_action_tokens": 10.0, "policy_loss": (0.0, 10.0), "fail_code": (3.0, 1.0)}
        ),
        {"policy_loss": torch.tensor(0.0), "fail_code": torch.tensor([1.0, 2.0])},
        {"policy_loss": "token", "fail_code": "sample"},
        n_tokens=10.0,
        n_samples=4.0,
    )
    assert merged["fail_code"] == pytest.approx((1.0 + 2.0 + 3.0) / 3)


def _gloo_worker(rank, world, port):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        stub = SimpleNamespace()
        stub.all_reduce_dict = types.MethodType(FsdpStrategy.all_reduce_dict, stub)
        # rank 0 carries image_tokens, rank 1 does not; shared keys arrive in different orders
        data = {"kl": (1.0, 10.0), "image_tokens": (100.0, 2.0)} if rank == 0 else {"loss": 5.0, "kl": (3.0, 30.0)}
        total = stub.all_reduce_dict(data)
        assert total == {"kl": (4.0, 40.0), "image_tokens": (100.0, 2.0), "loss": 5.0}, total
    finally:
        dist.destroy_process_group()


def test_all_reduce_dict_merges_different_key_sets_across_a_live_gloo_group():
    import torch.multiprocessing as mp

    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs >= 2 CPUs for a 2-rank gloo group")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(_gloo_worker, args=(2, port), nprocs=2, join=True)
