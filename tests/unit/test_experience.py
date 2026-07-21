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

from types import SimpleNamespace

import torch

import molt.trainer.algorithm.experience as experience_mod
from molt.trainer.algorithm.experience import (
    Experience,
    balance_experiences,
    get_model_parallel_size,
    make_experience_batch,
)
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker


def _args(cp=1, tp=1, ep=1, actor_gpus=1):
    return SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=actor_gpus),
        fsdp=SimpleNamespace(cp_size=cp, tp_size=tp, ep_size=ep),
    )


def test_model_parallel_size_excludes_ep():
    assert get_model_parallel_size(_args(cp=2, tp=3, ep=4)) == 6


def _len_sample(length, idx):
    return Experience(total_length=torch.tensor([length]), rollout_ids=[f"r{idx}"])


def test_balance_experiences_excludes_ep_and_returns_flat_samples():
    # DP size = actor_gpus // (cp*tp) = 4 (EP is excluded from the divisor). 6 per-sample
    # experiences -> keep (6 // 4) * 4 = 4 (drop the 2-sample remainder). If EP were wrongly in
    # the divisor, dp would be 2 and all 6 would be kept.
    samples = [_len_sample(6 - i, i) for i in range(6)]
    balanced = balance_experiences(samples, _args(ep=2, actor_gpus=4))

    assert len(balanced) == 4  # flat, equal count per rank (1×4), remainder dropped — not concatenated
    ids = {id(s) for s in samples}
    assert all(id(b) in ids for b in balanced)  # returns the input samples themselves (no concat)


def test_balance_experiences_equal_count_and_length_sorted():
    # 10 samples across 4 DP ranks: the 2-sample remainder is dropped so every rank receives the
    # SAME count (2) — unequal counts desync num_steps and deadlock the world all_reduce. The flat
    # result is contiguous per-rank blocks; within each block samples are sorted by length descending
    # (k-th microbatch size-matched across ranks so no straggler trips the NCCL watchdog).
    samples = [_len_sample(10 - i, i) for i in range(10)]
    balanced = balance_experiences(samples, _args(actor_gpus=4))

    assert len(balanced) == 8  # 4 ranks × 2 samples; trailing 2-sample remainder dropped
    assert len({id(b) for b in balanced}) == 8  # no duplicates
    for rank in range(4):
        block = balanced[rank * 2 : rank * 2 + 2]
        block_lengths = [int(b.total_length.item()) for b in block]
        assert block_lengths == sorted(block_lengths, reverse=True)


def _grpo_maker():
    """A duck-typed RemoteExperienceMaker self with just the fields compute_advantages reads."""
    maker = SimpleNamespace(
        advantage_estimator="grpo",
        kl_ctl=SimpleNamespace(value=0.0),
        args=SimpleNamespace(
            reward=SimpleNamespace(clip_range=None),
            algo=SimpleNamespace(advantage=SimpleNamespace(gamma=1.0, lam=1.0, no_whiten=False)),
            rollout=SimpleNamespace(n_samples_per_prompt=4),
        ),
    )
    # Bind the two instance methods the estimator path dispatches through (each reads only its args).
    maker._merge_rollout_rewards = RemoteExperienceMaker._merge_rollout_rewards.__get__(maker)
    maker.compute_advantages_and_returns = RemoteExperienceMaker.compute_advantages_and_returns.__get__(maker)
    return maker


def _sample(idx, group_id, reward, length=6):
    # Equal length so the concatenated path stacks without padding — the two paths are
    # then directly comparable row-by-row.
    return Experience(
        action_mask=torch.ones(1, length, dtype=torch.bool),
        kl=torch.zeros(1, length),
        rewards=torch.tensor([float(reward)]),
        index=[idx],
        group_ids=[group_id],
        rollout_ids=[f"r{idx}"],
        info={},
    )


def test_distributed_advantages_match_materialized():
    # The distributed path runs compute_advantages_and_returns on per-sample "light"
    # Experiences; the materialized path runs it on the concatenated batch. Advantages are
    # a function of (reward, group, mask) only, so the per-sample results must be identical
    # — this is what lets the trainer skip gathering the heavy batch on the controller.
    rewards = [1.0, 0.0, 0.5, 0.25, 0.9, 0.1, 0.4, 0.6]
    groups = ["g0"] * 4 + ["g1"] * 4

    per_sample = [_sample(i, groups[i], rewards[i]) for i in range(8)]
    RemoteExperienceMaker.compute_advantages_and_returns(_grpo_maker(), per_sample)

    concat = make_experience_batch([_sample(i, groups[i], rewards[i]) for i in range(8)])
    RemoteExperienceMaker.compute_advantages_and_returns(_grpo_maker(), [concat])

    for i in range(8):
        assert torch.allclose(per_sample[i].advantages[0], concat.advantages[i])
        assert torch.allclose(per_sample[i].returns[0], concat.returns[i])


def test_experience_offload_reload_roundtrip(monkeypatch):
    # offload() moves the heavy fields into the object store (leaving a ref) and keeps the
    # lightweight ones in place; reload() restores them exactly. The controller only ever reads
    # the light fields, so a batch of these handles never fetches an image.
    store = {}

    def fake_put(obj):
        store[len(store)] = obj
        return len(store) - 1

    monkeypatch.setattr(experience_mod.ray, "put", fake_put)
    monkeypatch.setattr(experience_mod.ray, "get", lambda key: store[key])

    exp = Experience(
        sequences=torch.arange(6).view(1, 6),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        action_mask=torch.ones(1, 5, dtype=torch.bool),
        rewards=torch.tensor([1.0]),
        mm_train_inputs=[{"pixel_values": torch.zeros(2, 3)}],
    )
    exp.offload()
    assert exp.heavy_ref is not None
    assert exp.sequences is None and exp.attention_mask is None and exp.mm_train_inputs is None
    assert exp.action_mask is not None and exp.rewards is not None  # light fields untouched

    # offload() is idempotent: a second call must not re-put the now-nulled fields and clobber the
    # ref (the reload below would then restore Nones and lose the sample's tensors).
    first_ref = exp.heavy_ref
    exp.offload()
    assert exp.heavy_ref == first_ref

    exp.reload()
    assert exp.heavy_ref is None
    assert torch.equal(exp.sequences, torch.arange(6).view(1, 6))
    assert torch.equal(exp.mm_train_inputs[0]["pixel_values"], torch.zeros(2, 3))
    exp.reload()  # idempotent — second call is a no-op
