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

import sys
import types
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

if "ray" not in sys.modules:
    fake_ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(obj):
            return obj

        return decorator

    fake_ray.remote = remote
    fake_ray.get = MagicMock()
    fake_ray.wait = MagicMock()
    fake_ray.cancel = MagicMock()
    fake_util = types.ModuleType("ray.util")
    fake_placement_group = types.ModuleType("ray.util.placement_group")
    fake_placement_group.PlacementGroup = type("PlacementGroup", (), {})
    fake_placement_group.placement_group = MagicMock()
    fake_util.placement_group = fake_placement_group
    fake_scheduling = types.ModuleType("ray.util.scheduling_strategies")
    fake_scheduling.PlacementGroupSchedulingStrategy = type("PlacementGroupSchedulingStrategy", (), {})
    fake_util.scheduling_strategies = fake_scheduling
    fake_ray.util = fake_util
    sys.modules["ray"] = fake_ray
    sys.modules["ray.util"] = fake_util
    sys.modules["ray.util.placement_group"] = fake_placement_group
    sys.modules["ray.util.scheduling_strategies"] = fake_scheduling
if "vllm" not in sys.modules:
    fake_vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_vllm.SamplingParams = SamplingParams
    sys.modules["vllm"] = fake_vllm

from molt.agents.base import Trajectory
from molt.trainer.rollout import samples_generator
from molt.trainer.rollout.samples_generator import SamplesGenerator


def _sample(group_id):
    return SimpleNamespace(group_ids=[group_id])


def _prompt_loader(num_prompts):
    """A dataloader yielding one prompt per item, as (index, prompts, labels, images, tools)."""
    return [(i, [f"p{i}"], [f"l{i}"], [None], [None]) for i in range(num_prompts)]


def _wire_fake_vllm(generator, monkeypatch, to_sample):
    """Wire the streaming generator to an in-memory vLLM that finishes rollouts FIFO.

    Each dispatched prompt becomes one in-flight rollout handle tagged with its prompt
    string; ray.wait hands them back in dispatch order and ray.get turns a handle into
    the runner's result — a list of ``(light, drop_reason)`` (here one usable light that
    ``to_sample`` maps from the prompt tag).
    """
    generator._dispatch_to_agent_runners = lambda prompts, labels, images, **kw: [
        SimpleNamespace(group_id=prompt) for prompt in prompts
    ]
    monkeypatch.setattr(
        samples_generator.ray, "wait", lambda handles, num_returns=1: ([handles[0]], list(handles[1:]))
    )
    monkeypatch.setattr(samples_generator.ray, "get", lambda handle: [(to_sample(handle.group_id), None)])


def test_generate_samples_returns_batch_as_rollouts_finish_and_keeps_pool_saturated(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    samples, rollout_metrics, prompts_dispatched, exhausted = generator.generate_samples()

    # Returns exactly batch_size (3) finished groups, in completion (= dispatch) order.
    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1", "p2"]
    # Pool stays saturated: 5 dispatched up front, then one refill per completion →
    # 7 dispatched total, the 4 unclaimed rollouts stay in flight for the next call.
    assert prompts_dispatched == 7
    assert [handle.group_id for handle in generator._inflight_rollouts] == ["p3", "p4", "p5", "p6"]
    assert generator._finished_samples == []
    # No drops and no dynamic filtering → no rollout metrics emitted.
    assert rollout_metrics == {}
    assert exhausted is False


def test_generate_samples_emits_short_batch_when_dataloader_exhausted(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=4, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(2)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    samples, _, prompts_dispatched, exhausted = generator.generate_samples()

    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1"]
    assert prompts_dispatched == 2
    assert exhausted is True
    assert generator._inflight_rollouts == []


def test_generate_samples_pool_persists_across_calls(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    first, *_ = generator.generate_samples()
    second, _, prompts_dispatched, _ = generator.generate_samples()

    assert [sample.group_ids[0] for sample in first] == ["p0", "p1", "p2"]
    # The second batch is served from rollouts already in flight after the first call
    # (p3-p5) — vLLM never drained between steps — and the pool is topped back up.
    assert [sample.group_ids[0] for sample in second] == ["p3", "p4", "p5"]
    assert prompts_dispatched == 3  # only the 3 refills, not a fresh batch of 5
    assert [handle.group_id for handle in generator._inflight_rollouts] == ["p6", "p7", "p8", "p9"]


def test_generator_keeps_no_checkpoint_state_and_resumes_from_dataloader(monkeypatch):
    """The in-flight pool is intentionally NOT persisted.

    Persisting the in-flight (prompt, label, images) payloads bloated checkpoints
    ~1000x (22-78 MB vs ~7 KB) and crashed the driver on resume, so the generator
    is stateless across checkpoints: state_dict() is empty and load_state_dict is a
    no-op tolerant of None/{}. The StatefulDataLoader cursor already points past the
    in-flight prefetch, so on resume those few prompts are skipped (a bounded loss,
    negligible for multi-epoch RL) rather than redispatched.
    """
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    first, *_ = generator.generate_samples()
    assert [sample.group_ids[0] for sample in first] == ["p0", "p1", "p2"]
    # p3-p6 are in flight at the checkpoint; the generator carries no state for them.
    assert generator.state_dict() == {}

    restored = object.__new__(SamplesGenerator)
    restored.args = generator.args
    # The StatefulDataLoader cursor in the checkpoint already points past the
    # prefetched prompts (p0-p6 were read), so resume starts at p7.
    restored.prompts_dataloader = [(i, [f"p{i}"], [f"l{i}"], [None], [None]) for i in range(7, 10)]
    _wire_fake_vllm(restored, monkeypatch, _sample)
    restored.load_state_dict(None)  # tolerate a missing payload
    restored.load_state_dict({})  # and an empty one

    second, _, newly_dispatched, _ = restored.generate_samples()
    # Resume continues from the dataloader cursor; the in-flight p3-p6 are not retrained.
    assert [sample.group_ids[0] for sample in second] == ["p7", "p8", "p9"]
    assert newly_dispatched == 3


def test_generate_samples_drops_filtered_groups_and_refills_their_slots(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=2, n_samples_per_prompt=1, vllm_generate_batch_size=2),
        algo=SimpleNamespace(dynamic_filtering_enable=True, dynamic_filtering_range=(0.0, 1.0)),
    )
    generator.prompts_dataloader = _prompt_loader(10)

    # p1's mean score (1.0) sits on the boundary of the open range (0, 1) → filtered out.
    group_score = {"p0": 0.5, "p1": 1.0, "p2": 0.5, "p3": 0.5}

    def scored_sample(group_id):
        return SimpleNamespace(group_ids=[group_id], scores=[torch.tensor(group_score[group_id])])

    _wire_fake_vllm(generator, monkeypatch, scored_sample)

    samples, rollout_metrics, prompts_dispatched, _ = generator.generate_samples()

    # p1 is dropped; p2 (refilled into p1's freed slot) completes the batch.
    assert [sample.group_ids[0] for sample in samples] == ["p0", "p2"]
    assert prompts_dispatched == 4  # p0,p1 up front; p2,p3 refilled one per completion
    assert rollout_metrics["dynamic_filtering_pass_rate"] == 2 / 4 * 100
    # The filtered group is tallied by reason for observability.
    assert rollout_metrics["rollout/dropped/dynamic_filter"] == 1.0
    assert rollout_metrics["rollout/dropped/total"] == 1.0


def test_dynamic_filtering_counts_compaction_segments_once_per_rollout(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(n_samples_per_prompt=2),
        algo=SimpleNamespace(dynamic_filtering_range=(0.4, 0.6)),
    )
    samples = [
        SimpleNamespace(rollout_ids=[rollout_id], scores=torch.tensor([score]))
        for rollout_id, score in [("a", 1.0), ("a", 1.0), ("a", 1.0), ("b", 0.0)]
    ]
    # run_group now builds each response into a (light Experience, drop_reason) on the runner;
    # _filter_group just unpacks and applies the DAPO group filter.
    monkeypatch.setattr(samples_generator.ray, "get", lambda _: [(sample, None) for sample in samples])
    score_stats = defaultdict(float)

    kept = generator._filter_group(object(), True, defaultdict(int), score_stats=score_stats)

    assert kept == samples
    assert dict(score_stats) == {"score_sum": 1.0, "score_n": 2, "groups": 1.0, "all_pass": 0.0, "all_fail": 0.0}


def test_process_response_counts_only_action_tokens_for_multiturn_lengths():
    experience, drop_reason = SamplesGenerator._process_response_into_experience(
        Trajectory(
            prompt="p",
            label="l",
            images=None,
            observation_text="",
            observation_tokens=list(range(8)),
            action_ranges=[(2, 4), (6, 7)],
            rollout_log_probs=[float(i) for i in range(8)],
            reward=1.0,
            scores=1.0,
        ),
        media_ids=set(),
        truncate_length=8,
    )

    assert drop_reason is None
    assert experience.response_length.item() == 3
    assert experience.action_mask.sum().item() == 3
    torch.testing.assert_close(
        experience.action_mask,
        torch.tensor([[False, True, True, False, False, True, False]]),
    )
    torch.testing.assert_close(
        experience.rollout_log_probs,
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
    )


def test_process_response_rejects_action_ranges_outside_trajectory():
    with pytest.raises(ValueError, match="Invalid action range"):
        SamplesGenerator._process_response_into_experience(
            Trajectory(
                prompt="p",
                label="l",
                images=None,
                observation_text="",
                observation_tokens=[0, 1, 2],
                action_ranges=[(2, 4)],
                rollout_log_probs=[0.0, 0.0, 0.0],
                reward=1.0,
                scores=1.0,
            ),
            media_ids=set(),
            truncate_length=8,
        )


def test_process_response_skips_misaligned_rollout_logprobs():
    experience, drop_reason = SamplesGenerator._process_response_into_experience(
        Trajectory(
            prompt="p",
            label="l",
            images=None,
            observation_text="",
            observation_tokens=[0, 1, 2],
            action_ranges=[(1, 3)],
            rollout_log_probs=[0.0, 0.0],
            reward=1.0,
            scores=1.0,
        ),
        media_ids=set(),
        truncate_length=8,
    )

    assert experience is None
    assert drop_reason == "logprob_misalign"
