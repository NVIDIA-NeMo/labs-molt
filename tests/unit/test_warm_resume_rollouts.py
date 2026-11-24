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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from molt.trainer import rl_trainer
from molt.trainer.algorithm.experience import Experience
from molt.trainer.rollout.samples_generator import (
    SamplesGenerator,
    WarmRolloutWriter,
    load_warm_rollout,
)


def _prompt_loader(n: int) -> StatefulDataLoader:
    rows = [("ds", f"prompt-{i}", f"label-{i}", None, None) for i in range(n)]
    return StatefulDataLoader(rows, batch_size=1, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))


def _make_experience(idx: int) -> Experience:
    return Experience(sequences=torch.tensor([[idx]]), group_ids=[f"g{idx}"], rollout_ids=[f"r{idx}"])


def _make_generator_actor(
    tmp_path: Path,
    *,
    dataset_len: int,
    batch_size: int = 2,
    inflight_capacity: int = 3,
    num_episodes: int = 1,
    seed: int = 42,
    reward_ctx: str | float = 0.0,
    warm_enabled: bool = True,
    monkeypatch,
):
    dataset_path = tmp_path / "train.jsonl"
    if not dataset_path.exists():
        dataset_path.write_text('{"prompt": "p"}\n')
    ckpt_path = str(tmp_path / "checkpoints")

    args = SimpleNamespace(
        actor=SimpleNamespace(model_name_or_path="same-model", num_nodes=1, num_gpus_per_node=1),
        fsdp=SimpleNamespace(cp_size=1, tp_size=1),
        train=SimpleNamespace(
            rollout_replay_dir=None,
            rollout_dump_dir=None,
            partial_rollout_enable=True,
            num_episodes=num_episodes,
            seed=seed,
            agent_path="same-agent",
        ),
        data=SimpleNamespace(prompt_dataset=str(dataset_path)),
        rollout=SimpleNamespace(
            batch_size=batch_size,
            n_samples_per_prompt=1,
            num_runners=2,
            vllm_generate_batch_size=inflight_capacity,
        ),
        vllm=SimpleNamespace(num_engines=4, router_policy="consistent_hash"),
        reward=SimpleNamespace(policy_failure={"context": reward_ctx}),
        algo=SimpleNamespace(dynamic_filtering_enable=False, dynamic_filtering_range=(0, 1)),
        eval=SimpleNamespace(steps=float("inf")),
        ckpt=SimpleNamespace(warm_resume_rollouts=warm_enabled, path=ckpt_path),
    )

    loader = _prompt_loader(dataset_len)
    sg = object.__new__(SamplesGenerator)
    sg.args = args
    sg.prompts_dataloader = loader
    sg.eval_dataloader = None

    def dispatch(prompts, labels, images=None, tools=None, **_kwargs):
        refs = []
        for p in prompts:
            idx = int(p.split("-")[1])
            refs.append((idx, [_make_experience(idx)]))
        return refs

    sg._dispatch_to_agent_runners = dispatch
    sg._filter_group = lambda finished_rollout, *_a, **_kw: finished_rollout[1]

    monkeypatch.setattr(rl_trainer.ray, "get", lambda value: value)
    monkeypatch.setattr(rl_trainer.ray, "put", lambda value: value)
    monkeypatch.setattr(
        "molt.trainer.rollout.samples_generator.ray.wait",
        lambda refs, num_returns=1: (refs[:num_returns], refs[num_returns:]),
    )

    actor = object.__new__(rl_trainer.GenerateSamplesActor.__ray_metadata__.modified_class)
    actor.args = args
    actor.generate_kwargs = {"temperature": 1.0}
    actor.prompts_dataloader = loader
    actor.eval_dataloader = None
    actor._partial_rollout = True
    actor.samples_generator = sg
    actor.vllm_lock = SimpleNamespace(acquire=SimpleNamespace(remote=Mock()), release=SimpleNamespace(remote=Mock()))
    payloads = []
    actor.rollout_queue = SimpleNamespace(put=lambda item, *, block: payloads.append(item))
    actor.rollout_slots = SimpleNamespace(get=lambda *, block: 0)
    return actor, payloads


def test_full_batch_warm_resume_under_partial_rollout(tmp_path, monkeypatch):
    """Under partial_rollout, completed batches persist atomically and resume with zero regen."""
    actor1, payloads1 = _make_generator_actor(
        tmp_path, dataset_len=8, batch_size=2, inflight_capacity=3, monkeypatch=monkeypatch
    )
    actor1.fit(episode=0, total_consumed_prompts=0)

    # Batches produced: step 1 (0,1), step 2 (2,3), step 3 (4,5), step 4 (6,7), then "done".
    assert payloads1[-1] == "done"
    assert len(payloads1) == 5
    step1_cs = payloads1[0][1]
    step2_samples = payloads1[1][0]
    step2_cs = payloads1[1][1]
    assert [s.sequences.item() for s in step2_samples] == [2, 3]
    # With partial_rollout inflight_capacity=3: iter1 dispatches 0,1,2 & finishes 0; iter2 refills 3 & finishes 1 -> cursor=4.
    assert step1_cs["total_consumed_prompts"] == 4
    assert step2_cs["total_consumed_prompts"] == 6

    # Simulate resuming from checkpoint saved after step 1 (episode=0, cursor=4), including a
    # stale legacy buffer that must be cleared when the complete warm sidecar hits.
    actor2, payloads2 = _make_generator_actor(
        tmp_path, dataset_len=8, batch_size=2, inflight_capacity=3, monkeypatch=monkeypatch
    )
    actor2.prompts_dataloader.load_state_dict(step1_cs["data_loader_state_dict"])
    actor2.samples_generator._resumed_samples = [_make_experience(99)]
    gen_spy = Mock(wraps=actor2.samples_generator.generate_samples)
    actor2.samples_generator.generate_samples = gen_spy

    actor2.fit(episode=step1_cs["episode"], total_consumed_prompts=step1_cs["total_consumed_prompts"])

    # First post-resume payload must come straight from the warm sidecar (generation_time == 0.0)
    # with exact step-2 samples, then normal generation continues from cursor 6 without duplicates.
    first_samples, first_cs, _, first_gen_time, _ = payloads2[0]
    assert first_gen_time == 0.0
    assert [s.reload().sequences.item() for s in first_samples] == [2, 3]
    assert first_cs["total_consumed_prompts"] == 6

    # Remaining prompts 6, 7 generated normally (1 remaining batch, no stale sample 99).
    rest_ids = [s.sequences.item() for item in payloads2[1:-1] for s in item[0]]
    assert rest_ids == [6, 7]
    assert gen_spy.call_count == 1


def test_partial_mid_batch_warm_resume_after_walltime_kill(tmp_path, monkeypatch):
    """If walltime kills a job mid-batch (2/4 groups finished), resume restores those 2 and only generates 2 more."""
    actor1, _ = _make_generator_actor(
        tmp_path, dataset_len=10, batch_size=4, inflight_capacity=4, monkeypatch=monkeypatch
    )

    # Advance cursor to 2 to simulate checkpoint saved at step 1 (cursor=2).
    it = iter(actor1.prompts_dataloader)
    next(it)
    next(it)
    saved_dl = actor1.prompts_dataloader.state_dict()
    actor1.prompts_dataloader.load_state_dict(saved_dl)

    # Simulate mid-batch crash after 2 of 4 groups finish inside generate_samples.
    calls = {"n": 0}
    orig_append = WarmRolloutWriter.append_group

    def crash_after_two(self, *args, **kwargs):
        orig_append(self, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated walltime SIGKILL")

    monkeypatch.setattr(WarmRolloutWriter, "append_group", crash_after_two)
    with pytest.raises(RuntimeError, match="simulated walltime SIGKILL"):
        actor1.fit(episode=0, total_consumed_prompts=2)

    # Restore normal append_group and resume from step-1 checkpoint (cursor=2).
    monkeypatch.setattr(WarmRolloutWriter, "append_group", orig_append)
    actor2, payloads2 = _make_generator_actor(
        tmp_path, dataset_len=10, batch_size=4, inflight_capacity=4, monkeypatch=monkeypatch
    )
    actor2.prompts_dataloader.load_state_dict(saved_dl)
    actor2.fit(episode=0, total_consumed_prompts=2)

    # Batch 2 should contain the 2 restored groups (2, 3) + 2 freshly generated groups starting past dispatched cursor (7, 8).
    first_batch_ids = [s.sequences.item() for s in payloads2[0][0]]
    assert len(first_batch_ids) == 4
    assert first_batch_ids[:2] == [2, 3]
    assert first_batch_ids[2:] == [7, 8]
    # And the manifest is now finalized as complete=True on disk.
    hit = load_warm_rollout(
        actor2.args.ckpt.path,
        episode=0,
        consumed_prompts=2,
        recipe=actor2._build_rollout_recipe(),
    )
    assert hit is not None and hit["complete"] is True
    assert [s.sequences.item() for s in hit["samples"]] == [2, 3, 7, 8]


def test_complete_warm_resume_at_episode_boundary_starts_next_episode_from_zero(tmp_path, monkeypatch):
    """A complete warm hit that drains episode 0 must reset the dataloader before episode 1 starts."""
    actor1, payloads1 = _make_generator_actor(
        tmp_path, dataset_len=4, batch_size=2, inflight_capacity=2, num_episodes=1, monkeypatch=monkeypatch
    )
    actor1.fit(episode=0, total_consumed_prompts=0)
    step1_cs = payloads1[0][1]
    # Under partial_rollout with inflight_capacity=2: iter1 dispatches 0,1 & finishes 0; iter2 refills 2 & finishes 1.
    assert step1_cs["episode"] == 0 and step1_cs["total_consumed_prompts"] == 3
    assert payloads1[1][1]["episode"] == 1 and payloads1[1][1]["data_loader_state_dict"] == {}

    # Resume a 2-episode run from step 1 (episode=0, cursor=3): step 2 comes from the warm sidecar
    # and drains episode 0, then episode 1 must start fresh at prompt 0 (not skip 0..2).
    actor2, payloads2 = _make_generator_actor(
        tmp_path, dataset_len=4, batch_size=2, inflight_capacity=2, num_episodes=2, monkeypatch=monkeypatch
    )
    actor2.prompts_dataloader.load_state_dict(step1_cs["data_loader_state_dict"])
    actor2.fit(episode=0, total_consumed_prompts=step1_cs["total_consumed_prompts"])

    assert payloads2[-1] == "done"
    batches = [[s.reload().sequences.item() for s in item[0]] for item in payloads2[:-1]]
    assert batches == [[2, 3], [0, 1], [2, 3]]


def test_warm_resume_rejects_mismatched_recipe_and_corrupt_files(tmp_path, monkeypatch):
    """Changing reward/seed rejects stale warm sidecars; missing group files degrade safely."""
    actor1, payloads1 = _make_generator_actor(
        tmp_path, dataset_len=6, batch_size=2, inflight_capacity=2, reward_ctx="drop", monkeypatch=monkeypatch
    )
    actor1.fit(episode=0, total_consumed_prompts=0)
    step1_cs = payloads1[0][1]

    # Recipe change (context="drop" -> 0.0) must NOT reuse stale warm rollouts.
    actor2, _ = _make_generator_actor(
        tmp_path, dataset_len=6, batch_size=2, inflight_capacity=2, reward_ctx=0.0, monkeypatch=monkeypatch
    )
    actor2.prompts_dataloader.load_state_dict(step1_cs["data_loader_state_dict"])
    gen_spy = Mock(wraps=actor2.samples_generator.generate_samples)
    actor2.samples_generator.generate_samples = gen_spy
    actor2.fit(episode=step1_cs["episode"], total_consumed_prompts=step1_cs["total_consumed_prompts"])
    assert gen_spy.call_count == 2  # regenerated both remaining batches instead of reusing stale drop-recipe batch

    # Missing group file degrades gracefully to None without crashing.
    warm_dir = tmp_path / "rollout_warm" / f"step_ep0_cursor{step1_cs['total_consumed_prompts']}"
    (warm_dir / "group_0001.pt").unlink()
    assert (
        load_warm_rollout(
            actor2.args.ckpt.path,
            episode=0,
            consumed_prompts=step1_cs["total_consumed_prompts"],
            recipe=actor2._build_rollout_recipe(),
        )
        is None
    )


def test_warm_writer_prunes_old_batches_and_preserves_lazy_experience(tmp_path, monkeypatch):
    """Disk stays bounded to keep_batches=3 and materializing copies does not mutate lazy Experiences."""
    ckpt_path = str(tmp_path / "checkpoints")
    recipe = {"seed": 1}

    payload = {"sequences": torch.tensor([[7, 8]])}
    monkeypatch.setattr("molt.trainer.algorithm.experience.ray.get", lambda ref: ref)
    lazy = Experience(sequences=None, group_ids=["g7"], rollout_ids=["r7"], heavy_ref=payload)

    for cursor in (10, 20, 30, 40, 50):
        w = WarmRolloutWriter(
            ckpt_path, source_episode=0, source_consumed_prompts=cursor, recipe=recipe, keep_batches=3
        )
        w.append_group([lazy], prompts_dispatched=10, data_loader_state_dict={"c": cursor + 10}, drained=False)
        w.finalize(
            client_states={"episode": 0, "total_consumed_prompts": cursor + 10, "data_loader_state_dict": {}},
            metrics={},
        )

    # Original lazy Experience remained untouched.
    assert lazy.heavy_ref is payload and lazy.sequences is None

    warm_dirs = sorted(p.name for p in (tmp_path / "rollout_warm").iterdir() if p.is_dir())
    assert len(warm_dirs) == 3
    assert warm_dirs == ["step_ep0_cursor30", "step_ep0_cursor40", "step_ep0_cursor50"]

    hit = load_warm_rollout(ckpt_path, episode=0, consumed_prompts=50, recipe=recipe)
    assert hit is not None and hit["complete"] is True
    assert hit["samples"][0].heavy_ref is None
    assert torch.equal(hit["samples"][0].sequences, torch.tensor([[7, 8]]))


def test_legacy_state_dict_pruning_ignores_warm_subdirectories(tmp_path):
    """Pruning legacy rollout_buffer.pt.* files must skip subdirectories in rollout_warm/."""
    warm_root = tmp_path / "rollout_warm"
    subdir = warm_root / "step_ep0_cursor10"
    subdir.mkdir(parents=True)

    gen = object.__new__(SamplesGenerator)
    gen.args = SimpleNamespace(ckpt=SimpleNamespace(warm_resume_rollouts=True, path=str(tmp_path / "checkpoints")))
    gen._finished_samples = [_make_experience(0)]

    for _ in range(4):
        sd = gen.state_dict()
        assert sd.get("buffer_file") and Path(sd["buffer_file"]).is_file()

    assert subdir.is_dir()
    buffers = [p for p in warm_root.iterdir() if p.is_file() and p.name.startswith(SamplesGenerator._BUFFER)]
    assert len(buffers) == 3
