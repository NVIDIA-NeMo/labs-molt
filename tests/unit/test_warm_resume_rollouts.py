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
    _warm_root,
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
    partial_rollout: bool = True,
    force_sync_mode: bool = False,
    routing_replay: bool = False,
    is_correction_level: str = "off",
    model_name: str = "same-model",
    actor_gpus: int = 1,
    ckpt_name: str = "checkpoints",
    dataset_path: Path | None = None,
    wait_fn=None,
    slot_step_fn=None,
    monkeypatch,
):
    if dataset_path is None:
        dataset_path = tmp_path / "train.jsonl"
        if not dataset_path.exists():
            dataset_path.write_text('{"prompt": "p"}\n')
    ckpt_path = str(tmp_path / ckpt_name)

    args = SimpleNamespace(
        actor=SimpleNamespace(model_name_or_path=model_name, num_nodes=1, num_gpus_per_node=actor_gpus),
        fsdp=SimpleNamespace(cp_size=1, tp_size=1),
        train=SimpleNamespace(
            rollout_replay_dir=None,
            rollout_dump_dir=None,
            partial_rollout_enable=partial_rollout,
            force_sync_mode=force_sync_mode,
            routing_replay=routing_replay,
            full_determinism_enable=False,
            async_queue_size=1,
            num_episodes=num_episodes,
            seed=seed,
            agent_path="same-agent",
        ),
        data=SimpleNamespace(prompt_dataset=str(dataset_path), dataloader_num_workers=0),
        rollout=SimpleNamespace(
            batch_size=batch_size,
            n_samples_per_prompt=1,
            num_runners=2,
            vllm_generate_batch_size=inflight_capacity,
        ),
        vllm=SimpleNamespace(num_engines=4, router_policy="consistent_hash"),
        reward=SimpleNamespace(policy_failure={"context": reward_ctx}),
        algo=SimpleNamespace(
            dynamic_filtering_enable=False,
            dynamic_filtering_range=(0, 1),
            advantage=SimpleNamespace(is_correction_level=is_correction_level, score_centering_top_k=0),
            kl=SimpleNamespace(init_coef=0.0, ref="model"),
        ),
        eval=SimpleNamespace(steps=float("inf")),
        ckpt=SimpleNamespace(warm_resume_rollouts=warm_enabled, path=ckpt_path),
    )

    loader = _prompt_loader(dataset_len)
    sg = object.__new__(SamplesGenerator)
    sg.args = args
    sg.prompts_dataloader = loader
    sg.eval_dataloader = None

    dispatched_prompts: list[int] = []

    def dispatch(prompts, labels, images=None, tools=None, **_kwargs):
        refs = []
        for p in prompts:
            idx = int(p.split("-")[1])
            dispatched_prompts.append(idx)
            refs.append((idx, [_make_experience(idx)]))
        return refs

    sg._dispatch_to_agent_runners = dispatch
    sg._filter_group = lambda finished_rollout, *_a, **_kw: finished_rollout[1]

    monkeypatch.setattr(rl_trainer.ray, "get", lambda value: value)
    monkeypatch.setattr(rl_trainer.ray, "put", lambda value: value)
    monkeypatch.setattr(
        "molt.trainer.rollout.samples_generator.ray.wait",
        wait_fn or (lambda refs, num_returns=1: (refs[:num_returns], refs[num_returns:])),
    )

    actor = object.__new__(rl_trainer.GenerateSamplesActor.__ray_metadata__.modified_class)
    actor.args = args
    actor.generate_kwargs = {"temperature": 1.0}
    actor.prompts_dataloader = loader
    actor.eval_dataloader = None
    actor._partial_rollout = partial_rollout
    actor.samples_generator = sg
    actor.vllm_lock = SimpleNamespace(acquire=SimpleNamespace(remote=Mock()), release=SimpleNamespace(remote=Mock()))
    payloads = []
    actor.rollout_queue = SimpleNamespace(put=lambda item, *, block: payloads.append(item))

    slot_counter = {"n": 0}

    def _default_slot_step(*, block):
        idx = slot_counter["n"]
        slot_counter["n"] += 1
        if slot_step_fn is not None:
            return slot_step_fn(idx)
        return idx if force_sync_mode else max(0, idx - 1)

    actor.rollout_slots = SimpleNamespace(get=_default_slot_step, put=lambda *_a, **_kw: None)
    actor._dispatched_prompts = dispatched_prompts
    return actor, payloads


def test_full_batch_warm_resume_under_partial_rollout(tmp_path, monkeypatch):
    """Under partial_rollout, completed batches persist atomically inside ckpt_path and resume with zero regen."""
    actor1, payloads1 = _make_generator_actor(
        tmp_path, dataset_len=8, batch_size=2, inflight_capacity=3, monkeypatch=monkeypatch
    )
    actor1.fit(episode=0, total_consumed_prompts=0, global_step=0, warm_run_id="run-1")

    # Batches produced: step 1 (0,1), step 2 (2,3), step 3 (4,5), step 4 (6,7), then "done".
    assert payloads1[-1] == "done"
    assert len(payloads1) == 5
    step1_cs = payloads1[0][1]
    step2_samples = payloads1[1][0]
    step2_cs = payloads1[1][1]
    assert step1_cs["warm_run_id"] == "run-1"
    assert [s.sequences.item() for s in step2_samples] == [2, 3]
    # With partial_rollout inflight_capacity=3: iter1 dispatches 0,1,2 & finishes 0; iter2 refills 3 & finishes 1 -> cursor=4.
    assert step1_cs["total_consumed_prompts"] == 4
    assert step2_cs["total_consumed_prompts"] == 6

    # Simulate resuming from checkpoint saved after step 1 (global_step=1, episode=0, cursor=4),
    # including a stale legacy buffer that must be cleared when the complete warm sidecar hits.
    actor2, payloads2 = _make_generator_actor(
        tmp_path, dataset_len=8, batch_size=2, inflight_capacity=3, monkeypatch=monkeypatch
    )
    actor2.prompts_dataloader.load_state_dict(step1_cs["data_loader_state_dict"])
    actor2.samples_generator._resumed_samples = [_make_experience(99)]
    gen_spy = Mock(wraps=actor2.samples_generator.generate_samples)
    actor2.samples_generator.generate_samples = gen_spy

    actor2.fit(
        episode=step1_cs["episode"],
        total_consumed_prompts=step1_cs["total_consumed_prompts"],
        global_step=1,
        warm_run_id="run-1",
    )

    # First post-resume payload comes straight from the warm sidecar (generation_time == 0.0)
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
    """Mid-batch kill (2/4 finished) restores those 2 and regenerates the 2 unfinished prompts without skipping."""
    actor1, _ = _make_generator_actor(
        tmp_path, dataset_len=10, batch_size=4, inflight_capacity=4, monkeypatch=monkeypatch
    )

    # Advance cursor to 2 to simulate checkpoint saved at step 1 (cursor=2).
    it = iter(actor1.prompts_dataloader)
    next(it)
    next(it)
    saved_dl = actor1.prompts_dataloader.state_dict()
    actor1.prompts_dataloader.load_state_dict(saved_dl)

    # Simulate mid-batch crash after 2 of 4 groups (prompts 2, 3) finish inside generate_samples.
    calls = {"n": 0}
    orig_append = WarmRolloutWriter.append_group

    def crash_after_two(self, *args, **kwargs):
        orig_append(self, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated walltime SIGKILL")

    monkeypatch.setattr(WarmRolloutWriter, "append_group", crash_after_two)
    with pytest.raises(RuntimeError, match="simulated walltime SIGKILL"):
        actor1.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-1")

    # Restore normal append_group and resume from step-1 checkpoint (cursor=2).
    monkeypatch.setattr(WarmRolloutWriter, "append_group", orig_append)
    actor2, payloads2 = _make_generator_actor(
        tmp_path, dataset_len=10, batch_size=4, inflight_capacity=4, monkeypatch=monkeypatch
    )
    actor2.prompts_dataloader.load_state_dict(saved_dl)
    actor2.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-1")

    # Batch 2 must contain restored (2, 3) + regenerated unfinished (4, 5) — NEVER skipping 4, 5, 6!
    first_batch_ids = [s.sequences.item() for s in payloads2[0][0]]
    assert first_batch_ids == [2, 3, 4, 5]
    # Prompts 2 and 3 were skipped (restored from sidecar); 4..9 dispatched once each.
    assert actor2._dispatched_prompts == [4, 5, 6, 7, 8, 9]
    second_batch_ids = [s.sequences.item() for s in payloads2[1][0]]
    assert second_batch_ids == [6, 7, 8, 9]

    # And the manifest is now finalized as complete=True on disk.
    hit = load_warm_rollout(
        actor2.args.ckpt.path,
        episode=0,
        consumed_prompts=2,
        global_step=1,
        run_id="run-1",
        max_policy_lag=1,
        recipe=actor2._build_rollout_recipe(),
    )
    assert hit is not None and hit["complete"] is True
    assert [s.sequences.item() for s in hit["samples"]] == [2, 3, 4, 5]


def test_partial_resume_out_of_order_completion_does_not_skip_or_duplicate(tmp_path, monkeypatch):
    """Out-of-order completion (prompts 5 & 3 finish before 2 & 4) resumes only 2 & 4 with zero skips/dups."""

    def finish_5_then_3(refs, num_returns=1):
        # First pop prompt 5 (last), then prompt 3, regardless of arrival order.
        for target in (5, 3):
            for i, ref in enumerate(refs):
                if ref[0] == target:
                    return [ref], refs[:i] + refs[i + 1 :]
        return refs[:num_returns], refs[num_returns:]

    actor1, _ = _make_generator_actor(
        tmp_path,
        dataset_len=10,
        batch_size=4,
        inflight_capacity=4,
        partial_rollout=False,
        wait_fn=finish_5_then_3,
        monkeypatch=monkeypatch,
    )
    it = iter(actor1.prompts_dataloader)
    next(it)
    next(it)
    saved_dl = actor1.prompts_dataloader.state_dict()
    actor1.prompts_dataloader.load_state_dict(saved_dl)

    calls = {"n": 0}
    orig_append = WarmRolloutWriter.append_group

    def crash_after_two(self, *args, **kwargs):
        orig_append(self, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("walltime SIGKILL")

    monkeypatch.setattr(WarmRolloutWriter, "append_group", crash_after_two)
    with pytest.raises(RuntimeError, match="walltime SIGKILL"):
        actor1.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-ooo")

    # Resume with normal FIFO completion: must restore [5, 3], regenerate ONLY [2, 4],
    # advance cursor past already-completed trailing offset (prompt 5) to 6, then emit [6, 7, 8, 9].
    monkeypatch.setattr(WarmRolloutWriter, "append_group", orig_append)
    actor2, payloads2 = _make_generator_actor(
        tmp_path,
        dataset_len=10,
        batch_size=4,
        inflight_capacity=4,
        partial_rollout=False,
        monkeypatch=monkeypatch,
    )
    actor2.prompts_dataloader.load_state_dict(saved_dl)
    actor2.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-ooo")

    batches = [[s.sequences.item() for s in item[0]] for item in payloads2[:-1]]
    assert batches == [[5, 3, 2, 4], [6, 7, 8, 9]]
    # Prompts 3 and 5 were NEVER re-dispatched; prompts 2, 4, 6, 7, 8, 9 each dispatched once.
    assert actor2._dispatched_prompts == [2, 4, 6, 7, 8, 9]
    assert payloads2[0][1]["total_consumed_prompts"] == 6


def test_partial_resume_survives_two_consecutive_restarts(tmp_path, monkeypatch):
    """Kill #1 after [5, 3] finish, Kill #2 after [4] finishes, Run #3 only generates [2] and completes."""

    def finish_highest_first(refs, num_returns=1):
        best_i = max(range(len(refs)), key=lambda i: refs[i][0])
        return [refs[best_i]], refs[:best_i] + refs[best_i + 1 :]

    actor1, _ = _make_generator_actor(
        tmp_path,
        dataset_len=10,
        batch_size=4,
        inflight_capacity=4,
        partial_rollout=False,
        wait_fn=finish_highest_first,
        monkeypatch=monkeypatch,
    )
    it = iter(actor1.prompts_dataloader)
    next(it)
    next(it)
    saved_dl = actor1.prompts_dataloader.state_dict()
    actor1.prompts_dataloader.load_state_dict(saved_dl)

    orig_append = WarmRolloutWriter.append_group

    # Run 1: prompts 2, 3, 4, 5 dispatched; highest-first completes 5 (offset 3), then 4 (offset 2) -> KILL #1.
    c1 = {"n": 0}

    def kill_1(self, *args, **kwargs):
        orig_append(self, *args, **kwargs)
        c1["n"] += 1
        if c1["n"] == 2:
            raise RuntimeError("kill #1")

    monkeypatch.setattr(WarmRolloutWriter, "append_group", kill_1)
    with pytest.raises(RuntimeError, match="kill #1"):
        actor1.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-2x")

    # Run 2: restores [5, 4] (offsets [2, 3]); dispatches [2, 3]; highest-first completes 3 (offset 1) -> KILL #2.
    actor2, _ = _make_generator_actor(
        tmp_path,
        dataset_len=10,
        batch_size=4,
        inflight_capacity=4,
        partial_rollout=False,
        wait_fn=finish_highest_first,
        monkeypatch=monkeypatch,
    )
    actor2.prompts_dataloader.load_state_dict(saved_dl)
    c2 = {"n": 0}

    def kill_2(self, *args, **kwargs):
        orig_append(self, *args, **kwargs)
        c2["n"] += 1
        if c2["n"] == 1:
            raise RuntimeError("kill #2")

    monkeypatch.setattr(WarmRolloutWriter, "append_group", kill_2)
    with pytest.raises(RuntimeError, match="kill #2"):
        actor2.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-2x")
    assert actor2._dispatched_prompts == [2, 3]

    # Run 3: restores [5, 4, 3] (offsets [1, 2, 3]); dispatches ONLY [2], advances cursor to 6, then [6, 7, 8, 9].
    monkeypatch.setattr(WarmRolloutWriter, "append_group", orig_append)
    actor3, payloads3 = _make_generator_actor(
        tmp_path,
        dataset_len=10,
        batch_size=4,
        inflight_capacity=4,
        partial_rollout=False,
        monkeypatch=monkeypatch,
    )
    actor3.prompts_dataloader.load_state_dict(saved_dl)
    actor3.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-2x")

    batches = [[s.sequences.item() for s in item[0]] for item in payloads3[:-1]]
    assert batches == [[5, 4, 3, 2], [6, 7, 8, 9]]
    assert actor3._dispatched_prompts == [2, 6, 7, 8, 9]
    assert payloads3[0][1]["total_consumed_prompts"] == 6


def test_complete_warm_resume_at_episode_boundary_starts_next_episode_from_zero(tmp_path, monkeypatch):
    """A complete warm hit that drains episode 0 must reset the dataloader before episode 1 starts."""
    actor1, payloads1 = _make_generator_actor(
        tmp_path, dataset_len=4, batch_size=2, inflight_capacity=2, num_episodes=1, monkeypatch=monkeypatch
    )
    actor1.fit(episode=0, total_consumed_prompts=0, global_step=0, warm_run_id="run-ep")
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
    actor2.fit(
        episode=0,
        total_consumed_prompts=step1_cs["total_consumed_prompts"],
        global_step=1,
        warm_run_id="run-ep",
    )

    assert payloads2[-1] == "done"
    batches = [[s.reload().sequences.item() for s in item[0]] for item in payloads2[:-1]]
    assert batches == [[2, 3], [0, 1], [2, 3]]


def test_epoch_tail_and_same_cursor_snapshots_never_clobber_or_evict(tmp_path, monkeypatch):
    """Post-drain inflight flushes, empty epoch probes, and dropped <dp tails must not clobber or evict."""
    # 1) partial_rollout drains all 6 prompts in batch 1, then flushes 2 more batches at cursor=6:
    # post-drain flush batches must not create colliding cursor-6 snapshots or clobber batch 1.
    actor_drain, payloads_drain = _make_generator_actor(
        tmp_path,
        dataset_len=6,
        batch_size=2,
        inflight_capacity=6,
        partial_rollout=True,
        ckpt_name="ckpt_drain",
        monkeypatch=monkeypatch,
    )
    actor_drain.fit(episode=0, total_consumed_prompts=0, global_step=0, warm_run_id="run-drain")
    assert len(payloads_drain) == 4  # 3 batches + "done"
    drain_dirs = sorted(p.name for p in _warm_root(actor_drain.args.ckpt.path).iterdir() if p.is_dir())
    assert drain_dirs == ["step_ep0_cursor0_step0"]

    # 2) Empty end-of-epoch probe (dataset_len=4, batch_size=2, partial_rollout=False, keep_batches=2):
    # the 3rd generate_samples call returns [] and must neither create a dir nor evict step 0 or step 1.
    orig_init = WarmRolloutWriter.__init__

    def keep_two(self, *args, **kwargs):
        kwargs["keep_batches"] = 2
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(WarmRolloutWriter, "__init__", keep_two)
    actor_probe, payloads_probe = _make_generator_actor(
        tmp_path,
        dataset_len=4,
        batch_size=2,
        inflight_capacity=2,
        partial_rollout=False,
        ckpt_name="ckpt_probe",
        monkeypatch=monkeypatch,
    )
    actor_probe.fit(episode=0, total_consumed_prompts=0, global_step=0, warm_run_id="run-probe")
    assert len(payloads_probe) == 3  # 2 batches + "done"
    probe_dirs = sorted(p.name for p in _warm_root(actor_probe.args.ckpt.path).iterdir() if p.is_dir())
    assert probe_dirs == ["step_ep0_cursor0_step0", "step_ep0_cursor2_step1"]
    monkeypatch.setattr(WarmRolloutWriter, "__init__", orig_init)

    # 3) Dropped < dp epoch tail discards its unfinalized dir cleanly without leaving garbage.
    actor_drop, payloads_drop = _make_generator_actor(
        tmp_path,
        dataset_len=5,
        batch_size=4,
        inflight_capacity=4,
        partial_rollout=False,
        actor_gpus=2,
        ckpt_name="ckpt_drop",
        monkeypatch=monkeypatch,
    )
    actor_drop.fit(episode=0, total_consumed_prompts=0, global_step=0, warm_run_id="run-drop")
    assert len(payloads_drop) == 2  # batch [0..3] + "done" (1-sample tail dropped since < dp=2)
    drop_dirs = sorted(p.name for p in _warm_root(actor_drop.args.ckpt.path).iterdir() if p.is_dir())
    assert drop_dirs == ["step_ep0_cursor0_step0"]

    # 4) Epoch-tail partial kill (3 prompts remain: 2, 3, 4; prompt 4 finishes before kill):
    # resume regenerates only 2 and 3 and emits the complete 3-sample tail with exhausted=True.
    def finish_4_first(refs, num_returns=1):
        for i, ref in enumerate(refs):
            if ref[0] == 4:
                return [ref], refs[:i] + refs[i + 1 :]
        return refs[:num_returns], refs[num_returns:]

    actor_tail1, _ = _make_generator_actor(
        tmp_path,
        dataset_len=5,
        batch_size=4,
        inflight_capacity=4,
        partial_rollout=False,
        ckpt_name="ckpt_tail",
        wait_fn=finish_4_first,
        monkeypatch=monkeypatch,
    )
    it = iter(actor_tail1.prompts_dataloader)
    next(it)
    next(it)
    saved_dl = actor_tail1.prompts_dataloader.state_dict()
    actor_tail1.prompts_dataloader.load_state_dict(saved_dl)

    orig_append = WarmRolloutWriter.append_group

    def kill_after_one(self, *args, **kwargs):
        orig_append(self, *args, **kwargs)
        raise RuntimeError("tail kill")

    monkeypatch.setattr(WarmRolloutWriter, "append_group", kill_after_one)
    with pytest.raises(RuntimeError, match="tail kill"):
        actor_tail1.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-tail")

    monkeypatch.setattr(WarmRolloutWriter, "append_group", orig_append)
    actor_tail2, payloads_tail2 = _make_generator_actor(
        tmp_path,
        dataset_len=5,
        batch_size=4,
        inflight_capacity=4,
        partial_rollout=False,
        ckpt_name="ckpt_tail",
        monkeypatch=monkeypatch,
    )
    actor_tail2.prompts_dataloader.load_state_dict(saved_dl)
    actor_tail2.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="run-tail")
    assert [s.sequences.item() for s in payloads_tail2[0][0]] == [4, 2, 3]
    assert payloads_tail2[0][1]["episode"] == 1
    assert actor_tail2._dispatched_prompts == [2, 3]


def test_legacy_buffer_groups_are_persisted_and_survive_double_interrupt(tmp_path, monkeypatch):
    """Pre-seeded legacy buffer groups must be written to WarmRolloutWriter and survive mid-batch kill."""
    # Create a legacy buffer file (even in an old parent-dir location to verify backwards-compat load).
    legacy_buf_path = tmp_path / "old_rollout_buffer.pt"
    torch.save([_make_experience(90)], legacy_buf_path)
    legacy_rg_state = {"buffer_file": str(legacy_buf_path)}

    actor1, _ = _make_generator_actor(
        tmp_path,
        dataset_len=8,
        batch_size=3,
        inflight_capacity=3,
        partial_rollout=False,
        ckpt_name="ckpt_legacy",
        monkeypatch=monkeypatch,
    )
    it = iter(actor1.prompts_dataloader)
    next(it)
    next(it)
    saved_dl = actor1.prompts_dataloader.state_dict()
    actor1.load_dataloader_state_dict(saved_dl, legacy_rg_state)

    # Kill after pre-seeded [90] AND newly-finished [2] are persisted (2 append_group calls), before [3] finishes.
    orig_append = WarmRolloutWriter.append_group
    calls = {"n": 0}

    def kill_after_legacy_plus_one(self, *args, **kwargs):
        orig_append(self, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("legacy mid-batch kill")

    monkeypatch.setattr(WarmRolloutWriter, "append_group", kill_after_legacy_plus_one)
    with pytest.raises(RuntimeError, match="legacy mid-batch kill"):
        actor1.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="legacy-step-1")

    # Partial manifest on disk must already contain BOTH legacy [90] and generated [2] (offset 0).
    partial_hit = load_warm_rollout(
        actor1.args.ckpt.path,
        episode=0,
        consumed_prompts=2,
        global_step=1,
        run_id="legacy-step-1",
        max_policy_lag=1,
        recipe=actor1._build_rollout_recipe(),
    )
    assert partial_hit is not None and partial_hit["complete"] is False
    assert [s.sequences.item() for s in partial_hit["samples"]] == [90, 2]
    assert partial_hit["completed_offsets"] == [0]

    # Second restart from the same legacy checkpoint: must keep [90, 2], skip prompt 2, and only generate prompt 3.
    monkeypatch.setattr(WarmRolloutWriter, "append_group", orig_append)
    actor2, payloads2 = _make_generator_actor(
        tmp_path,
        dataset_len=8,
        batch_size=3,
        inflight_capacity=3,
        partial_rollout=False,
        ckpt_name="ckpt_legacy",
        monkeypatch=monkeypatch,
    )
    actor2.load_dataloader_state_dict(saved_dl, legacy_rg_state)
    actor2.fit(episode=0, total_consumed_prompts=2, global_step=1, warm_run_id="legacy-step-1")

    assert [s.sequences.item() for s in payloads2[0][0]] == [90, 2, 3]
    assert actor2._dispatched_prompts[:1] == [3]  # prompt 2 was skipped; prompt 3 completed the batch

    # Finalized complete manifest holds ALL 3 groups [90, 2, 3] and can be warm-loaded directly.
    full_hit = load_warm_rollout(
        actor2.args.ckpt.path,
        episode=0,
        consumed_prompts=2,
        global_step=1,
        run_id="legacy-step-1",
        max_policy_lag=1,
        recipe=actor2._build_rollout_recipe(),
    )
    assert full_hit is not None and full_hit["complete"] is True
    assert [s.sequences.item() for s in full_hit["samples"]] == [90, 2, 3]


def test_sibling_checkpoints_under_same_parent_are_isolated(tmp_path, monkeypatch):
    """Runs sharing a parent directory must isolate warm snapshots and legacy buffers under their own ckpt.path."""
    ckpt_a = str(tmp_path / "run_a")
    ckpt_b = str(tmp_path / "run_b")
    recipe = {"seed": 42}

    # Run A writes a warm batch and a legacy buffer.
    wa = WarmRolloutWriter(
        ckpt_a,
        source_episode=0,
        source_consumed_prompts=10,
        source_global_step=1,
        source_policy_step=1,
        run_id="run-a",
        recipe=recipe,
        keep_batches=2,
    )
    wa.append_group([_make_experience(1)], prompt_offset=0)
    wa.finalize(
        client_states={"episode": 0, "total_consumed_prompts": 12, "data_loader_state_dict": {}},
        metrics={},
    )
    gen_a = object.__new__(SamplesGenerator)
    gen_a.args = SimpleNamespace(ckpt=SimpleNamespace(warm_resume_rollouts=True, path=ckpt_a))
    gen_a._finished_samples = [_make_experience(101)]
    sd_a = gen_a.state_dict()
    assert Path(sd_a["buffer_file"]).is_relative_to(Path(ckpt_a) / "rollout_warm")

    # Run B writes & prunes 5 warm batches and 5 legacy buffers under the SAME parent directory.
    for step in range(5):
        wb = WarmRolloutWriter(
            ckpt_b,
            source_episode=0,
            source_consumed_prompts=10 + step * 2,
            source_global_step=step + 1,
            source_policy_step=step + 1,
            run_id="run-b",
            recipe=recipe,
            keep_batches=2,
        )
        wb.append_group([_make_experience(20 + step)], prompt_offset=0)
        wb.finalize(
            client_states={"episode": 0, "total_consumed_prompts": 12 + step * 2, "data_loader_state_dict": {}},
            metrics={},
        )
    gen_b = object.__new__(SamplesGenerator)
    gen_b.args = SimpleNamespace(ckpt=SimpleNamespace(warm_resume_rollouts=True, path=ckpt_b))
    gen_b._finished_samples = [_make_experience(201)]
    for _ in range(5):
        gen_b.state_dict()

    # Shared parent directory must NOT have a rollout_warm directory.
    assert not (tmp_path / "rollout_warm").exists()

    # Run A's warm snapshot and legacy buffer remain completely intact, and Run B cannot read Run A.
    hit_a = load_warm_rollout(
        ckpt_a, episode=0, consumed_prompts=10, global_step=1, run_id="run-a", max_policy_lag=1, recipe=recipe
    )
    assert hit_a is not None and [s.sequences.item() for s in hit_a["samples"]] == [1]
    assert Path(sd_a["buffer_file"]).is_file()
    assert (
        load_warm_rollout(
            ckpt_b, episode=0, consumed_prompts=10, global_step=1, run_id="run-a", max_policy_lag=1, recipe=recipe
        )
        is None
    )


def test_warm_resume_checks_recipe_policy_and_checkpoint_identity(tmp_path, monkeypatch):
    """Recipe, policy-lag, checkpoint step, run_id, staged dir fingerprinting, and corrupt fallback checks."""
    # Directory dataset staged in two different locations must produce identical fingerprints.
    ds_dir_a = tmp_path / "stage_a" / "ds"
    ds_dir_b = tmp_path / "stage_b" / "ds"
    for d in (ds_dir_a, ds_dir_b):
        d.mkdir(parents=True)
        (d / "shard_0.jsonl").write_text('{"prompt": "hello"}\n')

    actor1, payloads1 = _make_generator_actor(
        tmp_path,
        dataset_len=6,
        batch_size=2,
        inflight_capacity=2,
        reward_ctx="drop",
        dataset_path=ds_dir_a,
        monkeypatch=monkeypatch,
    )
    actor1.fit(episode=0, total_consumed_prompts=0, global_step=0, warm_run_id="run-1")
    step1_cs = payloads1[0][1]
    base_kwargs = dict(
        episode=0,
        consumed_prompts=step1_cs["total_consumed_prompts"],
        global_step=1,
        run_id="run-1",
        max_policy_lag=1,
    )

    # Identical staged dataset directory in a different path still hits!
    actor_same_staged, _ = _make_generator_actor(
        tmp_path,
        dataset_len=6,
        batch_size=2,
        inflight_capacity=2,
        reward_ctx="drop",
        dataset_path=ds_dir_b,
        monkeypatch=monkeypatch,
    )
    assert (
        load_warm_rollout(
            actor_same_staged.args.ckpt.path, recipe=actor_same_staged._build_rollout_recipe(), **base_kwargs
        )
        is not None
    )

    # Recipe mismatches (reward, routing_replay, logprobs_mode, model, force_sync_mode) must reject.
    for override in (
        {"reward_ctx": 0.0},
        {"routing_replay": True},
        {"is_correction_level": "token"},
        {"model_name": "other-model"},
        {"force_sync_mode": True},
    ):
        actor_bad, _ = _make_generator_actor(
            tmp_path,
            dataset_len=6,
            batch_size=2,
            inflight_capacity=2,
            reward_ctx=override.get("reward_ctx", "drop"),
            routing_replay=override.get("routing_replay", False),
            is_correction_level=override.get("is_correction_level", "off"),
            model_name=override.get("model_name", "same-model"),
            force_sync_mode=override.get("force_sync_mode", False),
            dataset_path=ds_dir_a,
            monkeypatch=monkeypatch,
        )
        assert (
            load_warm_rollout(actor_bad.args.ckpt.path, recipe=actor_bad._build_rollout_recipe(), **base_kwargs)
            is None
        )

    # Policy/checkpoint identity mismatches (run_id, global_step, strict policy lag=0 vs async policy_step=0).
    recipe = actor1._build_rollout_recipe()
    assert load_warm_rollout(actor1.args.ckpt.path, recipe=recipe, **{**base_kwargs, "run_id": "other-run"}) is None
    assert load_warm_rollout(actor1.args.ckpt.path, recipe=recipe, **{**base_kwargs, "global_step": 2}) is None
    assert load_warm_rollout(actor1.args.ckpt.path, recipe=recipe, **{**base_kwargs, "max_policy_lag": 0}) is None

    # End-to-end safe fallback on recipe change regenerates cleanly.
    actor2, _ = _make_generator_actor(
        tmp_path,
        dataset_len=6,
        batch_size=2,
        inflight_capacity=2,
        reward_ctx=0.0,
        dataset_path=ds_dir_a,
        monkeypatch=monkeypatch,
    )
    actor2.prompts_dataloader.load_state_dict(step1_cs["data_loader_state_dict"])
    gen_spy = Mock(wraps=actor2.samples_generator.generate_samples)
    actor2.samples_generator.generate_samples = gen_spy
    actor2.fit(
        episode=step1_cs["episode"],
        total_consumed_prompts=step1_cs["total_consumed_prompts"],
        global_step=1,
        warm_run_id="run-1",
    )
    assert gen_spy.call_count == 2

    # Legacy checkpoint run_id helper is deterministic for the same extra_state.pt and changes if overwritten.
    actor_ckpt_dir = tmp_path / "legacy_run" / "_actor" / "global_step3"
    actor_ckpt_dir.mkdir(parents=True)
    (tmp_path / "legacy_run" / "_actor" / "latest").write_text("global_step3")
    torch.save({"client_state": {"global_step": 3}}, actor_ckpt_dir / "extra_state.pt")
    id1 = rl_trainer._legacy_checkpoint_run_id(str(tmp_path / "legacy_run"), 3)
    id2 = rl_trainer._legacy_checkpoint_run_id(str(tmp_path / "legacy_run"), 3)
    assert id1 == id2
    torch.save({"client_state": {"global_step": 3, "episode": 1}}, actor_ckpt_dir / "extra_state.pt")
    id3 = rl_trainer._legacy_checkpoint_run_id(str(tmp_path / "legacy_run"), 3)
    assert id3 != id1

    # Missing group file degrades gracefully to None without crashing.
    warm_dir = _warm_root(actor1.args.ckpt.path) / f"step_ep0_cursor{step1_cs['total_consumed_prompts']}_step1"
    (warm_dir / "group_0001.pt").unlink()
    assert load_warm_rollout(actor1.args.ckpt.path, recipe=recipe, **base_kwargs) is None

    # Corrupt/legacy partial manifest missing completed_offsets degrades safely to None.
    bad_partial = _warm_root(actor1.args.ckpt.path) / "step_ep0_cursor99_step1"
    bad_partial.mkdir(parents=True, exist_ok=True)
    torch.save([_make_experience(99)], bad_partial / "group_0000.pt")
    torch.save(
        {
            "format": "warm_rollout_v3",
            "complete": False,
            "recipe": recipe,
            "run_id": "run-1",
            "source_episode": 0,
            "source_consumed_prompts": 99,
            "source_global_step": 1,
            "source_policy_step": 1,
            "group_files": ["group_0000.pt"],
        },
        bad_partial / "manifest.pt",
    )
    assert (
        load_warm_rollout(
            actor1.args.ckpt.path,
            episode=0,
            consumed_prompts=99,
            global_step=1,
            run_id="run-1",
            max_policy_lag=1,
            recipe=recipe,
        )
        is None
    )

    # RLTrainer.fit() generates a fresh warm_run_id on fresh start and preserves checkpoint warm_run_id on resume.
    rl_cls = rl_trainer.RLTrainer.__ray_metadata__.modified_class
    for ckpt_state, expect_id in (
        ({"episode": 0, "global_step": 0, "total_consumed_prompts": 0, "data_loader_state_dict": {}}, None),
        (
            {
                "episode": 0,
                "global_step": 4,
                "total_consumed_prompts": 16,
                "data_loader_state_dict": {"k": 1},
                "warm_run_id": "persisted-run-id",
            },
            "persisted-run-id",
        ),
    ):
        controller = object.__new__(rl_cls)
        controller.args = SimpleNamespace(ckpt=SimpleNamespace(path=str(tmp_path / "legacy_run")))
        gen_fit = Mock()
        controller.generator_actor = SimpleNamespace(
            load_dataloader_state_dict=SimpleNamespace(remote=Mock()),
            fit=SimpleNamespace(remote=gen_fit),
        )
        controller.trainer_actor = SimpleNamespace(
            load_checkpoint_states_or_default=SimpleNamespace(remote=lambda s=ckpt_state: s),
            restore_best_metric_tracker=SimpleNamespace(remote=Mock()),
            broadcast_to_vllm=SimpleNamespace(remote=Mock()),
            fit=SimpleNamespace(remote=Mock()),
        )
        controller.fit()
        call_kw = gen_fit.call_args.kwargs
        assert call_kw["global_step"] == ckpt_state["global_step"]
        if expect_id is not None:
            assert call_kw["warm_run_id"] == expect_id
        else:
            assert isinstance(call_kw["warm_run_id"], str) and len(call_kw["warm_run_id"]) == 32


def test_warm_writer_prunes_old_batches_and_preserves_lazy_experience(tmp_path, monkeypatch):
    """Disk stays bounded to keep_batches=3 inside ckpt_path and materializing copies does not mutate lazy Experiences."""
    ckpt_path = str(tmp_path / "checkpoints")
    recipe = {"seed": 1}

    payload = {"sequences": torch.tensor([[7, 8]])}
    monkeypatch.setattr("molt.trainer.algorithm.experience.ray.get", lambda ref: ref)
    lazy = Experience(sequences=None, group_ids=["g7"], rollout_ids=["r7"], heavy_ref=payload)

    for step, cursor in enumerate((10, 20, 30, 40, 50), start=1):
        w = WarmRolloutWriter(
            ckpt_path,
            source_episode=0,
            source_consumed_prompts=cursor,
            source_global_step=step,
            source_policy_step=step,
            run_id="run-prune",
            recipe=recipe,
            keep_batches=3,
        )
        w.append_group([lazy], prompt_offset=0)
        w.finalize(
            client_states={"episode": 0, "total_consumed_prompts": cursor + 10, "data_loader_state_dict": {}},
            metrics={},
        )

    # Original lazy Experience remained untouched.
    assert lazy.heavy_ref is payload and lazy.sequences is None

    warm_dirs = sorted(p.name for p in _warm_root(ckpt_path).iterdir() if p.is_dir())
    assert len(warm_dirs) == 3
    assert warm_dirs == ["step_ep0_cursor30_step3", "step_ep0_cursor40_step4", "step_ep0_cursor50_step5"]

    hit = load_warm_rollout(
        ckpt_path, episode=0, consumed_prompts=50, global_step=5, run_id="run-prune", max_policy_lag=1, recipe=recipe
    )
    assert hit is not None and hit["complete"] is True
    assert hit["samples"][0].heavy_ref is None
    assert torch.equal(hit["samples"][0].sequences, torch.tensor([[7, 8]]))


def test_legacy_state_dict_pruning_ignores_warm_subdirectories(tmp_path):
    """Pruning legacy rollout_buffer.pt.* files must skip subdirectories in ckpt_path/rollout_warm/."""
    ckpt_path = str(tmp_path / "checkpoints")
    warm_root = _warm_root(ckpt_path)
    subdir = warm_root / "step_ep0_cursor10_step1"
    subdir.mkdir(parents=True)

    gen = object.__new__(SamplesGenerator)
    gen.args = SimpleNamespace(ckpt=SimpleNamespace(warm_resume_rollouts=True, path=ckpt_path))
    gen._finished_samples = [_make_experience(0)]

    for _ in range(4):
        sd = gen.state_dict()
        assert sd.get("buffer_file") and Path(sd["buffer_file"]).is_file()

    assert subdir.is_dir()
    buffers = [p for p in warm_root.iterdir() if p.is_file() and p.name.startswith(SamplesGenerator._BUFFER)]
    assert len(buffers) == 3
