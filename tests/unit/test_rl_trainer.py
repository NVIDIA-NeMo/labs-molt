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

import asyncio
import errno
import os
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import molt.trainer.rl_trainer as rl_trainer
from molt.trainer.fsdp.checkpoint import CheckpointManager


class _Queue:
    def __init__(self, values=()):
        self.values = deque(values)
        self.put_values = []

    def get(self, block=True):
        return self.values.popleft()

    def put(self, value, block=True):
        self.put_values.append(value)


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _VLLMLock:
    def __init__(self, policy_step):
        self.policy_step = policy_step
        self.acquire_calls = 0
        self.release_calls = 0
        self.acquire = _RemoteMethod(self._acquire)
        self.release = _RemoteMethod(self._release)

    def _acquire(self):
        self.acquire_calls += 1
        return self.policy_step()

    def _release(self, policy_step=None):
        self.release_calls += 1
        if policy_step is not None:
            self.policy_step = lambda: policy_step


class _ActorGroup:
    def __init__(self, policy_state):
        self.policy_state = policy_state
        self.checkpoints = {}
        self.save_calls = []

    def async_run_method(self, method_name, **kwargs):
        if method_name == "save_checkpoint":
            self.save_calls.append(kwargs)
            client_step = (kwargs.get("client_states") or {}).get("global_step")
            self.checkpoints[kwargs["tag"]] = (self.policy_state["step"], client_step)
        return []


class _Strategy:
    def __init__(self):
        self.checkpoint = CheckpointManager(self)

    def print(self, *_args):
        pass


def _write_checkpoint(path, client_state=None, model=None, optimizer=None):
    (path / "model").mkdir(parents=True)
    if model is not None:
        (path / "model" / "shard").write_text(model)
    if optimizer is not None:
        (path / "optimizer").mkdir()
        (path / "optimizer" / "shard").write_text(optimizer)
    torch.save({"client_state": client_state or {}}, path / "extra_state.pt")


def _transaction_tree(tmp_path, previous_step):
    source_tag = "global_step5"
    sources = {}
    for role in ("actor", "critic"):
        source = tmp_path / f"_{role}" / source_tag
        _write_checkpoint(source, {"global_step": 5}, f"new-{role}", f"{role}-optimizer-step5")
        _write_checkpoint(
            tmp_path / f"_{role}" / f"best_global_step{previous_step}",
            {"global_step": previous_step},
            f"old-{role}",
        )
        sources[role] = source
    hf_root = tmp_path / "_hf"
    sources["hf"] = hf_root / source_tag
    sources["hf"].mkdir(parents=True)
    (sources["hf"] / "model.safetensors").write_text("new-hf")
    previous_hf = hf_root / f"best_global_step{previous_step}"
    previous_hf.mkdir()
    (previous_hf / "model.safetensors").write_text("old-hf")
    return source_tag, sources, previous_hf


def _controller_strategy(
    *,
    eval_dataset="eval.jsonl",
    eval_steps=5,
    save_steps=1,
    best_metric_key="",
    partial=False,
    force_sync=False,
    queue_size=1,
    max_num=3,
    dcp_max_num=None,
    max_mem=float("inf"),
    save_hf=False,
):
    if dcp_max_num is None:
        dcp_max_num = max_num
    return SimpleNamespace(
        args=SimpleNamespace(
            eval=SimpleNamespace(dataset=eval_dataset, steps=eval_steps),
            ckpt=SimpleNamespace(
                save_steps=save_steps,
                best_metric_key=best_metric_key,
                max_num=max_num,
                dcp_max_num=dcp_max_num,
                max_mem=max_mem,
                save_hf=save_hf,
            ),
            train=SimpleNamespace(
                partial_rollout_enable=partial,
                force_sync_mode=force_sync,
                async_queue_size=queue_size,
            ),
        )
    )


def _trainer(tmp_path, actor_group, critic_group=None, *, force_sync=False, partial=False, load=False, queue=()):
    trainer_cls = rl_trainer.TrainingActor.__ray_metadata__.modified_class
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(force_sync_mode=force_sync, partial_rollout_enable=partial),
        eval=SimpleNamespace(dataset="eval.jsonl", steps=5, eval_at_start=False),
        ckpt=SimpleNamespace(path=str(tmp_path), save_hf=False, load_enable=load),
    )
    trainer.strategy = _Strategy()
    trainer.actor_model_group = actor_group
    trainer.critic_model_group = critic_group
    trainer.vllm_lock = _VLLMLock(lambda: actor_group.policy_state["step"])
    trainer.rollout_queue = _Queue(queue)
    trainer.rollout_slots = _Queue()
    trainer.wandb_logger = None
    trainer.tensorboard_logger = None
    trainer.best_eval_metric_value = float("-inf")
    trainer.best_eval_metric_key = "eval_pass1"
    return trainer


def test_eval_uses_vllm_version_when_broadcast_wins_slot_race(monkeypatch, tmp_path):
    dataloader = MagicMock()
    dataloader.__len__.return_value = 1
    dataloader.sampler = None
    dataloader.state_dict.return_value = {}
    samples_generator = MagicMock()
    samples_generator.generate_eval_samples.return_value = ["sample"]
    monkeypatch.setattr(rl_trainer.ray, "get", lambda refs: refs)
    monkeypatch.setattr(rl_trainer, "tqdm", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(rl_trainer, "compute_eval_metrics", lambda *_args: {"eval_pass1": 1.0})

    generator_cls = rl_trainer.GenerateSamplesActor.__ray_metadata__.modified_class
    generator = generator_cls.__new__(generator_cls)
    generator.args = SimpleNamespace(
        train=SimpleNamespace(num_episodes=1, rollout_replay_dir=str(tmp_path), rollout_dump_dir=None),
        eval=SimpleNamespace(
            steps=5,
            eval_at_start=False,
            n_samples_per_prompt=None,
            temperature=None,
            top_p=None,
            max_new_tokens=None,
        ),
        rollout=SimpleNamespace(n_samples_per_prompt=1),
    )
    generator.prompts_dataloader = dataloader
    generator.eval_dataloader = object()
    generator.samples_generator = samples_generator
    generator.generate_kwargs = {}
    generator.vllm_lock = _VLLMLock(lambda: 6)
    generator._partial_rollout = False
    generator.rollout_queue = _Queue()
    generator.rollout_slots = _Queue([5, 6])
    generator._last_eval_step = -1
    generator._next_eval_step = 5
    generator._eval_at_start = False

    generator.fit(episode=0, total_consumed_prompts=0)

    assert generator.rollout_queue.put_values[0] == ("eval", 6, {"eval_pass1": 1.0})
    assert generator.vllm_lock.acquire_calls == 1
    assert generator.vllm_lock.release_calls == 1


def test_vllm_lock_returns_the_version_published_by_the_broadcast_that_won_the_race():
    lock_cls = rl_trainer.VLLMLock.__ray_metadata__.modified_class

    async def run_race():
        lock = lock_cls()
        assert await lock.acquire() == 0
        waiting_eval = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0)
        await lock.release(policy_step=6)
        assert await waiting_eval == 6
        await lock.release()

    asyncio.run(run_race())


@pytest.mark.parametrize(
    ("force_sync", "event_order"),
    [
        (False, ("rollout-step4", "rollout-step5", "eval-step5")),
        (True, ("rollout-step4", "eval-step5", "rollout-step5")),
        (
            False,
            (
                "rollout-step4",
                "rollout-step5",
                "rollout-step6",
                "rollout-step7",
                "rollout-step8",
                "rollout-step9",
                "rollout-step10",
                "eval-step5",
                "eval-step10",
            ),
        ),
    ],
)
def test_async_eval_saves_the_policy_that_produced_the_metric(monkeypatch, tmp_path, force_sync, event_order):
    """A delayed step-5 metric must keep step-5 actor and client state."""
    monkeypatch.setattr(rl_trainer.ray, "get", lambda refs: refs)

    policy_state = {"step": 4}
    actor_group = _ActorGroup(policy_state)
    critic_group = _ActorGroup(policy_state)
    events = {
        **{f"rollout-step{step}": (f"rollout-step{step}", {"episode": 0}, {}, 0.0, 0.0) for step in range(4, 11)},
        "eval-step5": ("eval", 5, {"eval_pass1": 1.0}),
        "eval-step10": ("eval", 10, {"eval_pass1": 0.5}),
    }

    trainer = _trainer(
        tmp_path,
        actor_group,
        critic_group,
        force_sync=force_sync,
        queue=[events[name] for name in event_order] + ["done"],
    )

    def train_step(_rollout_samples, global_step):
        policy_state["step"] += 1
        return {}, global_step + 1

    def save_checkpoint(global_step, *_args, **_kwargs):
        if global_step % 5:
            return
        for role in ("_actor", "_critic"):
            saved = tmp_path / role / f"global_step{global_step}"
            (saved / "model").mkdir(parents=True)
            (saved / "optimizer").mkdir()
            (saved / "model" / "policy_step").write_text(str(policy_state["step"]))
            (saved / "optimizer" / "state").write_text(f"optimizer-step{global_step}")
            torch.save({"client_state": {"global_step": global_step}}, saved / "extra_state.pt")

    trainer.train_step = train_step
    trainer.save_logs_and_checkpoints = save_checkpoint

    trainer.fit(global_step=4)

    if force_sync:
        assert actor_group.checkpoints["best_global_step5"] == (5, 5)
        assert critic_group.checkpoints["best_global_step5"] == (5, None)
    else:
        for role in ("_actor", "_critic"):
            best = tmp_path / role / "best_global_step5"
            assert (best / "model" / "policy_step").read_text() == "5"
            assert (best / "optimizer" / "state").read_text() == "optimizer-step5"
            state = torch.load(best / "extra_state.pt", weights_only=False)["client_state"]
            assert state["global_step"] == 5


def test_partial_rollout_does_not_select_a_best_checkpoint(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(rl_trainer.ray, "get", lambda refs: refs)

    policy_state = {"step": 5}
    actor_group = _ActorGroup(policy_state)
    trainer = _trainer(
        tmp_path,
        actor_group,
        partial=True,
        queue=[("eval", 5, {"eval_pass1": 1.0}), "done"],
    )

    trainer.fit(global_step=5)

    assert actor_group.checkpoints == {}
    assert trainer.best_eval_metric_value == float("-inf")
    assert "one evaluation can span multiple policy versions" in caplog.text


def test_delayed_best_copies_actor_critic_and_hf_from_the_eval_step(tmp_path):
    source_tag, sources, old_hf = _transaction_tree(tmp_path, previous_step=1)

    trainer = _trainer(tmp_path, object(), object())
    trainer.args.ckpt.save_hf = True

    trainer.save_best_checkpoint(
        {"eval_pass1": 0.75},
        5,
        source_tag=source_tag,
    )

    for role in ("actor", "critic"):
        promoted = tmp_path / f"_{role}" / "best_global_step5"
        assert (promoted / "model" / "shard").read_text() == f"new-{role}"
        assert (promoted / "optimizer" / "shard").read_text() == f"{role}-optimizer-step5"
        state = torch.load(promoted / "extra_state.pt", weights_only=False)["client_state"]
        assert state["global_step"] == 5
        assert state["best_eval_metric_value"] == 0.75
        assert not (tmp_path / f"_{role}" / "best_global_step1").exists()
        assert (sources[role] / "model" / "shard").exists()

    assert (tmp_path / "_hf" / "best_global_step5" / "model.safetensors").read_text() == "new-hf"
    assert (sources["hf"] / "model.safetensors").exists()
    assert not old_hf.exists()
    assert not [path for path in tmp_path.rglob("*") if path.name.endswith((".tmp", ".old"))]


@pytest.mark.parametrize("failed_role", ["actor", "critic", "hf"])
def test_delayed_best_copy_failure_leaves_no_partial_promotion(monkeypatch, tmp_path, failed_role):
    source_tag, sources, old_hf = _transaction_tree(tmp_path, previous_step=1)

    copytree = rl_trainer.shutil.copytree

    def fail_copy(source, destination, *args, **kwargs):
        if os.fspath(source) == str(sources[failed_role]):
            os.makedirs(destination)
            raise OSError(errno.ENOSPC, "injected copy failure")
        return copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(rl_trainer.shutil, "copytree", fail_copy)
    trainer = _trainer(tmp_path, object(), object())
    trainer.args.ckpt.save_hf = True
    trainer.best_eval_metric_key = ""

    with pytest.raises(OSError, match="injected copy failure"):
        trainer.save_best_checkpoint({"eval_pass1": 0.75}, 5, source_tag=source_tag)

    assert all((tmp_path / f"_{role}" / "best_global_step1").exists() for role in ("actor", "critic"))
    assert old_hf.exists()
    assert not list(tmp_path.rglob("best_global_step5"))
    assert not (tmp_path / "_hf" / "best_global_step5").exists()
    assert not [path for path in tmp_path.rglob("*") if path.name.endswith((".tmp", ".old"))]
    assert trainer.best_eval_metric_key == ""
    assert trainer.best_eval_metric_value == float("-inf")


def test_delayed_best_recovers_an_orphaned_backup_before_copying(monkeypatch, tmp_path):
    source = tmp_path / "_actor" / "global_step5"
    backup = tmp_path / "_actor" / "best_global_step5.old"
    _write_checkpoint(source, {"global_step": 5}, "new-actor")
    _write_checkpoint(backup, {"global_step": 5}, "old-actor")

    def fail_copy(_source, destination):
        os.makedirs(destination)
        raise OSError(errno.ENOSPC, "injected copy failure")

    monkeypatch.setattr(rl_trainer.shutil, "copytree", fail_copy)
    trainer = _trainer(tmp_path, object())

    with pytest.raises(OSError, match="injected copy failure"):
        trainer.save_best_checkpoint({"eval_pass1": 0.75}, 5, source_tag="global_step5")

    best = tmp_path / "_actor" / "best_global_step5"
    assert (best / "model" / "shard").read_text() == "old-actor"
    assert not backup.exists()
    assert not (tmp_path / "_actor" / "best_global_step5.tmp").exists()
    assert trainer.best_eval_metric_value == float("-inf")


@pytest.mark.parametrize("failed_role", ["actor", "critic", "hf"])
def test_delayed_best_publish_failure_rolls_back_every_role(monkeypatch, tmp_path, failed_role):
    source_tag, _, previous_hf = _transaction_tree(tmp_path, previous_step=5)

    replace = rl_trainer.os.replace
    pending = {
        "actor": tmp_path / "_actor" / "best_global_step5.tmp",
        "critic": tmp_path / "_critic" / "best_global_step5.tmp",
        "hf": tmp_path / "_hf" / "best_global_step5.tmp",
    }

    def fail_publish(source, destination):
        if source == str(pending[failed_role]):
            raise OSError("injected publish failure")
        return replace(source, destination)

    monkeypatch.setattr(rl_trainer.os, "replace", fail_publish)
    trainer = _trainer(tmp_path, object(), object())
    trainer.args.ckpt.save_hf = True
    trainer.best_eval_metric_key = ""

    with pytest.raises(OSError, match="injected publish failure"):
        trainer.save_best_checkpoint({"eval_pass1": 0.75}, 5, source_tag=source_tag)

    for role in ("actor", "critic"):
        assert (tmp_path / f"_{role}" / "best_global_step5" / "model" / "shard").read_text() == f"old-{role}"
    assert (previous_hf / "model.safetensors").read_text() == "old-hf"
    assert not [path for path in tmp_path.rglob("*") if path.name.endswith((".tmp", ".old"))]
    assert trainer.best_eval_metric_key == ""
    assert trainer.best_eval_metric_value == float("-inf")


def test_nan_eval_metric_is_not_promoted(tmp_path):
    actor_group = _ActorGroup({"step": 5})
    trainer = _trainer(tmp_path, actor_group)

    trainer.save_best_checkpoint({"eval_pass1": float("nan")}, 5)

    assert actor_group.checkpoints == {}
    assert trainer.best_eval_metric_value == float("-inf")


def test_delayed_best_is_skipped_when_the_eval_step_was_not_saved(tmp_path, caplog):
    actor_group = _ActorGroup({"step": 6})
    trainer = _trainer(tmp_path, actor_group)
    trainer.best_eval_metric_key = ""

    trainer.save_best_checkpoint({"eval_pass1": 1.0}, 5, source_tag="global_step5")

    assert actor_group.checkpoints == {}
    assert trainer.best_eval_metric_key == ""
    assert trainer.best_eval_metric_value == float("-inf")
    assert "matching saved checkpoint global_step5 is unavailable" in caplog.text


def test_restore_prefers_durable_best_metadata_over_stale_latest(tmp_path):
    best = tmp_path / "_actor" / "best_global_step5"
    _write_checkpoint(
        best,
        {
            "best_eval_metric_key": "eval_pass1",
            "best_eval_metric_value": 0.75,
        },
    )
    trainer = _trainer(tmp_path, _ActorGroup({"step": 6}), load=True)
    trainer.args.ckpt.save_hf = True
    orphan_hf = tmp_path / "_hf" / "best_global_step1"
    orphan_hf.mkdir(parents=True)
    (orphan_hf / "model.safetensors").write_text("orphan")
    for suffix in (".tmp", ".old"):
        pending = tmp_path / "_actor" / f"best_global_step9{suffix}"
        _write_checkpoint(pending, {"best_eval_metric_key": "wrong", "best_eval_metric_value": 1.0})

    trainer.restore_best_metric_tracker(
        {
            "best_eval_metric_key": "eval_pass1",
            "best_eval_metric_value": 0.5,
        }
    )

    assert trainer.best_eval_metric_key == "eval_pass1"
    assert trainer.best_eval_metric_value == 0.75
    assert not orphan_hf.exists()


def test_fresh_run_ignores_best_metadata_left_in_output_directory(tmp_path):
    best = tmp_path / "_actor" / "best_global_step5"
    _write_checkpoint(
        best,
        {
            "best_eval_metric_key": "eval_pass1",
            "best_eval_metric_value": 0.75,
        },
    )
    trainer = _trainer(tmp_path, _ActorGroup({"step": 0}), load=False)

    trainer.restore_best_metric_tracker({})

    assert trainer.best_eval_metric_value == float("-inf")


def test_hf_exports_follow_retained_actor_checkpoints(tmp_path):
    hf_root = tmp_path / "_hf"
    retained = {"global_step2", "global_step3", "best_global_step2"}
    for tag in retained:
        checkpoint = tmp_path / "_actor" / tag
        _write_checkpoint(checkpoint)
        export = hf_root / tag
        export.mkdir(parents=True)
        (export / "model.safetensors").write_text(tag)
    for tag in ("global_step1", "best_global_step1"):
        export = hf_root / tag
        export.mkdir()
        (export / "model.safetensors").write_text(tag)

    trainer = _trainer(tmp_path, _ActorGroup({"step": 3}))
    trainer.args.ckpt.save_hf = True
    trainer._prune_hf_checkpoints()

    assert {path.name for path in hf_root.iterdir()} == retained


def test_rolling_retention_uses_monotonic_best_metric(monkeypatch, tmp_path):
    monkeypatch.setattr(rl_trainer.ray, "get", lambda refs: refs)
    actor_group = _ActorGroup({"step": 6})
    trainer = _trainer(tmp_path, actor_group)
    trainer.args.logger = SimpleNamespace(logging_steps=1)
    trainer.args.ckpt.save_steps = 1
    trainer.best_eval_metric_value = 0.75

    trainer.save_logs_and_checkpoints(6, client_states={"global_step": 6})

    assert actor_group.save_calls[0]["metric_value"] == 0.75


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"queue_size": 2}, id="async-best"),
        pytest.param(
            {"queue_size": 2, "dcp_max_num": 3, "max_num": 1},
            id="dcp-retention-decoupled-from-hf",
        ),
        pytest.param({"save_steps": -1, "force_sync": True}, id="force-sync"),
        pytest.param({"save_steps": 5, "partial": True, "max_num": 1, "max_mem": 100}, id="partial"),
        pytest.param({"eval_steps": 25, "save_steps": 1000, "best_metric_key": "none"}, id="best-disabled"),
        pytest.param({"eval_dataset": "", "eval_steps": -1, "save_steps": 2}, id="eval-disabled"),
    ],
)
def test_best_checkpoint_validation_accepts_supported_modes(monkeypatch, config):
    strategy = _controller_strategy(**config)
    monkeypatch.setattr(rl_trainer, "Queue", lambda maxsize: _Queue())
    monkeypatch.setattr(rl_trainer, "VLLMLock", SimpleNamespace(remote=lambda: object()))
    monkeypatch.setattr(rl_trainer, "GenerateSamplesActor", SimpleNamespace(remote=lambda **_kwargs: object()))
    monkeypatch.setattr(rl_trainer, "TrainingActor", SimpleNamespace(remote=lambda **_kwargs: object()))
    trainer_cls = rl_trainer.RLTrainer.__ray_metadata__.modified_class

    trainer_cls(None, strategy, None, None, None)


@pytest.mark.parametrize(
    "config",
    [
        {"save_steps": -1},
        {"save_steps": 3},
        {"save_steps": 5},
        {"save_steps": 5, "force_sync": True, "queue_size": 2},
    ],
)
def test_async_best_requires_a_rolling_checkpoint_for_every_policy_step(config):
    strategy = _controller_strategy(**config)
    trainer_cls = rl_trainer.RLTrainer.__ray_metadata__.modified_class

    with pytest.raises(ValueError, match="save_steps 1"):
        trainer_cls(None, strategy, None, None, None)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"queue_size": 2, "dcp_max_num": 2}, "dcp_max_num"),
        (
            {"queue_size": 2, "dcp_max_num": 3, "max_num": 2, "save_hf": True},
            "--ckpt.max_num",
        ),
        ({"queue_size": 2, "max_mem": 100}, "max_mem"),
    ],
)
def test_async_best_rejects_retention_that_can_evict_an_inflight_source(config, message):
    strategy = _controller_strategy(**config)
    trainer_cls = rl_trainer.RLTrainer.__ray_metadata__.modified_class

    with pytest.raises(ValueError, match=message):
        trainer_cls(None, strategy, None, None, None)
