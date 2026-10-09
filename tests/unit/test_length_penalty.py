# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise length shaping through the real reward hook and trainer metrics."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import molt.trainer.rollout.experience_maker as em
from molt.agents.base import Trajectory
from molt.trainer.algorithm.experience import Experience
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker


def _maker(estimator="dr_grpo", **reward_kwargs):
    reward = dict(
        clip_range=None, overlong_buffer_len=None, overlong_penalty_factor=1.0, stop_properly_penalty_coef=None
    )
    reward.update(reward_kwargs)
    maker = SimpleNamespace(
        advantage_estimator=estimator,
        kl_ctl=SimpleNamespace(value=0.0),
        args=SimpleNamespace(
            reward=SimpleNamespace(**reward),
            algo=SimpleNamespace(advantage=SimpleNamespace(gamma=1.0, lam=1.0, no_whiten=True)),
            rollout=SimpleNamespace(n_samples_per_prompt=2, max_new_tokens=None, batch_size=1),
            data=SimpleNamespace(max_len=2048),
            actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=1),
            fsdp=SimpleNamespace(cp_size=1, tp_size=1),
            train=SimpleNamespace(force_on_policy=True),
        ),
    )
    maker._merge_rollout_rewards = RemoteExperienceMaker._merge_rollout_rewards.__get__(maker)
    maker._per_sample_rewards = RemoteExperienceMaker._per_sample_rewards
    maker.compute_advantages_and_returns = RemoteExperienceMaker.compute_advantages_and_returns.__get__(maker)
    return maker


def _sample(rid, gid, reward, response_length, total_length, truncated, idx):
    return Experience(
        sequences=torch.zeros(1, 5, dtype=torch.long),
        action_mask=torch.ones(1, 4, dtype=torch.bool),
        kl=torch.zeros(1, 4),
        rewards=torch.tensor([float(reward)]),
        response_length=torch.tensor([response_length]),
        total_length=torch.tensor([total_length]),
        truncated=torch.tensor([truncated]),
        index=[idx],
        group_ids=[gid],
        rollout_ids=[rid],
        info={"reward": torch.tensor([float(reward)])},
    )


@pytest.fixture
def record_rewards(monkeypatch):
    captured = {}

    def record(rewards, groups, ctx):
        captured["rewards"] = rewards.clone()
        zeros = [torch.zeros_like(m, dtype=torch.float32) for m in ctx.action_masks]
        return zeros, zeros

    monkeypatch.setattr(em, "get_advantage_estimator", lambda name: record)
    return captured


@pytest.mark.parametrize(
    "estimator,magnitude", [("grpo", 2**-0.5), ("dr_grpo", 0.5), ("reinforce_baseline", 0.5), ("rloo", 1.0)]
)
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0)])
def test_real_group_advantages_are_order_invariant(estimator, magnitude, order):
    samples = [
        _sample("A", "g", 1, 100, 600, False, 0),
        _sample("A", "g", 1, 1000, 2048, True, 1),
        _sample("B", "g", 1, 100, 600, False, 2),
    ]
    maker = _maker(estimator, overlong_buffer_len=200, stop_properly_penalty_coef=0.5)
    maker.compute_advantages_and_returns([samples[i] for i in order])
    for s, expected in zip(samples, [-magnitude, -magnitude, magnitude]):
        torch.testing.assert_close(s.advantages, torch.full((1, 4), expected), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(s.info["reward"], torch.tensor([1.0]))
    assert samples[0].info["length_penalty"].item() == -1
    assert samples[1].info["length_penalty"].item() == -1


def test_compaction_does_not_sum_separate_context_windows(record_rewards):
    samples = [_sample("A", "g", 1, 900, 1400, False, 0), _sample("A", "g", 1, 900, 1500, False, 1)]
    _maker(overlong_buffer_len=200).compute_advantages_and_returns(samples)
    torch.testing.assert_close(record_rewards["rewards"], torch.tensor([1.0]))


def test_multiturn_generation_uses_context_budget(record_rewards):
    maker = _maker(overlong_buffer_len=1024)
    maker.args.data.max_len = 16384
    maker.args.rollout.max_new_tokens = 8192
    sample = _sample("A", "g", 1, 5 * 2000, 11000, False, 0)
    maker.compute_advantages_and_returns([sample])
    torch.testing.assert_close(record_rewards["rewards"], torch.tensor([1.0]))


def test_none_generation_cap_filled_context_is_penalized(record_rewards):
    maker = _maker(overlong_buffer_len=512)
    maker.args.data.max_len = 4096
    sample = _sample("A", "g", 1, 3096, 4096, True, 0)
    maker.compute_advantages_and_returns([sample])
    torch.testing.assert_close(record_rewards["rewards"], torch.tensor([0.0]))


def test_tool_feedback_in_largest_segment_counts(record_rewards):
    samples = [_sample("A", "g", 1, 100, 600, False, 0), _sample("A", "g", 1, 100, 1908, False, 1)]
    _maker(overlong_buffer_len=200).compute_advantages_and_returns(samples)
    torch.testing.assert_close(record_rewards["rewards"], torch.tensor([0.7]))


def test_penalties_before_clip_keep_raw_reward(record_rewards):
    s = _sample("A", "g", 1, 1408, 1908, True, 0)
    _maker(
        overlong_buffer_len=200, stop_properly_penalty_coef=0.5, clip_range=(0.0, 0.5)
    ).compute_advantages_and_returns([s])
    torch.testing.assert_close(record_rewards["rewards"], torch.tensor([0.35]))
    torch.testing.assert_close(s.rewards, torch.tensor([1.0]))
    torch.testing.assert_close(s.info["reward"], torch.tensor([1.0]))
    torch.testing.assert_close(s.info["length_penalty"], torch.tensor([-0.65]))
    torch.testing.assert_close(s.info["overlong_penalty"], torch.tensor([-0.3]))
    torch.testing.assert_close(s.info["stop_properly_penalty"], torch.tensor([-0.35]))


@pytest.mark.parametrize("coef,expected", [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (-2.0, -2.0)])
def test_stop_any_segment_needs_only_truncation(record_rewards, coef, expected):
    samples = [
        _sample("A", "g", 1, 100, 600, False, 0),
        _sample("A", "g", 1, 100, 600, True, 1),
        _sample("B", "g", 1, 100, 600, False, 2),
    ]
    for s in samples:
        s.total_length = s.response_length = None
    _maker(stop_properly_penalty_coef=coef).compute_advantages_and_returns(samples)
    torch.testing.assert_close(record_rewards["rewards"], torch.tensor([expected, 1.0]))
    torch.testing.assert_close(samples[0].info["stop_properly_penalty"], torch.tensor([expected - 1]))


@pytest.mark.parametrize("estimator", ["reinforce", "gae"])
def test_per_sample_estimators_keep_distinct_rewards(record_rewards, estimator):
    samples = [_sample("A", "g", 1, 100, 1908, True, 0), _sample("A", "g", 1, 100, 600, False, 1)]
    _maker(estimator, overlong_buffer_len=200, stop_properly_penalty_coef=-2).compute_advantages_and_returns(samples)
    torch.testing.assert_close(record_rewards["rewards"], torch.tensor([-2.0, 1.0]))


def test_disabled_preserves_reward_dtype_and_needs_no_lengths(record_rewards):
    s = _sample("A", "g", 1, 100, 600, False, 0)
    s.rewards = s.rewards.bfloat16()
    s.total_length = s.response_length = s.truncated = None
    _maker().compute_advantages_and_returns([s])
    assert record_rewards["rewards"].dtype == torch.bfloat16
    assert "length_penalty" not in s.info


def test_real_sample_conversion_counts_image_budget(record_rewards, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace, __version__="0.21.0"))
    from molt.trainer.rollout.samples_generator import SamplesGenerator

    traj = Trajectory(
        prompt="p",
        label="l",
        images=None,
        observation_text="",
        observation_tokens=[1] * 1400,
        action_ranges=[(500, 1400)],
        image_budget=648,
        reward=1.0,
        group_id="g",
        rollout_id="A",
    )
    s, reason = SamplesGenerator._process_response_into_experience(traj, None, 2048)
    assert reason is None
    s.index = [0]
    s.kl = torch.zeros_like(s.action_mask, dtype=torch.float32)
    assert s.total_length.item() == 1400
    assert s.info["context_length"].item() == 2048
    _maker(overlong_buffer_len=200).compute_advantages_and_returns([s])
    torch.testing.assert_close(record_rewards["rewards"], torch.tensor([0.0]))


def test_trainer_metrics_deduplicate_segments_and_keep_dropped_rewards(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace, __version__="0.21.0"))
    import molt.trainer.rl_trainer as rt

    maker = _maker(overlong_buffer_len=200, stop_properly_penalty_coef=0.5)
    maker.args.actor.num_gpus_per_node = 2
    samples = [
        _sample("A", "g", 1, 100, 600, False, 0),
        _sample("A", "g", 1, 1000, 2048, True, 1),
        _sample("A", "g", 1, 100, 600, False, 2),
        _sample("B", "g", 1, 100, 600, False, 3),
        _sample("dropped", "g", 0, 100, 600, False, 4),
    ]
    trainer = SimpleNamespace(
        args=maker.args,
        experience_maker=SimpleNamespace(build_experiences=maker.compute_advantages_and_returns),
        tokenizer=SimpleNamespace(decode=lambda *a, **k: "sample"),
        actor_model_group=SimpleNamespace(async_run_method_batch=lambda **k: []),
        critic_model_group=None,
        freezing_actor_steps=0,
        vllm_engines=None,
        kl_ctl=SimpleNamespace(value=0.0),
        policy_train=lambda **k: {},
    )
    monkeypatch.setattr(rt.ray, "get", lambda refs: refs)
    stats, step = rt.BaseRLTrainer.train_step(trainer, samples, 0)
    assert step == 1
    assert stats["rollout/reward_mean"] == pytest.approx(2 / 3)
    assert stats["rollout/overlong_frac"] == 0.5  # one of two kept rollouts, not three of four segments
    assert samples[-1].info["reward"].item() == 0
    assert "length_penalty" not in samples[-1].info


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--reward.overlong_buffer_len", "0"], "overlong_buffer_len must be finite and in"),
        (["--reward.overlong_buffer_len", "4096"], "overlong_buffer_len must be finite and in"),
        (["--reward.overlong_buffer_len", "nan"], "overlong_buffer_len must be finite and in"),
        (["--reward.overlong_buffer_len", "200", "--reward.overlong_penalty_factor", "-1"], "factor must be finite"),
        (["--reward.overlong_buffer_len", "200", "--reward.overlong_penalty_factor", "nan"], "factor must be finite"),
        (["--reward.stop_properly_penalty_coef", "1.5"], "coef must be finite and <= 1"),
        (["--reward.stop_properly_penalty_coef", "nan"], "coef must be finite and <= 1"),
        (["--reward.overlong_buffer_len", "200"], "--train.agent_path is required"),
    ],
)
def test_cli_validates_before_training(flags, message):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "molt.cli.train_rl_ray",
            "--actor.model_name_or_path",
            "unused",
            "--vllm.num_engines",
            "1",
            "--data.max_len",
            "2048",
            *flags,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert message in result.stderr
