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
from types import SimpleNamespace

import pytest
import torch

from molt.agents.base import Env, Result, StepEnvRunner, _extract_generation_logprobs


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.tensor([[ord(ch) for ch in text]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


class _OneStepEnv(Env):
    async def step(self, state, **kwargs):
        state["sampling_params"].max_tokens = 1
        return Result(reward=[2.0], score=[3.0], observation="!", terminated=True)


class _Engine:
    def __init__(self):
        self.seen_sampling_params = []

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None, session_id=None):
        self.seen_sampling_params.append(sampling_params)
        # generate() returns (RequestOutput, off_policy_len); off_policy_len=0 = no
        # mid-generation weight broadcast (on-policy), the partial_rollout-off case.
        return (
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[65], text="A", finish_reason="stop", logprobs=None)]),
            0,
        )


def test_step_env_runner_isolates_sampling_params_per_trajectory():
    """Validates list-shaped reward/score unwrap to scalars without a dedicated
    normaliser, and that two concurrent rollouts do not share sampling-param state."""
    params = SimpleNamespace(max_tokens=8, logprobs=None)
    engine = _Engine()
    runner = StepEnvRunner(_OneStepEnv)

    async def _run():
        return await asyncio.gather(
            runner.execute("p", "l", params, 64, _Tokenizer(), engine),
            runner.execute("p", "l", params, 64, _Tokenizer(), engine),
        )

    outputs = asyncio.run(_run())

    assert params.max_tokens == 8
    assert len({id(item) for item in engine.seen_sampling_params}) == 2
    assert [output.reward for output in outputs] == [2.0, 2.0]
    assert [output.scores for output in outputs] == [3.0, 3.0]


def test_generation_logprobs_fail_fast_when_vllm_omits_them():
    with pytest.raises(RuntimeError, match="did not return token logprobs"):
        _extract_generation_logprobs([1], None)


class _ClosingEnv(Env):
    closes = 0

    async def close(self):
        type(self).closes += 1


class _TerminatesEnv(_ClosingEnv):
    async def step(self, state, **kwargs):
        return Result(reward=[1.0], observation="!", terminated=True)


class _NeverEndsEnv(_ClosingEnv):
    async def step(self, state, **kwargs):
        return Result(reward=[0.0], observation="x" * 8, terminated=False)


class _StepRaisesEnv(_ClosingEnv):
    async def step(self, state, **kwargs):
        raise RuntimeError("boom in step")


class _ResetRaisesEnv(_ClosingEnv):
    async def reset(self, state):
        raise RuntimeError("boom in reset")

    async def step(self, state, **kwargs):
        return Result(reward=[0.0], observation="", terminated=True)


class _CloseRaisesEnv(_TerminatesEnv):
    async def close(self):
        await super().close()
        raise RuntimeError("boom in close")


def _execute(env_cls, max_length=64):
    params = SimpleNamespace(max_tokens=8, logprobs=None)
    return asyncio.run(StepEnvRunner(env_cls).execute("p", "l", params, max_length, _Tokenizer(), _Engine()))


def test_env_close_runs_once_on_every_exit_path():
    """Teardown must not depend on step() returning terminated/truncated: context
    exhaustion and reset/step exceptions are routine exits for OS-resource envs."""
    for env_cls, raises in [
        (_TerminatesEnv, None),
        (_NeverEndsEnv, None),  # exits via remaining-context exhaustion
        (_StepRaisesEnv, "boom in step"),
        (_ResetRaisesEnv, "boom in reset"),
    ]:
        env_cls.closes = 0
        if raises:
            with pytest.raises(RuntimeError, match=raises):
                _execute(env_cls, max_length=32)
        else:
            _execute(env_cls, max_length=32)
        assert env_cls.closes == 1, env_cls.__name__


def test_env_close_failure_keeps_the_trajectory():
    _CloseRaisesEnv.closes = 0
    trajectory = _execute(_CloseRaisesEnv)
    assert trajectory.reward == 1.0
    assert _CloseRaisesEnv.closes == 1
