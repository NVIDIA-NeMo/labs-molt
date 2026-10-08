# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from molt.trainer.algorithm.experience import Experience
from molt.trainer.algorithm.length_penalty import (
    apply_length_penalties,
    apply_overlong_penalty,
    apply_stop_properly_penalty,
)


def _exp(rewards, response_lengths=None, truncated=None, info=None):
    return Experience(
        rewards=torch.tensor(rewards, dtype=torch.float32),
        response_length=torch.tensor(response_lengths, dtype=torch.long) if response_lengths is not None else None,
        truncated=torch.tensor(truncated, dtype=torch.bool) if truncated is not None else None,
        info=info or {},
    )


def _penalty_args(**overrides):
    reward = SimpleNamespace(
        overlong_buffer_len=None,
        overlong_penalty_factor=1.0,
        stop_properly_penalty_coef=None,
    )
    for k, v in overrides.items():
        setattr(reward, k, v)
    return SimpleNamespace(
        reward=reward,
        rollout=SimpleNamespace(max_new_tokens=1000),
        data=SimpleNamespace(max_len=2048),
    )


def test_overlong_penalty_math():
    # max_new_tokens=1000, buffer=200 -> expected_len=800.
    # lengths [700, 900, 1100]: only the last two exceed; penalties are
    # -(100/200)*1.0 = -0.5 and -(200/200 capped)*1.0 = -1.0.
    exps = [_exp([1.0, 1.0, 1.0], [700, 900, 1100])]
    n = apply_overlong_penalty(exps, max_new_tokens=1000, overlong_buffer_len=200)
    assert n == 2
    assert torch.allclose(exps[0].rewards, torch.tensor([1.0, 0.5, 0.0]))


def test_overlong_penalty_factor_scales():
    exps = [_exp([0.0], [900])]
    apply_overlong_penalty(exps, max_new_tokens=1000, overlong_buffer_len=200, overlong_penalty_factor=2.0)
    assert torch.allclose(exps[0].rewards, torch.tensor([-1.0]))


def test_overlong_penalty_no_penalty_within_limit():
    exps = [_exp([0.5, 0.5], [100, 800])]
    n = apply_overlong_penalty(exps, max_new_tokens=1000, overlong_buffer_len=200)
    assert n == 0
    assert torch.allclose(exps[0].rewards, torch.tensor([0.5, 0.5]))


def test_stop_properly_scales_truncated():
    exps = [_exp([1.0, 0.5], truncated=[True, False])]
    n = apply_stop_properly_penalty(exps, stop_properly_penalty_coef=0.5)
    assert n == 1
    assert torch.allclose(exps[0].rewards, torch.tensor([0.5, 0.5]))


def test_stop_properly_negative_overrides():
    exps = [_exp([1.0, 0.5], truncated=[True, False])]
    n = apply_stop_properly_penalty(exps, stop_properly_penalty_coef=-0.5)
    assert n == 1
    assert torch.allclose(exps[0].rewards, torch.tensor([-0.5, 0.5]))


def test_stop_properly_skips_missing_truncated():
    exps = [_exp([1.0])]
    n = apply_stop_properly_penalty(exps, stop_properly_penalty_coef=0.5)
    assert n == 0
    assert torch.allclose(exps[0].rewards, torch.tensor([1.0]))


def test_apply_length_penalties_noop_when_unset():
    exps = [_exp([1.0], [1500], [True], info={"reward": torch.tensor([1.0])})]
    apply_length_penalties(exps, _penalty_args())
    assert torch.allclose(exps[0].rewards, torch.tensor([1.0]))
    assert torch.allclose(exps[0].info["reward"], torch.tensor([1.0]))


def test_apply_length_penalties_both_and_info_sync():
    exps = [_exp([1.0], [950], [True], info={"reward": torch.tensor([1.0])})]
    args = _penalty_args(overlong_buffer_len=200, stop_properly_penalty_coef=0.5)
    # overlong: exceed = 950 - 800 = 150 -> penalty -150/200 = -0.75 -> reward 0.25;
    # stop-properly: truncated -> 0.25 * 0.5 = 0.125.
    apply_length_penalties(exps, args)
    assert torch.allclose(exps[0].rewards, torch.tensor([0.125]))
    assert torch.allclose(exps[0].info["reward"], torch.tensor([0.125]))
