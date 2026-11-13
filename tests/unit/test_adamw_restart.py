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

"""Native Adam restart keeps the BF16 working-weight trajectory and resume phase."""

import copy
import os
import subprocess
import sys

import pytest
import torch
from nemo_automodel.components.optim.optimizer import AdamWConfig

from molt.trainer.workers.policy_actor import _restart_adamw


def _build(rounding: str, amsgrad: bool = False) -> tuple[list[torch.nn.Parameter], torch.optim.AdamW]:
    quantized = torch.nn.Parameter((torch.arange(128).float() / 64 + 0.5).bfloat16().float())
    protected = torch.nn.Parameter(torch.full((8,), 0.1234567))
    unused = torch.nn.Parameter(torch.ones(8))
    optimizer = AdamWConfig(
        lr=0.0013,
        betas=(0.8, 0.97),
        weight_decay=0.1,
        amsgrad=amsgrad,
        weight_rounding=rounding,
        rounding_seed=7,
    ).build_from_param_groups(
        [{"params": [quantized, unused], "quantize": True}, {"params": [protected], "quantize": False}]
    )
    assert type(optimizer) is torch.optim.AdamW
    return [quantized, protected, unused], optimizer


def _advance(optimizer: torch.optim.AdamW, parameters: list[torch.nn.Parameter], steps: int) -> None:
    for step in range(steps):
        parameters[0].grad = torch.full_like(parameters[0], 0.2 + step / 16)
        parameters[1].grad = torch.full_like(parameters[1], -0.1 - step / 32)
        optimizer.step()


def _copy_state(optimizer: torch.optim.AdamW, parameters: list[torch.nn.Parameter]) -> list[dict[str, torch.Tensor]]:
    return [{key: value.clone() for key, value in optimizer.state.get(p, {}).items()} for p in parameters]


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.detach().reshape(-1).view(torch.uint8), expected.detach().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
@pytest.mark.parametrize("amsgrad", [False, True])
def test_restart_preserves_weights_master_counter_and_uses_native_adam(rounding: str, amsgrad: bool) -> None:
    parameters, optimizer = _build(rounding, amsgrad)
    _advance(optimizer, parameters, 3)
    original_weights = [parameter.detach().clone() for parameter in parameters]
    original_state = _copy_state(optimizer, parameters)
    original_storage = [(id(p), p.data_ptr()) for p in parameters]
    rng = torch.get_rng_state().clone()

    reference = [
        torch.nn.Parameter(original_state[0]["master"].clone()),
        torch.nn.Parameter(original_weights[1].clone()),
    ]
    oracle = torch.optim.AdamW(
        reference, lr=0.0013, betas=(0.8, 0.97), weight_decay=0.1, amsgrad=amsgrad, foreach=False, fused=False
    )
    assert _restart_adamw(optimizer, loaded_global_step=3, reset_at=3)
    assert [(id(p), p.data_ptr()) for p in parameters] == original_storage
    _assert_bits(torch.get_rng_state(), rng)
    for parameter, original in zip(parameters, original_weights, strict=True):
        _assert_bits(parameter, original)
    _assert_bits(optimizer.state[parameters[0]]["master"], original_state[0]["master"])
    assert parameters[2] not in optimizer.state
    for parameter in parameters[:2]:
        for key, value in optimizer.state[parameter].items():
            if key != "master":
                assert torch.count_nonzero(value) == 0
    for group in optimizer.param_groups:
        assert group["quantization_step"] == 3
        assert (group["faircast_phase"], group["faircast_phase_start_global_step"]) == (1, 3)

    for parameter, expected in zip(parameters[:2], reference, strict=True):
        parameter.grad = torch.full_like(parameter, 0.25)
        expected.grad = parameter.grad.clone()
    optimizer.step()
    oracle.step()
    _assert_bits(optimizer.state[parameters[0]]["master"], reference[0])
    _assert_bits(parameters[1], reference[1])
    for parameter, expected in zip(parameters[:2], reference, strict=True):
        for key, value in oracle.state[expected].items():
            _assert_bits(optimizer.state[parameter][key], value)
    assert [group["quantization_step"] for group in optimizer.param_groups] == [4, 4]


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_restart_phase_round_trip_is_idempotent(rounding: str) -> None:
    parameters, optimizer = _build(rounding)
    _advance(optimizer, parameters, 3)
    before = _copy_state(optimizer, parameters)
    assert not _restart_adamw(optimizer, loaded_global_step=3, reset_at=0)
    for parameter, state in zip(parameters, before, strict=True):
        for key, value in state.items():
            _assert_bits(optimizer.state[parameter][key], value)
    assert _restart_adamw(optimizer, loaded_global_step=3, reset_at=3)
    _advance(optimizer, parameters, 1)

    restored_parameters, restored = _build(rounding)
    with torch.no_grad():
        for target, source in zip(restored_parameters, parameters, strict=True):
            target.copy_(source)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    saved = _copy_state(restored, restored_parameters)
    weights = [parameter.detach().clone() for parameter in restored_parameters]
    assert not _restart_adamw(restored, loaded_global_step=4, reset_at=3)
    for parameter, original, state in zip(restored_parameters, weights, saved, strict=True):
        _assert_bits(parameter, original)
        for key, value in state.items():
            _assert_bits(restored.state[parameter][key], value)
    with pytest.raises(ValueError, match="original restart boundary"):
        _restart_adamw(restored, loaded_global_step=4, reset_at=0)
    with pytest.raises(ValueError, match="original restart boundary"):
        _restart_adamw(restored, loaded_global_step=2, reset_at=3)


@pytest.mark.parametrize("boundary", [-1, 2, 4])
def test_restart_rejects_wrong_boundary_before_mutation(boundary: int) -> None:
    parameters, optimizer = _build("stochastic")
    _advance(optimizer, parameters, 3)
    before = _copy_state(optimizer, parameters)
    with pytest.raises(ValueError):
        _restart_adamw(optimizer, loaded_global_step=3, reset_at=boundary)
    for parameter, state in zip(parameters, before, strict=True):
        for key, value in state.items():
            _assert_bits(optimizer.state[parameter][key], value)
    assert [group["faircast_phase"] for group in optimizer.param_groups] == [0, 0]


def test_restart_requires_initialized_state_but_disabled_restart_stays_lazy() -> None:
    _, optimizer = _build("stochastic")
    assert not _restart_adamw(optimizer, loaded_global_step=0, reset_at=0)
    assert not optimizer.state
    with pytest.raises(ValueError, match="initialized native Adam checkpoint state"):
        _restart_adamw(optimizer, loaded_global_step=3, reset_at=3)
    assert not optimizer.state


def test_restart_validates_every_state_before_clearing_any_moment() -> None:
    parameters, optimizer = _build("stochastic")
    _advance(optimizer, parameters, 3)
    del optimizer.state[parameters[1]]["exp_avg_sq"]
    before = _copy_state(optimizer, parameters)
    with pytest.raises(ValueError, match="incomplete or unrecognized"):
        _restart_adamw(optimizer, loaded_global_step=3, reset_at=3)
    for parameter, state in zip(parameters, before, strict=True):
        for key, value in state.items():
            _assert_bits(optimizer.state[parameter][key], value)
    assert [group["faircast_phase"] for group in optimizer.param_groups] == [0, 0]


@pytest.mark.parametrize(
    "flags,model_only,message",
    [
        (["--actor.reset_optimizer_at_step", "-1"], False, "must be nonnegative"),
        (["--actor.reset_optimizer_at_step", "3"], False, "full native checkpoint loading"),
        (
            ["--actor.reset_optimizer_at_step", "3", "--ckpt.load_enable"],
            False,
            "BF16 weight rounding",
        ),
        (
            [
                "--actor.reset_optimizer_at_step",
                "3",
                "--ckpt.load_enable",
                "--actor.adam.weight_rounding",
                "stochastic",
            ],
            True,
            "full native checkpoint loading",
        ),
    ],
)
def test_cli_restart_requires_full_native_rounded_checkpoint(flags: list[str], model_only: bool, message: str) -> None:
    env = {**os.environ, "LOAD_MODEL_ONLY": "1" if model_only else "0"}
    result = subprocess.run(
        [sys.executable, "-m", "molt.cli.train_rl_ray", *flags],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr
