# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

import pytest

from molt.cli.train_rl_ray import _ray_runtime_env_vars, _rollout_logprobs_mode, train


def test_ray_runtime_env_forwards_wandb_settings(monkeypatch):
    expected = {
        "WANDB_API_KEY": "test-api-key",
        "WANDB_ENTITY": "test-entity",
        "WANDB_MODE": "offline",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    env_vars = _ray_runtime_env_vars()

    assert {name: env_vars[name] for name in expected} == expected


def test_ray_runtime_env_omits_empty_optional_settings(monkeypatch):
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)

    env_vars = _ray_runtime_env_vars()

    assert not {"WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE"} & env_vars.keys()


@pytest.mark.parametrize(
    "temperature,expected",
    [
        (1.0, None),  # raw: the full vocabulary, untouched by temperature or any sampling mask
        (0.8, "processed_logprobs"),
        (1.2, "processed_logprobs"),
    ],
)
def test_rollout_logprobs_mode_follows_the_training_forward(temperature, expected):
    """The forward recomputes the sampled token's log-prob over the full vocabulary at
    --rollout.temperature. vLLM's raw log-probs are that vocabulary before temperature, so they match
    exactly at 1; its processed ones are after temperature but also after top_p/top_k and the
    min_tokens stop mask, so they are the closer match only away from 1."""
    assert _rollout_logprobs_mode(temperature) == expected


def test_rollout_engines_are_created_with_that_mode():
    """The engines have to be created with the mode above. They used to get vLLM's raw default unless
    the IS correction, score centering or a rollout-referenced KL asked for the processed one, which
    left mismatch/* comparing a pre-temperature log-prob against a post-temperature one. Asserted on
    the call because creating engines needs vLLM and GPUs."""
    call = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(train)))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "create_vllm_engines"
    )
    mode = next((kw.value for kw in call.keywords if kw.arg == "logprobs_mode"), None)

    assert isinstance(mode, ast.Call), "logprobs_mode must come from _rollout_logprobs_mode"
    assert mode.func.id == "_rollout_logprobs_mode"
