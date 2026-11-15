# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

from molt.cli.train_rl_ray import _ray_runtime_env_vars, train


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


def test_rollout_engines_are_asked_for_post_temperature_logprobs():
    """Every rollout records the sampled token's log-prob, and the training forward recomputes it at
    --rollout.temperature, so mismatch/* compares like with like only when vLLM returns processed
    (post-temperature) log-probs; its own default is raw. The engines used to get the raw default
    unless the IS correction, score centering or a rollout-referenced KL asked otherwise. Asserted on
    the call rather than by running it: creating engines needs vLLM and GPUs."""
    call = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(train)))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "create_vllm_engines"
    )
    mode = next((kw.value for kw in call.keywords if kw.arg == "logprobs_mode"), None)

    assert isinstance(mode, ast.Constant), "logprobs_mode must be passed, unconditionally"
    assert mode.value == "processed_logprobs"
