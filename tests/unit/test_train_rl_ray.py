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


def test_rollout_engines_are_created_with_processed_logprobs():
    """The IS correction, score centering and a rollout-referenced KL read the engine's log-probs as
    the density the tokens were sampled from, which is post-temperature and post-mask, so the mode
    cannot depend on the rest of the configuration. It used to: with all three off the engines got
    vLLM's raw default, and mismatch/* then compared a pre-temperature log-prob against a
    post-temperature one. Asserted on the call because creating engines needs vLLM and GPUs."""
    call = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(train)))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "create_vllm_engines"
    )
    mode = next((kw.value for kw in call.keywords if kw.arg == "logprobs_mode"), None)

    assert isinstance(mode, ast.Constant), "logprobs_mode must be passed, unconditionally"
    assert mode.value == "processed_logprobs"
