# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from molt.cli.train_rl_ray import _ray_runtime_env_vars


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
