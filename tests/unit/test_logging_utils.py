# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

from molt.utils.logging_utils import WandbLogger


def test_wandb_eval_metrics_use_global_step(monkeypatch):
    wandb = SimpleNamespace(
        api=SimpleNamespace(api_key="already-set"),
        login=Mock(),
        init=Mock(),
        define_metric=Mock(),
        Table=Mock(return_value=SimpleNamespace(columns=[], data=[])),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", wandb)
    args = SimpleNamespace(
        logger=SimpleNamespace(
            wandb=SimpleNamespace(key="env", org=None, project="test-project", group=None, run_name="test")
        )
    )

    WandbLogger(args)

    wandb.define_metric.assert_any_call("eval/global_step")
    wandb.define_metric.assert_any_call("eval/*", step_metric="eval/global_step", step_sync=True)
