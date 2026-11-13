# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoLT configuration and checkpoint boundaries for AutoModel BF16 rounding."""

import argparse
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from molt.cli.common_args import add_optimizer_args
from molt.trainer.fsdp.checkpoint import CheckpointManager
from molt.trainer.fsdp.strategy import FsdpStrategy


def _build(model, *, rounding=None, kind="adam", offload="none", seed=73):
    strategy = FsdpStrategy(seed=seed, args=SimpleNamespace(fsdp=SimpleNamespace(offload=offload)))
    _, optimizer, scheduler = strategy.prepare(
        (
            model,
            {
                "optim": kind,
                "adam": {
                    "lr": 6e-6,
                    "betas": (0.9, 0.98),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "weight_rounding": rounding,
                },
                "scheduler_steps": 8,
                "lr_warmup_ratio": 0,
            },
        )
    )
    return strategy, optimizer, scheduler


@pytest.mark.parametrize("prefix", ["", "actor."])
@pytest.mark.parametrize("alias", ["weight_rounding", "rounding"])
def test_cli_rounding_aliases_and_disabled_default(prefix, alias):
    parser = argparse.ArgumentParser()
    add_optimizer_args(parser, prefix=prefix)
    defaults = vars(parser.parse_args([]))
    assert defaults[f"{prefix}optim"] == "adam"
    assert defaults[f"{prefix}adam.weight_rounding"] is None
    args = vars(
        parser.parse_args(
            [
                f"--{prefix}optim",
                "faircast_adamw",
                f"--{prefix}adam.{alias}",
                "stochastic",
            ]
        )
    )
    assert args[f"{prefix}optim"] == "faircast_adamw"
    assert args[f"{prefix}adam.weight_rounding"] == "stochastic"
    assert f"{prefix}adam.rounding" not in args


def test_disabled_rounding_matches_plain_native_adamw():
    model = nn.Linear(4, 4)
    reference = copy.deepcopy(model)
    _, optimizer, _ = _build(model)
    native = torch.optim.AdamW(
        reference.parameters(),
        lr=6e-6,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    assert type(optimizer) is torch.optim.AdamW
    assert not optimizer.state
    assert "rounding" not in optimizer.param_groups[0]
    for step in range(3):
        for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
            parameter.grad = torch.full_like(parameter, (step + 1) / 7)
            other.grad = parameter.grad.clone()
        optimizer.step()
        native.step()
        for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
            assert torch.equal(parameter, other)
            assert optimizer.state[parameter].keys() == native.state[other].keys()
            for key in optimizer.state[parameter]:
                assert torch.equal(optimizer.state[parameter][key], native.state[other][key])


@pytest.mark.parametrize("metadata", ["strict", "regular_set", "absent"])
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_rounding_keeps_protected_parameters_and_native_updates(metadata, rounding):
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(8192))
    model.sensitive = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.sensitive.weight.fill_(1)
    if metadata == "strict":
        model._keep_in_fp32_modules_strict = ["sensitive"]
    elif metadata == "regular_set":
        model._keep_in_fp32_modules = {"sensitive"}
    _, optimizer, _ = _build(model, rounding=rounding)
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["rounding_seed"] == 73
    assert optimizer.param_groups[0]["params"][0] is model.weight
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    rng = torch.get_rng_state().clone()
    optimizer.step()
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(model.weight, model.weight.bfloat16().float())
    assert not torch.equal(optimizer.state[model.weight]["master"], model.weight)
    if metadata == "absent":
        assert len(optimizer.param_groups) == 1
        assert "master" in optimizer.state[model.sensitive.weight]
    else:
        assert optimizer.param_groups[1]["params"][0] is model.sensitive.weight
        assert optimizer.param_groups[1]["quantize"] is False
        assert "master" not in optimizer.state[model.sensitive.weight]
        assert not torch.equal(model.sensitive.weight, model.sensitive.weight.bfloat16().float())


def test_legacy_optimizer_name_defaults_to_nearest():
    _, optimizer, _ = _build(nn.Linear(4, 4), kind="faircast_adamw")
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["rounding"] == "nearest"


def test_rounding_rejects_muon_optimizer_offload_and_bf16_storage():
    with pytest.raises(ValueError, match="requires the AdamW optimizer"):
        _build(nn.Linear(4, 4), kind="muon", rounding="stochastic")
    with pytest.raises(NotImplementedError, match="optimizer-only offload"):
        _build(nn.Linear(4, 4), rounding="stochastic", offload="optimizer")
    with pytest.raises(TypeError, match="FP32 parameter storage"):
        _build(nn.Linear(4, 4).bfloat16(), rounding="stochastic")


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_cli_reapplication_preserves_loaded_rounding_state(rounding):
    model = nn.Linear(4, 4)
    _, optimizer, scheduler = _build(model, rounding=rounding)
    flags = copy.deepcopy([{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups])
    for _ in range(3):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    group = optimizer.param_groups[0]
    group["faircast_phase"] = 1
    group["faircast_phase_start_global_step"] = 5000
    masters = [optimizer.state[p]["master"].clone() for p in model.parameters()]
    flags[0]["initial_lr"] *= 2
    flags[0]["weight_decay"] = 0.01
    CheckpointManager._reapply_cli_optimizer_hyperparams(optimizer, scheduler, flags)
    assert group["quantization_step"] == 3
    assert group["faircast_phase"] == 1
    assert group["faircast_phase_start_global_step"] == 5000
    assert group["lr"] == 1.2e-5
    assert group["weight_decay"] == 0.01
    for parameter, master in zip(model.parameters(), masters, strict=True):
        assert torch.equal(optimizer.state[parameter]["master"], master)


@pytest.mark.parametrize("key,new_value", [("rounding", "nearest"), ("rounding_seed", 99)])
def test_resume_rejects_changed_rounding_mode_or_seed(key, new_value):
    _, optimizer, scheduler = _build(nn.Linear(4, 4), rounding="stochastic")
    flags = copy.deepcopy([{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups])
    flags[0][key] = new_value
    with pytest.raises(ValueError, match="mode and seed must match"):
        CheckpointManager._reapply_cli_optimizer_hyperparams(optimizer, scheduler, flags)


def test_rounded_checkpoint_missing_master_is_rejected(tmp_path, monkeypatch):
    """Exercise the real DCP optimizer planner; model loading is mocked in this unit test."""
    import torch.distributed.checkpoint as dcp
    from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
    from torch.distributed.checkpoint.api import CheckpointException
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    model = nn.Linear(4, 4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1)
    strategy, optimizer, _ = _build(model, rounding="stochastic")
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    saved = OptimizerState(model, optimizer).state_dict()
    master_keys = [key for key in saved["optim"] if key.endswith(".master")]
    assert master_keys
    saved["optim"] = {key: value for key, value in saved["optim"].items() if key not in master_keys}
    checkpoint_dir = tmp_path / "checkpoint"
    dcp.save(saved, checkpoint_id=checkpoint_dir / "optim")

    restored_model = copy.deepcopy(model)
    _, restored, _ = _build(restored_model, rounding="stochastic")
    skeleton = OptimizerState(restored_model, restored).state_dict()
    before = {key: skeleton["optim"][key].clone() for key in master_keys}
    # Negative control: the former partial-load policy silently keeps placeholder masters.
    dcp.load(skeleton, checkpoint_id=checkpoint_dir / "optim", planner=DefaultLoadPlanner(allow_partial_load=True))
    for key in master_keys:
        assert torch.equal(skeleton["optim"][key], before[key])
    manager = CheckpointManager(strategy)
    monkeypatch.setattr(manager, "_resolve_ckpt_load_dir", lambda _: str(checkpoint_dir))
    monkeypatch.setattr(manager, "_build_checkpointer", lambda *args, **kwargs: MagicMock())
    with pytest.raises(CheckpointException, match="master"):
        manager.load_ckpt(restored_model, str(tmp_path), optimizer=restored)


@pytest.mark.parametrize(
    "saved_rounding,configured_rounding",
    [("nearest", None), ("stochastic", None), (None, "nearest"), (None, "stochastic")],
)
def test_checkpoint_rejects_enabling_or_omitting_rounding(tmp_path, monkeypatch, saved_rounding, configured_rounding):
    """Real DCP optimizer metadata must reject mode changes; model loading is mocked."""
    import torch.distributed.checkpoint as dcp
    from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    model = nn.Linear(4, 4)
    strategy, optimizer, _ = _build(model, rounding=saved_rounding)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    checkpoint_dir = tmp_path / "checkpoint"
    dcp.save(OptimizerState(model, optimizer).state_dict(), checkpoint_id=checkpoint_dir / "optim")

    restored_model = copy.deepcopy(model)
    _, restored, _ = _build(restored_model, rounding=configured_rounding)
    if configured_rounding is None:
        # Negative control: the former partial load ignores all saved rounding/master keys.
        optimizer_state = OptimizerState(restored_model, restored)
        skeleton = optimizer_state.state_dict()
        dcp.load(skeleton, checkpoint_id=checkpoint_dir / "optim", planner=DefaultLoadPlanner(allow_partial_load=True))
        optimizer_state.load_state_dict(skeleton)
        assert all("rounding" not in group for group in restored.param_groups)
        assert all("master" not in restored.state[parameter] for parameter in restored_model.parameters())
        assert all(restored.state[parameter]["step"].item() == 1 for parameter in restored_model.parameters())

    manager = CheckpointManager(strategy)
    monkeypatch.setattr(manager, "_resolve_ckpt_load_dir", lambda _: str(checkpoint_dir))
    monkeypatch.setattr(manager, "_build_checkpointer", lambda *args, **kwargs: MagicMock())
    with pytest.raises(ValueError, match="BF16 weight rounding setting must match"):
        manager.load_ckpt(restored_model, str(tmp_path), optimizer=restored)
