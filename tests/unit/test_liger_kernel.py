# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from molt.cli.common_args import add_fsdp_args
from molt.models import base


def _stub_model(patched: bool):
    class Qwen3MLP(nn.Module):
        pass

    class Qwen3RMSNorm(nn.Module):
        pass

    model = nn.Module()
    model.config = SimpleNamespace(use_cache=True, model_type="qwen3")
    model.mlp = Qwen3MLP()
    model.norm = Qwen3RMSNorm()

    def forward(self, **kwargs):
        return kwargs

    model.forward = MethodType(forward, model)
    for module in (model.mlp, model.norm):
        module.forward = MethodType(forward, module)
    if patched:
        for module in (model.mlp, model.norm):
            module.forward.__func__.__module__ = "liger_kernel.transformers.qwen3"
    return model


def _mesh(tp_size=1, cp_size=1):
    mesh = MagicMock()
    mesh.mesh_dim_names = ("tp", "cp")
    sizes = {"tp": tp_size, "cp": cp_size}
    mesh.__getitem__.side_effect = lambda name: SimpleNamespace(size=lambda: sizes[name])
    return mesh


@pytest.fixture
def loader_env(monkeypatch):
    state = {
        "model": _stub_model(patched=True),
        "predicted_hf": True,
        "loaded_native": False,
        "is_vlm": False,
    }
    calls = []

    class StubLoader:
        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            calls.append(kwargs)
            return state["model"]

    nemo = ModuleType("nemo_automodel")
    nemo.NeMoAutoModelForCausalLM = StubLoader
    nemo.NeMoAutoModelForImageTextToText = StubLoader
    config = ModuleType("nemo_automodel.components.distributed.config")
    config.DistributedSetup = lambda **kwargs: SimpleNamespace(**kwargs)
    mesh = ModuleType("nemo_automodel.components.distributed.mesh")
    mesh.MeshContext = SimpleNamespace(from_meshes=lambda *meshes: meshes)
    liger_patch = ModuleType("liger_kernel.transformers.monkey_patch")

    def apply_liger_kernel_to_qwen3(*, model, **_kwargs):
        for module in model.modules():
            if type(module).__name__ in {"Qwen3MLP", "Qwen3RMSNorm"}:
                module.forward.__func__.__module__ = "liger_kernel.transformers.qwen3"

    liger_patch.apply_liger_kernel_to_qwen3 = apply_liger_kernel_to_qwen3
    for name, module in {
        "nemo_automodel": nemo,
        "nemo_automodel.components": ModuleType("nemo_automodel.components"),
        "nemo_automodel.components.distributed": ModuleType("nemo_automodel.components.distributed"),
        "nemo_automodel.components.distributed.config": config,
        "nemo_automodel.components.distributed.mesh": mesh,
        "liger_kernel": ModuleType("liger_kernel"),
        "liger_kernel.transformers": ModuleType("liger_kernel.transformers"),
        "liger_kernel.transformers.monkey_patch": liger_patch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    from molt.utils import utils

    monkeypatch.setattr(base, "find_spec", lambda _name: object())
    monkeypatch.setattr(base, "_detect_moe_arch", lambda _path: False)
    monkeypatch.setattr(base, "_will_use_hf_model", lambda _path: state["predicted_hf"])
    monkeypatch.setattr(base, "_validate_attn_implementation", lambda _attn: None)
    monkeypatch.setattr(base, "_mtp_off_kwargs", lambda _path: {})
    monkeypatch.setattr(base, "move_model_to_cpu_for_offload", lambda model, _config: model)
    monkeypatch.setattr(base, "configure_nemo_moe_aux_loss", lambda *_args: None)
    monkeypatch.setattr(base, "is_automodel_custom_model", lambda _model: state["loaded_native"])
    monkeypatch.setattr(utils, "is_vlm_model", lambda _path: state["is_vlm"])
    return state, calls


def test_shared_parser_defaults_off_and_accepts_opt_in():
    parser = argparse.ArgumentParser()
    add_fsdp_args(parser)

    assert vars(parser.parse_args([]))["fsdp.use_liger_kernel"] is False
    assert vars(parser.parse_args(["--fsdp.use_liger_kernel"]))["fsdp.use_liger_kernel"] is True


@pytest.mark.parametrize(("enabled", "patched"), [(False, False), (True, False)])
def test_loader_defers_liger_patching_until_after_load(loader_env, enabled, patched):
    state, calls = loader_env
    state["model"] = _stub_model(patched)

    base.BaseModel("qwen3", use_liger_kernel=enabled)

    assert calls[0]["use_liger_kernel"] is False


@pytest.mark.parametrize("unsupported", ["missing", "tp", "cp", "ep", "sp", "vlm", "instance"])
def test_liger_rejects_unsupported_request_before_model_load(loader_env, monkeypatch, unsupported):
    state, calls = loader_env
    kwargs = {"use_liger_kernel": True}
    model = "qwen3"

    if unsupported == "missing":
        monkeypatch.setattr(base, "find_spec", lambda _name: None)
    elif unsupported == "tp":
        kwargs["device_mesh"] = _mesh(tp_size=2)
    elif unsupported == "cp":
        kwargs["device_mesh"] = _mesh(cp_size=2)
    elif unsupported == "ep":
        kwargs["moe_mesh"] = object()
    elif unsupported == "sp":
        kwargs["distributed_config"] = SimpleNamespace(sequence_parallel=True)
    elif unsupported == "vlm":
        state["is_vlm"] = True
    else:
        model = nn.Linear(2, 2)

    with pytest.raises((RuntimeError, ValueError), match="use_liger_kernel"):
        base.BaseModel(model, **kwargs)

    assert calls == []


def test_liger_rejects_missing_qwen3_layers(loader_env, capsys):
    state, calls = loader_env
    state["model"] = nn.Module()
    state["model"].config = SimpleNamespace(use_cache=True, model_type="qwen3")

    with pytest.raises(RuntimeError, match="did not patch every Qwen3"):
        base.BaseModel("qwen3", use_liger_kernel=True)

    assert len(calls) == 1
    assert "[Liger] enabled" not in capsys.readouterr().out


def test_liger_reports_rank_zero_success_after_patch_evidence(loader_env, capsys):
    _state, calls = loader_env

    base.BaseModel("qwen3", use_liger_kernel=True)

    assert len(calls) == 1
    assert capsys.readouterr().out.count("[Liger] enabled on hf Qwen3 backend.") == 1
