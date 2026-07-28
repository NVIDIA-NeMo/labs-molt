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
    model = nn.Module()
    model.config = SimpleNamespace(use_cache=True)

    def forward(self, **kwargs):
        return kwargs

    if patched:
        forward.__module__ = "liger_kernel.transformers.qwen3"
    model.forward = MethodType(forward, model)
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
    for name, module in {
        "nemo_automodel": nemo,
        "nemo_automodel.components": ModuleType("nemo_automodel.components"),
        "nemo_automodel.components.distributed": ModuleType("nemo_automodel.components.distributed"),
        "nemo_automodel.components.distributed.config": config,
        "nemo_automodel.components.distributed.mesh": mesh,
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


@pytest.mark.parametrize(("enabled", "patched"), [(False, False), (True, True)])
def test_loader_receives_exact_liger_setting(loader_env, enabled, patched):
    state, calls = loader_env
    state["model"] = _stub_model(patched)

    base.BaseModel("qwen3", use_liger_kernel=enabled)

    assert calls[0]["use_liger_kernel"] is enabled


@pytest.mark.parametrize("unsupported", ["missing", "native", "tp", "cp", "ep", "sp", "vlm", "instance"])
def test_liger_rejects_unsupported_request_before_model_load(loader_env, monkeypatch, unsupported):
    state, calls = loader_env
    kwargs = {"use_liger_kernel": True}
    model = "qwen3"

    if unsupported == "missing":
        monkeypatch.setattr(base, "find_spec", lambda _name: None)
    elif unsupported == "native":
        state["predicted_hf"] = False
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


@pytest.mark.parametrize(("patched", "loaded_native"), [(False, False), (True, True)])
def test_liger_rejects_unpatched_or_native_loader_result(loader_env, capsys, patched, loaded_native):
    state, calls = loader_env
    state["model"] = _stub_model(patched)
    state["loaded_native"] = loaded_native

    with pytest.raises(RuntimeError, match="without observable Liger patch evidence"):
        base.BaseModel("qwen3", use_liger_kernel=True)

    assert len(calls) == 1
    assert "[Liger] enabled" not in capsys.readouterr().out


def test_liger_reports_rank_zero_success_after_patch_evidence(loader_env, capsys):
    _state, calls = loader_env

    base.BaseModel("qwen3", use_liger_kernel=True)

    assert len(calls) == 1
    assert capsys.readouterr().out.count("[Liger] enabled on HF backend.") == 1
