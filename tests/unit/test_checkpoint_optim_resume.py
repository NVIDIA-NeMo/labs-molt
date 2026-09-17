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

"""``load_ckpt`` must go through the same AutoModel Checkpointer as ``save_ckpt``.

Save uses ``Checkpointer.save_optimizer`` which picks a state-dict shape from
``is_peft`` + ``has_expert_parallelism`` (PEFT+EP writes the *native* AdamW state
dict; everything else writes DCP). A prior hand-rolled ``dcp.load`` on load bypassed
those flags and always requested the DCP shape, so under PEFT+EP the on-disk keys
never matched and ``allow_partial_load=True`` silently returned zero moments — a
resumed AdamW that carried no momentum. The fix routes load through the same
``Checkpointer.load_optimizer`` API. This test pins that wiring: patch
``_build_checkpointer`` and assert ``load_ckpt`` invokes ``load_optimizer`` (with
model + optimizer + scheduler + weights_path) instead of hand-rolling a DCP call.
"""

import os

import torch

from molt.trainer.fsdp.checkpoint import CheckpointManager


class _FakeStrategy:
    def __init__(self):
        self.offloaded = []

    def _unwrap_model(self, model):
        return model

    def offload_moments_to_cpu(self, optimizer):
        self.offloaded.append(optimizer)

    def print(self, *msg):
        pass


class _SpyCheckpointer:
    """Records save/load calls for both halves of the round-trip."""

    def __init__(self):
        self.calls = []

    def save_model(self, **kwargs):
        self.calls.append(("save_model", kwargs))

    def save_optimizer(self, **kwargs):
        self.calls.append(("save_optimizer", kwargs))

    def load_model(self, **kwargs):
        self.calls.append(("load_model", kwargs))

    def load_optimizer(self, **kwargs):
        self.calls.append(("load_optimizer", kwargs))


def _prepare_ckpt(tmp_path, tag: str = "step-1"):
    """Fake a loadable DCP checkpoint layout (``extra_state.pt`` + ``model/`` + ``optim/``)."""
    ckpt_dir = os.path.join(tmp_path, tag)
    os.makedirs(os.path.join(ckpt_dir, "model"))
    os.makedirs(os.path.join(ckpt_dir, "optim"))
    torch.save({"client_state": {"global_step": 1}}, os.path.join(ckpt_dir, "extra_state.pt"))
    with open(os.path.join(tmp_path, "latest"), "w") as f:
        f.write(tag)
    return ckpt_dir


def test_load_ckpt_delegates_optimizer_load_to_checkpointer(tmp_path, monkeypatch):
    """Regression: load must call ``Checkpointer.load_optimizer`` (symmetric with save),
    not a hand-rolled ``dcp.load`` that ignores ``is_peft`` / ``has_expert_parallelism``.
    Under PEFT+EP those flags gate the native vs DCP state-dict shape; a mismatch
    silently restores zero AdamW moments (exp_avg / exp_avg_sq)."""
    _prepare_ckpt(str(tmp_path))

    strategy = _FakeStrategy()
    cm = CheckpointManager(strategy)
    spy = _SpyCheckpointer()
    monkeypatch.setattr(cm, "_build_checkpointer", lambda *a, **kw: spy)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = object()  # opaque; only threaded through unchanged

    load_dir, states = cm.load_ckpt(model, str(tmp_path), optimizer=optimizer, scheduler=scheduler)

    assert load_dir is not None and states == {"global_step": 1}
    kinds = [name for name, _ in spy.calls]
    assert kinds == ["load_model", "load_optimizer"]
    load_optimizer_kwargs = spy.calls[1][1]
    assert load_optimizer_kwargs["optimizer"] is optimizer
    assert load_optimizer_kwargs["model"] is model
    assert load_optimizer_kwargs["scheduler"] is scheduler
    assert load_optimizer_kwargs["weights_path"] == load_dir
    assert strategy.offloaded == [optimizer]  # still pages moments back to CPU under --fsdp.offload


def test_load_ckpt_skips_optimizer_when_optim_dir_missing(tmp_path, monkeypatch):
    """Sanity: a checkpoint that only holds model weights (no ``optim/``) must not
    invoke ``load_optimizer``; the pre-fix path had the same guard and we keep it."""
    ckpt_dir = os.path.join(str(tmp_path), "step-1")
    os.makedirs(os.path.join(ckpt_dir, "model"))
    torch.save({"client_state": {}}, os.path.join(ckpt_dir, "extra_state.pt"))
    with open(os.path.join(str(tmp_path), "latest"), "w") as f:
        f.write("step-1")

    strategy = _FakeStrategy()
    cm = CheckpointManager(strategy)
    spy = _SpyCheckpointer()
    monkeypatch.setattr(cm, "_build_checkpointer", lambda *a, **kw: spy)

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cm.load_ckpt(model, str(tmp_path), optimizer=optimizer)

    assert [name for name, _ in spy.calls] == ["load_model"]
    assert strategy.offloaded == []
