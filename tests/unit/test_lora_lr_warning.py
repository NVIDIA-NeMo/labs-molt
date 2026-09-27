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

"""A LoRA run that keeps a full-fine-tune LR is warned about, not left to train nothing.

AutoModel's PEFT recipes force bf16 master weights, and bf16 AdamW rounds away a step far
below the weight's ULP — which is where the default LRs sit (SFT 5e-6, RL actor 1e-6).
"""

import pytest
import torch
import torch.nn as nn

from molt.trainer.fsdp.strategy import FsdpStrategy


class _Stub(nn.Module):
    """Trainable-parameter holder; ``peft_config`` marks it as a LoRA model."""

    def __init__(self, lora: bool):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        if lora:
            self.peft_config = object()


def _strategy():
    strategy = object.__new__(FsdpStrategy)
    strategy._max_norm_by_optimizer = {}
    strategy.max_norm = 1.0
    strategy.device_mesh = None
    strategy.offload_optimizer = False
    return strategy


def _cfg(lr: float, optim: str = "adam"):
    return {
        "optim": optim,
        "adam": {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0},
        "muon": {"lr": 0.02},
        "scheduler_steps": 10,
        "max_norm": 1.0,
    }


def test_warns_when_lora_keeps_a_full_fine_tune_lr(capsys):
    _strategy()._init_train_model(_Stub(lora=True), _cfg(1e-6))
    assert "rounds AdamW updates away" in capsys.readouterr().out


def test_silent_at_a_lora_scale_lr(capsys):
    _strategy()._init_train_model(_Stub(lora=True), _cfg(1e-4))
    assert capsys.readouterr().out == ""


def test_silent_without_lora_at_the_same_lr(capsys):
    _strategy()._init_train_model(_Stub(lora=False), _cfg(1e-6))
    assert capsys.readouterr().out == ""


def test_silent_for_muon(capsys, monkeypatch):
    """Muon's own lr drives the 2D LoRA weights; the Adam lr is only the aux group."""
    import molt.trainer.fsdp.muon as muon_module

    monkeypatch.setattr(
        muon_module,
        "build_automodel_muon_optimizer",
        lambda model, *args, **kwargs: torch.optim.AdamW(list(model.parameters()), lr=0.02),
    )
    _strategy()._init_train_model(_Stub(lora=True), _cfg(1e-6, optim="muon"))
    assert capsys.readouterr().out == ""
