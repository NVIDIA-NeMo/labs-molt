# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from molt.cli.common_args import validate_ep_gradient_checkpoint


def test_ep1_full_allowed():
    validate_ep_gradient_checkpoint(1, "full")


def test_ep_gt1_full_hybridep_rejected(monkeypatch):
    monkeypatch.setenv("MOLT_MOE_DISPATCHER", "hybridep")
    with pytest.raises(ValueError, match="incompatible with expert-parallel"):
        validate_ep_gradient_checkpoint(4, "full")


def test_ep_gt1_full_deepep_rejected(monkeypatch):
    monkeypatch.setenv("MOLT_MOE_DISPATCHER", "deepep")
    with pytest.raises(ValueError, match="incompatible with expert-parallel"):
        validate_ep_gradient_checkpoint(8, "full")


def test_ep_gt1_none_allowed(monkeypatch):
    monkeypatch.setenv("MOLT_MOE_DISPATCHER", "hybridep")
    validate_ep_gradient_checkpoint(4, "none")


def test_ep_gt1_full_torch_allowed(monkeypatch):
    monkeypatch.setenv("MOLT_MOE_DISPATCHER", "torch")
    validate_ep_gradient_checkpoint(8, "full")


def test_ep_gt1_full_default_dispatcher_rejected(monkeypatch):
    monkeypatch.delenv("MOLT_MOE_DISPATCHER", raising=False)
    with pytest.raises(ValueError, match="incompatible with expert-parallel"):
        validate_ep_gradient_checkpoint(2, "full")


def test_dense_model_full_allowed(monkeypatch):
    monkeypatch.delenv("MOLT_MOE_DISPATCHER", raising=False)
    validate_ep_gradient_checkpoint(1, "full")


def test_ep_gt1_selective_allowed(monkeypatch):
    monkeypatch.setenv("MOLT_MOE_DISPATCHER", "hybridep")
    validate_ep_gradient_checkpoint(4, "selective")


def test_ep_gt1_bool_true_rejected(monkeypatch):
    monkeypatch.setenv("MOLT_MOE_DISPATCHER", "hybridep")
    with pytest.raises(ValueError, match="incompatible with expert-parallel"):
        validate_ep_gradient_checkpoint(4, True)


def test_ep_gt1_bool_false_allowed(monkeypatch):
    monkeypatch.setenv("MOLT_MOE_DISPATCHER", "hybridep")
    validate_ep_gradient_checkpoint(4, False)
