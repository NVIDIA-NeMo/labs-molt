# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from torch import nn

from molt.models.base import BaseModel, _automodel_supports_thd_packing


class _DeclaredTHDModel(nn.Module):
    _molt_automodel_custom = True

    class ModelCapabilities:
        supports_tp = False
        supports_cp = False
        supports_pp = False
        supports_ep = False
        supports_thd = True


def test_thd_support_uses_automodel_capability_declaration():
    assert _automodel_supports_thd_packing(_DeclaredTHDModel())


def test_non_automodel_module_never_claims_thd_support():
    assert not _automodel_supports_thd_packing(nn.Linear(2, 2))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"packing_samples": True}, "THD sequence packing"),
        ({"moe_aux_loss_coef": 0.01}, "MoE auxiliary loss"),
    ],
)
def test_hf_fallback_features_fail_fast(kwargs, message):
    with pytest.raises(NotImplementedError, match=message):
        BaseModel(nn.Linear(2, 2), **kwargs)


def test_hf_fallback_moe_model_fails_fast_without_aux_loss_or_ep():
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(
        architectures=["MixtralForCausalLM"],
        text_config=SimpleNamespace(num_local_experts=8),
    )
    with pytest.raises(NotImplementedError, match="MoE model training"):
        BaseModel(model)


def test_automodel_full_cpu_offload_is_allowed_for_dense_model():
    wrapped = BaseModel(
        _DeclaredTHDModel(),
        distributed_config=SimpleNamespace(offload_policy=object()),
    )

    assert isinstance(wrapped.model, _DeclaredTHDModel)


def test_automodel_full_cpu_offload_is_allowed_for_custom_moe():
    model = _DeclaredTHDModel()
    model.config = SimpleNamespace(num_local_experts=8)

    wrapped = BaseModel(model, distributed_config=SimpleNamespace(offload_policy=object()))

    assert wrapped.model is model
