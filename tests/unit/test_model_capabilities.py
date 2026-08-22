# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

from molt.models.base import _automodel_supports_thd_packing


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
