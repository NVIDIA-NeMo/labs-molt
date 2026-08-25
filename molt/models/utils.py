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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


def unshard_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a DTensor as a plain, unsharded tensor on every rank."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def is_automodel_custom_model(model: Any) -> bool:
    """Recognize AutoModel-native modules, including FSDP dynamic subclasses."""
    for cls in type(model).__mro__:
        if getattr(cls, "_molt_automodel_custom", False):
            return True
        if issubclass(cls, nn.Module) and cls.__module__.startswith("nemo_automodel.components.models"):
            return True
    return False


def resolve_ac_mode(value: Union[bool, str, None]) -> Union[bool, str]:
    """Normalize the gradient_checkpoint CLI value into AutoModel's
    ActivationCheckpointingMode (``bool | "selective"``).

    The flag is ``nargs="?"`` with ``const="full"``: a bare
    ``--…gradient_checkpoint`` -> ``"full"``, or an explicit mode string.
    ``"full"``/``"true"`` -> ``True`` (full-block AC, the value every
    AutoModel MoE/deepep recipe uses), ``"selective"`` -> per-op AC, and
    falsy words -> ``False``. Used by BOTH the MoE path (actor.py ->
    DistributedSetup) and the dense/HF path (strategy.py -> FSDP2Config) so they
    never disagree on the AC mode.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "selective":
            return "selective"
        return v not in ("", "false", "none", "off", "0")
    return bool(value)


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    log_ratio = torch.nan_to_num(
        log_probs.float() - log_probs_base.float(),
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    )

    if kl_estimator == "k1":
        # Signed log-ratio p - q, returned unclamped: the nan_to_num above already
        # bounds true infinities to ±30, and on-policy distillation consumes this as
        # a dense per-token reward (advantage = -kl_coef * kl) that must not be capped
        # on the most-divergent tokens (matches slime, which never clamps it). Only the
        # non-negative loss-side estimators (k2/k3) get the ±10 bound below.
        return log_ratio

    if kl_estimator == "k2":
        # Non-negative KL approximation: (p - q)^2 / 2
        # http://joschu.net/blog/kl-approx.html
        # Approximately equivalent to one-step KL penalty with k1
        # used in https://arxiv.org/pdf/2310.10505.
        log_ratio = log_ratio**2 / 2.0
    elif kl_estimator == "k3":
        # Non-negative KL approximation: exp(q - p) - 1 - (q - p)
        # http://joschu.net/blog/kl-approx.html
        log_ratio = (-log_ratio).exp() - 1 + log_ratio
    else:
        raise ValueError(f"Unknown kl_estimator: {kl_estimator}")

    return log_ratio.clamp(min=-10, max=10)


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    valid = torch.where(mask.bool(), tensor, torch.zeros_like(tensor))
    denom = mask.sum(dim=dim).clamp_min(1)
    return valid.sum(dim=dim) / denom


def configure_nemo_moe_aux_loss(model: nn.Module, aux_loss_coef: float) -> bool:
    """Use NeMo's MoE aux-loss autograd path with Molt's CLI coefficient."""
    coef = float(aux_loss_coef or 0.0)
    gates = [
        module
        for module in model.modules()
        if (
            type(module).__name__ == "Gate"
            and type(module).__module__.startswith("nemo_automodel.components.moe")
            and hasattr(module, "aux_loss_coeff")
        )
    ]
    if not gates:
        return False

    for gate in gates:
        gate.aux_loss_coeff = coef

    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "router_aux_loss_coef"):
        config.router_aux_loss_coef = coef
    for module in model.modules():
        moe_config = getattr(module, "moe_config", None)
        if moe_config is not None and hasattr(moe_config, "aux_loss_coeff"):
            moe_config.aux_loss_coeff = coef

    return coef > 0
