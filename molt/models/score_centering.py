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
"""Score centering (https://arxiv.org/abs/2609.20807): off-policy REINFORCE without an
importance ratio.

With rollouts sampled from ``q`` (the inference engine) and the gradient taken under the trained
policy ``p``, plain REINFORCE picks up the drift term ``E_q[grad log p] != 0``. Score centering
subtracts the sampler's expected score instead of reweighting by ``p/q``::

    L = -A * (log p_y  -  sum_{v in H} sg[q_v - rho * p_v] * log p_v)              (Eq. 9/12)

``H`` is the sampler's top-k ("head") and ``rho = (1 - sum_H q) / (1 - sum_H p)`` reconstructs the
tail of ``q`` as a rescaled ``p`` (Eq. 10). Composed with a per-token importance weight ``f``
(Eq. 14) every candidate carries its own weight and the tail folds into ``alpha``::

    L = -A * (f(p_y/q_y) log p_y  -  sum_{v in H} sg[q_v f(p_v/q_v) - alpha p_v] log p_v),
    alpha = rho * f(1/rho)

so that ``E_q[grad L] = 0`` exactly when ``H`` is the whole vocabulary, for any ``f``.

The functions here are pure tensor math on ``[..., k]`` head log-probs so the loss, the actor and
the tests share one definition.
"""

import torch

IS_WEIGHT_MODES = ("none", "mask", "clip", "trunc")


def importance_weights(ratio: torch.Tensor, mode: str = "none", low=None, high=None) -> torch.Tensor:
    """Per-token importance weight ``f(ratio)`` in the vocabulary of ``--algo.advantage.is_correction_mode``.

    ``none`` -> 1 (vanilla score centering); ``trunc`` -> ``min(r, high)`` (TIS); ``clip`` ->
    ``clamp(r, low, high)``; ``mask`` -> ``r * 1[low <= r <= high]`` (MIS). The same ``f`` must weight
    the sampled token and the head candidates, which is why it lives here and not in the loss.
    """
    if mode == "none":
        return torch.ones_like(ratio)
    if mode == "trunc":
        return ratio.clamp(max=high)
    if mode == "clip":
        return ratio.clamp(min=low, max=high)
    if mode == "mask":
        return torch.where((ratio >= low) & (ratio <= high), ratio, torch.zeros_like(ratio))
    raise ValueError(f"unknown importance weight mode {mode!r}; expected one of {IS_WEIGHT_MODES}")


def score_centering_correction(
    policy_head_log_probs: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    *,
    mode: str = "none",
    low=None,
    high=None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The head-only centering term and the detached head masses.

    Args:
        policy_head_log_probs: ``[..., k]`` log-probs of the head ids under the trained policy
            (differentiable; this is the only input the gradient flows through).
        sampler_head_log_probs: ``[..., k]`` log-probs of the SAME ids under the sampler.
            Both are normalised over the full vocabulary, not over the head.
        mode, low, high: the importance weight ``f`` (see :func:`importance_weights`).
        eps: floor on the tail masses (paper App. A.3) so ``rho`` stays finite when the head
            covers (numerically) everything.

    Returns:
        ``(correction, sampler_head_mass, policy_head_mass)``; the loss adds
        ``advantages * correction`` to the (weighted) REINFORCE term ``-advantages * f * log p_y``.
    """
    with torch.no_grad():
        p = policy_head_log_probs.float().exp()
        q = sampler_head_log_probs.float().exp()
        p_mass, q_mass = head_mass(policy_head_log_probs), head_mass(sampler_head_log_probs)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = rho * importance_weights(rho.reciprocal(), mode, low, high)
        head_ratio = (policy_head_log_probs.float() - sampler_head_log_probs.float()).exp()
        weights = q * importance_weights(head_ratio, mode, low, high) - alpha.unsqueeze(-1) * p
    correction = (weights * policy_head_log_probs.float()).sum(-1)
    return correction, q_mass, p_mass


def head_mass(head_log_probs: torch.Tensor) -> torch.Tensor:
    """Probability mass a distribution puts on its ``[..., k]`` head: ``sum_v exp(log_prob_v)``."""
    return head_log_probs.float().exp().sum(-1)
