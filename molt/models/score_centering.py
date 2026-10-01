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
"""Score centering (https://arxiv.org/abs/2609.20807): off-policy REINFORCE without an importance ratio.

Rollouts come from the sampler ``q`` (vLLM); the gradient is taken under the trained policy ``p``.
Instead of reweighting by ``p/q``, subtract the sampler's expected score over its top-k head ``H``::

    L = -A * ( f(p_y/q_y) log p_y  -  sum_{v in H} sg[ q_v f(p_v/q_v) - alpha p_v ] log p_v )
    alpha = rho * f(1/rho),   rho = (1 - sum_H q) / (1 - sum_H p)

``rho`` models the tail of ``q`` as rescaled ``p`` (Eq. 10). ``f`` is the per-token importance weight:
1 for vanilla score centering (Eq. 9/12), truncated / clipped / masked for the TIS and MIS variants
(Eq. 14). ``E_q[grad L] = 0`` for any ``f`` once ``H`` covers the vocabulary.
"""

import torch


def importance_weights(ratio: torch.Tensor, mode: str = "none", low=None, high=None) -> torch.Tensor:
    """``f(ratio)`` named like ``--algo.advantage.is_correction_mode``: none -> 1, trunc -> min(r, high),
    clip -> clamp(r, low, high), mask -> r if low <= r <= high else 0."""
    if mode == "none":
        return torch.ones_like(ratio)
    if mode == "trunc":
        return ratio.clamp(max=high)
    if mode == "clip":
        return ratio.clamp(min=low, max=high)
    if mode == "mask":
        return torch.where((ratio >= low) & (ratio <= high), ratio, torch.zeros_like(ratio))
    raise ValueError(f"unknown importance weight mode {mode!r}")


def score_centering_correction(
    policy_head_log_probs: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    *,
    mode: str = "none",
    low=None,
    high=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """The head term ``sum_H sg[q_v f(p_v/q_v) - alpha p_v] log p_v`` from ``[..., k]`` log-probs of the
    sampler's top-k ids under the trained policy (the only differentiable input) and under the sampler,
    both normalised over the full vocabulary. The loss adds ``advantages * correction``. Tail masses are
    floored at ``eps`` (paper App. A.3) so ``rho`` stays finite when the head covers everything."""
    with torch.no_grad():
        p, q = policy_head_log_probs.float().exp(), sampler_head_log_probs.float().exp()
        rho = (1 - q.sum(-1)).clamp_min(eps) / (1 - p.sum(-1)).clamp_min(eps)
        alpha = rho * importance_weights(rho.reciprocal(), mode, low, high)
        head_ratio = (policy_head_log_probs.float() - sampler_head_log_probs.float()).exp()
        weights = q * importance_weights(head_ratio, mode, low, high) - alpha.unsqueeze(-1) * p
    return (weights * policy_head_log_probs.float()).sum(-1)
