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

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .utils import masked_mean


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Masked sum.

    NaNs outside the mask are zeroed before summation so padding-only garbage
    cannot contaminate the result.
    """
    valid_values = torch.where(mask.bool(), values, 0.0)
    return valid_values.sum(dim=dim)


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[torch.Tensor | int] = None,
    global_batch_size: Optional[torch.Tensor | int] = None,
    loss_scale_factor: Optional[int] = None,
) -> torch.Tensor:
    """Aggregate token/sequence losses following the ``agg_loss`` contract.

    The returned scalar is invariant to DP/FSDP averaging when callers provide
    global batch metadata. For ``token-mean`` this is exactly:
    ``masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size``.
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss_sum = masked_sum(loss_mat, loss_mask)
        denom = torch.as_tensor(batch_num_tokens, device=loss_sum.device, dtype=loss_sum.dtype)
        if denom.item() == 0:
            return loss_sum * 0.0
        return loss_sum / denom * dp_size

    if loss_agg_mode in {"seq-mean-token-sum", "seq-mean-token-sum-norm"}:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss_sum = masked_sum(seq_losses, seq_mask)
        denom = torch.as_tensor(global_batch_size, device=loss_sum.device, dtype=loss_sum.dtype)
        if denom.item() == 0:
            return loss_sum * 0.0
        loss = loss_sum / denom * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss /= loss_scale_factor if loss_scale_factor is not None else loss_mask.shape[-1]
        return loss

    if loss_agg_mode == "seq-mean-token-mean":
        seq_token_counts = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_token_counts + 1e-8)
        seq_mask = (seq_token_counts > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss_sum = masked_sum(seq_losses, seq_mask)
        denom = torch.as_tensor(global_batch_size, device=loss_sum.device, dtype=loss_sum.dtype)
        if denom.item() == 0:
            return loss_sum * 0.0
        return loss_sum / denom * dp_size

    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


class SFTLoss(nn.Module):
    """
    SFT Loss
    """

    def __init__(self, token_level_loss: bool = True, loss_agg_mode: str = "token-mean"):
        super().__init__()
        self.token_level_loss = token_level_loss
        self.loss_agg_mode = loss_agg_mode

    def forward(
        self,
        per_token_logps: torch.Tensor,
        loss_mask: torch.Tensor,
        dp_size: int = 1,
        batch_num_tokens: Optional[torch.Tensor] = None,
        global_batch_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.token_level_loss:
            return agg_loss(
                -per_token_logps,
                loss_mask,
                self.loss_agg_mode,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
            )
        return masked_mean(-per_token_logps, loss_mask, dim=-1).mean()


class ValueLoss(nn.Module):
    """Clipped value-function loss for PPO.

    Regresses the critic's current ``values`` onto the GAE targets ``returns``, with
    the prediction symmetrically clipped around the collection-time ``old_values`` and
    the larger of the clipped/unclipped squared error taken, scaled by ``0.5`` (the
    standard PPO / verl value-clip objective). Inputs live on the dense next-token step
    axis; ``action_mask`` selects the value-bearing positions, and aggregation reuses
    the same DP/FSDP-invariant ``agg_loss`` contract as ``PolicyLoss``.
    """

    def __init__(self, value_clip: Optional[float] = 0.2, loss_agg_mode: str = "token-mean") -> None:
        super().__init__()
        self.value_clip = value_clip
        self.loss_agg_mode = loss_agg_mode

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[torch.Tensor] = None,
        global_batch_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        values = values.float()
        returns = returns.float()
        surr_unclipped = (values - returns).pow(2)
        if self.value_clip is not None:
            old_values = old_values.float()
            values_clipped = old_values + (values - old_values).clamp(-self.value_clip, self.value_clip)
            surr_clipped = (values_clipped - returns).pow(2)
            loss_mat = 0.5 * torch.max(surr_unclipped, surr_clipped)
            # Fraction of positions where the prediction was clipped (for logging).
            clip_frac = masked_mean((torch.abs(values - old_values) > self.value_clip).float(), action_mask, dim=None)
        else:
            loss_mat = 0.5 * surr_unclipped
            clip_frac = None

        reported_loss = masked_mean(loss_mat.detach(), action_mask, dim=None)
        loss = agg_loss(
            loss_mat,
            action_mask,
            self.loss_agg_mode,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )
        return loss, reported_loss, clip_frac


# ──────────────── policy-loss surrogates (registry, mirrors advantage.py) ────────────────
# A surrogate maps the per-token IS ``ratio`` + advantages to a per-token loss and the
# clip-fraction metric; PolicyLoss.forward owns everything shared (ratio, IS correction,
# aggregation). Register a new surrogate the same way advantage estimators register, so it
# plugs in without editing forward:  @register_policy_loss("gspo") def ...(...).
PolicyLossFn = Callable[..., Tuple[torch.Tensor, torch.Tensor]]
POLICY_LOSSES: Dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy-loss surrogate under `name`."""

    def decorator(fn: PolicyLossFn) -> PolicyLossFn:
        if name in POLICY_LOSSES and POLICY_LOSSES[name] is not fn:
            raise ValueError(f"Policy loss '{name}' is already registered")
        POLICY_LOSSES[name] = fn
        return fn

    return decorator


def get_policy_loss(name: str) -> PolicyLossFn:
    if name not in POLICY_LOSSES:
        raise ValueError(f"Unknown policy loss '{name}'. Registered: {sorted(POLICY_LOSSES)}")
    return POLICY_LOSSES[name]


@register_policy_loss("ppo")
def ppo_policy_loss(ratio, advantages, log_probs, action_mask, *, clip_eps_low, clip_eps_high, dual_clip, **_):
    """Clipped PPO surrogate `-min(r·A, clip(r)·A)`, optionally dual-clipped for A<0."""
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
    if dual_clip is None:
        loss = -torch.min(surr1, surr2)
    else:
        clip1 = torch.min(surr1, surr2)
        # Dual-clip: extra lower bound for negative advantages (clip2 for A<0, clip1 for A>=0).
        clip2 = torch.max(clip1, dual_clip * advantages)
        loss = -torch.where(advantages < 0, clip2, clip1)
    clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
    return loss, clip_ratio


@register_policy_loss("cispo")
def cispo_policy_loss(ratio, advantages, log_probs, action_mask, *, clip_eps_high, **_):
    """CISPO (https://arxiv.org/abs/2506.13585): stop-gradient upper-clipped IS weight,
    REINFORCE through log-probs so clipped tokens still contribute gradient."""
    clipped_ratio = ratio.clamp_max(clip_eps_high).detach()
    loss = -clipped_ratio * advantages * log_probs
    clip_ratio = masked_mean((ratio > clip_eps_high).float(), action_mask, dim=None)
    return loss, clip_ratio


class PolicyLoss(nn.Module):
    """
    Clipped policy-gradient loss for non-critic RL.

    Inputs live on the dense next-token step axis. For multi-turn trajectories,
    action_mask selects generated action tokens and excludes observation/tool
    feedback slots without changing tensor alignment.
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        is_correction_threshold: list = None,
        is_correction_level: str = "off",
        is_correction_mode: str = "mask",
        loss_agg_mode: str = "token-mean",
        loss_mode: str = "ppo",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        # Train/rollout (FSDP-actor vs vLLM) logprob-mismatch correction, applied to
        # the per-token off-policy IS ratio pi_train/pi_rollout. Two knobs:
        #   is_correction_level: granularity of the ratio that gets gated —
        #     off   -> correction disabled
        #     token -> each token's own ratio (robust to a single outlier token)
        #     seq   -> product of a sequence's token ratios = exp(sum) (unbiased, high var)
        #     geo   -> geometric mean = exp(mean) (per-seq, balanced bias/variance)
        #   is_correction_mode: what to do with a unit outside [low, high] bounds —
        #     mask  -> drop it (zero gradient); keep the per-token IS weight otherwise
        #     clip  -> keep it, clamp its weight into [low, high]
        #     trunc -> keep it, clamp only the upper tail (small weights unchanged)
        self.is_correction_threshold = is_correction_threshold
        self.is_correction_level = is_correction_level
        self.is_correction_mode = is_correction_mode
        self.loss_agg_mode = loss_agg_mode
        self.loss_mode = loss_mode
        self.policy_loss_fn = get_policy_loss(loss_mode)  # raises on an unregistered name
        # Dual-clip policy objective: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"
        if self.loss_mode == "cispo":
            # CISPO clamps the raw ratio to an absolute ceiling (not a +offset like PPO's
            # surr2), so clip_eps_high here must be passed as that ceiling (e.g. 1.2).
            assert clip_eps_high >= 1.0, f"clip_eps_high must be >= 1.0 if loss_mode='cispo', got {clip_eps_high}"
            # dual_clip only has meaning for the min(surr1, surr2) PPO surrogate CISPO
            # doesn't compute; reject rather than silently ignore it.
            if dual_clip is not None:
                raise ValueError("dual_clip is a PPO-only extra bound; it has no effect under loss_mode='cispo'")

        if self.is_correction_level not in {"off", "token", "seq", "geo"}:
            raise ValueError(f"is_correction_level must be off/token/seq/geo, got {self.is_correction_level}")
        if self.is_correction_mode not in {"mask", "clip", "trunc"}:
            raise ValueError(f"is_correction_mode must be mask/clip/trunc, got {self.is_correction_mode}")
        # seq/geo aggregate the ratio into a per-sequence GATE; survivors keep their
        # per-token IS weight, so only mask (reject out-of-band sequences) is meaningful.
        # clip/trunc would replace the per-token weight with one clamped sequence weight,
        # discarding the per-token IS — reject that combination.
        if self.is_correction_level in {"seq", "geo"} and self.is_correction_mode != "mask":
            raise ValueError(
                f"is_correction_level={self.is_correction_level} only supports is_correction_mode=mask "
                f"(seq/geo are rejection filters, not per-token weights); got mode={self.is_correction_mode}"
            )

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[torch.Tensor] = None,
        global_batch_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_ratio_limit = 30.0
        policy_log_ratio = torch.nan_to_num(
            log_probs.float() - old_log_probs.float(),
            nan=0.0,
            posinf=log_ratio_limit,
            neginf=-log_ratio_limit,
        )
        advantages = torch.nan_to_num(advantages.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if action_mask is not None:
            mask = action_mask.bool()
            policy_log_ratio = torch.where(mask, policy_log_ratio, torch.zeros_like(policy_log_ratio))
            advantages = torch.where(mask, advantages, torch.zeros_like(advantages))

        ratio = policy_log_ratio.clamp(min=-log_ratio_limit, max=log_ratio_limit).exp()
        loss, clip_ratio = self.policy_loss_fn(
            ratio,
            advantages,
            log_probs,
            action_mask,
            clip_eps_low=self.clip_eps_low,
            clip_eps_high=self.clip_eps_high,
            dual_clip=self.dual_clip,
        )

        vllm_kl = None
        is_filter_ratio = None
        if self.is_correction_level != "off":
            if rollout_log_probs is None:
                raise ValueError("rollout_log_probs is required when IS correction is enabled")
            low, high = self.is_correction_threshold
            # per-token off-policy log-ratio  log(pi_train / pi_rollout)
            is_log_ratio = torch.nan_to_num(
                old_log_probs.float() - rollout_log_probs.float(),
                nan=0.0,
                posinf=log_ratio_limit,
                neginf=-log_ratio_limit,
            ).clamp(min=-log_ratio_limit, max=log_ratio_limit)
            token_ratio = torch.exp(is_log_ratio).detach()

            # (1) per-UNIT off-policy ratio (unit = token, or per-sequence for
            # seq/geo — kept at [B, 1] so the filter metric can stay per-sequence).
            # is_correction_level selects the aggregation.
            if self.is_correction_level == "token":
                unit_ratio = token_ratio
            elif self.is_correction_level == "seq":
                # product of a sequence's token ratios = exp(sum of log-ratios).
                seq_log = (is_log_ratio * action_mask.float()).sum(dim=-1, keepdim=True)
                unit_ratio = torch.exp(seq_log.clamp(min=-log_ratio_limit, max=log_ratio_limit))
            else:  # "geo" — per-sequence geometric mean = exp(mean of log-ratios).
                seq_log = masked_mean(is_log_ratio, action_mask, dim=-1).unsqueeze(-1)
                unit_ratio = torch.exp(seq_log)

            # (2) gate the per-unit ratio (is_correction_mode) -> per-token coefficient
            # + per-unit filtered flag.
            if self.is_correction_mode == "mask":
                # Drop out-of-band units; weight the survivors by their per-token ratio.
                keep = (unit_ratio >= low) & (unit_ratio <= high)
                coef = torch.where(keep.expand_as(token_ratio), token_ratio, torch.zeros_like(token_ratio))
                unit_filtered = ~keep
            elif self.is_correction_mode == "clip":
                # Keep every unit, clamp its weight into [low, high] (applied per-token).
                coef = unit_ratio.clamp(min=low, max=high).expand_as(token_ratio)
                unit_filtered = (unit_ratio < low) | (unit_ratio > high)
            else:  # "trunc" — cap only the upper tail; small weights unchanged.
                coef = unit_ratio.clamp(max=high).expand_as(token_ratio)
                unit_filtered = unit_ratio > high

            loss = coef * loss
            # Filter fraction reported at the unit's own granularity: per (masked)
            # token for token-level, per sequence for seq/geo (the latter matches the
            # original seq-mask-tis: unit_filtered.mean() == 1 - seq_mask.mean()).
            if self.is_correction_level == "token":
                is_filter_ratio = masked_mean(unit_filtered.float(), action_mask, dim=None)
            else:
                is_filter_ratio = unit_filtered.float().mean()

            vllm_logprob_diff = torch.nan_to_num(
                rollout_log_probs.float() - old_log_probs.float(),
                nan=0.0,
                posinf=log_ratio_limit,
                neginf=-log_ratio_limit,
            )
            vllm_kl = masked_mean(vllm_logprob_diff, action_mask, dim=None)

        # Reported loss is a plain per-token mean, decoupled from the gradient
        # normalization below. With the slime "global token-mean" denominator
        # (batch_num_tokens = whole optimizer-step batch), the aggregated loss is
        # a fraction-of-window, not a per-token mean — so we report this instead
        # to keep logged values interpretable and comparable across configs.
        reported_loss = masked_mean(loss.detach(), action_mask, dim=None)

        if self.token_level_loss:
            loss = agg_loss(
                loss,
                action_mask,
                self.loss_agg_mode,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
            )
        else:
            loss = agg_loss(
                loss,
                action_mask,
                "seq-mean-token-mean",
                dp_size=dp_size,
                global_batch_size=global_batch_size,
            )
        policy_kl = masked_mean(-policy_log_ratio.detach(), action_mask, dim=None)
        return loss, reported_loss, clip_ratio, policy_kl, vllm_kl, is_filter_ratio
