# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from typing import Optional

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
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        vllm_is_correction_type: str = "tis",
        loss_agg_mode: str = "token-mean",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type
        self.loss_agg_mode = loss_agg_mode

        # Dual-clip policy objective: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

        if self.vllm_is_correction_type not in {"tis", "icepop", "seq-mask-tis"}:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {self.vllm_is_correction_type}, must be one of tis/icepop/seq-mask-tis"
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

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            loss = -torch.min(surr1, surr2)
        else:
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        vllm_kl = None
        is_filter_ratio = None
        if self.enable_vllm_is_correction:
            if rollout_log_probs is None:
                raise ValueError("rollout_log_probs is required when vLLM importance sampling correction is enabled")
            low_threshold, high_threshold = self.vllm_is_truncated_threshold
            is_log_ratio = torch.nan_to_num(
                old_log_probs.float() - rollout_log_probs.float(),
                nan=0.0,
                posinf=log_ratio_limit,
                neginf=-log_ratio_limit,
            ).clamp(min=-log_ratio_limit, max=log_ratio_limit)
            vllm_is_ratio = torch.exp(is_log_ratio).detach()
            if self.vllm_is_correction_type == "icepop":
                # ICEPOP: token-level filtering (set coefficients outside the interval to 0)
                mask = (vllm_is_ratio >= low_threshold) & (vllm_is_ratio <= high_threshold)
                loss = torch.where(mask, vllm_is_ratio * loss, torch.zeros_like(loss))
                is_filter_ratio = 1.0 - masked_mean(mask.float(), action_mask, dim=None)
            elif self.vllm_is_correction_type == "seq-mask-tis":
                seq_log_ratio = masked_mean(is_log_ratio, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
                loss = torch.where(seq_mask.unsqueeze(-1), vllm_is_ratio * loss, torch.zeros_like(loss))
                # Fraction of sequences dropped by per-sequence geom-mean threshold.
                is_filter_ratio = 1.0 - seq_mask.float().mean()
            else:
                # TIS truncation semantics: cap large off-policy weights, but
                # keep small weights small instead of
                # raising them to the ICE-POP lower bound.
                too_large = vllm_is_ratio > high_threshold
                vllm_is_ratio = vllm_is_ratio.clamp(max=high_threshold)
                loss = vllm_is_ratio * loss
                # Tokens whose IS ratio exceeded the upper cap and got clamped.
                is_filter_ratio = masked_mean(too_large.float(), action_mask, dim=None)
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
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        policy_kl = masked_mean(-policy_log_ratio.detach(), action_mask, dim=None)
        return loss, reported_loss, clip_ratio, policy_kl, vllm_kl, is_filter_ratio
