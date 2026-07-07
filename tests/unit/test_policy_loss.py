import math

import torch

from molt.models import PolicyLoss
from molt.models.utils import compute_approx_kl


def test_seq_mask_tis_uses_raw_token_importance_weights_for_kept_sequences():
    loss_fn = PolicyLoss(
        enable_vllm_is_correction=True,
        vllm_is_truncated_threshold=[0.5, 2.0],
        vllm_is_correction_type="seq-mask-tis",
    )
    rollout_log_probs = torch.tensor([[-math.log(3.0), math.log(3.0)]])

    loss, *_ = loss_fn(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        rollout_log_probs=rollout_log_probs,
    )

    torch.testing.assert_close(loss, torch.tensor(-(3.0 + 1.0 / 3.0) / 2.0))


def test_force_on_policy_old_equals_action_gives_reinforce_gradient():
    # force_on_policy skips the old-logprob forward and sets old = action.detach()
    # (policy_actor). The PPO ratio is then exactly 1, so the surrogate's gradient is
    # REINFORCE: proportional to -advantage. Verify the grad has that structure.
    logp = torch.randn(3, 4, requires_grad=True)
    adv = torch.tensor([[1.0, -2.0, 0.5, -1.5]]).repeat(3, 1)
    mask = torch.ones(3, 4, dtype=torch.bool)
    loss_fn = PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.2)  # no IS, no dual-clip

    loss, *_ = loss_fn(logp, logp.detach(), adv, action_mask=mask)
    loss.backward()

    # grad_i = c * (-adv_i) for one shared token-mean weight c > 0 -> ratio is constant.
    ratio = logp.grad / (-adv)
    torch.testing.assert_close(ratio, ratio.flatten()[:1].expand_as(ratio).contiguous())
    assert (ratio > 0).all()


def test_force_on_policy_reinforce_still_applies_is_correction():
    # With old = action.detach(), the IS correction still runs vs the rollout log-probs
    # (behavior policy = vLLM). vllm_kl = mean(rollout - old).
    logp = torch.zeros(1, 2)
    rollout = torch.tensor([[-0.1, -0.1]])  # old(0) - rollout(-0.1) = 0.1
    loss_fn = PolicyLoss(
        enable_vllm_is_correction=True,
        vllm_is_truncated_threshold=[0.0, 100.0],
        vllm_is_correction_type="tis",
    )

    _, _, _, _, vllm_kl, _ = loss_fn(
        logp,
        logp.detach(),
        torch.ones(1, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        rollout_log_probs=rollout,
    )

    torch.testing.assert_close(vllm_kl, torch.tensor(-0.1))


def test_tis_caps_large_importance_weights_without_flooring_small_weights():
    loss_fn = PolicyLoss(
        enable_vllm_is_correction=True,
        vllm_is_truncated_threshold=[0.5, 2.0],
        vllm_is_correction_type="tis",
    )

    loss, *_ = loss_fn(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        rollout_log_probs=torch.tensor([[math.log(10.0), -math.log(10.0)]]),
    )

    torch.testing.assert_close(loss, torch.tensor(-(0.1 + 2.0) / 2.0))


def test_icepop_importance_weights_do_not_nan_on_overflow():
    loss_fn = PolicyLoss(
        enable_vllm_is_correction=True,
        vllm_is_truncated_threshold=[0.5, 2.0],
        vllm_is_correction_type="icepop",
    )

    loss, *_ = loss_fn(
        torch.zeros(1, 1),
        torch.full((1, 1), 1000.0),
        torch.ones(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        rollout_log_probs=torch.zeros(1, 1),
    )

    assert torch.isfinite(loss)


def test_policy_loss_zero_advantage_handles_extreme_policy_ratio():
    loss_fn = PolicyLoss()

    loss, *_ = loss_fn(
        torch.full((1, 1), 1000.0),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_policy_loss_zero_advantage_handles_equal_negative_infinity_logprobs():
    loss_fn = PolicyLoss()

    loss, _reported, clip_ratio, policy_kl, *_ = loss_fn(
        torch.full((1, 1), -torch.inf),
        torch.full((1, 1), -torch.inf),
        torch.zeros(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert torch.isfinite(clip_ratio)
    assert torch.isfinite(policy_kl)


def test_policy_loss_empty_action_mask_returns_zero():
    loss_fn = PolicyLoss()

    loss, _reported, clip_ratio, policy_kl, *_ = loss_fn(
        torch.full((1, 2), torch.nan),
        torch.full((1, 2), torch.nan),
        torch.ones(1, 2),
        action_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(clip_ratio, torch.tensor(0.0))
    torch.testing.assert_close(policy_kl, torch.tensor(0.0))


def test_tis_caps_overflowed_importance_weights_at_high_threshold():
    loss_fn = PolicyLoss(
        enable_vllm_is_correction=True,
        vllm_is_truncated_threshold=[0.5, 2.0],
        vllm_is_correction_type="tis",
    )

    loss, *_ = loss_fn(
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        rollout_log_probs=torch.full((1, 1), -1000.0),
    )

    torch.testing.assert_close(loss, torch.tensor(-2.0))


def test_seq_mask_tis_drops_overflowed_importance_weight_without_nan():
    loss_fn = PolicyLoss(
        enable_vllm_is_correction=True,
        vllm_is_truncated_threshold=[0.5, 2.0],
        vllm_is_correction_type="seq-mask-tis",
    )

    loss, *_, is_filter_ratio = loss_fn(
        torch.full((1, 1), 1000.0),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        rollout_log_probs=torch.full((1, 1), -1000.0),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(is_filter_ratio, torch.tensor(1.0))


def test_policy_loss_sanitizes_nonfinite_vllm_kl_metric():
    loss_fn = PolicyLoss(
        enable_vllm_is_correction=True,
        vllm_is_truncated_threshold=[0.5, 2.0],
        vllm_is_correction_type="tis",
    )

    loss, *_, vllm_kl, is_filter_ratio = loss_fn(
        torch.zeros(1, 1),
        torch.full((1, 1), -torch.inf),
        torch.zeros(1, 1),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        rollout_log_probs=torch.full((1, 1), -torch.inf),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert torch.isfinite(vllm_kl)
    assert torch.isfinite(is_filter_ratio)


def test_compute_approx_kl_sanitizes_equal_negative_infinity_logprobs():
    kl = compute_approx_kl(
        torch.full((1, 1), -torch.inf),
        torch.full((1, 1), -torch.inf),
        kl_estimator="k2",
    )

    torch.testing.assert_close(kl, torch.zeros(1, 1))
