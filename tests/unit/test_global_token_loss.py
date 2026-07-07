"""slime-style global token-mean normalization for the RL policy loss.

The RL trainer feeds every microbatch of one optimizer-step batch the *same*
``batch_num_tokens`` (the action-token count of the whole batch, summed over all
microbatches and DP ranks) and skips the ``/accum`` rescale. Summing the
per-microbatch ``agg_loss`` over the window therefore yields slime's
``calculate_per_token_loss=True`` objective: ``Σ (loss*mask).sum() / Σ mask.sum()``.
"""

import torch

from molt.models import PolicyLoss
from molt.models.loss import agg_loss, masked_sum


def _window_loss(losses, masks, batch_num_tokens, dp_size=1):
    # Mirror policy_train: one shared denominator for every microbatch, summed.
    return sum(
        agg_loss(loss, mask, "token-mean", dp_size=dp_size, batch_num_tokens=batch_num_tokens)
        for loss, mask in zip(losses, masks)
    )


def test_window_token_mean_matches_slime_sum_of_token():
    losses = [torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[4.0, 5.0]]), torch.tensor([[6.0]])]
    masks = [torch.tensor([[1.0, 1.0, 0.0]]), torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])]
    n_window = sum(m.sum() for m in masks)

    got = _window_loss(losses, masks, n_window)
    want = sum(masked_sum(loss, mask) for loss, mask in zip(losses, masks)) / n_window
    torch.testing.assert_close(got, want)


def test_uniform_tokens_match_old_per_microbatch_mean():
    # Equal action tokens per microbatch: the whole-batch token-mean reduces to
    # the old "per-microbatch token-mean, averaged over microbatches" — so the
    # change is a no-op for balanced batches (regression-safe).
    losses = [torch.tensor([[1.0, 3.0]]), torch.tensor([[5.0, 7.0]])]
    masks = [torch.ones(1, 2), torch.ones(1, 2)]
    n_window = sum(m.sum() for m in masks)

    new = _window_loss(losses, masks, n_window)
    old = sum(agg_loss(loss, m, "token-mean", batch_num_tokens=m.sum()) for loss, m in zip(losses, masks)) / len(
        losses
    )
    torch.testing.assert_close(new, old)


def test_unequal_tokens_differ_from_per_microbatch_mean():
    # Long microbatch (3 tokens) + short one (1 token). Global token-mean weights
    # by token, not by microbatch, so it differs from the old per-microbatch mean.
    losses = [torch.tensor([[2.0, 2.0, 2.0]]), torch.tensor([[10.0]])]
    masks = [torch.ones(1, 3), torch.ones(1, 1)]
    n_window = sum(m.sum() for m in masks)

    new = _window_loss(losses, masks, n_window)  # (2+2+2+10)/4 = 4.0
    old = sum(agg_loss(loss, m, "token-mean", batch_num_tokens=m.sum()) for loss, m in zip(losses, masks)) / len(
        losses
    )
    torch.testing.assert_close(new, torch.tensor(4.0))
    assert not torch.allclose(new, old)  # old = (2 + 10)/2 = 6.0


def test_accumulated_gradient_equals_global_token_mean():
    # End-to-end accumulation contract: per-microbatch backward with NO /accum
    # (scale_loss_by_accumulation=False), grads summed into .grad, must yield the
    # whole-batch global token-mean gradient ∇(Σ_tokens loss)/N_window — and NOT
    # the old per-microbatch-mean gradient.
    x = [torch.tensor([[2.0, 4.0, 6.0]]), torch.tensor([[8.0]])]  # 3 + 1 tokens
    masks = [torch.ones(1, 3), torch.ones(1, 1)]
    n_window = sum(m.sum() for m in masks)

    w = torch.zeros((), requires_grad=True)
    for xi, mi in zip(x, masks):
        loss = agg_loss(w * xi, mi, "token-mean", dp_size=1, batch_num_tokens=n_window)
        loss.backward()  # accumulates into w.grad, no /accum rescale

    want = sum(xi.sum() for xi in x) / n_window  # d/dw of Σ (w*x)/N_window
    torch.testing.assert_close(w.grad, want)

    # Old per-microbatch mean (denominator = each microbatch, then /num_mbs).
    w_old = torch.zeros((), requires_grad=True)
    for xi, mi in zip(x, masks):
        (agg_loss(w_old * xi, mi, "token-mean", batch_num_tokens=mi.sum()) / len(x)).backward()
    assert not torch.allclose(w.grad, w_old.grad)


def test_policy_loss_reports_per_token_mean_independent_of_denominator():
    # The reported loss is a plain per-token mean; only the gradient term carries
    # the global token denominator. With ratio==1, per-token loss = -advantage.
    loss_fn = PolicyLoss()
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    agg, reported, *_ = loss_fn(
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        torch.ones(1, 4),
        action_mask=mask,
        batch_num_tokens=torch.tensor(100.0),
    )
    torch.testing.assert_close(reported, torch.tensor(-1.0))
    torch.testing.assert_close(agg, torch.tensor(-3.0 / 100.0))
