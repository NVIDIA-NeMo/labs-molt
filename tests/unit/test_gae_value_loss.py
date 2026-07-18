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

"""Numerical-equivalence tests for the PPO `gae` estimator and `ValueLoss`.

Both are checked against a plain in-test reference implementation of the standard
GAE recursion and the PPO clipped value objective. The modules are loaded by file
path so the test runs on the host without the in-container `nemo_automodel`
dependency that `molt.models.__init__` would otherwise pull in.
"""

import importlib.util
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(modname: str, relpath: str):
    spec = importlib.util.spec_from_file_location(modname, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# Load loss.py / advantage.py by path WITHOUT executing molt/models/__init__.py
# (it imports Actor -> nemo_automodel, unavailable on the host). We register bare
# parent packages + the by-path modules in sys.modules, then restore it so these
# stubs don't shadow the real molt.models when a sibling test is collected next.
_MISSING = object()
_MODULE_NAMES = ("molt", "molt.models", "molt.models.utils", "molt.models.loss", "advantage")
_saved_modules = {name: sys.modules.get(name, _MISSING) for name in _MODULE_NAMES}
try:
    for _pkg in ("molt", "molt.models"):
        if _pkg not in sys.modules:
            _m = types.ModuleType(_pkg)
            _m.__path__ = []  # mark as a package
            sys.modules[_pkg] = _m

    _load("molt.models.utils", "molt/models/utils.py")
    loss_mod = _load("molt.models.loss", "molt/models/loss.py")
    adv_mod = _load("advantage", "molt/trainer/algorithm/advantage.py")

    AdvantageContext = adv_mod.AdvantageContext
    gae = adv_mod.get_advantage_estimator("gae")
    ValueLoss = loss_mod.ValueLoss
finally:
    for name, previous in _saved_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _reference_gae(rewards, values, gamma, lambd):
    """Plain reference GAE recursion over (B, T) for the equivalence check."""
    B, T = rewards.shape
    lastgaelam = torch.zeros(B, dtype=rewards.dtype)
    adv_rev = []
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)
    advantages = torch.stack(adv_rev[::-1], dim=1)
    returns = advantages + values
    return advantages, returns


def _ctx(masks, kls, values, kl_coef, gamma, lam):
    """A single-experience AdvantageContext (one rollout per sample, no merge)."""
    n = masks[0].size(0)
    return AdvantageContext(
        sample_to_rollout=torch.arange(n),
        exp_len=[n],
        action_masks=masks,
        kl_coef=kl_coef,
        gamma=gamma,
        kls=kls,
        lam=lam,
        values=values,
    )


def test_no_whiten_uses_raw_returns():
    """no_whiten=True skips cross-batch whitening entirely — advantages ARE the raw returns (no
    mean-center, no std); the default whitens to ~0 mean / ~1 std over the action tokens."""
    adv = [torch.tensor([[1.0, 2.0, 3.0, 4.0]])]
    masks = [torch.ones(1, 4)]

    ctx_raw = _ctx(masks, [torch.zeros(1, 4)], None, 0.0, 1.0, 1.0)
    ctx_raw.no_whiten = True
    assert torch.equal(adv_mod.normalize_advantages(adv, ctx_raw)[0], adv[0])  # unchanged

    whitened = adv_mod.normalize_advantages(adv, _ctx(masks, [torch.zeros(1, 4)], None, 0.0, 1.0, 1.0))[0]
    assert abs(whitened.mean().item()) < 1e-5 and abs(whitened.std(unbiased=False).item() - 1.0) < 1e-4


# gae whitens its advantages unconditionally, so the recursion is
# checked through the un-whitened ``returns`` (= A_raw + V, which fully pins the raw
# advantage since V is known); the whitening itself has its own test below.
def test_gae_recursion_returns_match_reference():
    """gae() returns must reproduce the standard GAE recursion on a fully-active
    response (no interior masking), for both lam<1 and the lam=1 default."""
    torch.manual_seed(0)
    B, L = 4, 7
    mask = torch.ones(B, L)
    kl = torch.randn(B, L)
    values = torch.randn(B, L)
    reward = torch.randn(B)  # outcome reward per rollout
    kl_coef, gamma = 0.1, 0.97

    for lam in (0.95, 1.0):
        ctx = _ctx([mask], [kl], [values], kl_coef, gamma, lam)
        _adv, ret = gae(reward, [[i] for i in range(B)], ctx)
        ret = ret[0]

        # Reference per-token reward: -kl_coef*kl everywhere + reward on last token.
        token_reward = -kl_coef * kl
        token_reward[:, -1] = token_reward[:, -1] + reward
        _ref_adv, ref_ret = _reference_gae(token_reward, values, gamma, lam)

        assert torch.allclose(ret, ref_ret, atol=1e-5), f"ret mismatch at lam={lam}"


def test_gae_lambda1_returns_are_montecarlo():
    """At gamma=lam=1, returns = G_t (the discounted Monte-Carlo return), independent
    of the interior values — the property that lets us avoid bootstrapping V across
    multi-turn observation tokens. (Advantages are A_t = G_t - V(s_t) before the
    whitening that gae applies unconditionally.)"""
    torch.manual_seed(1)
    B, L = 3, 6
    mask = torch.ones(B, L)
    kl = torch.randn(B, L)
    values = torch.randn(B, L)
    reward = torch.randn(B)
    kl_coef = 0.05

    ctx = _ctx([mask], [kl], [values], kl_coef, gamma=1.0, lam=1.0)
    _adv, ret = gae(reward, [[i] for i in range(B)], ctx)
    ret = ret[0]

    token_reward = -kl_coef * kl
    token_reward[:, -1] = token_reward[:, -1] + reward
    # G_t = reverse cumulative sum of token rewards (gamma=1).
    g = token_reward.flip(1).cumsum(1).flip(1)
    assert torch.allclose(ret, g, atol=1e-5)
    # Raw advantage A_t = G_t - V(s_t) is recoverable from the un-whitened return.
    assert torch.allclose(ret - values, g - values, atol=1e-5)


def _reference_gae_masked(token_reward, values, mask, gamma, lambd):
    """Masked GAE: masked tokens are transparent — the bootstrap value and the
    running gae are carried across them."""
    B, T = token_reward.shape
    adv = torch.zeros(B, T, dtype=token_reward.dtype)
    lastgaelam = torch.zeros(B, dtype=token_reward.dtype)
    nextvalue = torch.zeros(B, dtype=token_reward.dtype)
    for t in reversed(range(T)):
        delta = token_reward[:, t] + gamma * nextvalue - values[:, t]
        lastgaelam_ = delta + gamma * lambd * lastgaelam
        m = mask[:, t]
        nextvalue = values[:, t] * m + (1 - m) * nextvalue
        lastgaelam = lastgaelam_ * m + (1 - m) * lastgaelam
        adv[:, t] = lastgaelam
    return adv, adv + values


def test_gae_lambda_lt1_trailing_pad_is_terminal():
    """For right-padding (no action token follows), the last action token is genuinely
    terminal (bootstrap V=0), and the pad's garbage values never leak into earlier
    action tokens. Carry-forward and terminal agree here, so this still holds at lam<1.
    Checked via the un-whitened returns (gae whitens its advantages)."""
    B, L = 1, 5
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])  # 3 action tokens + 2 right-pad
    kl = torch.zeros(B, L)
    values = torch.tensor([[0.5, -0.3, 0.9, 4.0, -7.0]])  # last two are pad garbage
    reward = torch.tensor([1.0])
    gamma, lam = 1.0, 0.95

    ctx = _ctx([mask], [kl], [values], kl_coef=0.0, gamma=gamma, lam=lam)
    _adv, ret = gae(reward, [[0]], ctx)
    ret = ret[0]

    token_reward = torch.zeros(B, L)
    token_reward[:, 2] = reward  # reward on the last ACTION token
    _ref_adv, ref_ret = _reference_gae_masked(token_reward, values * mask, mask, gamma, lam)
    assert torch.allclose(ret, ref_ret * mask, atol=1e-5)
    # Last action token is terminal: ret = A + V = (r - V) + V = r = 1.0, and the raw
    # advantage A = r - V = 1.0 - 0.9 = 0.1 is NOT bootstrapped into the padding values.
    assert torch.allclose(ret[0, 2], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose((ret - values * mask)[0, 2], torch.tensor(0.1), atol=1e-5)


def test_gae_lambda_lt1_interior_gap_bootstraps_across_observation():
    """At lam<1, an action token BEFORE an interior masked gap (multi-turn observation
    /tool tokens) must bootstrap from the NEXT action token's value across the gap —
    not from a spurious terminal V=0. This is the bug the carry-forward fixes; it is
    invisible at lam=1 (telescoping) and only this interior-gap + lam<1 case exposes it.
    Checked via the un-whitened returns (gae whitens its advantages)."""
    B, L = 1, 6
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])  # 2 action | 2 obs | 2 action
    kl = torch.zeros(B, L)
    values = torch.tensor([[0.5, -0.3, 9.0, -9.0, 0.7, 0.2]])  # obs values are garbage
    reward = torch.tensor([1.0])
    gamma, lam = 1.0, 0.95

    ctx = _ctx([mask], [kl], [values], kl_coef=0.0, gamma=gamma, lam=lam)
    _adv, ret = gae(reward, [[0]], ctx)
    ret = ret[0]

    token_reward = torch.zeros(B, L)
    token_reward[:, 5] = reward  # reward on the last action token
    # Correct masked-carry reference.
    _ref_adv, ref_ret = _reference_gae_masked(token_reward, values * mask, mask, gamma, lam)
    assert torch.allclose(ret, ref_ret * mask, atol=1e-5)
    # The action token before the gap (idx 1) bootstraps from V(idx4)=0.7, giving
    # delta = gamma*0.7 - (-0.3) = 1.0 — NOT the terminal-V=0 delta of 0.3 the old code
    # produced. The two interpretations must disagree here (proves the carry matters).
    _bug_adv, bug_ret = _reference_gae(token_reward, values * mask, gamma, lam)
    assert not torch.allclose(ret, bug_ret * mask, atol=1e-3)


def test_gae_masks_interior_observation_tokens():
    """With a multi-turn action_mask (observation tokens masked off), advantages
    and returns are zero on the non-action steps."""
    B, L = 2, 6
    mask = torch.tensor([[1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1]], dtype=torch.float32)
    kl = torch.zeros(B, L)
    values = torch.randn(B, L)
    reward = torch.tensor([1.0, -1.0])

    ctx = _ctx([mask], [kl], [values], kl_coef=0.0, gamma=1.0, lam=1.0)
    adv, ret = gae(reward, [[0], [1]], ctx)
    adv, ret = adv[0], ret[0]
    off = mask == 0
    assert torch.all(adv[off] == 0) and torch.all(ret[off] == 0)


def test_reinforce_returns_and_advantages():
    """reinforce (refactored onto the shared helpers) still gives the discounted
    return broadcast over the response, whitened into advantages."""
    reinforce = adv_mod.get_advantage_estimator("reinforce")
    mask = torch.ones(2, 3)
    ctx = AdvantageContext(
        sample_to_rollout=torch.arange(2),
        exp_len=[2],
        action_masks=[mask],
        kl_coef=0.0,
        gamma=1.0,
        lam=1.0,
        kls=[torch.zeros(2, 3)],
        values=None,
    )
    rewards = torch.tensor([2.0, -1.0])
    advantages, returns = reinforce(rewards, [[0], [1]], ctx)
    # gamma=1, kl=0 -> each token carries its rollout's terminal reward.
    assert torch.allclose(returns[0], torch.tensor([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]]))
    # whitened across the 6 masked tokens (mean 0.5, std 1.5).
    assert torch.allclose(advantages[0], torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]), atol=1e-5)


def test_value_loss_matches_reference():
    """ValueLoss must equal 0.5 * max(clipped^2, unclipped^2) (the standard PPO
    0.5 factor), aggregated as a plain masked mean over the active tokens."""
    torch.manual_seed(2)
    B, L = 5, 9
    mask = (torch.rand(B, L) > 0.3).float()
    values = torch.randn(B, L)
    old_values = values + 0.5 * torch.randn(B, L)
    returns = torch.randn(B, L)
    value_clip = 0.2

    loss, reported, clip_frac = ValueLoss(value_clip=value_clip, loss_agg_mode="token-mean")(
        values, old_values, returns, action_mask=mask
    )

    # Reference clipped value objective (with the 0.5 factor), masked-mean reduction.
    v_clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
    surr1 = (v_clipped - returns).pow(2)
    surr2 = (values - returns).pow(2)
    ref = 0.5 * torch.max(surr1, surr2)
    ref_loss = (ref * mask).sum() / mask.sum()
    ref_clipfrac = ((torch.abs(values - old_values) > value_clip).float() * mask).sum() / mask.sum()

    assert torch.allclose(loss, ref_loss, atol=1e-6)
    assert torch.allclose(reported, ref_loss, atol=1e-6)  # token-mean == reported masked-mean (dp_size=1)
    assert torch.allclose(clip_frac, ref_clipfrac, atol=1e-6)


def test_gae_whitens_advantages_and_masks():
    """gae batch-whitens its advantages (zero mean, unit population std over the
    action tokens) and re-masks off-action positions to 0, while leaving
    returns = A + V(s) as the un-whitened value-regression target."""
    torch.manual_seed(3)
    B, L = 4, 7
    # Mask off the last two columns so we can check both whitening and re-masking.
    mask = torch.ones(B, L)
    mask[:, -2:] = 0.0
    kl = torch.randn(B, L)
    values = torch.randn(B, L)
    reward = torch.randn(B)
    kl_coef, gamma, lam = 0.1, 0.97, 0.95
    groups = [[i] for i in range(B)]

    adv, ret = gae(reward, groups, _ctx([mask], [kl], [values], kl_coef, gamma, lam))
    adv, ret = adv[0], ret[0]

    on = mask.bool()
    # Whitened over the action tokens: ~zero mean, ~unit population std.
    active = adv[on]
    assert abs(active.mean().item()) < 1e-5
    assert abs(active.std(unbiased=False).item() - 1.0) < 1e-4
    # Re-masked: off-action advantages are exactly 0 (not the -mean*rstd whitening leaves).
    assert torch.all(adv[~on] == 0)
    # Returns are the un-whitened value target and stay 0 off-action.
    assert torch.all(ret[~on] == 0)
    # Returns are NOT whitened (their active mean is generally far from 0).
    assert ret[on].std(unbiased=False).item() > 1e-3


def test_gae_length_adaptive_lambda_policy_decouples_from_critic():
    """lam_alpha sets a per-sequence lambda_policy = 1 - 1/(alpha*l) for the advantages,
    while returns keep lambda_critic (ctx.lam). Two rows of different action length get
    different lambda_policy. no_whiten exposes the raw advantages for a direct check."""
    B, L = 2, 5
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]])  # lengths 5, 3
    kl = torch.zeros(B, L)
    values = torch.randn(B, L)
    reward = torch.tensor([1.0, -1.0])
    gamma, alpha, lam_critic = 1.0, 1.5, 1.0

    ctx = _ctx([mask], [kl], [values], kl_coef=0.0, gamma=gamma, lam=lam_critic)
    ctx.lam_alpha = alpha
    ctx.no_whiten = True
    adv, ret = gae(reward, [[0], [1]], ctx)
    adv, ret = adv[0], ret[0]

    token_reward = torch.zeros(B, L)
    token_reward[0, 4] = reward[0]  # reward on each row's last action token
    token_reward[1, 2] = reward[1]

    lam_policy = 1.0 - 1.0 / (alpha * mask.sum(1))  # per row: [0.8667, 0.7778]
    ref_adv, _ = _reference_gae_masked(token_reward, values * mask, mask, gamma, lam_policy)
    _, ref_ret = _reference_gae_masked(token_reward, values * mask, mask, gamma, torch.tensor(lam_critic))

    assert torch.allclose(adv, ref_adv * mask, atol=1e-5)
    assert torch.allclose(ret, ref_ret * mask, atol=1e-5)
    assert not torch.allclose(lam_policy, torch.full((B,), float(lam_critic)))  # lambdas differ


def test_gae_lam_alpha_none_uses_single_lambda():
    """lam_alpha=None keeps the pre-existing single-lambda behaviour: advantages and
    returns share ctx.lam, so returns == advantage_recursion(lam) + V."""
    B, L = 1, 4
    mask = torch.ones(B, L)
    values = torch.randn(B, L)
    reward = torch.tensor([1.0])
    gamma, lam = 1.0, 0.9

    ctx = _ctx([mask], [torch.zeros(B, L)], [values], kl_coef=0.0, gamma=gamma, lam=lam)
    ctx.no_whiten = True
    adv, ret = gae(reward, [[0]], ctx)

    token_reward = torch.zeros(B, L)
    token_reward[0, -1] = reward[0]
    ref_adv, ref_ret = _reference_gae_masked(token_reward, values * mask, mask, gamma, lam)
    assert torch.allclose(adv[0], ref_adv * mask, atol=1e-5)
    assert torch.allclose(ret[0], ref_ret * mask, atol=1e-5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
