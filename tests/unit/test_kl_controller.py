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

"""Unit tests for the KL coefficient controllers (Ziegler et al. 2019,
https://arxiv.org/pdf/1909.08593.pdf).

``AdaptiveKLController`` adjusts the coefficient by::

    1 + clip(current / target - 1, -0.2, 0.2) * n_steps / horizon

The tests cover update direction (rise above target, fall below, unchanged at
target), the ±0.2 proportional-error clip, the ``n_steps / horizon`` scaling,
and the fixed controller's no-op. Expected values are derived directly from the
formula (not captured from the code).
"""

from pytest import approx

from molt.trainer.algorithm.kl_controller import AdaptiveKLController, FixedKLController


def test_adaptive_coef_rises_when_kl_above_target():
    # 10% above target -> error +0.1, inside the ±0.2 clip (unsaturated), so this
    # pins the proportional term itself: mult = 1 + 0.1 = 1.1. A stub that always
    # applied the +0.2 bound for any positive error would fail here.
    ctl = AdaptiveKLController(init_kl_coef=0.1, target=0.01, horizon=100)
    ctl.update(current=0.011, n_steps=100)
    assert ctl.value == approx(0.11)


def test_adaptive_coef_falls_when_kl_below_target():
    ctl = AdaptiveKLController(init_kl_coef=0.1, target=0.01, horizon=100)
    ctl.update(current=0.0, n_steps=100)  # well under -> clip -0.2 -> mult 0.8
    assert ctl.value == approx(0.08)


def test_adaptive_at_target_leaves_coef_unchanged():
    ctl = AdaptiveKLController(init_kl_coef=0.1, target=0.01, horizon=100)
    ctl.update(current=0.01, n_steps=100)  # zero error -> mult 1.0
    assert ctl.value == approx(0.1)


def test_adaptive_positive_error_is_clipped():
    # A KL 100x the target must not move the coefficient more than the +0.2 clip
    # allows: error saturates at +0.2 -> mult 1.2. The lower bound is exercised by
    # current=0.0 above.
    ctl = AdaptiveKLController(init_kl_coef=0.1, target=0.01, horizon=100)
    ctl.update(current=1.0, n_steps=100)
    assert ctl.value == approx(0.12)


def test_adaptive_update_scales_with_steps_over_horizon():
    # A KL 2x the target gives a +0.2 clipped error, but n_steps/horizon = 10/100 = 0.1
    # scales the move: mult = 1 + 0.2 * 0.1 = 1.02. Distinct from the 0.12 full-horizon
    # result, so this pins the n_steps/horizon factor (its removal would not pass here).
    ctl = AdaptiveKLController(init_kl_coef=0.1, target=0.01, horizon=100)
    ctl.update(current=0.02, n_steps=10)
    assert ctl.value == approx(0.102)


def test_fixed_controller_update_is_noop():
    ctl = FixedKLController(kl_coef=0.1)
    ctl.update(current=999.0, n_steps=100)  # any input
    assert ctl.value == 0.1
