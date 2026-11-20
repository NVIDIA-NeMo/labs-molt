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

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from molt.models.utils import mismatch_metrics


@pytest.mark.parametrize(
    "kl,exit_code",
    [(-0.5, 1), (0.5, 1), (-0.01, 0), (0.01, 0), (-0.009, 0), (0.009, 0), (0.0, 0), (None, 0)],
)
def test_e2e_summary_checks_magnitude_of_signed_kl(tmp_path, kl, exit_code):
    metrics = {"reward": 1.0, "actor_grad_norm": 1.0}
    if kl is not None:
        metrics["vllm_kl"] = kl
    log = tmp_path / "train.log"
    log.write_text(f"Global step 1: {metrics!r}\nGlobal step 2: {metrics!r}\n")
    summary = tmp_path / "summary.md"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "e2e/summarize_metrics.py"),
            str(log),
            "--out",
            str(summary),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == exit_code, result.stderr
    assert summary.read_text().strip() == result.stdout.strip()
    if exit_code:
        assert f"vllm_kl mean {kl:.2e}" in result.stdout
        assert "rollout and trainer log-probs disagree" in result.stdout
    else:
        assert "No alerts." in result.stdout


def test_bitwise_frac_counts_last_bit_differences_that_a_printed_value_hides():
    rollout = torch.tensor([[-0.5, -1.25, -2.0, -3.0, -4.0]])
    train = rollout.clone()
    train[0, 2] = torch.nextafter(train[0, 2], torch.tensor(0.0))  # one ulp away: same 6-decimal print
    # The masked token is never counted, and neither is the turn's first step (see below), so the
    # three compared tokens are 1, 2 and 3.
    mask = torch.tensor([[True, True, True, True, False]])
    m = mismatch_metrics(rollout, train, mask)
    assert abs(m["mismatch/bitwise_frac"].item() - 1 / 3) < 1e-6
    assert m["mismatch/abs_diff_mean"].item() > 0
    assert m["mismatch/k3_kl"].item() >= 0


def test_identical_bytes_read_exactly_zero_everywhere():
    rollout = torch.randn(2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    m = mismatch_metrics(rollout, rollout.clone(), mask)
    assert all(v.item() == 0.0 for v in m.values())


def test_a_turns_first_generated_token_is_not_compared():
    """vLLM masks the stop tokens while the request is below min_tokens, which at molt's min_tokens=1
    is each turn's first generated step: its log-prob is renormalized over the rest of the vocabulary,
    so it is a different quantity from the forward's and cannot be read as a mismatch."""
    rollout = torch.zeros(1, 7)
    train = rollout.clone()
    mask = torch.tensor([[True, True, True, False, True, True, False]])  # two turns, then padding
    train[0, 0] = 1.0  # first turn's first step
    train[0, 4] = 1.0  # second turn's first step

    m = mismatch_metrics(rollout, train, mask)
    assert all(v.item() == 0.0 for v in m.values())

    train[0, 5] = 1.0  # a step inside the second turn does count
    m = mismatch_metrics(rollout, train, mask)
    assert abs(m["mismatch/bitwise_frac"].item() - 1 / 3) < 1e-6  # compared: steps 1, 2 and 5


def test_nucleus_sampling_leaves_nothing_to_compare():
    """With --rollout.top_p below 1 vLLM renormalizes EVERY step over the nucleus it kept, so the
    metrics would report that renormalization as train/rollout drift. Report nothing instead."""
    rollout = torch.zeros(1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)

    assert mismatch_metrics(rollout, rollout + 0.3, mask, top_p=0.95) == {}
    assert mismatch_metrics(rollout, rollout + 0.3, mask, top_p=1.0)["mismatch/abs_diff_mean"].item() > 0
