#!/usr/bin/env python3
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
"""Summarize an RL run from its log: every numeric metric of the `Global step N: {...}` lines as a
markdown table (first / last / min / max / change over the run), then exit 1 on the few signals that
mean the run is broken rather than merely small: a non-finite value, no reward on any update, no
gradient on any update, an exploding gradient, or a rollout-vs-trainer log-prob gap (vllm_kl, IS
rejection rate) far above the bf16 mismatch floor.

    python3 tests/e2e/summarize_metrics.py train.log [--out metrics_summary.md]
"""

import argparse
import ast
import math
import re
import sys

STEP_RE = re.compile(r"Global step (\d+): (\{.*\})")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--out", help="also write the markdown here (e.g. for GITHUB_STEP_SUMMARY)")
    args = parser.parse_args()

    steps = {}
    for line in open(args.log, errors="ignore"):
        if m := STEP_RE.search(ANSI_RE.sub("", line)):
            # nan/inf are not Python literals; 1e999 overflows to inf and still fails the finite check.
            steps[int(m.group(1))] = ast.literal_eval(re.sub(r"\b(nan|inf)\b", "1e999", m.group(2)))
    if not steps:
        sys.exit(f"no 'Global step' line in {args.log}")
    series = {}
    for step in sorted(steps):
        for k, v in steps[step].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                series.setdefault(k, []).append(float(v))

    rows = ["| metric | first | last | min | max | change |", "|---|---|---|---|---|---|"]
    for k in sorted(series):
        v = series[k]
        rows.append(f"| {k} | {v[0]:.4g} | {v[-1]:.4g} | {min(v):.4g} | {max(v):.4g} | {v[-1] - v[0]:+.4g} |")

    def mean(k):
        return sum(series[k]) / len(series[k]) if k in series else None

    alerts = []
    bad = sorted(k for k, v in series.items() if not all(math.isfinite(x) for x in v))
    if bad:
        alerts.append(f"non-finite values in {', '.join(bad)}")
    if mean("reward") == 0:
        alerts.append("reward is 0 on every update: grader or answer format is broken")
    if "actor_grad_norm" in series and max(series["actor_grad_norm"]) == 0:
        alerts.append("actor_grad_norm is 0 on every update: no learning signal reaches the actor")
    if "actor_grad_norm" in series and max(series["actor_grad_norm"]) > 100:
        alerts.append(f"actor_grad_norm peaks at {max(series['actor_grad_norm']):.3g}: exploding gradient")
    if (mean("vllm_kl") or 0) > 1e-2:
        alerts.append(
            f"vllm_kl mean {mean('vllm_kl'):.2e} > 1e-2: rollout and trainer log-probs disagree (refit or log-prob path)"
        )
    if (mean("is_filter_ratio") or 0) > 0.2:
        alerts.append(f"is_filter_ratio mean {mean('is_filter_ratio'):.2f}: the IS gate rejects too many sequences")
    if (mean("response_clip_ratio") or 0) > 0.9:
        alerts.append("over 90% of responses hit max_new_tokens: generation does not terminate")

    text = f"## RL e2e metrics ({len(steps)} updates)\n\n" + "\n".join(rows) + "\n\n"
    text += ("### ALERTS\n" + "".join(f"- {a}\n" for a in alerts)) if alerts else "No alerts.\n"
    print(text)
    if args.out:
        open(args.out, "w").write(text)
    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
