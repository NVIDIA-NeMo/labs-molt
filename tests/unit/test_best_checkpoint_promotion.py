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

"""The best checkpoint is the evaluated step's rolling checkpoint, hard-linked under the best tag with the eval
result stamped on; a rolling checkpoint that was already pruned reports False so the caller can fall back."""

import json
import os

import torch

from molt.trainer.fsdp.checkpoint import CheckpointManager


class _FakeStrategy:
    def print(self, *msg):
        pass


def _rolling_checkpoint(root, step):
    path = root / f"global_step{step}"
    (path / "model").mkdir(parents=True)
    (path / "model" / "w.bin").write_bytes(b"x" * 16)
    torch.save({"client_state": {"global_step": step}}, path / "extra_state.pt")
    return path


def test_promote_best_links_the_evaluated_step(tmp_path):
    src = _rolling_checkpoint(tmp_path, 4)
    (tmp_path / "best_global_step2").mkdir()  # the previous best is rotated out
    cm = CheckpointManager(_FakeStrategy())

    assert cm.promote_best(
        str(tmp_path), "global_step4", "best_global_step4", {"best_eval_metric_key": "pass1"}, 0.5, "pass1"
    )

    dst = tmp_path / "best_global_step4"
    assert os.path.samefile(src / "model" / "w.bin", dst / "model" / "w.bin")
    assert torch.load(dst / "extra_state.pt", weights_only=False)["client_state"] == {
        "global_step": 4,
        "best_eval_metric_key": "pass1",
    }
    assert torch.load(src / "extra_state.pt", weights_only=False)["client_state"] == {"global_step": 4}
    assert json.load(open(dst / "metric.json")) == {"metric_key": "pass1", "metric_value": 0.5}
    assert not (tmp_path / "best_global_step2").exists()


def test_promote_best_reports_a_pruned_rolling_checkpoint(tmp_path):
    cm = CheckpointManager(_FakeStrategy())
    assert cm.promote_best(str(tmp_path), "global_step9", "best_global_step9", {}, 0.5, "pass1") is False
    assert not (tmp_path / "best_global_step9").exists()
