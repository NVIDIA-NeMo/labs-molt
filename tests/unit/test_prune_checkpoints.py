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

"""``_prune_checkpoints`` disk-budget observability.

``overflow_mem`` sums every subdir (incl. ``current_tag`` and ``best*``), but the
eviction candidates (``regular_subdirs``) exclude both — they are protected: a new
``best*`` save deletes older ``best*`` dirs itself (the ``is_best`` branch), and
regular pruning never touches ``best*`` at all. If current+best alone already
exceed ``--ckpt.max_mem``, the pre-fix loop found no candidates and broke silently,
leaving the dir over budget forever with no warning. The fix only adds the missing
warning — it does not evict ``best*``/``current_tag``, since that would silently
delete the one retained best checkpoint (a retention-policy change, not a bug fix).
"""

import os
import time

from molt.trainer.fsdp.checkpoint import CheckpointManager


class _FakeStrategy:
    def __init__(self):
        self.messages = []

    def print(self, *msg):
        self.messages.append(" ".join(str(m) for m in msg))


def _cm(strategy=None) -> CheckpointManager:
    return CheckpointManager(strategy or _FakeStrategy())


def _make_ckpt_dir(root: str, name: str, size_bytes: int, age_s: float = 0.0) -> str:
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    # Sparse file: _dir_size uses os.path.getsize, which honors the logical size
    # `truncate` sets without actually writing/allocating size_bytes of data.
    with open(os.path.join(path, "weights.bin"), "wb") as f:
        f.truncate(size_bytes)
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
    return path


def test_regular_checkpoints_evicted_under_max_mem(tmp_path):
    """Baseline: max_mem eviction still works via the regular (non-best, non-current) path."""
    root = str(tmp_path)
    _make_ckpt_dir(root, "step-1", 10 * 1024**2, age_s=20)
    _make_ckpt_dir(root, "step-2", 10 * 1024**2, age_s=10)
    _make_ckpt_dir(root, "step-3", 10 * 1024**2, age_s=0)

    cm = _cm()
    # ~25MB budget: one regular checkpoint (oldest first) must go.
    cm._prune_checkpoints(root, current_tag="step-3", max_num=0, max_mem=25 / 1024, is_best=False)

    remaining = set(os.listdir(root))
    assert "step-1" not in remaining  # oldest evicted first
    assert "step-3" in remaining  # current_tag always survives


def test_warns_after_regular_candidates_are_exhausted(tmp_path):
    """The representative regression case: a regular checkpoint IS evicted first
    (the loop re-evaluates candidates/usage after each deletion), but current+best
    alone still exceed the budget once that's exhausted.

    Before the fix, the loop broke silently the moment `candidates` went empty, with
    no warning. The fix must warn here — but must NOT evict best-eval or step-5,
    which would silently delete the one retained best checkpoint.
    """
    root = str(tmp_path)
    _make_ckpt_dir(root, "step-1", 5 * 1024**2, age_s=20)
    _make_ckpt_dir(root, "best-eval", 10 * 1024**2, age_s=10)
    _make_ckpt_dir(root, "step-5", 10 * 1024**2, age_s=0)

    strategy = _FakeStrategy()
    cm = _cm(strategy)
    # ~15MB budget: step-1 (5MB) is evicted first (25MB -> 20MB), still over budget,
    # but best-eval+step-5 (20MB) alone leaves no further regular candidates.
    cm._prune_checkpoints(root, current_tag="step-5", max_num=0, max_mem=15 / 1024, is_best=False)

    remaining = set(os.listdir(root))
    assert remaining == {"best-eval", "step-5"}  # step-1 evicted; both protected dirs survive
    assert any(
        "only protected current/best checkpoints remain" in m for m in strategy.messages
    )  # warns instead of silently stopping


def test_no_warning_when_regular_eviction_satisfies_budget(tmp_path):
    """Sanity check: the warning is specific to the "nothing left to evict" case, not
    fired whenever max_mem is set.
    """
    root = str(tmp_path)
    _make_ckpt_dir(root, "step-1", 10 * 1024**2, age_s=10)
    _make_ckpt_dir(root, "step-2", 10 * 1024**2, age_s=0)

    strategy = _FakeStrategy()
    cm = _cm(strategy)
    # ~15MB budget: evicting step-1 (the regular candidate) satisfies it.
    cm._prune_checkpoints(root, current_tag="step-2", max_num=0, max_mem=15 / 1024, is_best=False)

    assert set(os.listdir(root)) == {"step-2"}
    assert not strategy.messages
