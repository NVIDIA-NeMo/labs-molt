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

"""`--train.check_weight_update_equal`: the digests vLLM reports before and after a sync must agree."""

import pytest

from molt.trainer.workers import policy_actor
from molt.trainer.workers.policy_actor import PolicyTrainer

LOADED = {"dp0:pp0:pcp0:tp0:model.w": "aa", "dp0:pp0:pcp0:tp0:model.b": "bb"}


def _verify(monkeypatch, loaded: dict, synced: dict):
    """Run the check on one engine that reports `loaded` before the sync and `synced` after it."""
    checksums = iter([[loaded], [synced]])
    monkeypatch.setattr(
        policy_actor,
        "batch_vllm_engine_call",
        lambda engines, method: next(checksums) if method == "weight_checksums" else None,
    )
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    trainer.vllm_engines = [object()]
    trainer.broadcast_to_vllm = lambda: None
    trainer.verify_weight_sync()


def test_passes_when_the_sync_reproduces_every_digest(monkeypatch):
    _verify(monkeypatch, LOADED, dict(LOADED))


def test_fails_on_a_tensor_with_another_digest(monkeypatch):
    with pytest.raises(RuntimeError, match="1 of 2 .*engine 0 dp0:pp0:pcp0:tp0:model.w"):
        _verify(monkeypatch, LOADED, {**LOADED, "dp0:pp0:pcp0:tp0:model.w": "00"})


def test_fails_on_a_tensor_missing_after_the_sync(monkeypatch):
    with pytest.raises(RuntimeError, match="engine 0 dp0:pp0:pcp0:tp0:model.b"):
        _verify(monkeypatch, LOADED, {"dp0:pp0:pcp0:tp0:model.w": "aa"})


def test_fails_when_vllm_reports_no_digests(monkeypatch):
    with pytest.raises(RuntimeError, match="no weight checksums"):
        _verify(monkeypatch, {}, {})
