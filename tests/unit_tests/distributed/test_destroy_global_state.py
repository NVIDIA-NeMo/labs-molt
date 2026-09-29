# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``destroy_global_state`` DeepEP-aware teardown ordering."""

import nemo_automodel.components.distributed.init_utils as init_utils
import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


def _patch_common(monkeypatch, calls, *, initialized: bool):
    monkeypatch.setattr(init_utils.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(fused_a2a, "free_buffer", lambda: calls.append("free_buffer"))
    monkeypatch.setattr(init_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(init_utils.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(init_utils.torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(init_utils.torch.distributed, "destroy_process_group", lambda: calls.append("destroy_pg"))


def test_frees_deepep_buffer_before_destroying_process_group(monkeypatch):
    # The DeepEP buffer MUST be freed before destroy_process_group(); otherwise destroy hangs on
    # DeepEP's leftover NCCL sub-group state.
    calls = []
    _patch_common(monkeypatch, calls, initialized=True)

    init_utils.destroy_global_state()

    assert calls == ["free_buffer", "destroy_pg"]


def test_frees_buffer_even_without_process_group(monkeypatch):
    # free_buffer is best-effort and still runs when no process group is initialized; the PG
    # destroy is then skipped.
    calls = []
    _patch_common(monkeypatch, calls, initialized=False)

    init_utils.destroy_global_state()

    assert calls == ["free_buffer"]


def test_buffer_free_failure_does_not_block_pg_destroy(monkeypatch):
    # A failing free_buffer (e.g. deep_ep not importable) must not prevent destroy_process_group.
    calls = []
    _patch_common(monkeypatch, calls, initialized=True)

    def _boom():
        raise RuntimeError("deep_ep unavailable")

    monkeypatch.setattr(fused_a2a, "free_buffer", _boom)

    init_utils.destroy_global_state()

    assert calls == ["destroy_pg"]
