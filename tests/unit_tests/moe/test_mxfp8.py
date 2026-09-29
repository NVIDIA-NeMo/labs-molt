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

"""CPU unit tests for the torchao MXFP8 grouped-GEMM plumbing (moe/mxfp8.py).

These exercise the dispatch/gate/relayout logic that runs on CPU; the actual SM100
kernel execution lines are marked ``# pragma: no cover`` in the module.
"""

import torch

from nemo_automodel.components.moe import mxfp8


class TestMxfp8GroupedMMOrNone:
    """The device/import gate."""

    def test_cpu_returns_none(self, monkeypatch):
        """No CUDA -> gate returns None and emits the one-time fallback warning."""
        monkeypatch.setattr(mxfp8.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(mxfp8, "_MXFP8_FALLBACK_WARNED", False)
        assert mxfp8._mxfp8_grouped_mm_or_none() is None

    def test_pre_sm100_capability_returns_none(self, monkeypatch):
        """CUDA present but compute capability < 10 (e.g. Hopper) -> None."""
        monkeypatch.setattr(mxfp8.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(mxfp8.torch.cuda, "get_device_capability", lambda: (9, 0))
        monkeypatch.setattr(mxfp8, "_MXFP8_FALLBACK_WARNED", False)
        assert mxfp8._mxfp8_grouped_mm_or_none() is None

    def test_sm100_with_resolver_returns_fn(self, monkeypatch):
        """SM100 + a resolvable callable -> the gate returns it (and announces once)."""
        sentinel = object()
        monkeypatch.setattr(mxfp8.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(mxfp8.torch.cuda, "get_device_capability", lambda: (10, 0))
        monkeypatch.setattr(mxfp8, "_resolve_mxfp8_grouped_mm", lambda: sentinel)
        monkeypatch.setattr(mxfp8, "_MXFP8_ACTIVE_ANNOUNCED", False)
        assert mxfp8._mxfp8_grouped_mm_or_none() is sentinel

    def test_sm100_but_no_torchao_returns_none(self, monkeypatch):
        """SM100 but the resolver finds no torchao API -> None (fallback)."""
        monkeypatch.setattr(mxfp8.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(mxfp8.torch.cuda, "get_device_capability", lambda: (10, 0))
        monkeypatch.setattr(mxfp8, "_resolve_mxfp8_grouped_mm", lambda: None)
        monkeypatch.setattr(mxfp8, "_MXFP8_FALLBACK_WARNED", False)
        assert mxfp8._mxfp8_grouped_mm_or_none() is None


class TestWeightRelayout:
    """_mxfp8_weight_relayout: values unchanged, transpose made contiguous."""

    def test_relayout_preserves_shape_and_values(self):
        B = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        out = mxfp8._mxfp8_weight_relayout(B)
        assert out.shape == B.shape
        assert torch.equal(out, B)

    def test_relayout_transpose_is_contiguous(self):
        # A plain [E,K,N] tensor's (-2,-1) transpose is NOT contiguous; after relayout it is.
        B = torch.randn(2, 3, 4)
        assert not B.transpose(-2, -1).is_contiguous()
        out = mxfp8._mxfp8_weight_relayout(B)
        assert out.transpose(-2, -1).is_contiguous()


class TestResolver:
    """_resolve_mxfp8_grouped_mm import ladder (runs in-container where torchao exists;
    on a torchao-less env it returns None via the ImportError guards)."""

    def test_resolver_is_idempotent_and_callable_or_none(self, monkeypatch):
        # Reset the module cache so the ladder actually runs.
        monkeypatch.setattr(mxfp8, "_MXFP8_RESOLVED", False)
        monkeypatch.setattr(mxfp8, "_MXFP8_GROUPED_MM", None)
        first = mxfp8._resolve_mxfp8_grouped_mm()
        assert first is None or callable(first)
        # Cached: second call returns the same object without re-running the ladder.
        second = mxfp8._resolve_mxfp8_grouped_mm()
        assert second is first

    def test_resolver_none_when_torchao_absent(self, monkeypatch):
        """Force every torchao import to fail -> resolver returns None."""
        import builtins

        real_import = builtins.__import__

        def _no_torchao(name, *args, **kwargs):
            if name.startswith("torchao"):
                raise ImportError("torchao not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_torchao)
        monkeypatch.setattr(mxfp8, "_MXFP8_RESOLVED", False)
        monkeypatch.setattr(mxfp8, "_MXFP8_GROUPED_MM", None)
        assert mxfp8._resolve_mxfp8_grouped_mm() is None
