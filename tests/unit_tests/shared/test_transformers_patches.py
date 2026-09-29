# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.shared.transformers_patches import patch_t5_layer_norm


class TestPatchT5LayerNormIdempotent:
    """Test that patch_t5_layer_norm is idempotent."""

    def _make_apex_mock(self):
        """Return mocks with FusedRMSNorm installed as T5LayerNorm."""
        fused_cls = type("FusedRMSNorm", (), {"__module__": "apex.normalization"})
        mock_modeling_t5 = MagicMock()
        mock_modeling_t5.T5LayerNorm = fused_cls
        mock_apex_norm = MagicMock()
        mock_apex_norm.FusedRMSNorm = fused_cls
        mock_t5_parent = MagicMock()
        mock_t5_parent.modeling_t5 = mock_modeling_t5
        return mock_modeling_t5, mock_apex_norm, mock_t5_parent, fused_cls

    def test_replaces_fused_rms_norm(self):
        """When T5LayerNorm IS FusedRMSNorm, it gets replaced with the native impl."""
        mock_modeling_t5, mock_apex_norm, mock_t5_parent, fused_cls = self._make_apex_mock()

        with patch.dict(
            "sys.modules",
            {
                "apex": MagicMock(),
                "apex.normalization": mock_apex_norm,
                "transformers.models.t5": mock_t5_parent,
                "transformers.models.t5.modeling_t5": mock_modeling_t5,
            },
        ):
            patch_t5_layer_norm()

        assert mock_modeling_t5.T5LayerNorm is not fused_cls
        assert mock_modeling_t5.T5LayerNorm.__name__ == "_NativeT5LayerNorm"

    def test_idempotent_when_already_patched(self):
        """Calling twice leaves T5LayerNorm as the already-replaced native class."""
        mock_modeling_t5, mock_apex_norm, mock_t5_parent, fused_cls = self._make_apex_mock()

        modules = {
            "apex": MagicMock(),
            "apex.normalization": mock_apex_norm,
            "transformers.models.t5": mock_t5_parent,
            "transformers.models.t5.modeling_t5": mock_modeling_t5,
        }
        with patch.dict("sys.modules", modules):
            patch_t5_layer_norm()
            first_replacement = mock_modeling_t5.T5LayerNorm
            patch_t5_layer_norm()

        assert mock_modeling_t5.T5LayerNorm is first_replacement

    def test_skips_when_not_apex(self):
        """When T5LayerNorm is already NOT from apex, no replacement happens."""
        original_norm = type("OriginalT5LayerNorm", (), {"__module__": "transformers"})
        mock_modeling_t5 = MagicMock()
        mock_modeling_t5.T5LayerNorm = original_norm
        mock_t5_parent = MagicMock()
        mock_t5_parent.modeling_t5 = mock_modeling_t5

        with patch.dict(
            "sys.modules",
            {
                "transformers.models.t5": mock_t5_parent,
                "transformers.models.t5.modeling_t5": mock_modeling_t5,
            },
        ):
            patch_t5_layer_norm()

        assert mock_modeling_t5.T5LayerNorm is original_norm

    def test_handles_missing_transformers(self):
        """When transformers is not installed, ImportError is caught silently."""
        with patch.dict("sys.modules", {"transformers": None, "transformers.models.t5": None}):
            patch_t5_layer_norm()  # must not raise


class TestNativeT5LayerNorm:
    """Test the _NativeT5LayerNorm implementation that replaces FusedRMSNorm."""

    def _get_native_norm_class(self):
        """Extract the _NativeT5LayerNorm class by running the patch against a mock."""
        fused_cls = type("FusedRMSNorm", (), {"__module__": "apex.normalization"})
        mock_modeling_t5 = MagicMock()
        mock_modeling_t5.T5LayerNorm = fused_cls

        mock_apex_norm = MagicMock()
        mock_apex_norm.FusedRMSNorm = fused_cls

        mock_t5_parent = MagicMock()
        mock_t5_parent.modeling_t5 = mock_modeling_t5

        with patch.dict(
            "sys.modules",
            {
                "apex": MagicMock(),
                "apex.normalization": mock_apex_norm,
                "transformers.models.t5": mock_t5_parent,
                "transformers.models.t5.modeling_t5": mock_modeling_t5,
            },
        ):
            patch_t5_layer_norm()

        return mock_modeling_t5.T5LayerNorm

    def test_forward_fp32(self):
        NormClass = self._get_native_norm_class()
        norm = NormClass(hidden_size=8, eps=1e-6)
        x = torch.randn(2, 4, 8)
        out = norm(x)
        assert out.shape == x.shape
        assert out.dtype == torch.float32
        assert not torch.isnan(out).any()

    def test_forward_bf16_weights(self):
        NormClass = self._get_native_norm_class()
        norm = NormClass(hidden_size=8, eps=1e-6)
        norm.weight = torch.nn.Parameter(norm.weight.to(torch.bfloat16))
        x = torch.randn(2, 4, 8)
        out = norm(x)
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16

    def test_forward_fp16_weights(self):
        NormClass = self._get_native_norm_class()
        norm = NormClass(hidden_size=8, eps=1e-6)
        norm.weight = torch.nn.Parameter(norm.weight.to(torch.float16))
        x = torch.randn(2, 4, 8)
        out = norm(x)
        assert out.shape == x.shape
        assert out.dtype == torch.float16

    def test_rms_norm_correctness(self):
        """Verify the RMS norm computation matches expected formula."""
        NormClass = self._get_native_norm_class()
        norm = NormClass(hidden_size=4, eps=1e-6)
        # Set weight to ones so we can verify normalization directly
        norm.weight = torch.nn.Parameter(torch.ones(4))

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = norm(x)

        # Manual RMS norm: x / sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        expected = x * torch.rsqrt(variance + 1e-6)
        torch.testing.assert_close(out, expected)

    def test_weight_initialization(self):
        NormClass = self._get_native_norm_class()
        norm = NormClass(hidden_size=16, eps=1e-5)
        assert norm.weight.shape == (16,)
        torch.testing.assert_close(norm.weight, torch.ones(16))
        assert norm.variance_epsilon == 1e-5

    @pytest.mark.run_only_on("GPU")
    def test_forward_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        NormClass = self._get_native_norm_class()
        norm = NormClass(hidden_size=16, eps=1e-6).cuda()
        norm.weight = torch.nn.Parameter(norm.weight.to(torch.bfloat16))
        x = torch.randn(2, 8, 16, device="cuda")
        out = norm(x)
        assert out.device.type == "cuda"
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on("GPU")
    def test_forward_gpu_fp16(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        NormClass = self._get_native_norm_class()
        norm = NormClass(hidden_size=32, eps=1e-6).cuda()
        norm.weight = torch.nn.Parameter(norm.weight.to(torch.float16))
        x = torch.randn(4, 8, 32, device="cuda", dtype=torch.float16)
        out = norm(x)
        assert out.device.type == "cuda"
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()
