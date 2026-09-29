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

import os
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

# Disable torch.compile for testing to avoid compilation overhead
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from nemo_automodel.components.utils.compile_utils import CompileConfig, apply_flash_attention_compile_fix, compile_model, configure_torch_dynamo, enable_torch_dynamo_scalar_outputs, patch_prepare_fa2_from_position_ids


class TestCompileConfig:
    """Test CompileConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CompileConfig()
        assert config.enabled is False
        assert config.mode == "default"
        assert config.fullgraph is False
        assert config.dynamic is False
        assert config.backend is None
        assert config.options == {}
        assert config.dynamo_cache_size_limit == 256

    def test_custom_config(self):
        """Test custom configuration values."""
        options = {"some_option": "value"}
        config = CompileConfig(
            enabled=True,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=True,
            backend="inductor",
            options=options,
            dynamo_cache_size_limit=512,
        )
        assert config.enabled is True
        assert config.mode == "reduce-overhead"
        assert config.fullgraph is True
        assert config.dynamic is True
        assert config.backend == "inductor"
        assert config.options == options
        assert config.dynamo_cache_size_limit == 512

    def test_to_dict(self):
        """Test to_dict method."""
        config = CompileConfig(
            enabled=True,
            mode="max-autotune",
            dynamo_cache_size_limit=128,
        )
        result = config.to_dict()
        expected = {
            "enabled": True,
            "mode": "max-autotune",
            "fullgraph": False,
            "dynamic": False,
            "backend": None,
            "options": {},
            "dynamo_cache_size_limit": 128,
        }
        assert result == expected


class TestConfigureTorchDynamo:
    """Test configure_torch_dynamo function."""

    @patch("torch._dynamo")
    def test_configure_dynamo_success(self, mock_dynamo):
        """Test successful dynamo configuration."""
        mock_config = MagicMock()
        mock_config.cache_size_limit = 8  # simulate PyTorch default
        mock_dynamo.config = mock_config

        configure_torch_dynamo(cache_size_limit=512, capture_scalar_outputs=True)

        assert mock_config.cache_size_limit == 512
        assert mock_config.capture_scalar_outputs is True

    @patch("torch._dynamo")
    def test_configure_dynamo_without_scalar_outputs(self, mock_dynamo):
        """Test dynamo configuration without scalar outputs."""
        mock_config = MagicMock()
        mock_config.cache_size_limit = 8
        mock_dynamo.config = mock_config

        configure_torch_dynamo(cache_size_limit=256, capture_scalar_outputs=False)

        assert mock_config.cache_size_limit == 256
        # When capture_scalar_outputs=False, the attribute is not set at all

    @patch("torch._dynamo")
    def test_configure_dynamo_never_lowers_limit(self, mock_dynamo):
        """Test that cache_size_limit is never lowered below the current value."""
        mock_config = MagicMock()
        mock_config.cache_size_limit = 512  # already higher
        mock_dynamo.config = mock_config

        configure_torch_dynamo(cache_size_limit=64)

        assert mock_config.cache_size_limit == 512  # preserved, not lowered

    def test_configure_dynamo_import_error(self):
        """Test handling of ImportError when torch._dynamo is not available."""
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            # Should not raise an exception
            configure_torch_dynamo()


class TestEnableTorchDynamoScalarOutputs:
    """Test enable_torch_dynamo_scalar_outputs function."""

    @patch("torch._dynamo")
    def test_enable_scalar_outputs_success(self, mock_dynamo):
        """Test successful enabling of scalar outputs."""
        mock_config = MagicMock()
        mock_dynamo.config = mock_config

        enable_torch_dynamo_scalar_outputs()

        assert mock_config.capture_scalar_outputs is True

    def test_enable_scalar_outputs_import_error(self):
        """Test handling of ImportError."""
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            # Should not raise an exception
            enable_torch_dynamo_scalar_outputs()


class TestPatchPrepareFA2FromPositionIds:
    """Test patch_prepare_fa2_from_position_ids function."""

    @patch("transformers.modeling_flash_attention_utils")
    def test_patch_success(self, mock_fa_utils):
        """Test successful patching."""
        result = patch_prepare_fa2_from_position_ids()
        assert result is True
        # Verify that the function was patched
        assert hasattr(mock_fa_utils, "prepare_fa2_from_position_ids")

    def test_patch_import_error(self):
        """Test handling of import error."""
        with patch("builtins.__import__") as mock_import:

            def side_effect(name, *args, **kwargs):
                if name == "transformers.modeling_flash_attention_utils":
                    raise ImportError("Module not found")
                return mock_import.return_value

            mock_import.side_effect = side_effect
            result = patch_prepare_fa2_from_position_ids()
            assert result is False

    def test_patch_general_exception(self):
        """Test handling of general exception during patching."""
        # Create a mock that raises an exception when accessed
        with patch("transformers.modeling_flash_attention_utils", new=None):
            result = patch_prepare_fa2_from_position_ids()
            assert result is False


class TestApplyFlashAttentionCompileFix:
    """Test apply_flash_attention_compile_fix function."""

    @patch("nemo_automodel.components.utils.compile_utils.enable_torch_dynamo_scalar_outputs")
    @patch("nemo_automodel.components.utils.compile_utils.patch_prepare_fa2_from_position_ids")
    def test_apply_fix_success(self, mock_patch, mock_enable):
        """Test successful application of fix."""
        mock_patch.return_value = True

        result = apply_flash_attention_compile_fix()

        mock_enable.assert_called_once()
        mock_patch.assert_called_once()
        assert result is True

    @patch("nemo_automodel.components.utils.compile_utils.enable_torch_dynamo_scalar_outputs")
    @patch("nemo_automodel.components.utils.compile_utils.patch_prepare_fa2_from_position_ids")
    def test_apply_fix_failure(self, mock_patch, mock_enable):
        """Test failure in applying fix."""
        mock_patch.return_value = False

        result = apply_flash_attention_compile_fix()

        mock_enable.assert_called_once()
        mock_patch.assert_called_once()
        assert result is False


class TestCompileModel:
    """Test compile_model function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    def test_compile_disabled(self):
        """Test that model is returned unchanged when compilation is disabled."""
        config = CompileConfig(enabled=False)
        result = compile_model(self.model, config)
        assert result is self.model

    @patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo")
    @patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix")
    @patch("nemo_automodel.components.utils.compile_utils.torch.compile")
    def test_compile_enabled_simple_model(self, mock_torch_compile, mock_fa_fix, mock_configure):
        """Test compilation with a simple model."""
        config = CompileConfig(enabled=True, mode="default", dynamo_cache_size_limit=512)
        mock_compiled_model = MagicMock()
        mock_torch_compile.return_value = mock_compiled_model

        result = compile_model(self.model, config)

        mock_configure.assert_called_once_with(cache_size_limit=512)
        mock_fa_fix.assert_called_once()
        mock_torch_compile.assert_called_once_with(self.model, mode="default", fullgraph=False, dynamic=False)
        assert result is mock_compiled_model

    @patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo")
    @patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix")
    @patch("nemo_automodel.components.utils.compile_utils.torch.compile")
    def test_compile_enabled_with_backend(self, mock_torch_compile, mock_fa_fix, mock_configure):
        """Test compilation with specific backend."""
        config = CompileConfig(enabled=True, mode="max-autotune", backend="inductor")
        mock_compiled_model = MagicMock()
        mock_torch_compile.return_value = mock_compiled_model

        result = compile_model(self.model, config)

        mock_configure.assert_called_once_with(cache_size_limit=256)
        mock_fa_fix.assert_called_once()
        mock_torch_compile.assert_called_once_with(
            self.model, mode="max-autotune", fullgraph=False, dynamic=False, backend="inductor"
        )
        assert result is mock_compiled_model

    @patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo")
    @patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix")
    @patch("nemo_automodel.components.utils.compile_utils.torch.compile")
    def test_compile_with_exception_fallback(self, mock_torch_compile, mock_fa_fix, mock_configure):
        """Test compilation fallback when torch.compile raises exception."""
        config = CompileConfig(enabled=True)
        mock_torch_compile.side_effect = Exception("compilation error")

        result = compile_model(self.model, config)

        # Should attempt compilation once and return original model
        assert mock_torch_compile.call_count == 1
        assert result is self.model

    @patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo")
    @patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix")
    @patch("nemo_automodel.components.utils.compile_utils.torch.compile")
    def test_compile_with_options(self, mock_torch_compile, mock_fa_fix, mock_configure):
        """Options go through torch.compile's `options` dict (never flattened), replacing `mode`."""
        config = CompileConfig(
            enabled=True, mode="default", fullgraph=True, dynamic=True, options={"some_option": "value"}
        )
        mock_compiled_model = MagicMock()
        mock_torch_compile.return_value = mock_compiled_model

        result = compile_model(self.model, config)

        mock_torch_compile.assert_called_once_with(
            self.model, fullgraph=True, dynamic=True, options={"some_option": "value"}
        )
        assert result is mock_compiled_model

    def test_options_with_non_default_mode_raises(self):
        """torch.compile treats mode and options as mutually exclusive; reject the combination."""
        from nemo_automodel.components.utils.compile_utils import _resolve_compile_kwargs

        config = CompileConfig(enabled=True, mode="max-autotune", options={"some_option": "value"})
        with pytest.raises(ValueError, match="mutually exclusive"):
            _resolve_compile_kwargs(config)


