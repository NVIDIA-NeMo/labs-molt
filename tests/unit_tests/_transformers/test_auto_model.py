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

import logging
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel._transformers.auto_model import (
    _MAX_BUILD_RETRIES,
    NeMoAutoModelForCausalLM,
    _alias_remote_auto_map_for_target,
    _BaseNeMoAutoModelClass,
    _consume_config_overrides,
    _get_next_fallback_attn,
    _init_model,
    _maybe_reject_tie_word_embeddings_flip,
    _patch_attention,
    _patch_remote_code_compat,
    _resolve_distributed_setup,
)
from nemo_automodel._transformers.model_init import (
    _filter_kwargs_for_init,
    _filter_meta_device_from_init_context,
    _get_hf_meta_device_disabled,
    _get_mixin_wrapped_class,
    _patched_get_init_context,
    no_hf_meta_device,
)
from nemo_automodel.components.checkpoint.utils import _get_checkpoint_tensor_dtypes
from nemo_automodel.components.distributed.config import (
    DistributedSetup,
    FSDP2Config,
    MoEParallelizerConfig,
)
from nemo_automodel.components.distributed.mesh import MeshAxisName, MeshContext
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin


class _FakeMesh:
    def __init__(self, sizes):
        self._sizes = sizes
        self.mesh_dim_names = tuple(sizes)

    def __getitem__(self, axis):
        return types.SimpleNamespace(size=lambda: self._sizes[axis])


class TestResolveMeshContext:

    def test_device_mesh_rejects_mesh_context(self):
        mesh_context = MeshContext()

        with pytest.raises(TypeError, match="DeviceMesh"):
            _resolve_distributed_setup(
                distributed_setup=None,
                device_mesh=mesh_context,
            )

    def test_distributed_setup_and_device_mesh_are_mutually_exclusive(self):
        device_mesh = _FakeMesh({MeshAxisName.DP_SHARD: 1, MeshAxisName.CP: 1, MeshAxisName.TP: 1})
        distributed_setup = DistributedSetup(mesh_context=MeshContext())

        with pytest.raises(ValueError, match="either distributed_setup or device_mesh"):
            _resolve_distributed_setup(
                distributed_setup=distributed_setup,
                device_mesh=device_mesh,
            )

    def test_distributed_setup_input_supplies_configs(self):
        device_mesh = _FakeMesh({MeshAxisName.DP_SHARD: 1, MeshAxisName.CP: 1, MeshAxisName.TP: 1})
        moe_mesh = _FakeMesh({MeshAxisName.EP_SHARD: 1, MeshAxisName.EP: 8})
        distributed_config = FSDP2Config(activation_checkpointing=True)
        moe_config = MoEParallelizerConfig()
        source_setup = DistributedSetup(
            mesh_context=MeshContext.from_meshes(device_mesh, moe_mesh),
            strategy_config=distributed_config,
            moe_parallel_config=moe_config,
            activation_checkpointing=True,
        )

        setup = _resolve_distributed_setup(
            distributed_setup=source_setup,
        )

        assert setup.mesh_context.device_mesh is device_mesh
        assert setup.mesh_context.moe_mesh is moe_mesh
        assert setup.strategy_config is distributed_config
        assert setup.moe_parallel_config is moe_config
        assert setup.activation_checkpointing is True

    def test_missing_distributed_setup_returns_empty_setup(self):
        setup = _resolve_distributed_setup(distributed_setup=None)

        assert isinstance(setup.mesh_context, MeshContext)
        assert setup.strategy_config is None
        assert setup.activation_checkpointing is False


class TestFromPretrainedDeviceMesh:


    def test_from_pretrained_rejects_distributed_setup_with_device_mesh(self):
        device_mesh = _FakeMesh({MeshAxisName.DP_SHARD: 1, MeshAxisName.CP: 1, MeshAxisName.TP: 1})
        distributed_setup = DistributedSetup(mesh_context=MeshContext())

        with pytest.raises(ValueError, match="either distributed_setup or device_mesh"):
            NeMoAutoModelForCausalLM.from_pretrained(
                "test-model",
                distributed_setup=distributed_setup,
                device_mesh=device_mesh,
            )

    def test_from_pretrained_rejects_separate_distributed_kwargs(self):
        with pytest.raises(TypeError, match="distributed_setup"):
            NeMoAutoModelForCausalLM.from_pretrained(
                "test-model",
                distributed_config=FSDP2Config(),
            )

    def test_from_pretrained_rejects_moe_mesh_kwarg(self):
        with pytest.raises(TypeError, match="distributed_setup"):
            NeMoAutoModelForCausalLM.from_pretrained(
                "test-model",
                moe_mesh=object(),
            )


class TestPatchAttention:
    """Test cases for _patch_attention function."""

    def test__patch_attention_basic(self):
        """Test basic _patch_attention functionality."""

        # Create a real object with a forward method to test the actual wrapping
        class DummyModule:
            def forward(self, x):
                """Dummy forward method."""
                return x * 2

        obj = DummyModule()
        original_forward = obj.forward

        with patch("nemo_automodel._transformers.kernel_patches.sdpa_kernel") as mock_sdpa_kernel:
            result = _patch_attention(obj)

            assert result is obj
            # Verify that the forward method was replaced
            assert obj.forward != original_forward
            # Verify the wrapper has the expected docstring prefix
            assert obj.forward.__doc__.startswith("SDPA kernel patch")

            # Call forward and verify sdpa_kernel was used as context manager
            output = obj.forward(5)
            assert output == 10  # Original forward logic still works
            mock_sdpa_kernel.assert_called_once()

    def test__patch_attention_with_custom_sdpa_method(self):
        """Test _patch_attention with custom SDPA method."""
        from torch.nn.attention import SDPBackend

        class DummyModule:
            def forward(self, x):
                """Dummy forward method."""
                return x + 1

        obj = DummyModule()
        custom_sdpa_method = [SDPBackend.FLASH_ATTENTION]

        with (
            patch("nemo_automodel._transformers.kernel_patches.sdpa_kernel") as mock_sdpa_kernel,
            patch("nemo_automodel._transformers.kernel_patches._set_global_sdpa_backends") as mock_global_sdpa,
        ):
            result = _patch_attention(obj, custom_sdpa_method)

            assert result is obj
            # Verify the wrapper has the expected docstring prefix
            assert obj.forward.__doc__.startswith("SDPA kernel patch")

            # Call forward and verify sdpa_kernel was called with the custom method
            output = obj.forward(5)
            assert output == 6  # Original forward logic still works
            mock_global_sdpa.assert_called_once_with(custom_sdpa_method)
            mock_sdpa_kernel.assert_called_once_with(custom_sdpa_method)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_assert_same_signature_matching(self):
        """Test _assert_same_signature with matching signatures."""
        from nemo_automodel._transformers.kernel_patches import _assert_same_signature

        def func1(a, b, c=None):
            pass

        def func2(a, b, c=None):
            pass

        # Should not raise an exception
        _assert_same_signature(func1, func2)

    def test_assert_same_signature_different(self):
        """Test _assert_same_signature with different signatures."""
        from nemo_automodel._transformers.kernel_patches import _assert_same_signature

        def func1(a, b, c=None):
            pass

        def func2(a, b, d=None):
            pass

        # Should raise an AssertionError
        with pytest.raises(AssertionError):
            _assert_same_signature(func1, func2)

    def test_get_next_fallback_attn_valid_priorities(self):
        """Test _get_next_fallback_attn with valid attention implementations."""
        # Test fallback from highest to lowest priority
        assert _get_next_fallback_attn("flash_attention_3") == "flash_attention_2"
        assert _get_next_fallback_attn("flash_attention_2") == "sdpa"
        assert _get_next_fallback_attn("sdpa") == "eager"

        # Test that eager falls back to itself (lowest priority)
        assert _get_next_fallback_attn("eager") == "eager"

    def test_get_next_fallback_attn_invalid_implementations(self):
        """Test _get_next_fallback_attn with invalid/unknown attention implementations."""
        # Test various invalid implementations all fall back to eager
        assert _get_next_fallback_attn("flash_attention_1") == "eager"
        assert _get_next_fallback_attn("unknown_attention") == "eager"
        assert _get_next_fallback_attn("custom_attention") == "eager"
        assert _get_next_fallback_attn("") == "eager"
        assert _get_next_fallback_attn("none") == "eager"
        assert _get_next_fallback_attn("legacy_attention") == "eager"

    @pytest.mark.parametrize(
        "attn_impl,expected",
        [
            ("flash_attention_3", "flash_attention_2"),
            ("flash_attention_2", "sdpa"),
            ("sdpa", "eager"),
            ("eager", "eager"),
            ("invalid", "eager"),
            ("custom_impl", "eager"),
            ("", "eager"),
        ],
    )
    def test_get_next_fallback_attn_parametrized(self, attn_impl, expected):
        """Parametrized test for _get_next_fallback_attn covering all scenarios."""
        assert _get_next_fallback_attn(attn_impl) == expected

    def test_get_next_fallback_attn_edge_cases(self):
        """Test _get_next_fallback_attn with edge cases and special inputs."""
        # Test with None (should be treated as unknown)
        assert _get_next_fallback_attn(None) == "eager"

        # Test case sensitivity (should be treated as unknown since not exact match)
        assert _get_next_fallback_attn("EAGER") == "eager"
        assert _get_next_fallback_attn("Flash_Attention_2") == "eager"
        assert _get_next_fallback_attn("SDPA") == "eager"

        # Test with whitespace (should be treated as unknown)
        assert _get_next_fallback_attn(" eager ") == "eager"
        assert _get_next_fallback_attn("sdpa ") == "eager"

        # Test with numeric strings
        assert _get_next_fallback_attn("123") == "eager"
        assert _get_next_fallback_attn("0") == "eager"


class TestModelRuntimePatches:
    """Test cases for model runtime patch dispatch."""

    class _DummyModel(torch.nn.Module):
        def __init__(self, architectures=None):
            super().__init__()
            self.config = types.SimpleNamespace(architectures=architectures)

    def test_apply_model_runtime_patches_dispatches_by_architecture(self):
        # The registry mechanism is exercised with a temporary test entry — the
        # built-in registry no longer ships any entries (Qwen3.5 builds its
        # CP/fp32-gate modules at construction instead of patching at load time).
        import nemo_automodel._transformers.kernel_patches as kp
        from nemo_automodel._transformers.kernel_patches import apply_model_runtime_patches

        model = self._DummyModel(["FakeArchForCausalLM"])
        mesh = types.SimpleNamespace(cp_size=1)
        calls = []

        def fake_hook(model, mesh):
            calls.append((model, mesh))
            return model

        fake_module = types.SimpleNamespace(apply_model_runtime_patches=fake_hook)
        test_registry = {"FakeArchForCausalLM": ("fake.module.path", "apply_model_runtime_patches")}

        with (
            patch.object(kp, "_MODEL_RUNTIME_PATCHES", test_registry),
            patch(
                "nemo_automodel._transformers.kernel_patches.importlib.import_module",
                return_value=fake_module,
            ) as mock_import,
        ):
            assert apply_model_runtime_patches(model, mesh) is model

        mock_import.assert_called_once_with("fake.module.path")
        assert calls == [(model, mesh)]

    def test_apply_model_runtime_patches_deduplicates_hook_specs(self):
        # Two architectures sharing one hook spec must invoke the hook once.
        import nemo_automodel._transformers.kernel_patches as kp
        from nemo_automodel._transformers.kernel_patches import apply_model_runtime_patches

        model = self._DummyModel(["FakeArchA", "FakeArchB"])
        mesh = types.SimpleNamespace(cp_size=2)
        calls = []

        def fake_hook(model, mesh):
            calls.append((model, mesh))
            return model

        fake_module = types.SimpleNamespace(apply_model_runtime_patches=fake_hook)
        shared_spec = ("fake.module.path", "apply_model_runtime_patches")
        test_registry = {"FakeArchA": shared_spec, "FakeArchB": shared_spec}

        with (
            patch.object(kp, "_MODEL_RUNTIME_PATCHES", test_registry),
            patch(
                "nemo_automodel._transformers.kernel_patches.importlib.import_module",
                return_value=fake_module,
            ),
        ):
            assert apply_model_runtime_patches(model, mesh) is model

        assert calls == [(model, mesh)]


    def test_apply_model_runtime_patches_skips_unavailable_hook(self):
        from nemo_automodel._transformers.kernel_patches import apply_model_runtime_patches

        model = self._DummyModel(["Qwen3_5ForCausalLM"])
        mesh = types.SimpleNamespace(cp_size=1)

        with patch(
            "nemo_automodel._transformers.kernel_patches.importlib.import_module",
            side_effect=ImportError,
        ):
            assert apply_model_runtime_patches(model, mesh) is model


class DummyModel(torch.nn.Module):
    """A tiny nn.Module that behaves enough like a HF/BERT style model."""

    def __init__(self):
        super().__init__()
        self.config = {}  # _patch_liger_kernel calls  model.config.update(...)
        self.called = False  # turned on by fake liger kernel

    def mark(self):
        self.called = True


def prepare_env(monkeypatch, target_mod, *, has_liger=True, apply_ok=True):
    """
    Patch every external symbol that _patch_liger_kernel touches.

    Parameters
    ----------
    has_liger : bool
        Value for HAS_LIGER_KERNEL global.
    apply_ok : bool
        Force liger_kernel_trf._apply_liger_kernel_to_instance to succeed/fail.
    """
    monkeypatch.setattr(target_mod, "HAS_LIGER_KERNEL", has_liger, raising=False)

    apply_mock = MagicMock()

    if apply_ok:
        # mark model when called so we can assert later
        apply_mock.side_effect = lambda *, model: model.mark()
    else:
        apply_mock.side_effect = RuntimeError("boom")

    liger_stub = types.SimpleNamespace(_apply_liger_kernel_to_instance=apply_mock)
    monkeypatch.setattr(target_mod, "liger_kernel_trf", liger_stub, raising=False)

    patch_attn_mock = MagicMock(side_effect=lambda *args, **kwargs: args[0])
    monkeypatch.setattr(target_mod, "_patch_attention", patch_attn_mock, raising=True)

    return apply_mock, patch_attn_mock


def test_patch_liger_kernel_success(monkeypatch):
    """Test _patch_liger_kernel successfully applies liger kernel when available."""
    import nemo_automodel._transformers.kernel_patches as tgt

    apply_mock, attn_mock = prepare_env(monkeypatch, tgt, has_liger=True, apply_ok=True)

    model = DummyModel()
    patched = tgt._patch_liger_kernel(model)

    # Returns same instance
    assert patched is model

    # Liger kernel was applied
    apply_mock.assert_called_once()
    assert model.called is True

    # SDPA not called inside _patch_liger_kernel (it's called separately)
    attn_mock.assert_not_called()


def test_liger_not_available(monkeypatch):
    """
    Asked for Liger but HAS_LIGER_KERNEL is False.
    Expect: return untouched model, _patch_attention still invoked,
            no exceptions thrown.
    """
    import nemo_automodel._transformers.kernel_patches as tgt

    apply_mock, attn_mock = prepare_env(
        monkeypatch,
        tgt,
        has_liger=False,  # unavailable
        apply_ok=True,
    )

    model = DummyModel()
    out = tgt._patch_liger_kernel(model)

    # untouched instance returned
    assert out is model
    assert model.called is False
    # _apply never called, because we short-circuit when HAS_LIGER_KERNEL==False
    apply_mock.assert_not_called()
    attn_mock.assert_not_called()


def test_liger_apply_failure_raises(monkeypatch):
    """
    If _apply_liger_kernel_to_instance throws, _patch_liger_kernel must
    clean up and raise RuntimeError.
    """
    import nemo_automodel._transformers.kernel_patches as tgt

    prepare_env(
        monkeypatch,
        tgt,
        has_liger=True,
        apply_ok=False,  # force failure
    )

    with pytest.raises(RuntimeError, match="Failed to patch model"):
        tgt._patch_liger_kernel(DummyModel())


def test_patch_liger_kernel_skips_non_nn_module(monkeypatch, caplog):
    """
    When model is not an nn.Module (e.g., a lightweight mock), _patch_liger_kernel
    should skip patching and return the model unchanged with a warning.
    """
    import nemo_automodel._transformers.kernel_patches as tgt

    apply_mock, attn_mock = prepare_env(
        monkeypatch,
        tgt,
        has_liger=True,
        apply_ok=True,
    )

    # Create a non-nn.Module mock object
    mock_model = MagicMock(spec=[])  # Empty spec means no nn.Module methods
    mock_model.config = {}

    with caplog.at_level(logging.WARNING):
        result = tgt._patch_liger_kernel(mock_model)

    # Should return the same mock unchanged
    assert result is mock_model
    # Liger kernel should NOT be applied
    apply_mock.assert_not_called()
    # Warning should be logged
    assert "Skipping Liger Kernel patch for non-nn.Module model" in caplog.text


# =============================================================================
# Tests for _get_mixin_wrapped_class
# =============================================================================


class TestGetMixinWrappedClass:
    """Test cases for _get_mixin_wrapped_class function."""

    def test_returns_original_if_already_has_mixin(self):
        """When model class already inherits from HFCheckpointingMixin, return it unchanged."""

        class ModelWithMixin(HFCheckpointingMixin, torch.nn.Module):
            pass

        result = _get_mixin_wrapped_class(ModelWithMixin)
        assert result is ModelWithMixin

    def test_creates_wrapper_for_hf_class_with_correct_attributes(self):
        """For HF model classes, create a wrapper inheriting from both and preserving attributes."""

        class PlainModel(torch.nn.Module):
            pass

        result = _get_mixin_wrapped_class(PlainModel)

        # Should be a new class inheriting from both
        assert result is not PlainModel
        assert issubclass(result, HFCheckpointingMixin)
        assert issubclass(result, PlainModel)
        # Should preserve original class attributes
        assert result.__module__ == PlainModel.__module__
        assert result.__qualname__ == PlainModel.__qualname__
        assert result.__name__ == PlainModel.__name__


# =============================================================================
# Tests for _apply_peft_and_lower_precision
# =============================================================================


# =============================================================================
# Tests for _consume_config_overrides and _filter_kwargs_for_init
# =============================================================================


class TestConsumeConfigOverrides:
    """Test cases for _consume_config_overrides function."""

    def test_consume_config_overrides_moves_config_keys_to_config(self):
        """Config-related kwargs are moved from kwargs dict to config object."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"output_hidden_states": True, "use_cache": True}

        kwargs = {"output_hidden_states": True, "some_other_arg": 42}

        _consume_config_overrides(mock_config, kwargs)

        # output_hidden_states should be moved to config
        assert "output_hidden_states" not in kwargs
        assert "some_other_arg" in kwargs
        mock_config.output_hidden_states = True  # Should be set on config

    def test_consume_config_overrides_preserves_init_param_names(self):
        """Keys that are in init_param_names are kept in kwargs."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"output_hidden_states": True}

        kwargs = {"output_hidden_states": True, "explicit_param": 42}

        _consume_config_overrides(mock_config, kwargs, init_param_names={"explicit_param", "output_hidden_states"})

        # Both should be kept since they're in init_param_names
        assert "output_hidden_states" in kwargs
        assert "explicit_param" in kwargs


class TestAliasRemoteAutoMapForTarget:
    """Tests for the auto_map alias fallback used when a trust-remote-code config
    ships only ``auto_map[AutoModel]`` and HF's ``AutoModelFor*`` resolution
    can't find the right class.
    """

    def test_aliases_automodel_to_target_when_conditions_met(self):
        cfg = MagicMock()
        cfg.auto_map = {"AutoModel": "modeling.MyModel"}
        result = _alias_remote_auto_map_for_target(
            ("/some/path",), {"trust_remote_code": True, "config": cfg}, "AutoModelForCausalLM"
        )
        assert result is cfg
        assert result.auto_map["AutoModelForCausalLM"] == "modeling.MyModel"

    def test_returns_none_when_trust_remote_code_not_set(self):
        cfg = MagicMock()
        cfg.auto_map = {"AutoModel": "modeling.MyModel"}
        result = _alias_remote_auto_map_for_target(("/some/path",), {"config": cfg}, "AutoModelForCausalLM")
        assert result is None


class TestGetCheckpointTensorDtypes:
    def test_uses_provided_state_dict_dtypes(self):
        state_dict = {
            "linear.weight": torch.empty(2, 2, dtype=torch.bfloat16),
            "norm.weight": torch.empty(2, dtype=torch.float32),
        }

        result = _get_checkpoint_tensor_dtypes("ignored", object(), {"state_dict": state_dict})

        assert result == {
            "linear.weight": torch.bfloat16,
            "norm.weight": torch.float32,
        }


class TestFilterKwargsForInit:
    """Test cases for _filter_kwargs_for_init function."""

    def test_filter_kwargs_for_init_removes_unknown_kwargs(self):
        """Filters out kwargs not in model __init__ signature."""

        class ModelWithSpecificInit:
            def __init__(self, config, a, b):
                pass

        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = _filter_kwargs_for_init(ModelWithSpecificInit, kwargs)

        assert "a" in result
        assert "b" in result
        assert "c" not in result
        assert "d" not in result

    def test_filter_kwargs_for_init_keeps_all_with_var_keyword(self):
        """If __init__ has **kwargs, returns all kwargs unchanged."""

        class ModelWithVarKwargs:
            def __init__(self, config, **kwargs):
                pass

        kwargs = {"a": 1, "b": 2, "c": 3}
        result = _filter_kwargs_for_init(ModelWithVarKwargs, kwargs)

        assert result == kwargs


# =============================================================================
# Tests for NEED_SETUP_CACHE_CLASSES_MAPPING backward compatibility shim
# =============================================================================


class TestNeedSetupCacheClassesMapping:
    """Test cases for the NEED_SETUP_CACHE_CLASSES_MAPPING backward compat shim."""

    def test_shim_does_not_overwrite_existing_attribute(self):
        """If NEED_SETUP_CACHE_CLASSES_MAPPING already exists, shim doesn't overwrite."""
        import importlib

        import transformers.generation.utils as gen_utils

        sentinel = {"test": "sentinel_value"}
        gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = sentinel

        # Re-import to trigger the shim code
        import nemo_automodel._transformers.auto_model as mod

        importlib.reload(mod)

        # The sentinel should still be there (shim didn't overwrite)
        assert gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING is sentinel

        # Clean up
        del gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING

    def test_shim_creates_attribute_when_missing(self):
        """If NEED_SETUP_CACHE_CLASSES_MAPPING is missing, shim creates it."""
        import importlib

        import transformers.generation.utils as gen_utils

        # Remove the attribute if it exists
        if hasattr(gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
            delattr(gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING")

        # Re-import to trigger the shim
        import nemo_automodel._transformers.auto_model as mod

        importlib.reload(mod)

        assert hasattr(gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING")
        mapping = gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING
        assert "static" in mapping

        # Clean up
        del gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING


# =============================================================================
# Tests for _model_mapping KeyError fallback in _init_model
# =============================================================================


class TestModelMappingKeyErrorFallback:
    """Test cases for _model_mapping KeyError fallback in _init_model."""

    def _make_cls(self, model_mapping_dict):
        """Create a mock cls with _model_mapping, parent class methods, etc."""
        cls = MagicMock()
        cls._model_mapping = model_mapping_dict
        return cls

    def test_force_hf_known_config_type(self):
        """force_hf path: _model_mapping lookup succeeds, class gets wrapped with mixin."""

        class FakeConfig:
            _commit_hash = None
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = FakeConfig()
        fake_model = FakeModel()

        cls = self._make_cls({FakeConfig: FakeModel})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,  # Pass config object directly (not str) to skip pretrained path
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=True,
            )

        assert is_custom is False
        mock_wrap.assert_called_once_with(FakeModel)

    def test_force_hf_unknown_config_falls_back_to_type_model(self):
        """force_hf path: KeyError in _model_mapping falls back to type(model)."""

        class UnknownConfig:
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = UnknownConfig()
        fake_model = FakeModel()

        # _model_mapping does NOT have UnknownConfig
        cls = self._make_cls({})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=True,
            )

        assert is_custom is False
        # Fallback: type(model) = FakeModel
        mock_wrap.assert_called_once_with(FakeModel)

    def test_force_hf_pretrained_restores_checkpoint_dtype_per_tensor(self):
        """force_hf pretrained path should restore each tensor dtype from the checkpoint."""

        class FakeConfig:
            _commit_hash = None
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.norm = torch.nn.LayerNorm(2)

        fake_config = FakeConfig()
        fake_model = FakeModel().to(torch.float32)

        cls = self._make_cls({})
        cls._from_pretrained_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch(
                "nemo_automodel.components.checkpoint.utils._get_checkpoint_tensor_dtypes",
                return_value={
                    "linear.weight": torch.bfloat16,
                    "norm.weight": torch.float32,
                },
            ),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                "test-model",
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=True,
            )

        assert is_custom is False
        assert cls._from_pretrained_parent_class.call_args.kwargs["dtype"] == "auto"
        assert fake_model.linear.weight.dtype == torch.bfloat16
        assert fake_model.norm.weight.dtype == torch.float32
        mock_wrap.assert_called_once_with(FakeModel)

    def test_force_hf_pretrained_explicit_fp32_promotes_all_to_fp32(self):
        """Explicit fp32 request unifies every floating tensor to fp32 (master weights)."""

        class FakeConfig:
            _commit_hash = None
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.norm = torch.nn.LayerNorm(2)

        fake_config = FakeConfig()
        # Simulate HF's mixed-dtype load: most params bf16, a stray param fp32.
        fake_model = FakeModel()
        fake_model.linear.to(torch.bfloat16)
        fake_model.norm.to(torch.float32)

        cls = self._make_cls({})
        cls._from_pretrained_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch(
                "nemo_automodel.components.checkpoint.utils._get_checkpoint_tensor_dtypes",
                return_value={
                    "linear.weight": torch.bfloat16,
                    "norm.weight": torch.float32,
                },
            ),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            _init_model(
                cls,
                "test-model",
                attn_implementation="eager",
                torch_dtype=torch.float32,
                quantization_config=None,
                force_hf=True,
            )

        assert fake_model.linear.weight.dtype == torch.float32
        assert fake_model.norm.weight.dtype == torch.float32
        # Storage was upcast to fp32, but the checkpoint's original (compute) dtype is
        # recorded so downstream sharding can keep the bulk in bf16 compute.
        assert fake_model.linear.weight._hf_compute_dtype == torch.bfloat16
        assert fake_model.norm.weight._hf_compute_dtype == torch.float32

    def test_force_hf_pretrained_explicit_bf16_preserves_intrinsic_fp32(self):
        """Explicit bf16 request keeps bf16 params bf16 but preserves intrinsically-fp32 params."""

        class FakeConfig:
            _commit_hash = None
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.norm = torch.nn.LayerNorm(2)

        fake_config = FakeConfig()
        fake_model = FakeModel()
        fake_model.linear.to(torch.bfloat16)
        fake_model.norm.to(torch.float32)

        cls = self._make_cls({})
        cls._from_pretrained_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch(
                "nemo_automodel.components.checkpoint.utils._get_checkpoint_tensor_dtypes",
                return_value={
                    "linear.weight": torch.bfloat16,
                    "norm.weight": torch.float32,
                },
            ),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            _init_model(
                cls,
                "test-model",
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                quantization_config=None,
                force_hf=True,
            )

        assert fake_model.linear.weight.dtype == torch.bfloat16
        # promote(fp32, bf16) == fp32 -> intrinsically-fp32 checkpoint param survives.
        assert fake_model.norm.weight.dtype == torch.float32

    def test_fallback_path_known_config_type(self):
        """Fallback (non-force_hf, no custom model) path: _model_mapping succeeds."""

        class FakeConfig:
            _commit_hash = None
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = FakeConfig()
        fake_model = FakeModel()

        cls = self._make_cls({FakeConfig: FakeModel})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch("nemo_automodel._transformers.model_init.get_architectures", return_value=[]),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=False,
            )

        assert is_custom is False
        mock_wrap.assert_called_once_with(FakeModel)

    def test_fallback_pretrained_restores_tied_weight_checkpoint_dtype(self):
        """Fallback pretrained path should preserve tied-weight checkpoint dtypes."""

        class FakeConfig:
            _commit_hash = None
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(4, 3)
                self.lm_head = torch.nn.Linear(3, 4, bias=False)
                self.lm_head.weight = self.embed_tokens.weight

        fake_config = FakeConfig()
        fake_model = FakeModel().to(torch.float32)

        cls = self._make_cls({})
        cls._from_pretrained_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch(
                "nemo_automodel.components.checkpoint.utils._get_checkpoint_tensor_dtypes",
                return_value={"lm_head.weight": torch.bfloat16},
            ),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                "test-model",
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                quantization_config=None,
                force_hf=False,
            )

        assert is_custom is False
        assert cls._from_pretrained_parent_class.call_args.kwargs["dtype"] == torch.bfloat16
        assert fake_model.lm_head.weight is fake_model.embed_tokens.weight
        assert fake_model.lm_head.weight.dtype == torch.bfloat16
        assert fake_model.embed_tokens.weight.dtype == torch.bfloat16
        mock_wrap.assert_called_once_with(FakeModel)

    def test_fallback_path_skips_incompatible_shared_arch_custom_model(self):
        """Shared architecture names should stay on HF when the config does not match our custom model."""

        class FakeConfig:
            _commit_hash = None
            name_or_path = "test-model"
            architectures = ["NemotronHForCausalLM"]

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = FakeConfig()
        fake_model = FakeModel()

        cls = self._make_cls({FakeConfig: FakeModel})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.ModelRegistry.has_custom_model", return_value=True),
            patch("nemo_automodel._transformers.model_init.ModelRegistry.resolve_custom_model_cls") as mock_resolve,
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=False,
            )

        assert is_custom is False
        mock_resolve.assert_not_called()
        mock_wrap.assert_called_once_with(FakeModel)

    def test_fallback_path_unknown_config_falls_back_to_type_model(self):
        """Fallback path: KeyError in _model_mapping falls back to type(model)."""

        class UnknownConfig:
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = UnknownConfig()
        fake_model = FakeModel()

        cls = self._make_cls({})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch("nemo_automodel._transformers.model_init.get_architectures", return_value=[]),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=False,
            )

        assert is_custom is False
        mock_wrap.assert_called_once_with(FakeModel)


class TestFromPretrainedSafetensorsFallback:
    """Tests for the use_safetensors=False fallback in _from_pretrained_parent_class (issue #1511)."""

    @staticmethod
    @contextmanager
    def _patch_parent_fp(side_effects):
        """Inject from_pretrained into AutoModelForCausalLM so super() finds it in the MRO.

        super() in _from_pretrained_parent_class (defined on _BaseNeMoAutoModelClass)
        starts resolution at AutoModelForCausalLM, so we place the mock there.
        """
        from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM

        had_original = "from_pretrained" in HFAutoModelForCausalLM.__dict__
        original = HFAutoModelForCausalLM.__dict__.get("from_pretrained")
        calls = []
        effects = list(side_effects)
        idx = [0]

        def fake_fp(cls_arg, *args, **kwargs):
            calls.append(kwargs.copy())
            i = idx[0]
            idx[0] += 1
            if i < len(effects):
                val = effects[i]
                if isinstance(val, Exception):
                    raise val
                return val
            raise RuntimeError("Unexpected extra call to from_pretrained")

        HFAutoModelForCausalLM.from_pretrained = classmethod(fake_fp)
        try:
            yield calls
        finally:
            if had_original:
                HFAutoModelForCausalLM.from_pretrained = original
            else:
                del HFAutoModelForCausalLM.from_pretrained

    def test_retries_with_use_safetensors_false_on_oserror(self):
        """When super().from_pretrained raises OSError, retry with use_safetensors=False."""
        sentinel = MagicMock()
        with self._patch_parent_fp([OSError("model.safetensors not found"), sentinel]) as calls:
            result = NeMoAutoModelForCausalLM._from_pretrained_parent_class("test-model")
        assert result is sentinel
        assert len(calls) == 2
        assert calls[1].get("use_safetensors") is False

    def test_reraises_oserror_when_safetensors_already_false(self):
        """When use_safetensors is already False, the OSError is not swallowed."""
        with self._patch_parent_fp([OSError("file not found")]) as calls:
            with pytest.raises(OSError, match="file not found"):
                NeMoAutoModelForCausalLM._from_pretrained_parent_class(
                    "test-model",
                    use_safetensors=False,
                )
        assert len(calls) == 1

    def test_restores_class_name_after_success(self):
        """cls.__name__ is restored after a successful load."""
        original_name = NeMoAutoModelForCausalLM.__name__
        with self._patch_parent_fp([MagicMock()]):
            NeMoAutoModelForCausalLM._from_pretrained_parent_class("test-model")
        assert NeMoAutoModelForCausalLM.__name__ == original_name

    def test_restores_class_name_after_oserror_fallback(self):
        """cls.__name__ is restored even when the OSError fallback path is taken."""
        original_name = NeMoAutoModelForCausalLM.__name__
        with self._patch_parent_fp([OSError("not found"), MagicMock()]):
            NeMoAutoModelForCausalLM._from_pretrained_parent_class("test-model")
        assert NeMoAutoModelForCausalLM.__name__ == original_name

    def test_restores_class_name_after_unrecoverable_error(self):
        """cls.__name__ is restored when the OSError cannot be recovered."""
        original_name = NeMoAutoModelForCausalLM.__name__
        with self._patch_parent_fp([OSError("permission denied")]):
            with pytest.raises(OSError):
                NeMoAutoModelForCausalLM._from_pretrained_parent_class(
                    "test-model",
                    use_safetensors=False,
                )
        assert NeMoAutoModelForCausalLM.__name__ == original_name

    def test_aliases_and_retries_on_unrecognized_config_error(self):
        """Unrecognized-config-class ValueError triggers auto_map alias + retry."""
        patched_cfg = MagicMock()
        sentinel = MagicMock()
        error = ValueError("Unrecognized configuration class <class 'Foo'> for AutoModelForCausalLM")
        with self._patch_parent_fp([error, sentinel]) as calls:
            with patch(
                "nemo_automodel._transformers.auto_model._alias_remote_auto_map_for_target",
                return_value=patched_cfg,
            ) as mock_alias:
                result = NeMoAutoModelForCausalLM._from_pretrained_parent_class("test-model", trust_remote_code=True)
        assert result is sentinel
        assert len(calls) == 2
        # Retry passes the patched config.
        assert calls[1].get("config") is patched_cfg
        mock_alias.assert_called_once()


class TestBuildModelRetryDepth:
    """Tests for _build_model retry depth limiting (issue #1510)."""

    @staticmethod
    def _make_build_kwargs():
        """Minimal kwargs for _build_model with all required parameters."""
        mock_config = MagicMock()
        mock_config.quantization_config = None
        mock_config._commit_hash = None
        mesh = MagicMock()
        mesh.tp_size = 1
        mesh.cp_size = 1
        return dict(
            is_hf_model=True,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            sdpa_method=None,
            torch_dtype="auto",
            attn_implementation="eager",
            quantization_config=None,
            force_hf=False,
            model_wrapper=None,
            autopipeline=None,
            parallelize_fn=None,
            qat_quantizer=None,
            mesh=mesh,
            loss_fn=None,
            peft_config=None,
            fp8_config=None,
            load_base_model=True,
        ), mock_config

    def test_raises_after_max_retries_instead_of_recursion(self):
        """Deterministic errors propagate after _MAX_BUILD_RETRIES (no RecursionError)."""
        build_kwargs, mock_config = self._make_build_kwargs()
        with (
            patch("nemo_automodel._transformers.auto_model._apply_preload_overrides", return_value=("eager", False)),
            patch("nemo_automodel._transformers.auto_model._init_model") as mock_init,
            patch("nemo_automodel._transformers.auto_model.get_world_size_safe", return_value=1),
            patch("torch.cuda.current_device", return_value=0),
        ):
            mock_init.side_effect = ValueError("model does not support sdpa")
            with pytest.raises(ValueError, match="does not support"):
                _BaseNeMoAutoModelClass._build_model(mock_config, **build_kwargs)
            assert mock_init.call_count == _MAX_BUILD_RETRIES + 1

    def test_immediate_raise_when_already_at_max_depth(self):
        """When _retry_depth is already at _MAX_BUILD_RETRIES, errors propagate on first attempt."""
        build_kwargs, mock_config = self._make_build_kwargs()
        build_kwargs["_retry_depth"] = _MAX_BUILD_RETRIES
        with (
            patch("nemo_automodel._transformers.auto_model._apply_preload_overrides", return_value=("eager", False)),
            patch("nemo_automodel._transformers.auto_model._init_model") as mock_init,
            patch("nemo_automodel._transformers.auto_model.get_world_size_safe", return_value=1),
            patch("torch.cuda.current_device", return_value=0),
        ):
            mock_init.side_effect = ValueError("model does not support sdpa")
            with pytest.raises(ValueError, match="does not support"):
                _BaseNeMoAutoModelClass._build_model(mock_config, **build_kwargs)
            assert mock_init.call_count == 1


class TestFilterMetaDeviceFromInitContext:
    def test_removes_meta_device(self):
        contexts = [torch.device("meta"), torch.float32]
        result = _filter_meta_device_from_init_context(contexts)
        assert torch.device("meta") not in result
        assert torch.float32 in result

    def test_keeps_non_meta_devices(self):
        contexts = [torch.device("cpu"), torch.device("cuda")]
        result = _filter_meta_device_from_init_context(contexts)
        assert len(result) == 2

    def test_empty_list(self):
        assert _filter_meta_device_from_init_context([]) == []


class TestPatchedGetInitContext:
    def test_forwards_extra_args(self):
        """Verify _patched_get_init_context forwards *args/**kwargs (transformers v5.3.0 compat)."""
        received_args = {}

        def mock_original(cls, dtype, is_quantized, _is_ds_init_called, *args, **kwargs):
            received_args["args"] = args
            received_args["kwargs"] = kwargs
            return []

        with patch.object(_patched_get_init_context, "__wrapped__", mock_original):
            _patched_get_init_context(None, torch.float32, False, False, True, extra_kwarg="test")

        assert received_args["args"] == (True,)
        assert received_args["kwargs"] == {"extra_kwarg": "test"}

    def test_forwards_allow_all_kernels(self):
        """Simulate the exact transformers v5.3.0 call with allow_all_kernels param."""
        received_args = {}

        def mock_original(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels):
            received_args["allow_all_kernels"] = allow_all_kernels
            return []

        with patch.object(_patched_get_init_context, "__wrapped__", mock_original):
            _patched_get_init_context(None, torch.float32, False, False, None)

        assert received_args["allow_all_kernels"] is None

    def test_strips_meta_device_when_disabled(self):
        """When no_hf_meta_device context is active, meta devices are filtered out."""

        def mock_original(cls, dtype, is_quantized, _is_ds_init_called, *args, **kwargs):
            return [torch.device("meta"), torch.float32]

        with patch.object(_patched_get_init_context, "__wrapped__", mock_original):
            with no_hf_meta_device():
                result = _patched_get_init_context(None, torch.float32, False, False)
            assert torch.device("meta") not in result
            assert torch.float32 in result

    def test_keeps_meta_device_by_default(self):
        """Without no_hf_meta_device, meta devices are preserved."""

        def mock_original(cls, dtype, is_quantized, _is_ds_init_called, *args, **kwargs):
            return [torch.device("meta"), torch.float32]

        with patch.object(_patched_get_init_context, "__wrapped__", mock_original):
            result = _patched_get_init_context(None, torch.float32, False, False)
        assert torch.device("meta") in result

    def test_patch_installed_on_pretrained_model(self):
        """Verify the patch is actually installed on PreTrainedModel."""
        from transformers import PreTrainedModel

        assert PreTrainedModel.get_init_context.__func__ is _patched_get_init_context


class TestNoHfMetaDevice:
    def test_context_manager_sets_and_restores(self):
        assert not _get_hf_meta_device_disabled()
        with no_hf_meta_device():
            assert _get_hf_meta_device_disabled()
        assert not _get_hf_meta_device_disabled()

    def test_nested_context_managers(self):
        with no_hf_meta_device():
            assert _get_hf_meta_device_disabled()
            with no_hf_meta_device():
                assert _get_hf_meta_device_disabled()
            assert _get_hf_meta_device_disabled()
        assert not _get_hf_meta_device_disabled()


# =============================================================================
# Tests for _patch_remote_code_compat
# =============================================================================


class TestPatchRemoteCodeCompat:
    """Tests for the remote-code compatibility patch."""

    @pytest.fixture(autouse=True)
    def reset_patch_state(self):
        """Reset the global patch state and restore original _finalize_model_loading."""
        from transformers import PreTrainedModel

        import nemo_automodel._transformers.auto_model as mod

        orig_finalize = PreTrainedModel._finalize_model_loading
        orig_flag = mod._remote_code_compat_applied
        mod._remote_code_compat_applied = False
        yield
        PreTrainedModel._finalize_model_loading = orig_finalize
        mod._remote_code_compat_applied = orig_flag

    def test_patch_is_idempotent(self):
        """Calling _patch_remote_code_compat twice should not double-wrap."""
        from transformers import PreTrainedModel

        _patch_remote_code_compat()
        first = PreTrainedModel._finalize_model_loading
        _patch_remote_code_compat()
        second = PreTrainedModel._finalize_model_loading
        assert first is second

    def test_sets_all_tied_weights_keys(self):
        """Patch should set all_tied_weights_keys if missing."""
        from transformers import PreTrainedModel

        _patch_remote_code_compat()

        # Create a mock model missing the attribute
        model = MagicMock(spec=[])
        model.get_expanded_tied_weights_keys = MagicMock(return_value=["some.key"])
        model.config = MagicMock()
        model.config.use_cache = False

        # The model class uses base tie_weights (no wrapping needed)
        model_cls = type(model)
        model_cls.tie_weights = PreTrainedModel.tie_weights

        load_config = MagicMock()
        loading_info = MagicMock()

        # The patched finalize will call _orig_finalize which may fail on mock,
        # but we can check the attribute was set before that point
        try:
            PreTrainedModel._finalize_model_loading(model, load_config, loading_info)
        except Exception:
            pass

        assert hasattr(model, "all_tied_weights_keys")

    def test_wraps_old_tie_weights_signature(self):
        """Patch should wrap tie_weights that lacks `missing_keys` param."""
        from transformers import PreTrainedModel

        _patch_remote_code_compat()

        # Create a real class with an old-style tie_weights (no missing_keys param)
        class OldModel(PreTrainedModel):
            config_class = type("DummyConfig", (), {"model_type": "dummy"})

            def __init__(self):
                # Skip PreTrainedModel.__init__ — we just need the class hierarchy
                pass

            def tie_weights(self):
                self.tied = True

        # Simulate what _compat_finalize does: inspect type(model) and wrap tie_weights
        model = object.__new__(OldModel)
        model.all_tied_weights_keys = []
        model.config = types.SimpleNamespace(use_cache=False)
        model.get_expanded_tied_weights_keys = lambda all_submodels=True: []

        load_config = MagicMock()
        loading_info = MagicMock()

        try:
            PreTrainedModel._finalize_model_loading(model, load_config, loading_info)
        except Exception:
            pass

        # The wrapper should now accept missing_keys without TypeError
        try:
            OldModel.tie_weights(model, missing_keys=["foo"])
        except TypeError:
            pytest.fail("Wrapped tie_weights should accept missing_keys kwarg")

    def test_sets_missing_config_defaults(self):
        """Patch should set use_cache=False if missing from config."""
        from transformers import PreTrainedModel

        _patch_remote_code_compat()

        model = MagicMock(spec=[])
        model.get_expanded_tied_weights_keys = MagicMock(return_value=[])
        model.config = types.SimpleNamespace()  # no use_cache attribute

        model_cls = type(model)
        model_cls.tie_weights = PreTrainedModel.tie_weights

        load_config = MagicMock()
        loading_info = MagicMock()

        try:
            PreTrainedModel._finalize_model_loading(model, load_config, loading_info)
        except Exception:
            pass

        assert hasattr(model.config, "use_cache")
        assert model.config.use_cache is False

    def test_compatible_model_unaffected(self):
        """A model that already has all required attributes should pass through unchanged."""
        from transformers import PreTrainedModel

        _patch_remote_code_compat()

        model = MagicMock(spec=[])
        model.all_tied_weights_keys = ["existing.key"]
        model.config = MagicMock()
        model.config.use_cache = True  # already set

        model_cls = type(model)
        model_cls.tie_weights = PreTrainedModel.tie_weights

        load_config = MagicMock()
        loading_info = MagicMock()

        try:
            PreTrainedModel._finalize_model_loading(model, load_config, loading_info)
        except Exception:
            pass

        # Existing values should be preserved
        assert model.all_tied_weights_keys == ["existing.key"]
        assert model.config.use_cache is True


class TestMaybeRejectTieWordEmbeddingsFlip:
    """Layer 2 from_pretrained flip guard (_maybe_reject_tie_word_embeddings_flip).

    from_pretrained accepts str | os.PathLike sources, so the guard must normalize
    path-like inputs with os.fspath() and enforce the checkpoint's raw
    tie_word_embeddings for both — a pathlib.Path local checkpoint must not bypass
    the check. Non-path sources are skipped and a failed raw-config re-read is
    conservative (never blocks the load).
    """

    @staticmethod
    def _requested_config(tied):
        return types.SimpleNamespace(tie_word_embeddings=tied, architectures=["DummyForCausalLM"])

    @staticmethod
    def _patch_raw_config(**kwargs):
        return patch(
            "nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained",
            **kwargs,
        )

    def test_str_source_flip_rejected(self):
        raw = types.SimpleNamespace(tie_word_embeddings=True)
        with self._patch_raw_config(return_value=raw):
            with pytest.raises(NotImplementedError, match="flipping the flag is not supported"):
                _maybe_reject_tie_word_embeddings_flip("org/tied-model", self._requested_config(tied=False), {})

    def test_pathlib_path_source_flip_rejected(self):
        from pathlib import Path

        raw = types.SimpleNamespace(tie_word_embeddings=True)
        with self._patch_raw_config(return_value=raw):
            with pytest.raises(NotImplementedError, match="flipping the flag is not supported"):
                _maybe_reject_tie_word_embeddings_flip(
                    Path("/ckpts/tied-model"), self._requested_config(tied=False), {}
                )

    def test_pathlib_path_source_matching_value_passes(self):
        from pathlib import Path

        raw = types.SimpleNamespace(tie_word_embeddings=True)
        with self._patch_raw_config(return_value=raw):
            _maybe_reject_tie_word_embeddings_flip(Path("/ckpts/tied-model"), self._requested_config(tied=True), {})

    def test_pathlib_path_normalized_to_str_for_raw_read(self):
        from pathlib import Path

        raw = types.SimpleNamespace(tie_word_embeddings=True)
        with self._patch_raw_config(return_value=raw) as mock_from_pretrained:
            _maybe_reject_tie_word_embeddings_flip(Path("/ckpts/tied-model"), self._requested_config(tied=True), {})
        (source,) = mock_from_pretrained.call_args.args
        assert isinstance(source, str)
        assert source == str(Path("/ckpts/tied-model"))

    def test_non_path_source_skipped(self):
        with self._patch_raw_config() as mock_from_pretrained:
            _maybe_reject_tie_word_embeddings_flip(None, self._requested_config(tied=False), {})
        mock_from_pretrained.assert_not_called()

    def test_raw_config_read_failure_does_not_block(self):
        with self._patch_raw_config(side_effect=OSError("offline")):
            _maybe_reject_tie_word_embeddings_flip("org/unreachable", self._requested_config(tied=False), {})
