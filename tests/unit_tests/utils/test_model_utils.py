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

from __future__ import annotations

import warnings
from types import MethodType
from typing import Dict

import pytest
import torch
import torch.nn as nn

import nemo_automodel.components.utils.model_utils as model_utils


@pytest.fixture()
def dummy_model() -> nn.Module:
    """
    Create a minimal but representative model containing:
    - An nn.Embedding layer
    - A `vision_tower` sub-module
    - Another module whose name contains “visual” (to test name-based pattern)
    - A `language_model` backbone
    - An additional unfrozen Linear layer (“other”) for sanity checks
    """

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embed = nn.Embedding(10, 3)  # embeddings
            self.vision_tower = nn.Sequential(nn.Linear(4, 4))  # vision tower attr
            self.visual_extra = nn.Sequential(nn.Linear(5, 5))  # triggers "visual" name pattern
            self.language_model = nn.Sequential(nn.Linear(6, 6))
            self.other = nn.Linear(7, 7)

        def forward(self, x):  # pragma: no cover
            pass

    return DummyModel()


def _all_requires_grad(module: nn.Module) -> bool:
    """Return True if every parameter in `module` requires gradients."""
    return all(p.requires_grad for p in module.parameters())


def _any_requires_grad(module: nn.Module) -> bool:
    """Return True if at least one parameter in `module` requires gradients."""
    return any(p.requires_grad for p in module.parameters())


def test_print_trainable_parameters_counts(dummy_model, caplog, monkeypatch):
    """
    Ensure the helper returns correct (trainable, total) counts
    and prints to stdout only when rank == 0.
    """
    import logging

    caplog.set_level(logging.DEBUG)
    dummy_model.other.weight.requires_grad = False
    trainable, total = model_utils.print_trainable_parameters(dummy_model)

    assert trainable == sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    assert total == sum(p.numel() for p in dummy_model.parameters())

    # Check logging output
    assert "Trainable parameters" in caplog.text
    assert "Total parameters" in caplog.text
    # Default header label
    assert "Model summary:" in caplog.text


def test_print_trainable_parameters_custom_name(dummy_model, caplog):
    """The ``name`` label customizes the summary header (e.g. the draft model
    in speculative-decoding training)."""
    import logging

    caplog.set_level(logging.DEBUG)
    model_utils.print_trainable_parameters(dummy_model, name="Draft")
    assert "Draft summary:" in caplog.text
    assert "Model summary:" not in caplog.text


def test_print_trainable_parameters_non_zero_rank(dummy_model, capsys, monkeypatch):
    """
    Helper must stay silent for non-zero ranks.
    """
    _ = model_utils.print_trainable_parameters(dummy_model)
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.parametrize(
    "freeze_cfg, expect",
    [
        (
            # freeze_vision_tower=False means vision stays trainable
            {"freeze_vision_tower": False, "freeze_language_model": False},
            {"vision": True, "lang": True, "other": True},
        ),
        (
            # freeze_vision_tower=True means vision is frozen
            {"freeze_vision_tower": True, "freeze_language_model": False},
            {"vision": False, "lang": True, "other": True},
        ),
        (
            # freeze_language_model=True means language_model is frozen
            {"freeze_vision_tower": False, "freeze_language_model": True},
            {"vision": True, "lang": False, "other": True},
        ),
        (
            # defaults: freeze_vision_tower=True, freeze_language_model=False
            {},
            {"vision": False, "lang": True, "other": True},
        ),
    ],
)
def test_apply_parameter_freezing(dummy_model, freeze_cfg: Dict, expect: Dict):
    """
    Parametrized test to verify that each freeze flag affects the right sub-modules.

    `expect` dict uses:
        vision-> require_grad status for *all* vision-related parameters
        lang  -> require_grad status for language_model
        other -> require_grad status for the unrelated `other` layer
    A value of True means gradients SHOULD be enabled; False means frozen.

    Note: freeze_embeddings was removed from apply_parameter_freezing.
    Embeddings are no longer frozen by this function.
    """
    # Reset all grads before every run (pytest reuses the same fixture instance)
    for p in dummy_model.parameters():
        p.requires_grad = True

    model_utils.apply_parameter_freezing(dummy_model, model_utils.parse_freeze_config(freeze_cfg))

    # vision tower(s)
    assert _all_requires_grad(dummy_model.vision_tower) is expect["vision"]
    assert _all_requires_grad(dummy_model.visual_extra) is expect["vision"]

    # language model
    assert _all_requires_grad(dummy_model.language_model) is expect["lang"]

    # unrelated layer (including embeddings - not frozen by this function)
    assert dummy_model.other.weight.requires_grad is expect["other"]
    assert dummy_model.other.bias.requires_grad is expect["other"]
    # embeddings are always trainable now (freeze_embeddings was removed)
    assert dummy_model.token_embed.weight.requires_grad is True


def test_apply_parameter_freezing_supports_generic_module_selectors():
    """Generic selectors match fully qualified module names and unfreeze wins on overlap."""

    class GenericModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.ModuleDict(
                {
                    "q_proj": nn.Linear(4, 4),
                    "k_proj": nn.Linear(4, 4),
                    "dense": nn.Linear(4, 4),
                }
            )
            self.backbone["k_proj"].requires_grad_(False)

    model = GenericModel()

    model_utils.apply_parameter_freezing(
        model,
        model_utils.parse_freeze_config(
            {
                "freeze_modules": [{"glob": "*_proj"}],
                "unfreeze_modules": [{"glob": "*.k_proj"}],
            }
        ),
    )

    assert not _any_requires_grad(model.backbone["q_proj"])
    assert _all_requires_grad(model.backbone["k_proj"])
    assert _all_requires_grad(model.backbone["dense"])


def test_generic_module_selector_matches_shared_module_alias():
    """A selector can address any fully qualified alias of a shared module."""

    class SharedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(4, 4)
            self.primary = shared
            self.alias = shared

    model = SharedModel()
    model_utils.apply_parameter_freezing(
        model,
        model_utils.FreezeConfig(freeze_modules=[model_utils.ModuleSelector(path="alias")]),
    )

    assert not _any_requires_grad(model.primary)


def test_generic_unfreeze_selector_overrides_legacy_freeze_alias(dummy_model):
    """Generic unfreeze selectors take precedence over legacy freeze aliases."""
    model_utils.apply_parameter_freezing(
        dummy_model,
        model_utils.parse_freeze_config(
            {
                "freeze_vision_tower": True,
                "unfreeze_modules": [{"path": "vision_tower"}],
            }
        ),
    )

    assert _all_requires_grad(dummy_model.vision_tower)
    assert not _any_requires_grad(dummy_model.visual_extra)


@pytest.mark.parametrize(
    "legacy_option",
    ["freeze_vision_tower", "freeze_audio_tower", "freeze_language_model", "freeze_video_embedder"],
)
def test_legacy_freeze_options_remain_supported_without_deprecation(dummy_model, legacy_option: str):
    """Legacy modality booleans are supported and do not emit deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model_utils.apply_parameter_freezing(
            dummy_model, model_utils.parse_freeze_config({legacy_option: False, "freeze_vision_tower": False})
        )

    assert _all_requires_grad(dummy_model.other)


def test_legacy_freeze_alias_uses_substring_matching(dummy_model):
    """Legacy aliases retain their original substring matcher ("visual" matches "visual_extra")."""
    model_utils.apply_parameter_freezing(dummy_model, model_utils.FreezeConfig(freeze_vision_tower=True))

    assert not _any_requires_grad(dummy_model.vision_tower)
    assert not _any_requires_grad(dummy_model.visual_extra)


def test_generic_selectors_disable_implicit_legacy_vision_freeze(dummy_model):
    """Explicit-selector configurations do not implicitly freeze vision modules."""
    model_utils.apply_parameter_freezing(
        dummy_model,
        model_utils.FreezeConfig(unfreeze_modules=[model_utils.ModuleSelector(path="other")]),
    )

    assert _all_requires_grad(dummy_model.vision_tower)
    assert _all_requires_grad(dummy_model.visual_extra)
    assert _all_requires_grad(dummy_model.other)


def test_empty_generic_selector_list_disables_implicit_legacy_vision_freeze(dummy_model):
    """An explicitly empty selector field opts out of legacy implicit freezing."""
    config = model_utils.parse_freeze_config({"freeze_modules": []})

    assert config.freeze_modules == []
    assert config.unfreeze_modules is None
    model_utils.apply_parameter_freezing(dummy_model, config)

    assert _all_requires_grad(dummy_model.vision_tower)
    assert _all_requires_grad(dummy_model.visual_extra)


def test_glob_selector_treats_dot_as_literal():
    """Glob matching is shell-style: `.` is literal, not a regex wildcard."""

    class DotModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.axb = nn.Linear(4, 4)  # would match the regular expression r"^a.b$"
            self.a = nn.ModuleDict({"b": nn.Linear(4, 4)})  # canonical module path "a.b"

    model = DotModel()
    model_utils.apply_parameter_freezing(
        model,
        model_utils.FreezeConfig(freeze_modules=[model_utils.ModuleSelector(glob="a.b")]),
    )

    assert not _any_requires_grad(model.a["b"])
    assert _all_requires_grad(model.axb)


def test_glob_selector_is_case_sensitive():
    """Glob matching uses fnmatchcase and never folds case."""
    assert model_utils.ModuleSelector(glob="*Encoder").matches("audio_encoder") is False
    assert model_utils.ModuleSelector(glob="*Encoder").matches("audio_Encoder") is True
    assert model_utils.ModuleSelector(path="audio_encoder").matches("model.audio_encoder") is False


class TestParseFreezeConfig:
    """Strict validation of the freeze_config schema."""

    def test_none_and_typed_passthrough(self):
        typed = model_utils.FreezeConfig()
        assert model_utils.parse_freeze_config(None) is None
        assert model_utils.parse_freeze_config(typed) is typed

    def test_empty_mapping_retains_legacy_selector_absence(self):
        config = model_utils.parse_freeze_config({})

        assert config.freeze_modules is None
        assert config.unfreeze_modules is None
        assert not config.has_generic_selectors()

    def test_parses_typed_selectors(self):
        config = model_utils.parse_freeze_config(
            {
                "freeze_modules": [{"path": "vision_tower"}, {"glob": "*.audio_encoder"}],
                "unfreeze_modules": [{"path": "multi_modal_projector"}],
            }
        )
        assert config.freeze_modules == [
            model_utils.ModuleSelector(path="vision_tower"),
            model_utils.ModuleSelector(glob="*.audio_encoder"),
        ]
        assert config.unfreeze_modules == [model_utils.ModuleSelector(path="multi_modal_projector")]

    def test_unknown_option_rejected(self):
        with pytest.raises(ValueError, match="unsupported option.*freeze_vision"):
            model_utils.parse_freeze_config({"freeze_vision": True})

    def test_hydra_meta_option_rejected(self):
        with pytest.raises(ValueError, match="unsupported option.*_target_"):
            model_utils.parse_freeze_config({"_target_": "example.FreezeConfig"})

    def test_bare_string_selector_rejected(self):
        with pytest.raises(ValueError, match="must be mappings with exactly one of `path` or `glob`"):
            model_utils.parse_freeze_config({"freeze_modules": ["vision_tower"]})

    def test_selector_list_required(self):
        with pytest.raises(ValueError, match="freeze_config.freeze_modules must be a list"):
            model_utils.parse_freeze_config({"freeze_modules": {"path": "vision_tower"}})
        with pytest.raises(ValueError, match="freeze_config.freeze_modules must be a list"):
            model_utils.parse_freeze_config({"freeze_modules": ({"path": "vision_tower"},)})

    @pytest.mark.parametrize("field_name", ["freeze_modules", "unfreeze_modules"])
    @pytest.mark.parametrize("invalid", [None, False, 0])
    def test_empty_like_selector_values_rejected(self, field_name, invalid):
        with pytest.raises(ValueError, match=rf"freeze_config\.{field_name} must be a list"):
            model_utils.parse_freeze_config({field_name: invalid})

    def test_selector_with_both_path_and_glob_rejected(self):
        with pytest.raises(ValueError, match="exactly one of `path` or `glob`"):
            model_utils.parse_freeze_config({"freeze_modules": [{"path": "a", "glob": "b"}]})

    def test_selector_with_neither_path_nor_glob_rejected(self):
        with pytest.raises(ValueError, match="exactly one of `path` or `glob`"):
            model_utils.parse_freeze_config({"unfreeze_modules": [{}]})

    def test_selector_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unsupported key.*regex"):
            model_utils.parse_freeze_config({"freeze_modules": [{"regex": "vision"}]})

    def test_non_boolean_legacy_option_rejected(self):
        with pytest.raises(ValueError, match="freeze_config.freeze_audio_tower must be a boolean"):
            model_utils.parse_freeze_config({"freeze_audio_tower": "yes"})

    def test_non_mapping_rejected(self):
        with pytest.raises(TypeError, match="freeze_config must be a mapping"):
            model_utils.parse_freeze_config("vision_tower")

    def test_direct_typed_config_rejects_untyped_values(self):
        with pytest.raises(TypeError, match="entries must be ModuleSelector"):
            model_utils.FreezeConfig(freeze_modules=[{"path": "vision_tower"}])
        with pytest.raises(TypeError, match="freeze_audio_tower must be a boolean"):
            model_utils.FreezeConfig(freeze_audio_tower="yes")

    def test_apply_parameter_freezing_accepts_raw_mapping_for_compatibility(self, dummy_model):
        model_utils.apply_parameter_freezing(dummy_model, {"freeze_modules": [{"path": "other"}]})
        assert not _any_requires_grad(dummy_model.other)

    def test_unmatched_selector_raises_in_strict_mode(self, dummy_model):
        config = model_utils.FreezeConfig(freeze_modules=[model_utils.ModuleSelector(path="nonexistent")])
        with pytest.raises(
            ValueError, match=r"freeze_config\.freeze_modules selector `path: nonexistent` matched no parameters"
        ):
            model_utils.apply_parameter_freezing(dummy_model, config, strict=True)

    def test_unmatched_selector_allowed_when_rebinding(self, dummy_model):
        config = model_utils.FreezeConfig(freeze_modules=[model_utils.ModuleSelector(path="nonexistent")])
        model_utils.apply_parameter_freezing(dummy_model, config, strict=False)
        assert _all_requires_grad(dummy_model.other)


def test_init_empty_weights_moves_params_to_meta_and_preserves_requires_grad():
    """
    Creating parameters inside the context should place them on meta device and
    preserve their requires_grad flags.
    """

    class CustomModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty(2, 3))  # requires_grad=True by default
            self.b = nn.Parameter(torch.empty(3), requires_grad=False)

    with model_utils.init_empty_weights():
        m = CustomModule()

    # Device moved to meta
    assert m.w.device.type == "meta"
    assert m.b.device.type == "meta"
    # requires_grad preserved
    assert m.w.requires_grad is True
    assert m.b.requires_grad is False
    # shapes and types preserved
    assert m.w.shape == (2, 3)
    assert isinstance(m.w, nn.Parameter)
    assert isinstance(m.b, nn.Parameter)


def test_init_empty_weights_restores_register_parameter_on_exception():
    """
    Even if an exception occurs inside the context, the original register_parameter
    must be restored.
    """
    original = nn.Module.register_parameter
    with pytest.raises(RuntimeError):
        with model_utils.init_empty_weights():
            raise RuntimeError("boom")
    assert nn.Module.register_parameter is original


def test_init_empty_weights_torchao_branch_with_fake_weight(monkeypatch):
    """
    Simulate the torchao branch by monkeypatching torch_ao and providing a
    fake WeightWithDynamicFloat8CastTensor that subclasses nn.Parameter.
    Verify:
      - parameters are moved to meta
      - requires_grad is preserved based on the wrapped tensor
      - mapped attributes (_linear_mm_config, _dtype, _precomputed_scale) are copied through
    """

    class FakeWeight(nn.Parameter):
        def __new__(
            cls,
            tensor: torch.Tensor,
            linear_mm_config=None,
            dtype=None,
            precomputed_scale=None,
        ):
            # Mirror torchao behavior: requires_grad comes from the wrapped tensor
            obj = nn.Parameter.__new__(cls, tensor, requires_grad=tensor.requires_grad)
            # store with underscore names to match model_utils' mapping lookup
            obj._linear_mm_config = linear_mm_config
            obj._dtype = dtype
            obj._precomputed_scale = precomputed_scale
            return obj

    class _DummyFSDPUtils:
        WeightWithDynamicFloat8CastTensor = FakeWeight

    class _DummyFloat8:
        fsdp_utils = _DummyFSDPUtils

    class _DummyTorchAO:
        float8 = _DummyFloat8

    monkeypatch.setattr(model_utils, "HAVE_TORCHAO", True, raising=True)
    monkeypatch.setattr(model_utils, "torch_ao", _DummyTorchAO, raising=True)

    class Mod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # create a FakeWeight so isinstance(..., WeightWithDynamicFloat8CastTensor) is True
            base = torch.empty(3, requires_grad=False)
            # requires_grad False to ensure preservation through the branch
            self.p = FakeWeight(
                base,
                "cfg",
                torch.float32,
                torch.tensor(1.0),
            )

    with model_utils.init_empty_weights():
        m = Mod()

    # type preserved as FakeWeight
    assert isinstance(m.p, FakeWeight)
    # moved to meta
    assert m.p.device.type == "meta"
    # requires_grad preserved from wrapped tensor
    assert m.p.requires_grad is False
    # mapped attributes preserved via mapping
    assert getattr(m.p, "_linear_mm_config") == "cfg"
    assert getattr(m.p, "_dtype") == torch.float32
    assert isinstance(getattr(m.p, "_precomputed_scale"), torch.Tensor)


def test_init_empty_weights_preserves_is_hf_initialized_attribute():
    """
    Test that _is_hf_initialized attribute is preserved when moving parameters
    to meta device. This attribute is used by HuggingFace transformers to track
    which parameters have been initialized.
    """

    class ModuleWithHFInitializedParam(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty(2, 3))
            # Simulate HF setting _is_hf_initialized on the parameter
            self.w._is_hf_initialized = True

    with model_utils.init_empty_weights():
        m = ModuleWithHFInitializedParam()

    # Device moved to meta
    assert m.w.device.type == "meta"
    # _is_hf_initialized should be preserved
    assert hasattr(m.w, "_is_hf_initialized")
    assert m.w._is_hf_initialized is True


def test_init_empty_weights_handles_missing_is_hf_initialized():
    """
    Test that parameters without _is_hf_initialized attribute work correctly
    and don't have the attribute added spuriously.
    """

    class ModuleWithoutHFInitialized(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty(2, 3))
            # No _is_hf_initialized attribute set

    with model_utils.init_empty_weights():
        m = ModuleWithoutHFInitialized()

    # Device moved to meta
    assert m.w.device.type == "meta"
    # _is_hf_initialized should NOT be present since it wasn't set originally
    assert not hasattr(m.w, "_is_hf_initialized")


# =============================================================================
# Tests for freeze_audio_tower with "speech" pattern
# =============================================================================


@pytest.fixture()
def audio_model() -> nn.Module:
    """Model with audio and speech submodules for freeze tests."""

    class AudioModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_encoder = nn.Linear(4, 4)
            self.speech_adapter = nn.Linear(4, 4)
            self.language_head = nn.Linear(4, 4)

        def forward(self, x):
            pass

    return AudioModel()


def test_freeze_audio_tower_freezes_speech_pattern(audio_model):
    """freeze_audio_tower=True should freeze modules matching 'audio' and 'speech'."""
    model_utils.apply_parameter_freezing(
        audio_model, model_utils.FreezeConfig(freeze_audio_tower=True, freeze_vision_tower=False)
    )

    assert not _any_requires_grad(audio_model.audio_encoder)
    assert not _any_requires_grad(audio_model.speech_adapter)
    assert _all_requires_grad(audio_model.language_head)


def test_freeze_audio_tower_false_keeps_speech_trainable(audio_model):
    """freeze_audio_tower=False should keep speech modules trainable."""
    model_utils.apply_parameter_freezing(
        audio_model, model_utils.FreezeConfig(freeze_audio_tower=False, freeze_vision_tower=False)
    )

    assert _all_requires_grad(audio_model.audio_encoder)
    assert _all_requires_grad(audio_model.speech_adapter)


def test_freeze_audio_encoder_with_generic_selector(audio_model):
    """A glob selector freezes only the matched module, without the legacy substrings."""
    model_utils.apply_parameter_freezing(
        audio_model,
        model_utils.FreezeConfig(freeze_modules=[model_utils.ModuleSelector(glob="*audio_encoder")]),
    )

    assert not _any_requires_grad(audio_model.audio_encoder)
    assert _all_requires_grad(audio_model.speech_adapter)


@pytest.fixture()
def sound_model() -> nn.Module:
    """Model with a NemotronOmni-style ``sound_*`` submodule."""

    class SoundModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sound_encoder = nn.Linear(4, 4)
            self.sound_projection = nn.Linear(4, 4)
            self.language_head = nn.Linear(4, 4)

        def forward(self, x):
            pass

    return SoundModel()


def test_freeze_audio_tower_freezes_sound_pattern(sound_model):
    """``freeze_audio_tower=True`` must also freeze NemotronOmni's ``sound_*`` modules."""
    model_utils.apply_parameter_freezing(
        sound_model, model_utils.FreezeConfig(freeze_audio_tower=True, freeze_vision_tower=False)
    )

    assert not _any_requires_grad(sound_model.sound_encoder)
    assert not _any_requires_grad(sound_model.sound_projection)
    assert _all_requires_grad(sound_model.language_head)


def test_freeze_audio_tower_false_keeps_sound_trainable(sound_model):
    """Sound modules stay trainable when audio freeze is off."""
    model_utils.apply_parameter_freezing(
        sound_model, model_utils.FreezeConfig(freeze_audio_tower=False, freeze_vision_tower=False)
    )

    assert _all_requires_grad(sound_model.sound_encoder)
    assert _all_requires_grad(sound_model.sound_projection)


# =============================================================================
# Tests for cast_mixed_dtype_params_to_bf16
# =============================================================================


def test_cast_mixed_dtype_params_to_bf16():
    """cast_mixed_dtype_params_to_bf16 converts fp32 params and buffers to bf16."""
    m = nn.Linear(4, 4)  # fp32 by default
    m.register_buffer("my_buf", torch.ones(3, dtype=torch.float32))
    assert m.weight.dtype == torch.float32
    assert m.my_buf.dtype == torch.float32

    model_utils.cast_mixed_dtype_params_to_bf16(m)

    assert m.weight.dtype == torch.bfloat16
    assert m.bias.dtype == torch.bfloat16
    assert m.my_buf.dtype == torch.bfloat16


def test_cast_mixed_dtype_preserves_bf16():
    """Already-bf16 params should not be changed."""
    m = nn.Linear(4, 4).to(torch.bfloat16)
    model_utils.cast_mixed_dtype_params_to_bf16(m)
    assert m.weight.dtype == torch.bfloat16


# =============================================================================
# Tests for phi4mm-specific logic in apply_parameter_freezing
# =============================================================================


def test_phi4mm_use_cache_disabled():
    """For phi4mm models, apply_parameter_freezing sets use_cache=False."""
    import types

    m = nn.Linear(4, 4)
    m.config = types.SimpleNamespace(model_type="phi4mm", use_cache=True)

    model_utils.apply_parameter_freezing(m, model_utils.FreezeConfig())
    assert m.config.use_cache is False


def test_non_phi4mm_use_cache_unchanged():
    """For non-phi4mm models, use_cache should not be changed."""
    import types

    m = nn.Linear(4, 4)
    m.config = types.SimpleNamespace(model_type="gemma3", use_cache=True)

    model_utils.apply_parameter_freezing(m, model_utils.FreezeConfig())
    assert m.config.use_cache is True


class TestFilterForwardKwargs:
    def test_drops_unsupported_kwargs(self):
        class Model(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                return input_ids

        model = Model()
        batch = {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "padding_mask": torch.zeros(2, 3, dtype=torch.bool),
        }

        filtered = model_utils.filter_forward_kwargs(model, batch)

        assert set(filtered.keys()) == {"input_ids", "attention_mask"}
        assert "padding_mask" in batch  # original dict is unchanged

    def test_keeps_kwargs_when_forward_accepts_var_keyword(self):
        class Model(nn.Module):
            def forward(self, input_ids, **kwargs):
                return input_ids, kwargs

        model = Model()
        batch = {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "padding_mask": torch.zeros(2, 3, dtype=torch.bool),
        }

        filtered = model_utils.filter_forward_kwargs(model, batch)

        assert filtered == batch

    def test_caches_signature_by_forward_callable(self, monkeypatch):
        class Model(nn.Module):
            def forward(self, input_ids):
                return input_ids

        model_utils._get_cached_forward_signature.cache_clear()
        signature = model_utils.inspect.signature
        calls = 0

        def counting_signature(callable_object):
            nonlocal calls
            calls += 1
            return signature(callable_object)

        monkeypatch.setattr(model_utils.inspect, "signature", counting_signature)
        try:
            model_utils.filter_forward_kwargs(Model(), {"input_ids": torch.ones(1)})
            model_utils.filter_forward_kwargs(Model(), {"input_ids": torch.ones(1)})
        finally:
            model_utils._get_cached_forward_signature.cache_clear()

        assert calls == 1

    def test_class_forward_patch_invalidates_cached_signature(self):
        class Model(nn.Module):
            def forward(self, input_ids):
                return input_ids

        model = Model()
        assert set(model_utils.filter_forward_kwargs(model, {"input_ids": 1, "pixel_values": 2})) == {"input_ids"}

        def replacement_forward(self, pixel_values):
            return pixel_values

        Model.forward = replacement_forward
        assert set(model_utils.filter_forward_kwargs(model, {"input_ids": 1, "pixel_values": 2})) == {"pixel_values"}

    def test_bound_forward_receiver_name_is_not_treated_as_kwarg(self):
        class Model(nn.Module):
            def forward(module, input_ids):
                return input_ids

        filtered = model_utils.filter_forward_kwargs(Model(), {"module": 1, "input_ids": 2})

        assert filtered == {"input_ids": 2}

    def test_instance_callable_forward_is_not_retained_in_cache(self):
        class ForwardCallable:
            def __call__(self, input_ids):
                return input_ids

        class Model(nn.Module):
            forward = ForwardCallable()

        model_utils._get_cached_forward_signature.cache_clear()
        model_utils.filter_forward_kwargs(Model(), {"input_ids": 1})

        assert model_utils._get_cached_forward_signature.cache_info().currsize == 0

    def test_instance_bound_forward_is_not_retained_in_cache(self):
        class Model(nn.Module):
            def forward(self, input_ids):
                return input_ids

        model = Model()

        def replacement_forward(owning_model, pixel_values):
            assert owning_model is model
            return pixel_values

        model.forward = MethodType(replacement_forward, model)
        model_utils._get_cached_forward_signature.cache_clear()

        filtered = model_utils.filter_forward_kwargs(model, {"input_ids": 1, "pixel_values": 2})

        assert filtered == {"pixel_values": 2}
        assert model_utils._get_cached_forward_signature.cache_info().currsize == 0


class TestSqueezeInputForThd:
    """``squeeze_input_for_thd`` strips the placeholder batch dim (``[1, T, ...] -> [T, ...]``)
    before THD attention/Mamba kernels see the inputs. The contract has to handle
    ``input_ids=None`` because callers feeding the model via ``inputs_embeds`` only
    (multimodal LMs, speech-language models) leave ``input_ids`` unset; the embeddings
    are squeezed inside the model forward instead.
    """

    def _attn_kwargs(self, *, padded: bool = False, with_max_seqlen: bool = True):
        """Build the kwargs dict in the canonical [1, num_seqs+1] layout, padded
        with the ``-1000`` sentinel that ``squeeze_input_for_thd`` filters out."""
        kwargs: dict = {
            "cu_seqlens": torch.tensor([[0, 3, 5, -1000]], dtype=torch.int32),
        }
        if padded:
            kwargs["cu_seqlens_padded"] = torch.tensor([[0, 4, 6, -1000]], dtype=torch.int32)
        if with_max_seqlen:
            kwargs["max_seqlen"] = torch.tensor([3])
        return kwargs

    def test_squeezes_input_ids_when_provided(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        padding_mask = torch.tensor([[False, False, False, False, False]])
        kwargs = self._attn_kwargs()

        ids, pos, mask, kw = model_utils.squeeze_input_for_thd(input_ids, position_ids, padding_mask, kwargs)

        assert ids.shape == (5,)
        assert pos.shape == (5,)
        assert mask.shape == (5,)
        # Sentinel filtered out and dtype/shape preserved.
        assert kw["cu_seqlens"].tolist() == [0, 3, 5]
        assert kw["cu_seqlens"].dtype == torch.int32
        # max_seqlen tensor → Python int.
        assert kw["max_seqlen"] == 3
        assert isinstance(kw["max_seqlen"], int)

    def test_accepts_input_ids_none_for_inputs_embeds_callers(self):
        """The bug: prior code did ``input_ids.squeeze(0)`` unconditionally and
        crashed when the caller used ``inputs_embeds`` only. The fix returns
        ``None`` for the ``input_ids`` slot and squeezes everything else."""
        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        padding_mask = torch.tensor([[False, False, False, False, False]])
        kwargs = self._attn_kwargs()

        ids, pos, mask, kw = model_utils.squeeze_input_for_thd(None, position_ids, padding_mask, kwargs)

        assert ids is None
        assert pos.shape == (5,)
        assert mask.shape == (5,)
        assert kw["cu_seqlens"].tolist() == [0, 3, 5]
        assert kw["max_seqlen"] == 3

    def test_padding_mask_none_is_passed_through(self):
        """Existing behavior: ``padding_mask`` may be ``None`` (unmasked path).
        The new ``input_ids=None`` branch must compose with this."""
        position_ids = torch.tensor([[0, 1, 2, 3, 4]])
        kwargs = self._attn_kwargs(with_max_seqlen=False)

        ids, pos, mask, kw = model_utils.squeeze_input_for_thd(None, position_ids, None, kwargs)

        assert ids is None
        assert mask is None
        assert pos.shape == (5,)

    def test_3d_inputs_embeds_via_input_ids_slot_still_works(self):
        """Belt-and-braces: the docstring claims ``input_ids`` may carry a 3D
        ``[1, T, H]`` embedding tensor. Squeezing dim 0 of that yields ``[T, H]``."""
        embeds = torch.randn(1, 5, 16)
        position_ids = torch.tensor([[0, 1, 2, 3, 4]])
        kwargs = self._attn_kwargs(with_max_seqlen=False)

        ids, pos, _mask, _kw = model_utils.squeeze_input_for_thd(embeds, position_ids, None, kwargs)

        assert ids.shape == (5, 16)
        assert pos.shape == (5,)

    def test_cu_seqlens_padded_filtered_alongside_cu_seqlens(self):
        """Both ``cu_seqlens`` and ``cu_seqlens_padded`` (CP path) get the
        sentinel filter — the bug fix must not regress this."""
        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        kwargs = self._attn_kwargs(padded=True, with_max_seqlen=False)

        _ids, _pos, _mask, kw = model_utils.squeeze_input_for_thd(None, position_ids, None, kwargs)

        assert kw["cu_seqlens"].tolist() == [0, 3, 5]
        assert kw["cu_seqlens_padded"].tolist() == [0, 4, 6]


# =============================================================================
# Tests for enable_radio_vit_fused_attn
# =============================================================================


def _build_radio_model(num_blocks: int = 3, attach_via_nested_model: bool = False, attn_present: bool = True):
    """Build a stub mirroring ``model.[model.]vision_model.radio_model.model.blocks[*].attn.fused_attn``."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.fused_attn = False

    class Block(nn.Module):
        def __init__(self, with_attn: bool):
            super().__init__()
            if with_attn:
                self.attn = Attn()
            # else: no `attn` attribute at all → exercises the `attn is None` skip

    class RadioInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(attn_present) for _ in range(num_blocks)])

    class RadioModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = RadioInner()

    class VisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.radio_model = RadioModel()

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            if attach_via_nested_model:

                class Inner(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.vision_model = VisionModel()

                self.model = Inner()
            else:
                self.vision_model = VisionModel()

    return Outer()


def test_enable_radio_vit_fused_attn_flips_blocks_top_level():
    """All blocks reachable via ``model.vision_model.radio_model.model.blocks`` flip to fused_attn=True."""
    model = _build_radio_model(num_blocks=3)
    model_utils.enable_radio_vit_fused_attn(model)

    for block in model.vision_model.radio_model.model.blocks:
        assert block.attn.fused_attn is True


def test_enable_radio_vit_fused_attn_flips_blocks_nested_model():
    """Resolves vision_model via the ``model.model.vision_model`` path used by HF wrappers."""
    model = _build_radio_model(num_blocks=2, attach_via_nested_model=True)
    model_utils.enable_radio_vit_fused_attn(model)

    for block in model.model.vision_model.radio_model.model.blocks:
        assert block.attn.fused_attn is True


def test_enable_radio_vit_fused_attn_noop_without_radio():
    """No vision tower → silent no-op (does not raise)."""
    model = nn.Linear(4, 4)
    model_utils.enable_radio_vit_fused_attn(model)


def test_enable_radio_vit_fused_attn_skips_blocks_without_attn():
    """Blocks without an ``attn`` attribute are skipped, not crashed on."""
    model = _build_radio_model(num_blocks=2, attn_present=False)
    model_utils.enable_radio_vit_fused_attn(model)  # must not raise


# =============================================================================
# Tests for the freeze_video_embedder compatibility alias
# =============================================================================


@pytest.fixture()
def video_embedder_model() -> nn.Module:
    """Model exposing a ``patch_generator.video_embedder`` parameter."""

    class PatchGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.video_embedder = nn.Linear(4, 4)
            self.image_embedder = nn.Linear(4, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_generator = PatchGenerator()
            self.language_head = nn.Linear(4, 4)

    return Model()


def test_freeze_video_embedder_true_freezes_only_video_embedder(video_embedder_model):
    """``freeze_video_embedder=True`` freezes ``patch_generator.video_embedder.*`` only."""
    model_utils.apply_parameter_freezing(
        video_embedder_model,
        model_utils.FreezeConfig(freeze_vision_tower=False, freeze_video_embedder=True),
    )

    assert not _any_requires_grad(video_embedder_model.patch_generator.video_embedder)
    assert _all_requires_grad(video_embedder_model.patch_generator.image_embedder)
    assert _all_requires_grad(video_embedder_model.language_head)


def test_freeze_video_embedder_false_keeps_video_embedder_trainable(video_embedder_model):
    """Default ``freeze_video_embedder=False`` leaves the video embedder trainable."""
    model_utils.apply_parameter_freezing(
        video_embedder_model,
        model_utils.FreezeConfig(freeze_vision_tower=False, freeze_video_embedder=False),
    )

    assert _all_requires_grad(video_embedder_model.patch_generator.video_embedder)
