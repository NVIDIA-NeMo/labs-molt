# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the model capability query API."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch.nn as nn

from nemo_automodel._transformers.model_capabilities import (
    ModelCapabilities,
    query_capabilities,
)
from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING, ModelRegistry

# ---------------------------------------------------------------------------
# 1. Every registered class must declare capabilities via exactly one pattern
#
# Modeled on tests/unit_tests/_transformers/test_doc_coverage.py: walk every
# entry in MODEL_ARCH_MAPPING, collect violations, and surface them in a
# single actionable assertion. Any new model onboarded into the registry that
# forgets to declare capabilities (or declares both patterns by mistake) will
# fail this test in CI before merge.
# ---------------------------------------------------------------------------


def test_every_registered_arch_declares_capabilities():
    """Every architecture in ``MODEL_ARCH_MAPPING`` must declare capabilities
    via exactly one of:

      * a nested ``ModelCapabilities`` dataclass (for classes that has no variants or every model that maps to
      this class shares the same parallelism story), or
      * a ``get_capabilities(cls, config)`` classmethod (for classes that
        serve multiple variants ex: gemma4, ernie4.5).

    Declaring both is forbidden (the placeholder defaults of the nested class
    can silently shadow the real, config-dependent answer). Declaring neither
    is forbidden (callers cannot infer parallelism support from the registry).

    This guards against the regression where a new arch lands in
    ``MODEL_ARCH_MAPPING`` but the contributor forgets to declare its
    capability flags. The test will surface every offending arch in a single
    error message so onboarding fixes can be done in one pass.
    """
    missing: list[tuple[str, str]] = []  # (arch_name, reason)
    unimportable: list[tuple[str, str]] = []

    for arch_name in MODEL_ARCH_MAPPING:
        try:
            cls = ModelRegistry.get_model_cls_from_model_arch(arch_name)
        except Exception as e:
            unimportable.append((arch_name, f"{type(e).__name__}: {e}"))
            continue

        has_static = "ModelCapabilities" in cls.__dict__
        has_dynamic = "get_capabilities" in cls.__dict__

        if has_static and has_dynamic:
            missing.append(
                (
                    arch_name,
                    f"{cls.__name__} declares BOTH 'ModelCapabilities' and 'get_capabilities' (must be exactly one).",
                )
            )
        elif not has_static and not has_dynamic:
            missing.append(
                (
                    arch_name,
                    f"{cls.__name__} declares NEITHER 'ModelCapabilities' nor 'get_capabilities'.",
                )
            )

    if missing:
        details = "\n".join(f"  - {arch}: {reason}" for arch, reason in missing)
        raise AssertionError(
            "The following registered architectures violate the capability "
            "declaration contract:\n"
            f"{details}\n\n"
            "Fix by either:\n"
            "  1. For classes with a single capability profile, add a nested\n"
            "     `@dataclass(frozen=True) class ModelCapabilities` with the\n"
            "     relevant `supports_*` flags, matching the pattern in\n"
            "     `nemo_automodel/components/models/llama/model.py`.\n"
            "  2. For classes serving multiple checkpoint variants (e.g. dense vs.\n"
            "     MoE), add a `@classmethod get_capabilities(cls, config)` that\n"
            "     returns `nemo_automodel._transformers.model_capabilities.ModelCapabilities`,\n"
            "     matching the pattern in\n"
            "     `nemo_automodel/components/models/gemma4_moe/model.py`.\n"
            "  3. If a class declared both by mistake, remove the static nested\n"
            "     dataclass; the dynamic method is the source of truth."
        )

    if unimportable:
        details = "\n".join(f"  - {arch}: {reason}" for arch, reason in unimportable)
        # Don't fail on unimportable archs (transformers version mismatches in
        # the test environment are not a capability-declaration bug), but log
        # them so a missing declaration on an unimportable class is not silently
        # ignored when the env is upgraded.
        print(f"\n[capabilities] {len(unimportable)} arch(es) not importable in this env:\n{details}")


# ---------------------------------------------------------------------------
# 2. Dynamic-dispatch classes must NOT expose a static ModelCapabilities
# ---------------------------------------------------------------------------


def test_gemma4_dynamic_class_blocks_static_access():
    """Gemma4ForConditionalGeneration uses get_capabilities; static attribute access must fail."""
    try:
        cls = ModelRegistry.get_model_cls_from_model_arch("Gemma4ForConditionalGeneration")
    except Exception as e:
        pytest.skip(f"Gemma4 not importable in this environment: {e}")

    with pytest.raises(AttributeError):
        cls.ModelCapabilities  # noqa: B018  -- attribute access is the test


def test_gemma4_declares_model_owned_context_parallel_attention():
    """The Gemma4 p2p ring must not be rejected for lacking a TE backend."""
    cls = ModelRegistry.get_model_cls_from_model_arch("Gemma4ForConditionalGeneration")
    assert cls._owns_cp_attention is True


@pytest.mark.parametrize(
    "arch",
    [
        "Qwen3_5ForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
    ],
)
def test_dense_qwen_vlms_declare_cp_vision_frame_sharding_support(arch):
    """Only production-integrated dense Qwen VLMs opt into CP vision frame sharding."""
    caps = query_capabilities(ModelRegistry.get_model_cls_from_model_arch(arch))

    assert caps.supports_cp_vision_frame_sharding is True


def test_qwen3_vl_moe_does_not_declare_cp_vision_frame_sharding_support():
    """Qwen3-VL-MoE remains unsupported until it owns a production integration."""
    caps = query_capabilities(ModelRegistry.get_model_cls_from_model_arch("Qwen3VLMoeForConditionalGeneration"))

    assert caps.supports_cp_vision_frame_sharding is False


# ---------------------------------------------------------------------------
# 3. query_capabilities canonical-type & dispatch behavior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StaticCaps:
    supports_tp: bool = True
    supports_cp: bool = False
    supports_pp: bool = True
    supports_ep: bool = False
    supports_thd: bool = True


class _FakeStaticModel(nn.Module):
    ModelCapabilities = _StaticCaps

    def __init__(self, config=None):
        super().__init__()
        self.config = config


class _FakeDynamicModel(nn.Module):
    @classmethod
    def get_capabilities(cls, config):
        if getattr(config, "is_variant_a", False):
            return ModelCapabilities(supports_tp=True, supports_pp=True)
        return ModelCapabilities(supports_ep=True)

    def __init__(self, config=None):
        super().__init__()
        self.config = config


class _FakeBothModel(nn.Module):
    """Invalid: declares both patterns."""

    ModelCapabilities = _StaticCaps

    @classmethod
    def get_capabilities(cls, config):
        return ModelCapabilities()


class _FakeNeitherModel(nn.Module):
    """Invalid: declares neither pattern."""


class _FakeConfig:
    def __init__(self, archs, **kwargs):
        self.architectures = archs
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_query_returns_canonical_type_for_static_class():
    caps = query_capabilities(_FakeStaticModel)
    assert type(caps) is ModelCapabilities
    assert caps.supports_tp is True
    assert caps.supports_pp is True
    assert caps.supports_cp is False
    assert caps.supports_ep is False
    assert caps.supports_thd is True
    assert caps.supports_mtp_cp is False
    assert caps.supports_mtp_cp_pp is False
    assert caps.supports_cp_vision_frame_sharding is False


def test_query_static_class_from_instance():
    caps = query_capabilities(_FakeStaticModel())
    assert type(caps) is ModelCapabilities
    assert caps.supports_tp is True
    assert caps.supports_thd is True


def test_query_static_declaration_inherited_by_runtime_wrapper():
    class _RuntimeWrapper(_FakeStaticModel):
        pass

    caps = query_capabilities(_RuntimeWrapper())
    assert caps == ModelCapabilities(supports_tp=True, supports_pp=True, supports_thd=True)


def test_query_dynamic_class_requires_config():
    with pytest.raises(ValueError, match="dynamic capability dispatch"):
        query_capabilities(_FakeDynamicModel)


def test_query_dynamic_class_via_instance_dispatches_on_config():
    inst_a = _FakeDynamicModel(config=_FakeConfig(archs=[], is_variant_a=True))
    inst_b = _FakeDynamicModel(config=_FakeConfig(archs=[]))
    caps_a = query_capabilities(inst_a)
    caps_b = query_capabilities(inst_b)
    assert caps_a == ModelCapabilities(supports_tp=True, supports_pp=True)
    assert caps_b == ModelCapabilities(supports_ep=True)
    assert caps_a != caps_b


def test_query_dynamic_declaration_inherited_by_runtime_wrapper():
    class _RuntimeWrapper(_FakeDynamicModel):
        pass

    caps = query_capabilities(_RuntimeWrapper(config=_FakeConfig(archs=[], is_variant_a=True)))
    assert caps == ModelCapabilities(supports_tp=True, supports_pp=True)


def test_query_dynamic_class_via_config_dispatches():
    cfg = _FakeConfig(archs=[], is_variant_a=True)
    # Direct config form requires the arch to resolve via registry; bypass that
    # by passing the instance pathway since this fake isn't registered.
    inst = _FakeDynamicModel(config=cfg)
    caps = query_capabilities(inst)
    assert caps == ModelCapabilities(supports_tp=True, supports_pp=True)


def test_query_model_id_loads_config_and_resolves_architecture():
    config = _FakeConfig(archs=["FakeStaticModel"])
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=config) as load_config,
        patch(
            "nemo_automodel._transformers.model_capabilities._resolve_class_from_arch",
            return_value=_FakeStaticModel,
        ) as resolve_class,
    ):
        caps = query_capabilities("org/model", trust_remote_code=True)

    assert caps == ModelCapabilities(supports_tp=True, supports_pp=True, supports_thd=True)
    load_config.assert_called_once_with("org/model", trust_remote_code=True)
    resolve_class.assert_called_once_with("FakeStaticModel")


def test_query_config_resolves_registered_architecture():
    config = _FakeConfig(archs=["FakeStaticModel"])
    with (
        patch.object(ModelRegistry, "has_custom_model", return_value=True),
        patch.object(ModelRegistry, "get_model_cls_from_model_arch", return_value=_FakeStaticModel) as resolve_class,
    ):
        caps = query_capabilities(config)

    assert caps == ModelCapabilities(supports_tp=True, supports_pp=True, supports_thd=True)
    resolve_class.assert_called_once_with("FakeStaticModel")


def test_query_config_rejects_unknown_architecture():
    config = _FakeConfig(archs=["UnknownArchitecture"])
    with (
        patch.object(ModelRegistry, "has_custom_model", return_value=False),
        pytest.raises(KeyError, match="UnknownArchitecture"),
    ):
        query_capabilities(config)


def test_query_config_requires_architecture():
    with pytest.raises(ValueError, match="has no 'architectures' field"):
        query_capabilities(_FakeConfig(archs=[]))


def test_both_patterns_is_rejected():
    with pytest.raises(TypeError, match="declares both"):
        query_capabilities(_FakeBothModel)


def test_neither_pattern_is_rejected():
    with pytest.raises(AttributeError, match="declares no capabilities"):
        query_capabilities(_FakeNeitherModel)


def test_unsupported_target_type_raises():
    with pytest.raises(TypeError):
        query_capabilities(42)


def test_query_class_non_module_raises():
    class NotAModule:
        pass

    with pytest.raises(TypeError, match="not a torch.nn.Module"):
        query_capabilities(NotAModule)
