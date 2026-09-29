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


import importlib
import importlib.metadata
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple, Type, Union

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

# Static mapping: architecture name → (module_path, class_name[, tags]).
# Analogous to HuggingFace transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.
# Models are loaded lazily on first access rather than imported at startup.
# Optional third element is a set of tags (e.g. {"retrieval"}) used by
# downstream code to classify model archs without importing them.
MODEL_ARCH_MAPPING = OrderedDict(
    [
        (
            "DeepseekV4ForCausalLM",
            ("nemo_automodel.components.models.deepseek_v4.model", "DeepseekV4ForCausalLM"),
        ),
        (
            "DeepseekV41ForCausalLM",
            ("nemo_automodel.components.models.deepseek_v41.model", "DeepseekV41ForCausalLM"),
        ),
        (
            "GlmMoeDsaForCausalLM",
            ("nemo_automodel.components.models.glm_moe_dsa.model", "GlmMoeDsaForCausalLM"),
        ),
        (
            "Glm5NextForConditionalGeneration",
            (
                "nemo_automodel.components.models.glm5_next.model",
                "Glm5NextForConditionalGeneration",
            ),
        ),
        (
            "Gemma4ForConditionalGeneration",
            ("nemo_automodel.components.models.gemma4_moe.model", "Gemma4ForConditionalGeneration"),
        ),
        (
            "Gemma4UnifiedForConditionalGeneration",
            (
                "nemo_automodel.components.models.gemma4_unified.model",
                "Gemma4UnifiedForConditionalGeneration",
            ),
        ),
        (
            "NemotronHForCausalLM",
            ("nemo_automodel.components.models.nemotron_v3.model", "NemotronHForCausalLM"),
        ),
        (
            "NemotronH_Nano_Omni_Reasoning_V3",
            (
                "nemo_automodel.components.models.nemotron_omni.model",
                "NemotronOmniForConditionalGeneration",
            ),
        ),
        (
            "NemotronH_Omni_Reasoning_V3",
            (
                "nemo_automodel.components.models.nemotron_omni.model",
                "NemotronOmniForConditionalGeneration",
            ),
        ),
        (
            "MuseGlimmerForConditionalGeneration",
            ("nemo_automodel.components.models.muse_glimmer.model", "MuseGlimmerForConditionalGeneration"),
        ),
        (
            "Qwen2ForCausalLM",
            ("nemo_automodel.components.models.qwen2.model", "Qwen2ForCausalLM"),
        ),
        (
            "Qwen3ForCausalLM",
            ("nemo_automodel.components.models.qwen3.model", "Qwen3ForCausalLM"),
        ),
        (
            "Qwen3MoeForCausalLM",
            ("nemo_automodel.components.models.qwen3_moe.model", "Qwen3MoeForCausalLM"),
        ),
        (
            "Qwen3NextForCausalLM",
            ("nemo_automodel.components.models.qwen3_next.model", "Qwen3NextForCausalLM"),
        ),
        (
            "Qwen3_5ForCausalLM",
            ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForCausalLM"),
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForConditionalGeneration"),
        ),
        (
            "Qwen3VLMoeForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_vl_moe.model", "Qwen3VLMoeForConditionalGeneration"),
        ),
        (
            "Qwen3VLForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_vl.model", "Qwen3VLForConditionalGeneration"),
        ),
        (
            "Qwen3_5MoeForCausalLM",
            ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForCausalLM"),
        ),
        (
            "Qwen3_5MoeForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForConditionalGeneration"),
        ),
        (
            "Qwen3_8_FlashNextForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_8_flash_next.model", "Qwen3_8_FlashNextForConditionalGeneration"),
        ),
        (
            # Immutable checkpoint dumps predate the Qwen3.8-Flash-Next rename.
            "Qwen4ExpForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_8_flash_next.model", "Qwen3_8_FlashNextForConditionalGeneration"),
        ),
        (
            "InklingForConditionalGeneration",
            ("nemo_automodel.components.models.inkling.model", "InklingForConditionalGeneration"),
        ),
    ]
)


# Custom model_type → config class for models that have auto_map in their
# checkpoint config.json.  Registered eagerly with AutoConfig so that
# AutoConfig.from_pretrained can resolve them without trust_remote_code.
_CUSTOM_CONFIG_REGISTRATIONS: Dict[str, Tuple[str, str]] = {
    "deepseek_v4": ("nemo_automodel.components.models.deepseek_v4.config", "DeepseekV4Config"),
    "deepseek_v41": ("nemo_automodel.components.models.deepseek_v41.config", "DeepseekV41Config"),
    "deepseek_v41_text": ("nemo_automodel.components.models.deepseek_v41.config", "DeepseekV41TextConfig"),
    "deepseek_v41_vision": ("nemo_automodel.components.models.deepseek_v41.config", "DeepseekV41VisionConfig"),
    "glm_moe_dsa": ("nemo_automodel.components.models.glm_moe_dsa.config", "GlmMoeDsaConfig"),
    "glm5_next": ("nemo_automodel.components.models.glm5_next.config", "Glm5NextConfig"),
    "inkling_mm_model": ("nemo_automodel.components.models.inkling.configuration", "InklingConfig"),
    "muse_glimmer": ("nemo_automodel.components.models.muse_glimmer.config", "MuseGlimmerConfig"),
    "qwen3_8_flash_next": ("nemo_automodel.components.models.qwen3_8_flash_next.config", "Qwen3_8_FlashNextConfig"),
    "qwen3_8_flash_next_text": (
        "nemo_automodel.components.models.qwen3_8_flash_next.config",
        "Qwen3_8_FlashNextTextConfig",
    ),
    # Immutable checkpoint dumps predate the Qwen3.8-Flash-Next rename.
    "qwen4_exp": (
        "nemo_automodel.components.models.qwen3_8_flash_next.config",
        "Qwen3_8_FlashNextLegacyConfig",
    ),
    "qwen4_exp_text": (
        "nemo_automodel.components.models.qwen3_8_flash_next.config",
        "Qwen3_8_FlashNextLegacyTextConfig",
    ),
}

# Checkpoint model_type to local image/text processor implementation.
_CUSTOM_PROCESSOR_REGISTRATIONS: dict[str, tuple[str, str]] = {
    "deepseek_v41": ("nemo_automodel.components.models.deepseek_v41.processing", "DeepseekV41Processor"),
}

# model_types whose custom model implementation should win over a transformers
# built-in config when one exists. A registration means the custom model reads
# fields only our config class provides, so the built-in must NOT win by
# default: when a transformers release starts shipping a config for a registered
# model_type (same model_type string, often the same class name) it would
# silently replace ours and break the model's field protocol at init -- e.g.
# transformers 5.12 added minimax_m3_vl whose vision config drops rope_theta
# (``'MiniMaxM3VLVisionConfig' object has no attribute 'rope_theta'``), and its
# deepseek_v4 config renames ``compress_ratios`` to ``compress_rates``, which
# silently collapses the DSV4 context-parallel shard multiple to 1.
# Remove a model_type only once its custom model is verified to run on the
# native config (mistral4 was written against it and runs green).
_CUSTOM_CONFIG_OVERRIDES_BUILTIN = {
    *(_CUSTOM_CONFIG_REGISTRATIONS.keys() - {"mistral4"}),
}


def resolve_custom_config_cls(model_type: str) -> Type[PretrainedConfig] | None:
    """Resolve Automodel's preferred config class for ``model_type`` if one applies."""
    if model_type not in _CUSTOM_CONFIG_REGISTRATIONS:
        return None

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    is_builtin = model_type in CONFIG_MAPPING
    if is_builtin and model_type not in _CUSTOM_CONFIG_OVERRIDES_BUILTIN:
        return None

    module_path, cls_name = _CUSTOM_CONFIG_REGISTRATIONS[model_type]
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)
    except Exception:
        logger.debug("Failed to resolve custom config for model_type=%s", model_type, exc_info=True)
        return None


def _register_custom_configs() -> None:
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for model_type in _CUSTOM_CONFIG_REGISTRATIONS:
        is_builtin = model_type in CONFIG_MAPPING
        cfg_cls = resolve_custom_config_cls(model_type)
        if cfg_cls is None:
            continue
        try:
            AutoConfig.register(model_type, cfg_cls, exist_ok=is_builtin)
        except Exception:
            logger.debug("Failed to register config for model_type=%s", model_type, exc_info=True)


def _register_custom_processors() -> None:
    """Attach local processor implementations to their registered HF config classes."""
    from transformers import AutoProcessor

    for model_type, (module_path, cls_name) in _CUSTOM_PROCESSOR_REGISTRATIONS.items():
        config_cls = resolve_custom_config_cls(model_type)
        if config_cls is None:
            continue
        module = importlib.import_module(module_path)
        processor_cls = getattr(module, cls_name)
        AutoProcessor.register(config_cls, processor_cls, exist_ok=True)


_register_custom_configs()
_register_custom_processors()


class _LazyArchMapping:
    """Lazy-loading mapping from architecture name to model class.

    Inspired by HuggingFace transformers' ``_LazyAutoMapping``.  Entries from the
    static ``auto_map`` are imported on first access and cached.  Additional entries
    can be added at runtime via ``register``.
    """

    def __init__(self, auto_map: Union[OrderedDict, Dict[str, tuple], None] = None):
        # Entries may be (module_path, class_name) or (module_path, class_name, tags).
        # Strip the optional tags and store them separately.
        self._auto_map: Dict[str, Tuple[str, str]] = OrderedDict()
        self._tags: Dict[str, set] = {}
        for key, value in (auto_map or {}).items():
            self._auto_map[key] = (value[0], value[1])
            if len(value) > 2:
                self._tags[key] = value[2]
        self._loaded: Dict[str, Type[nn.Module]] = {}
        self._extra: Dict[str, Type[nn.Module]] = {}
        self._modules: Dict[str, object] = {}

    def _load(self, key: str) -> Type[nn.Module]:
        if key in self._loaded:
            return self._loaded[key]
        module_path, class_name = self._auto_map[key]
        if module_path not in self._modules:
            self._modules[module_path] = importlib.import_module(module_path)
        cls = getattr(self._modules[module_path], class_name)
        self._loaded[key] = cls
        return cls

    def __contains__(self, key: str) -> bool:
        if key in self._extra or key in self._loaded:
            return True
        if key not in self._auto_map:
            return False
        try:
            self._load(key)
            return True
        except Exception:
            logger.debug("Model %s unavailable (import failed), removing from auto_map", key)
            self._auto_map.pop(key, None)
            return False

    def __getitem__(self, key: str) -> Type[nn.Module]:
        if key in self._extra:
            return self._extra[key]
        if key in self._auto_map:
            return self._load(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Type[nn.Module]) -> None:
        self._extra[key] = value

    def register(self, key: str, value: Type[nn.Module], exist_ok: bool = False) -> None:
        """Register a model class under the given architecture name.

        Conflicts, including built-in architectures, require ``exist_ok=True``.
        """
        if not exist_ok and (key in self._auto_map or key in self._extra):
            raise ValueError(f"Duplicated model implementation for {key}")

        self._auto_map.pop(key, None)
        self._loaded.pop(key, None)
        self._extra[key] = value

    def has_tag(self, key: str, tag: str) -> bool:
        """Return ``True`` if *key* was registered with *tag*."""
        return tag in self._tags.get(key, set())


    def keys(self):
        return set(self._auto_map.keys()) | set(self._extra.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"_LazyArchMapping(auto_map={len(self._auto_map)}, extra={len(self._extra)}, loaded={len(self._loaded)})"


@dataclass
class _ModelRegistry:
    model_arch_name_to_cls: _LazyArchMapping = field(default=None)

    def __post_init__(self):
        if self.model_arch_name_to_cls is None:
            self.model_arch_name_to_cls = _LazyArchMapping(MODEL_ARCH_MAPPING)
        self._discover_entry_points()

    def _discover_entry_points(self) -> None:
        """Register architecture entry points without importing their modules."""
        mapping = self.model_arch_name_to_cls
        for ep in importlib.metadata.entry_points(group="nemo_automodel.architectures"):
            if ep.name in mapping.keys():
                logger.warning("Architecture %s is already registered; skipping entry point %s", ep.name, ep.value)
                continue
            module_path, _, class_name = ep.value.rpartition(":")
            if not module_path or not class_name:
                raise ValueError(f"Entry point {ep.name!r} value must be module.path:ClassName, got {ep.value!r}")
            mapping._auto_map[ep.name] = (module_path, class_name)

    @property
    def supported_models(self):
        return self.model_arch_name_to_cls.keys()

    def get_model_cls_from_model_arch(self, model_arch: str) -> Type[nn.Module]:
        return self.model_arch_name_to_cls[model_arch]

    def has_custom_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* has a custom (non-HF) implementation."""
        return arch_name in self.model_arch_name_to_cls


    def resolve_custom_model_cls(self, architecture: str, config) -> Union[Type[nn.Module], None]:
        """Return the custom model class if it exists and supports *config*, else ``None``.

        Custom model classes may define a ``supports_config(config)`` classmethod
        to opt out for specific HF configs (e.g. a Mistral3 VLM with a dense
        Ministral3 text backbone instead of the expected Mistral4 MoE+MLA).
        """
        if architecture not in self.model_arch_name_to_cls:
            return None
        model_cls = self.model_arch_name_to_cls[architecture]
        if hasattr(model_cls, "supports_config") and not model_cls.supports_config(config):
            logger.info(
                "Custom model %s does not support config %s, falling back to HF",
                model_cls.__name__,
                type(config).__name__,
            )
            return None
        return model_cls

    def register(self, arch_name: str, model_cls: Type[nn.Module], exist_ok: bool = False) -> None:
        """Register a custom model class for a given architecture name."""
        self.model_arch_name_to_cls.register(arch_name, model_cls, exist_ok=exist_ok)


@lru_cache
def get_registry():
    return _ModelRegistry()


def register_architecture(arch_name: str, model_cls: type[nn.Module], *, exist_ok: bool = False) -> None:
    """Register a custom model class for an architecture name.

    Args:
        arch_name: Architecture name (e.g. ``"LlavaExampleNemotronForCausalLM"``).
        model_cls: The model class (not a string path; the class object itself).
        exist_ok: If True, replace an existing registration.

    Raises:
        ValueError: If *arch_name* is already a built-in or registered to a
            different class and *exist_ok* is False.
    """
    ModelRegistry.register(arch_name, model_cls, exist_ok=exist_ok)


ModelRegistry = get_registry()
