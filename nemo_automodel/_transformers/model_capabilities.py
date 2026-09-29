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

"""Programmatic query API for NeMo AutoModel's per-architecture capability registry.

Two declaration patterns are supported on registered model classes:

1. **Static** -- nested ``ModelCapabilities`` dataclass on the class. Used by
   classes whose capability profile does not depend on config.
2. **Dynamic** -- ``get_capabilities(cls, config)`` classmethod. Used by
   classes that serve multiple checkpoint variants (e.g. ``Gemma4`` MoE vs.
   dense vs. audio).

A class must declare capabilities via exactly one of these patterns.
:func:`query_capabilities` dispatches to whichever is present and always
returns the canonical :class:`ModelCapabilities` defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PretrainedConfig


__all__ = ["ModelCapabilities", "query_capabilities"]


@dataclass(frozen=True)
class ModelCapabilities:
    """Canonical parallelism capability flags for a model architecture.

    All fields are conservative defaults; a flag is ``True`` only when the
    corresponding feature has a working, verified implementation for that
    architecture (or variant, in the dynamic-dispatch case).

    Attributes:
        supports_tp: Tensor parallelism.
        supports_cp: Context parallelism.
        supports_pp: Pipeline parallelism.
        supports_ep: Expert parallelism (MoE).
        supports_thd: THD packed-sequence inputs.
        supports_mtp_cp: MTP training with context parallelism.
        supports_mtp_cp_pp: MTP training with context and pipeline parallelism.
        supports_cp_vision_frame_sharding: Frame-level vision-tower sharding over
            context-parallel ranks.
    """

    supports_tp: bool = False
    supports_cp: bool = False
    supports_pp: bool = False
    supports_ep: bool = False
    supports_thd: bool = False
    supports_mtp_cp: bool = False
    supports_mtp_cp_pp: bool = False
    supports_cp_vision_frame_sharding: bool = False


def _to_canonical(caps_obj) -> ModelCapabilities:
    """Re-pack an object with ``supports_*`` fields into the canonical type.

    Per-class nested ``ModelCapabilities`` dataclasses are their own types; this
    converts them to the canonical :class:`ModelCapabilities` so callers see a
    single, stable type.
    """
    if isinstance(caps_obj, ModelCapabilities):
        return caps_obj
    return ModelCapabilities(
        supports_tp=bool(caps_obj.supports_tp),
        supports_cp=bool(caps_obj.supports_cp),
        supports_pp=bool(caps_obj.supports_pp),
        supports_ep=bool(caps_obj.supports_ep),
        supports_thd=bool(getattr(caps_obj, "supports_thd", False)),
        supports_mtp_cp=bool(getattr(caps_obj, "supports_mtp_cp", False)),
        supports_mtp_cp_pp=bool(getattr(caps_obj, "supports_mtp_cp_pp", False)),
        supports_cp_vision_frame_sharding=bool(getattr(caps_obj, "supports_cp_vision_frame_sharding", False)),
    )


def _resolve_class_from_arch(arch: str):
    """Look up a registered model class by architecture name. Raises on miss."""
    from nemo_automodel._transformers.registry import ModelRegistry

    if not ModelRegistry.has_custom_model(arch):
        raise KeyError(f"Architecture {arch!r} is not registered in MODEL_ARCH_MAPPING. Cannot query capabilities.")
    return ModelRegistry.get_model_cls_from_model_arch(arch)


def _arch_from_config(config) -> str:
    """Extract the primary architecture name from an HF ``PretrainedConfig``."""
    archs = getattr(config, "architectures", None)
    if not archs:
        raise ValueError(
            f"Config of type {type(config).__name__} has no 'architectures' field. "
            f"Cannot resolve a model class for capability query."
        )
    return archs[0]


def _dispatch(model_cls, config) -> ModelCapabilities:
    """Apply the static/dynamic capability rules for a resolved class."""
    # Capability attachment wraps models in a thin runtime subclass, while the
    # declaration remains on the original model class.
    capability_cls = next(
        (cls for cls in model_cls.__mro__ if "get_capabilities" in cls.__dict__ or "ModelCapabilities" in cls.__dict__),
        None,
    )
    if capability_cls is None:
        raise AttributeError(
            f"{model_cls.__name__} declares no capabilities (neither nested "
            f"'ModelCapabilities' dataclass nor 'get_capabilities' classmethod)."
        )

    has_dynamic = "get_capabilities" in capability_cls.__dict__
    has_static = "ModelCapabilities" in capability_cls.__dict__

    if has_dynamic and has_static:
        raise TypeError(
            f"{capability_cls.__name__} declares both 'ModelCapabilities' and "
            f"'get_capabilities'. A class must declare exactly one."
        )
    if has_dynamic:
        if config is None:
            raise ValueError(
                f"{capability_cls.__name__} uses dynamic capability dispatch via "
                f"get_capabilities(config); a config is required. Pass an HF "
                f"model id, a PretrainedConfig, or a model instance instead of "
                f"the bare class."
            )
        return _to_canonical(capability_cls.get_capabilities(config))
    if has_static:
        return _to_canonical(capability_cls.ModelCapabilities())


def query_capabilities(
    target: Union[str, "PretrainedConfig", nn.Module, type],
    *,
    trust_remote_code: bool = False,
) -> ModelCapabilities:
    """Resolve declared parallelism capabilities for a model.

    Accepted input forms:

    * **HF model id (str)** -- ``AutoConfig.from_pretrained`` is used to load
      the config (no weights), the architecture is read from the config, and
      the registered NeMo class is looked up via :data:`ModelRegistry`.
    * **HF config object** -- the architecture is read from the config.
    * **Model instance (nn.Module)** -- ``type(target)`` resolves the class
      and ``target.config`` provides the variant info needed by dynamic
      dispatch.
    * **Registered model class (type)** -- usable only for classes with a
      static nested ``ModelCapabilities`` dataclass. Classes with dynamic
      dispatch raise :class:`ValueError` because no config is available.

    Args:
        target: HF model id, ``PretrainedConfig``, model instance, or
            registered NeMo model class.
        trust_remote_code: Forwarded to ``AutoConfig.from_pretrained`` when
            ``target`` is a string. Ignored otherwise.

    Returns:
        A :class:`ModelCapabilities` (canonical type, identical across all
        inputs and all model classes).

    Raises:
        TypeError: If ``target`` is not a supported input form.
        KeyError: If the resolved architecture is not registered.
        ValueError: If a dynamic-dispatch class is queried via the bare class
            (no config available) or a config has no ``architectures`` field.
        AttributeError: If the resolved class declares no capabilities.
    """
    if isinstance(target, str):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(target, trust_remote_code=trust_remote_code)
        arch = _arch_from_config(config)
        model_cls = _resolve_class_from_arch(arch)
        return _dispatch(model_cls, config)

    if isinstance(target, nn.Module):
        model_cls = type(target)
        config = getattr(target, "config", None)
        return _dispatch(model_cls, config)

    if isinstance(target, type):
        if not issubclass(target, nn.Module):
            raise TypeError(
                f"query_capabilities() received class {target.__name__!r} which is not a torch.nn.Module subclass."
            )
        return _dispatch(target, None)

    # Anything left that exposes 'architectures' is treated as an HF config.
    if hasattr(target, "architectures"):
        arch = _arch_from_config(target)
        model_cls = _resolve_class_from_arch(arch)
        return _dispatch(model_cls, target)

    raise TypeError(
        f"query_capabilities() expected a model id (str), a PretrainedConfig, "
        f"a model instance (nn.Module), or a registered model class. Got "
        f"{type(target).__name__}."
    )
