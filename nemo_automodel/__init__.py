# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import importlib.abc
import importlib.machinery
import sys
import warnings
from types import ModuleType
from typing import Any

# Pydantic v2 emits UnsupportedFieldAttributeWarning for Field(repr=...) /
# Field(frozen=...) used inside 3.12-style `type` aliases in third-party libs.
# Suppress early so any later import that triggers pydantic schema generation
# (e.g. transformers, huggingface_hub) won't emit these warnings.
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass

from .package_info import __package_name__, __version__

# Keep the base package import lightweight.
# Heavy dependencies (e.g., torch/transformers) are intentionally imported lazily
# via __getattr__ so importing tokenizers doesn't pull in the full training stack.

_SUBMODULES = {"shared", "components", "models"}

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "NeMoAutoConfig": ("nemo_automodel._transformers.auto_config", "NeMoAutoConfig"),
    "NeMoAutoModelForCausalLM": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForCausalLM"),
    "NeMoAutoModelForImageTextToText": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForImageTextToText"),
    "ModelCapabilities": ("nemo_automodel._transformers.model_capabilities", "ModelCapabilities"),
    "query_capabilities": ("nemo_automodel._transformers.model_capabilities", "query_capabilities"),
    "register_architecture": ("nemo_automodel._transformers.registry", "register_architecture"),
}

__all__ = sorted([*_SUBMODULES, "__version__", "__package_name__", *_LAZY_ATTRS.keys()])


# ---------------------------------------------------------------------------
# nemo_automodel.models → nemo_automodel.components.models alias
#
# Implemented as a meta-path finder so it works regardless of whether a
# physical nemo_automodel/models/ directory is shipped in the installation.
# The actual import of the canonical module happens inside exec_module so
# that _load_unlocked's pop-and-set pattern on sys.modules picks up the
# replacement correctly.
# ---------------------------------------------------------------------------

_MODELS_ALIAS = "nemo_automodel.models"
_MODELS_ALIAS_DOT = _MODELS_ALIAS + "."
_MODELS_TARGET = "nemo_automodel.components.models"
_MODELS_TARGET_DOT = _MODELS_TARGET + "."


class _AliasLoader(importlib.abc.Loader):
    """Loader that replaces the placeholder module in sys.modules with the
    canonical module during exec_module."""

    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        real = importlib.import_module(self._real_name)
        sys.modules[module.__name__] = real


class _ModelsAliasFinder(importlib.abc.MetaPathFinder):
    """Redirect ``nemo_automodel.models`` and ``nemo_automodel.models.*``
    to ``nemo_automodel.components.models`` and its subpackages.

    Installed at the *front* of ``sys.meta_path`` so it intercepts before the
    default ``PathFinder``.  Both import paths resolve to the exact same module
    objects, avoiding duplication.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname == _MODELS_ALIAS:
            return importlib.machinery.ModuleSpec(
                fullname,
                _AliasLoader(_MODELS_TARGET),
                is_package=True,
            )
        if fullname.startswith(_MODELS_ALIAS_DOT):
            real_name = _MODELS_TARGET_DOT + fullname[len(_MODELS_ALIAS_DOT) :]
            return importlib.machinery.ModuleSpec(
                fullname,
                _AliasLoader(real_name),
            )
        return None


sys.meta_path.insert(0, _ModelsAliasFinder())


def __getattr__(name: str) -> ModuleType | Any:
    """
    Lazily import and cache selected submodules / exported symbols when accessed.

    Raises:
        AttributeError if the name isn't in __all__.
    """
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Expose the names of all available submodules for auto-completion.
    """
    return sorted(__all__)
