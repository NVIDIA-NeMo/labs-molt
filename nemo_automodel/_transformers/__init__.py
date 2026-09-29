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

# Keep this package lightweight: importing `nemo_automodel._transformers.*` should not
# automatically pull in torch + all model code unless a specific symbol is accessed.

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "NeMoAutoModelForCausalLM": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForCausalLM"),
    "NeMoAutoModelForImageTextToText": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForImageTextToText"),
    "AutoMFU": ("nemo_automodel._transformers.mfu", "AutoMFU"),
}

__all__ = [
    "NeMoAutoModelForCausalLM",
    "NeMoAutoModelForImageTextToText",
    "AutoMFU",
]


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
