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

import re
from dataclasses import dataclass, field
from typing import List

import torch.nn as nn

from nemo_automodel.shared.import_utils import safe_import_te

HAS_TE, transformer_engine = safe_import_te()
import logging

logger = logging.getLogger(__name__)
from functools import lru_cache


def _is_linear_module(module):
    return isinstance(module, nn.Linear) or (HAS_TE and isinstance(module, transformer_engine.pytorch.Linear))


@lru_cache(maxsize=1000)
def _compile_wildcard_pattern(pattern):
    pattern = re.sub(r"(?<!\.)\*", r".*", pattern)  # replace [^\.]* with `.*` ie insert "." before "*"
    pattern = re.sub(r"\.\*", "(.*)", pattern)  # replace .* -> (.*)
    return re.compile("^" + pattern + "$")


def wildcard_match(pattern, key):
    """
    Return whether the pattern (target module to add LoRA) matches the key (model weight name).

    Example:
    --------
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.0.self_attention.linear_qkv")
        True
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.1.self_attention.linear_qkv")
        False
    """
    if key is None:
        return False
    regex_pattern = _compile_wildcard_pattern(pattern)
    match = regex_pattern.match(key)
    return match is not None


@dataclass
class ModuleMatcher:
    """
    Matches Modules to apply PEFT adapters on.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to an empty list.
            If empty and no other parameter is provided it will match to "*_proj".
            Target modules can also contain wildcards (e.g. "*.layers.0.*.linear_qkv"). For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names to exclude from applying LoRA to.
            Defaults to an empty list.
            Exclude modules can also contain wildcards (e.g. "*.lm_head"). For example, you can specify
                exclude_modules=['*.lm_head'] to exclude the lm_head.
        match_all_linear (bool, optional): Whether to match all linear layers.
            Defaults to False. Prefer using target_modules or exclude_modules to specify the modules to match,
            to avoid issues with downstream tools (e.g., vLLM, etc).
        is_causal_lm (bool, optional): Whether the model is a causal language model.
    """

    target_modules: List[str] = field(default_factory=list)
    exclude_modules: List[str] = field(default_factory=list)
    match_all_linear: bool = field(default=False)
    is_causal_lm: bool = field(default=False)

    def __post_init__(self):
        """
        Input validation.
        """
        if self.target_modules is None:
            self.target_modules = []
        if self.exclude_modules is None:
            self.exclude_modules = []
        if isinstance(self.target_modules, str):
            self.target_modules = [self.target_modules]
        if isinstance(self.exclude_modules, str):
            self.exclude_modules = [self.exclude_modules]
        if self.match_all_linear is False and len(self.target_modules) == 0 and len(self.exclude_modules) == 0:
            logger.warning(
                "No modules specified for LoRA. Will use target_modules='*_proj' by default."
                """
            Equivalent to the following YAML configuration:
            peft:
              target_modules: '*_proj'
            If this is not what you want, please specify target_modules or exclude_modules.
            """
            )
            self.target_modules = ["*_proj"]

        if self.target_modules and self.exclude_modules:
            raise ValueError(
                "target_modules and exclude_modules are mutually exclusive. Please provide only one of them."
            )
        if self.match_all_linear and (len(self.target_modules) > 0 or len(self.exclude_modules) > 0):
            raise ValueError(
                "Expected target_modules/exclude_modules to be empty when match_all_linear is true. Please provide only one of them."
            )
        if self.match_all_linear:
            logger.warning(
                "match_all_linear is true. This will match all linear layers in the model (including lm_head). "
                "Please consider using target_modules or exclude_modules to specify the modules to match, to avoid issues with downstream tools "
                "For example, to match all linear layers except the lm_head, you can use: "
                "peft: "
                "  target_modules: '*_proj' "
            )

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def match(self, m: nn.Module, name: str = None, prefix: str = None):
        """
        Return (pattern, full_name) if the module matches; otherwise None.
        """
        full_name = f"{prefix}.{name}" if prefix else name

        # 1. matching by layer type takes absolute precedence
        if self.match_all_linear and _is_linear_module(m):
            return True

        # 2. target_modules is the next most-specific rule set
        elif self.target_modules:
            assert not self.exclude_modules, "`exclude_modules` must be empty when `target_modules` is used."
            for pattern in self.target_modules:
                if name == pattern or wildcard_match(pattern, full_name):
                    return True
            return False
        # 3. Fallback: “all linear layers except those explicitly excluded”
        else:
            return (
                name not in self.exclude_modules
                and not any(wildcard_match(pattern, full_name) for pattern in self.exclude_modules)
                and _is_linear_module(m)
            )
