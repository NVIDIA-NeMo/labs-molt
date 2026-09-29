# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Registry guardrail: every head-owning registered model declares a TieSupport policy.

``reject_unsupported_tie_word_embeddings`` defaults *undeclared* classes to
``TieSupport.BOTH``, so this test is the only enforcement that a newly onboarded
causal-LM / conditional-generation model cannot silently skip declaring
``tie_word_embeddings_support``. Every ``MODEL_ARCH_MAPPING`` class must either
declare a policy or be listed in the explicit, commented exemption below.
"""

import importlib

from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING
from nemo_automodel.components.models.common.tie_word_embeddings import TieSupport

# Registered classes that do NOT own a causal LM head and are therefore exempt from
# declaring ``tie_word_embeddings_support``. Keep this explicit and commented so a
# head-owning model cannot be onboarded without either a declaration or a deliberate
# exemption reviewed here.
_TIE_SUPPORT_EXEMPT: dict[str, str] = {}


def _registered_classes():
    """Yield ``(arch, module_path, class_name)`` for each distinct registered class."""
    seen: set[str] = set()
    for arch, spec in MODEL_ARCH_MAPPING.items():
        module_path, class_name = spec[0], spec[1]
        if class_name in seen:
            continue
        seen.add(class_name)
        yield arch, module_path, class_name


def test_every_registered_lm_head_class_declares_tie_support():
    checked = 0
    for arch, module_path, class_name in _registered_classes():
        if class_name in _TIE_SUPPORT_EXEMPT:
            continue
        try:
            cls = getattr(importlib.import_module(module_path), class_name)
        except Exception:
            # Optional-dependency-gated arch not importable in this environment; it is
            # exercised in the fuller CI image. Skip rather than raise a false failure.
            continue
        support = getattr(cls, "tie_word_embeddings_support", None)
        assert isinstance(support, TieSupport), (
            f"{class_name} (registered as {arch!r}) does not declare a TieSupport policy. "
            f"Add `tie_word_embeddings_support: TieSupport = TieSupport.<BOTH|TIED_ONLY|UNTIED_ONLY>` "
            f"to the class, or add it to _TIE_SUPPORT_EXEMPT with a reason if it owns no causal lm_head."
        )
        if support in (TieSupport.BOTH, TieSupport.TIED_ONLY):
            assert callable(cls.__dict__.get("tie_weights")), (
                f"{class_name} declares {support.name} but does not define a model-local tie_weights(). "
                "Implement the exact lm_head/input-embedding alias instead of relying on an inherited HF method."
            )
        checked += 1
    assert checked > 0, "no registered classes were checked — MODEL_ARCH_MAPPING import likely broke"


def test_tie_support_exempt_list_has_no_stale_entries():
    """Every exempt name must still be a registered class (catch renames/removals)."""
    registered = {class_name for _, _, class_name in _registered_classes()}
    stale = set(_TIE_SUPPORT_EXEMPT) - registered
    assert not stale, f"stale exempt entries no longer in MODEL_ARCH_MAPPING: {sorted(stale)}"
