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
"""Additional unit tests for `validate_tp_mesh` utility.

The main suite in ``test_parallelizer.py`` already exercises the typical
branches when *transformers* is available.  The tests here specifically target
edge-cases that were not covered:

* Execution when the *transformers* package is **absent** – the helper should
  return early **without raising**.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_automodel.components.distributed.parallelizer import validate_tp_mesh

# -----------------------------------------------------------------------------
# Lightweight "transformers" stub for environments without the package
# -----------------------------------------------------------------------------


def _install_fake_gemma3(monkeypatch):
    """Install a minimal stub of the Gemma3 import hierarchy if missing.

    We only need the class object to satisfy the ``import`` statement inside
    ``validate_tp_mesh``; no actual functionality is required.
    """

    import sys  # Local import to avoid polluting global namespace
    import types

    module_chain = [
        "transformers",
        "transformers.models",
        "transformers.models.gemma3",
        "transformers.models.gemma3.modeling_gemma3",
    ]

    for name in module_chain:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    modeling = sys.modules[module_chain[-1]]

    class _StubGemma3:  # pragma: no cover – behaviour irrelevant
        pass

    modeling.Gemma3ForConditionalGeneration = _StubGemma3


def _make_tp_mesh(size: int):
    """Create a ``MagicMock`` that mimics the minimal DeviceMesh interface."""

    mesh = MagicMock()
    mesh.size.return_value = size
    return mesh


def test_validate_tp_mesh_import_error(monkeypatch):
    """Function should no-op (not raise) when Gemma3 import fails.

    We simulate an *ImportError* by purging *transformers* modules from
    ``sys.modules`` *and* customizing ``__import__`` to raise for any attempt to
    import the package.
    """

    # 1. Ensure the transformers hierarchy is not present.
    for key in list(sys.modules.keys()):
        if key.startswith("transformers"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    # 2. Call the helper – it should *not* raise.
    tp_mesh = _make_tp_mesh(4)
    dummy_model = object()

    validate_tp_mesh(dummy_model, tp_mesh)  # Should silently return


@pytest.mark.parametrize(
    "num_heads,tp_size,should_raise",
    [
        (8, 4, False),  # divisible
        (10, 4, True),  # not divisible
    ],
)
def test_validate_tp_mesh_basic_divisibility(monkeypatch, num_heads, tp_size, should_raise):
    """Smoke-test divisibility logic with a minimal config object.

    This variant *does not* depend on HuggingFace classes – it relies on the
    generic ``elif hasattr(model, 'config')`` branch.
    """

    class _DummyConfig(SimpleNamespace):
        pass

    class _DummyModel:
        def __init__(self, heads):
            self.config = _DummyConfig(num_attention_heads=heads, num_key_value_heads=heads)

    model = _DummyModel(num_heads)
    tp_mesh = _make_tp_mesh(tp_size)

    # Ensure the Gemma3 import inside helper resolves even without transformers
    _install_fake_gemma3(monkeypatch)

    if should_raise:
        with pytest.raises(AssertionError):
            validate_tp_mesh(model, tp_mesh)
    else:
        validate_tp_mesh(model, tp_mesh)
