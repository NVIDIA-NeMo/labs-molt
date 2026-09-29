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

"""Tests for _apply_transformers_compat_guards in checkpoint/addons.py."""

import sys
import types

from nemo_automodel.components.checkpoint.addons import _apply_transformers_compat_guards

_REMOVED_IMPORT = "from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING, GenerationMixin\n"


def test_inserts_guard_before_removed_symbol_import(tmp_path):
    p = tmp_path / "modeling_x.py"
    p.write_text("import os\n" + _REMOVED_IMPORT + "x = 1\n")

    _apply_transformers_compat_guards(str(p))
    text = p.read_text()

    assert "import transformers.generation.utils as _nemo_compat_mod" in text
    assert 'if not hasattr(_nemo_compat_mod, "NEED_SETUP_CACHE_CLASSES_MAPPING")' in text
    # the guard is inserted before the original import line
    assert text.index("_nemo_compat_mod") < text.index("from transformers.generation.utils import NEED_SETUP")
    # the patched file is still valid Python
    compile(text, str(p), "exec")


def test_idempotent_across_calls(tmp_path):
    p = tmp_path / "modeling_x.py"
    p.write_text(_REMOVED_IMPORT)

    _apply_transformers_compat_guards(str(p))
    after_once = p.read_text()
    _apply_transformers_compat_guards(str(p))

    assert p.read_text() == after_once
    assert after_once.count("import transformers.generation.utils as _nemo_compat_mod") == 1


def test_file_without_removed_symbol_is_unchanged(tmp_path):
    p = tmp_path / "plain.py"
    original = "import os\nfrom transformers import AutoModel\nx = 2\n"
    p.write_text(original)

    _apply_transformers_compat_guards(str(p))

    assert p.read_text() == original


def test_missing_file_is_noop(tmp_path):
    # unreadable / missing path must not raise
    _apply_transformers_compat_guards(str(tmp_path / "does_not_exist.py"))


def test_guard_defines_symbol_when_transformers_dropped_it(tmp_path, monkeypatch):
    """The patched preamble resolves the import even when transformers lacks the symbol."""
    p = tmp_path / "modeling_x.py"
    p.write_text(_REMOVED_IMPORT)
    _apply_transformers_compat_guards(str(p))

    # stub a transformers.generation.utils WITHOUT NEED_SETUP_CACHE_CLASSES_MAPPING
    tf = types.ModuleType("transformers")
    tf.__path__ = []
    gen = types.ModuleType("transformers.generation")
    gen.__path__ = []
    gu = types.ModuleType("transformers.generation.utils")
    gu.GenerationMixin = type("GenerationMixin", (), {})
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "transformers.generation", gen)
    monkeypatch.setitem(sys.modules, "transformers.generation.utils", gu)

    ns: dict = {}
    exec(p.read_text(), ns)
    assert ns["NEED_SETUP_CACHE_CLASSES_MAPPING"] == {}
