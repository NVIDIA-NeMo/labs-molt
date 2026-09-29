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

"""Tests for _prepopulate_remote_code_cache in _transformers/model_init.py."""

import os
import shutil
from unittest.mock import MagicMock

from nemo_automodel._transformers.model_init import _prepopulate_remote_code_cache


def test_copies_full_closure_into_cache_dir(tmp_path, monkeypatch):
    """Every .py from the checkpoint dir is copied into the resolved cache dir.

    HF's get_cached_module_file copies only the main file (direct imports); the
    function must add the rest so the transitive closure is complete.
    """
    src = tmp_path / "consolidated"
    src.mkdir()
    for name in ["modeling_x.py", "configuration_x.py", "shim_y.py"]:
        (src / name).write_text(f"# {name}\n")
    cache_root = tmp_path / "hf_modules"
    cache_root.mkdir()
    submodule = os.path.join("transformers_modules", "consolidated", "hash")

    def fake_get_cached_module_file(path, module_file, **kwargs):
        cache_dir = os.path.join(str(cache_root), submodule)
        os.makedirs(cache_dir, exist_ok=True)
        # simulate HF copying ONLY the requested module file
        shutil.copy2(os.path.join(path, module_file), os.path.join(cache_dir, module_file))
        return os.path.join(submodule, module_file)

    import transformers.dynamic_module_utils as dmu

    monkeypatch.setattr(dmu, "get_cached_module_file", fake_get_cached_module_file)
    monkeypatch.setattr(dmu, "HF_MODULES_CACHE", str(cache_root))

    cfg = MagicMock()
    cfg.auto_map = {"AutoModelForCausalLM": "modeling_x.MyModel"}
    _prepopulate_remote_code_cache(cfg, str(src), {})

    cache_dir = cache_root / "transformers_modules" / "consolidated" / "hash"
    copied = {p.name for p in cache_dir.glob("*.py")}
    assert {"modeling_x.py", "configuration_x.py", "shim_y.py"} <= copied


def test_noop_when_path_not_a_dir(tmp_path):
    cfg = MagicMock()
    cfg.auto_map = {"AutoModelForCausalLM": "modeling_x.MyModel"}
    # must not raise for a non-existent path (e.g. hub id loads)
    _prepopulate_remote_code_cache(cfg, str(tmp_path / "missing"), {})


def test_noop_when_no_auto_map(tmp_path):
    cfg = MagicMock()
    cfg.auto_map = None
    _prepopulate_remote_code_cache(cfg, str(tmp_path), {})
