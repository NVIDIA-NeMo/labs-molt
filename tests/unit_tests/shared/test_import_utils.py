# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import sys
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version as PkgVersion

import nemo_automodel.shared.import_utils as si


@pytest.fixture(scope="module")
def dummy_module() -> types.ModuleType:
    """
    Create an *empty* module that is present in ``sys.modules`` for the duration
    of the test session.  This allows us to test the *symbol-missing* branch of
    ``safe_import_from`` without provoking an ImportError.
    """
    name = f"dummy_{uuid.uuid4().hex}"
    module = types.ModuleType(name)
    sys.modules[name] = module
    yield module
    sys.modules.pop(name, None)


def _random_module_name() -> str:
    """Return a module name that is (very likely) missing."""
    return f"no_such_mod_{uuid.uuid4().hex}"


def test_safe_import_success():
    """Importing an existing std-lib module should succeed."""
    ok, mod = si.safe_import("math")
    assert ok is True
    assert mod.sqrt(4) == 2


def test_safe_import_alt_object():
    """
    When an ``alt`` object is supplied, it must be returned instead of the
    placeholder and the success flag must still be *False*.
    """
    sentinel = object()
    ok, obj = si.safe_import(_random_module_name(), alt=sentinel)
    assert ok is False
    assert obj is sentinel


def test_safe_import_from_success():
    """Happy path: import a known symbol from a known module."""
    ok, sqrt = si.safe_import_from("math", "sqrt")
    assert ok is True
    assert sqrt(9) == 3


def test_safe_import_from_fallback(dummy_module):
    """
    Symbol is missing from *primary* module but present in *fallback* module.
    This should succeed and return the symbol from the fallback.
    """
    ok, sym = si.safe_import_from(dummy_module.__name__, "sqrt", fallback_module="math")
    assert ok is True
    assert sym(16) == 4


def test_safe_import_from_alt_object():
    """Verify the ``alt`` path."""
    sentinel = object()
    ok, obj = si.safe_import_from(_random_module_name(), "foo", alt=sentinel)
    assert ok is False
    assert obj is sentinel


def test_get_torch_version_type():
    """
    ``get_torch_version`` should *never* raise – even when torch is unavailable
    while building docs – and must always return a ``packaging.version.Version``.
    """
    ver = si.get_torch_version()
    assert isinstance(ver, PkgVersion)


def test_get_transformers_version_type():
    """
    ``get_transformers_version`` should *never* raise – even when transformers is unavailable
    while building docs – and must always return a ``packaging.version.Version``.
    """
    ver = si.get_transformers_version()
    assert isinstance(ver, PkgVersion)


def test_is_transformers_min_version():
    """
    * A ridiculously low requirement must be satisfied.
    * A far-future version must *not* be satisfied.
    """
    assert si.is_transformers_min_version("0.0.0") is True
    assert si.is_transformers_min_version("9999.0.0", check_equality=False) is False


def test_get_check_model_inputs_decorator():
    """
    ``get_check_model_inputs_decorator`` should always return a callable decorator.
    """
    decorator = si.get_check_model_inputs_decorator()
    assert callable(decorator)


def test_get_check_model_inputs_decorator_new_api():
    """check_model_inputs() factory path (transformers >= 4.57.3)."""
    inner = lambda f: f  # noqa: E731
    mock_check = MagicMock(return_value=inner)
    mod = types.ModuleType("transformers.utils.generic")
    mod.check_model_inputs = mock_check

    with patch.dict(sys.modules, {"transformers.utils.generic": mod}):
        with patch.object(si, "is_transformers_min_version", return_value=True):
            result = si.get_check_model_inputs_decorator()

    assert result is inner
    mock_check.assert_called_once_with()


def test_get_check_model_inputs_decorator_old_api():
    """check_model_inputs direct-decorator path (transformers < 4.57.3)."""
    sentinel = lambda f: f  # noqa: E731
    mod = types.ModuleType("transformers.utils.generic")
    mod.check_model_inputs = sentinel

    with patch.dict(sys.modules, {"transformers.utils.generic": mod}):
        with patch.object(si, "is_transformers_min_version", return_value=False):
            result = si.get_check_model_inputs_decorator()

    assert result is sentinel


def test_get_check_model_inputs_decorator_combined():
    """Combined merge_with_config_defaults + capture_outputs path (transformers >= 5.2.0)."""
    mock_merge = MagicMock()
    mock_capture = MagicMock()

    generic_mod = types.ModuleType("transformers.utils.generic")
    generic_mod.merge_with_config_defaults = mock_merge
    # No check_model_inputs → first try-block raises ImportError

    output_mod = types.ModuleType("transformers.utils.output_capturing")
    output_mod.capture_outputs = mock_capture

    with patch.dict(
        sys.modules,
        {
            "transformers.utils.generic": generic_mod,
            "transformers.utils.output_capturing": output_mod,
        },
    ):
        decorator = si.get_check_model_inputs_decorator()

    assert callable(decorator)

    # Verify composition: capture_outputs first, then merge_with_config_defaults
    dummy = lambda x: x  # noqa: E731
    result = decorator(dummy)
    mock_capture.assert_called_once_with(dummy)
    mock_merge.assert_called_once_with(mock_capture.return_value)
    assert result is mock_merge.return_value


def test_get_check_model_inputs_decorator_fallback_to_null():
    """Falls back to null_decorator when all transformers imports fail."""
    generic_mod = types.ModuleType("transformers.utils.generic")
    output_mod = types.ModuleType("transformers.utils.output_capturing")
    # Neither module has the expected attributes → both try-blocks raise ImportError

    with patch.dict(
        sys.modules,
        {
            "transformers.utils.generic": generic_mod,
            "transformers.utils.output_capturing": output_mod,
        },
    ):
        result = si.get_check_model_inputs_decorator()

    assert result is si.null_decorator


def test_get_check_model_inputs_decorator_fallback_with_kwargs():
    """
    The ``null_decorator`` fallback (returned by ``get_check_model_inputs_decorator``
    when transformers decorators are unavailable) must work as a plain ``@decorator``
    on functions called with keyword arguments.

    This guards against a bug where a ``@contextmanager``-wrapped fallback
    produced a ``ContextDecorator`` whose ``__call__`` collided with model
    forward kwargs like ``input_ids``.
    """
    decorator = si.null_decorator

    @decorator
    def dummy(x=None):
        return x

    assert dummy(x=42) == 42


def test_null_decorator_as_direct_decorator():
    """
    ``null_decorator`` must be a valid no-op decorator in ``@decorator`` form.
    """

    @si.null_decorator
    def _identity(x):
        return x

    assert _identity(7) == 7


def test_null_decorator_as_factory():
    """
    ``null_decorator`` must also work in ``@decorator()`` form.
    """

    @si.null_decorator()
    def _identity(x):
        return x

    assert _identity(11) == 11
