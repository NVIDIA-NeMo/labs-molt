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
from __future__ import annotations

import uuid
from typing import Callable

import pytest

import nemo_automodel.shared.import_utils as si


def _random_module_name() -> str:
    return f"no_such_mod_{uuid.uuid4().hex}"


@pytest.fixture()
def placeholder_class():
    ok, placeholder = si.safe_import(_random_module_name())
    assert ok is False
    assert si.is_unavailable(placeholder) is True
    return placeholder


def test_placeholder_class_naming_and_msg(placeholder_class):
    # Name is prefixed and message is informative
    assert placeholder_class.__name__.startswith("MISSING")
    assert "could not be imported" in getattr(placeholder_class, "_msg", "")


def test_attribute_access_raises_unavailable_error(placeholder_class):
    with pytest.raises(si.UnavailableError):
        _ = getattr(placeholder_class, "nonexistent_attr")


def test_call_raises_unavailable_error(placeholder_class):
    with pytest.raises(si.UnavailableError):
        placeholder_class()


@pytest.mark.parametrize(
    "operation",
    [
        lambda c: c == object(),
        lambda c: c != object(),
        lambda c: c < 1,
        lambda c: c <= 1,
        lambda c: c > 1,
        lambda c: c >= 1,
    ],
)
def test_comparisons_raise_unavailable_error(placeholder_class, operation: Callable):
    with pytest.raises(si.UnavailableError):
        operation(placeholder_class)


@pytest.mark.parametrize(
    "operation",
    [
        lambda c: +c,  # triggers __pos__ via __add__ fallback; ensure coverage via raising methods
        lambda c: c + 1,
        lambda c: 1 + c,
        lambda c: c - 1,
        lambda c: 1 - c,
        lambda c: c * 2,
        lambda c: 2 * c,
        lambda c: c / 1,
        lambda c: 1 / c,
        lambda c: c // 1,
        lambda c: 1 // c,
        lambda c: c ** 2,
        lambda c: 2 ** c,
        lambda c: abs(c),
    ],
)
def test_arithmetic_like_ops_raise_unavailable_error(placeholder_class, operation: Callable):
    with pytest.raises((si.UnavailableError, TypeError)):
        operation(placeholder_class)


def test_len_iter_context_manager_and_hash_raise(placeholder_class):
    # __len__
    with pytest.raises(si.UnavailableError):
        len(placeholder_class)
    # __iter__
    with pytest.raises(si.UnavailableError):
        iter(placeholder_class)
    # __enter__ on the class should raise; __exit__ is not defined on placeholder
    with pytest.raises(si.UnavailableError):
        placeholder_class.__enter__()
    # __hash__
    with pytest.raises(si.UnavailableError):
        hash(placeholder_class)


def test_item_assignment_and_deletion_raise(placeholder_class):
    # __setitem__
    with pytest.raises(si.UnavailableError):
        placeholder_class["key"] = 1  # type: ignore[index]
    # __delitem__
    with pytest.raises(si.UnavailableError):
        del placeholder_class["key"]  # type: ignore[index]


def test_isinstance_checks_do_not_raise(placeholder_class):
    # The placeholder should not match any instance, and must not raise.
    assert isinstance(123, placeholder_class) is False


def test_unavailablemeta_default_msg_and_name_via_direct_metaclass():
    # Directly construct via metaclass without _msg to cover default path
    cls = si.UnavailableMeta("Foo", (), {})
    assert cls.__name__ == "MISSINGFoo"
    assert getattr(cls, "_msg") == "Foo could not be imported"


@pytest.mark.parametrize(
    "operation",
    [
        lambda c: (lambda x: None, c).__setitem__(1, None),  # noop pattern; we'll call methods below instead
    ],
)
@pytest.mark.skip(reason="placeholder - operations covered by explicit tests below")
def test_placeholder_skip_marker(placeholder_class, operation: Callable):
    pass


@pytest.mark.parametrize(
    "operation",
    [
        lambda c: (lambda v=c: None) or None,  # no-op to keep parametrize structure consistent
    ],
)
@pytest.mark.skip(reason="structure placeholder")
def test_structure_placeholder(placeholder_class, operation: Callable):
    pass


def test_inplace_ops_raise(placeholder_class):
    c = placeholder_class
    with pytest.raises((si.UnavailableError, TypeError)):
        c += 1  # __iadd__
    c = placeholder_class
    with pytest.raises((si.UnavailableError, TypeError)):
        c //= 1  # __ifloordiv__
    c = placeholder_class
    with pytest.raises((si.UnavailableError, TypeError)):
        c <<= 1  # __ilshift__
    c = placeholder_class
    with pytest.raises((si.UnavailableError, TypeError)):
        c **= 2  # __ipow__
    c = placeholder_class
    with pytest.raises((si.UnavailableError, TypeError)):
        c >>= 1  # __irshift__
    c = placeholder_class
    with pytest.raises((si.UnavailableError, TypeError)):
        c -= 1  # __isub__
    with pytest.raises((si.UnavailableError, TypeError)):
        c *= 1  # __imul__
    c = placeholder_class
    with pytest.raises((si.UnavailableError, TypeError)):
        c /= 1  # __itruediv__


def test_shift_ops_and_divmod_and_unary_raise(placeholder_class):
    with pytest.raises((si.UnavailableError, TypeError)):
        _ = placeholder_class << 1  # __lshift__
    with pytest.raises((si.UnavailableError, TypeError)):
        _ = 1 << placeholder_class  # __rlshift__
    with pytest.raises((si.UnavailableError, TypeError)):
        _ = placeholder_class >> 1  # __rshift__
    with pytest.raises((si.UnavailableError, TypeError)):
        _ = 1 >> placeholder_class  # __rrshift__
    with pytest.raises((si.UnavailableError, TypeError)):
        _ = divmod(placeholder_class, 1)  # __divmod__
    with pytest.raises((si.UnavailableError, TypeError)):
        _ = divmod(1, placeholder_class)  # __rdivmod__
    with pytest.raises((si.UnavailableError, TypeError)):
        _ = -placeholder_class  # __neg__
    with pytest.raises((si.UnavailableError, TypeError)):
        _ = ~placeholder_class  # __invert__


def test_index_get_delete_raise(placeholder_class):
    with pytest.raises((si.UnavailableError, TypeError)):
        placeholder_class.__index__()  # __index__
    with pytest.raises(si.UnavailableError):
        placeholder_class.__get__(None, None)  # __get__
    with pytest.raises(si.UnavailableError):
        placeholder_class.__delete__(None, None)  # __delete__
