# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Tests for the normalization of the call arguments."""

from __future__ import annotations

from typing import Any

import pytest

from gemseo.util._workflow_observer.interface import CallSpec
from gemseo.util._workflow_observer.interface import _get_signature_without_self
from gemseo.util._workflow_observer.interface import normalize_arguments_safely
from gemseo.util.testing.helper import assert_exception


def _method(self, x: int, y: int = 2, *, z: int = 3) -> None:  # noqa: ARG001
    """A method with a positional, a defaulted and a keyword-only parameter."""


def _variadic_positional_method(self, x: int, *args: Any) -> None:  # noqa: ARG001
    """A method with a variadic positional parameter."""


def _variadic_keyword_method(self, x: int, **kwargs: Any) -> None:  # noqa: ARG001
    """A method with a variadic keyword parameter."""


def _variadic_both_method(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
    """A method with both a variadic positional and a variadic keyword parameter."""


def _positional_only_keyword_method(self, x: int, /, **kwargs: Any) -> None:  # noqa: ARG001
    """A method with a positional-only parameter and a variadic keyword one."""


def test_create_safely_binds_the_positional_arguments_to_their_parameter_names():
    """Verify that an argument passed positionally is stored under its name."""
    call_spec = CallSpec.create_safely(_method, (1, 20), {})
    assert call_spec.kwargs == {"x": 1, "y": 20, "z": 3}


def test_create_safely_applies_the_default_values_of_the_omitted_parameters():
    """Verify that a parameter the caller omitted is stored with its default."""
    call_spec = CallSpec.create_safely(_method, (), {"x": 1})
    assert call_spec.kwargs == {"x": 1, "y": 2, "z": 3}


def test_create_safely_keeps_the_callable_and_ignores_the_first_parameter():
    """Verify that the callable is kept and that `self` is not an argument."""
    call_spec = CallSpec.create_safely(_method, (1,), {})
    assert call_spec.callable_ is _method
    assert "self" not in call_spec.kwargs


def test_create_safely_raises_for_arguments_not_matching_the_signature(snapshot):
    """Verify that arguments that the method could not accept are rejected.

    A non-variadic signature is bound as is: arguments that the method itself
    would reject are not silently recorded.
    """
    with assert_exception(TypeError, snapshot):
        CallSpec.create_safely(_method, (), {})


def test_get_signature_without_self_keeps_a_variadic_signature():
    """Verify that a variadic signature is returned rather than `None`."""
    signature_ = _get_signature_without_self(_variadic_positional_method)
    assert list(signature_.parameters) == ["x", "args"]


def test_get_signature_without_self_drops_the_first_parameter():
    """Verify that `self` is not part of the returned signature."""
    signature_ = _get_signature_without_self(_method)
    assert signature_ is not None
    assert list(signature_.parameters) == ["x", "y", "z"]


def test_normalize_arguments_safely_normalizes_a_non_variadic_signature():
    """Verify that a non-variadic signature is bound by parameter name."""
    assert normalize_arguments_safely(_method, (1, 20), {}) == {
        "x": 1,
        "y": 20,
        "z": 3,
    }


def test_normalize_arguments_safely_binds_a_variadic_positional_signature():
    """Verify that a `*args` signature binds its named parameter.

    The extra positional arguments are kept as a tuple under the name of the
    `*args` parameter prefixed with `*`, so that a trace shows which argument
    is variadic.
    """
    result = normalize_arguments_safely(_variadic_positional_method, (1, 2, 3), {})
    assert result == {"x": 1, "*args": (2, 3)}


def test_normalize_arguments_safely_flattens_a_variadic_keyword_signature():
    """Verify that a `**kwargs` signature binds its named parameter.

    The extra keyword arguments are flattened into the returned mapping
    instead of staying nested under the `**kwargs` parameter's own name.
    """
    result = normalize_arguments_safely(
        _variadic_keyword_method, (1,), {"y": 2, "z": 3}
    )
    assert result == {"x": 1, "y": 2, "z": 3}
    assert "kwargs" not in result


def test_normalize_arguments_safely_flattens_a_keyword_named_after_the_args():
    """Verify that a keyword argument named like the `*args` parameter is flattened.

    `f(self, *args, **kwargs)` called with `args=9` puts `args` into `kwargs`.
    The extra positional arguments are keyed by `"*args"`, not `"args"`, so
    flattening overwrites nothing.
    """
    result = normalize_arguments_safely(_variadic_both_method, (1, 2), {"args": 9})
    assert result == {"*args": (1, 2), "args": 9}


@pytest.mark.parametrize(
    ("callable_", "args", "kwargs", "expected"),
    [
        (
            _positional_only_keyword_method,
            (1,),
            {"x": 2},
            {"x": 1, "**kwargs": {"x": 2}},
        ),
        (
            _variadic_both_method,
            (1,),
            {"*args": 2},
            {"*args": (1,), "**kwargs": {"*args": 2}},
        ),
    ],
    ids=["positional-only parameter", "dictionary unpacking"],
)
def test_normalize_arguments_safely_keeps_the_keyword_arguments_nested_on_a_clash(
    callable_, args, kwargs, expected
):
    """Verify that flattening is skipped when it would overwrite an argument.

    Python accepts two calls where a keyword argument captured by `**kwargs`
    has the key of another argument: a keyword argument named after a
    positional-only parameter, and a key such as `"*args"` passed by
    dictionary unpacking. Flattening would then silently overwrite that
    argument, so the `**kwargs` entries are left nested under `"**kwargs"`
    in these cases only, and no error is raised.

    Args:
        callable_: The method called.
        args: The positional arguments of the call.
        kwargs: The keyword arguments of the call.
        expected: The expected normalized arguments.
    """
    assert normalize_arguments_safely(callable_, args, kwargs) == expected


def test_normalize_arguments_safely_introspects_a_signature_once():
    """Verify that the introspection of a signature is cached.

    Computing a signature is expensive relative to an observed call, e.g. an
    MDA iteration, hence the cache on `_get_signature_without_self`.
    """
    _get_signature_without_self.cache_clear()

    for _ in range(3):
        normalize_arguments_safely(_method, (1,), {})

    cache_info = _get_signature_without_self.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 2


def test_create_safely_binds_a_variadic_signature():
    """Verify that `create_safely` returns the bound mapping of a variadic signature."""
    call_spec = CallSpec.create_safely(_variadic_positional_method, (1, 2), {})
    assert call_spec.kwargs == {"x": 1, "*args": (2,)}
    assert call_spec.callable_ is _variadic_positional_method
