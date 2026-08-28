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
"""Tests for the Value collaborator."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import complex128
from numpy import float32
from numpy import float64
from numpy import int64
from numpy.testing import assert_equal

from gemseo.space._variable import ContinuousVariable
from gemseo.space._variable import IntegerVariable
from gemseo.space.design._bounds import Bounds
from gemseo.space.design._integer_rounder import IntegerRounder
from gemseo.space.design._normalizer import Normalizer
from gemseo.space.design._value import Value
from gemseo.space.design._variables import Variables
from gemseo.util.testing.helper import assert_exception


def _build_value(*names: str) -> tuple[Variables, Value]:
    """Build a Value over float variables of size 1 with bounds [0, 1].

    Args:
        names: The names of the variables.

    Returns:
        The variables and the value.
    """
    variables = Variables()
    for name in names:
        variables[name] = ContinuousVariable(size=1, lower_bound=0.0, upper_bound=1.0)
    bounds = Bounds(variables)
    normalizer = Normalizer(variables, bounds, IntegerRounder(variables))
    return variables, Value(variables, bounds, normalizer)


def _build_integer_value(name: str) -> tuple[Variables, Value]:
    """Build a Value over a single integer variable with bounds [-10, 10].

    Args:
        name: The name of the variable.

    Returns:
        The variables and the value.
    """
    variables = Variables()
    variables[name] = IntegerVariable(size=1, lower_bound=-10, upper_bound=10)
    bounds = Bounds(variables)
    normalizer = Normalizer(variables, bounds, IntegerRounder(variables))
    return variables, Value(variables, bounds, normalizer)


def _resize(variables: Variables, name: str, size: int) -> None:
    """Replace a variable with a bigger one, bumping the variables version.

    Args:
        variables: The variables.
        name: The name of the variable to resize.
        size: The new size of the variable.
    """
    variables[name] = ContinuousVariable(size=size, lower_bound=0.0, upper_bound=1.0)


@pytest.fixture
def value() -> Value:
    """A Value over a single float variable with bounds [0, 1]."""
    return _build_value("x")[1]


def test_set_variable_unknown_name(value, snapshot) -> None:
    """Check that setting the value of an unknown variable raises."""
    with assert_exception(KeyError, snapshot):
        value.set_variable("unknown", array([0.5]))


def test_set_variable_rejects_non_integer_value(snapshot) -> None:
    """Check that set_variable rejects a non-integer value for an integer variable."""
    _, value = _build_integer_value("i")

    with assert_exception(ValueError, snapshot):
        value.set_variable("i", array([2.7]))

    assert value.name_to_value == {}


@pytest.mark.parametrize("given_value", [array([3]), array([3.0])])
def test_set_variable_accepts_a_valid_integer_value(given_value) -> None:
    """Check that set_variable accepts a valid integer value and casts it to int64."""
    _, value = _build_integer_value("i")

    value.set_variable("i", given_value)

    assert value.name_to_value["i"].dtype == int64
    assert_equal(value.name_to_value["i"], array([3]))


def test_refresh_status_drops_resized_value(snapshot) -> None:
    """Check that a resizing replacement invalidates the stored value."""
    variables, value = _build_value("x")
    value.set_variable("x", array([0.5]))
    assert value.has_value

    _resize(variables, "x", 3)

    assert not value.has_value
    # The variable keeps its entry, marked as having no value.
    assert value.name_to_value["x"] is None
    with assert_exception(KeyError, snapshot):
        value.get()


@pytest.mark.parametrize("read_status_first", [False, True])
def test_resize_invalidation_survives_another_mutation(
    read_status_first, snapshot
) -> None:
    """Check that mutating a variable does not cancel a pending resize drop."""
    variables, value = _build_value("x", "y")
    value.set({"x": array([0.5]), "y": array([0.5])})

    _resize(variables, "x", 3)
    if read_status_first:
        # Reading the status applies the invalidation eagerly; the outcome must
        # not depend on whether such a read happened.
        assert not value.has_value

    value.set_variable("y", array([0.25]))

    assert not value.has_value
    assert value.name_to_value["x"] is None
    with assert_exception(KeyError, snapshot):
        value.get(["x"])


@pytest.mark.parametrize("dtype", [int64, float32, float64])
def test_set_casts_to_the_component_type(dtype) -> None:
    """Check that set casts the value of a float variable to float64."""
    _, value = _build_value("x")

    value.set({"x": array([1.0], dtype=dtype)})

    assert value.name_to_value["x"].dtype == float64


def test_set_does_not_alias_the_given_value() -> None:
    """Check that set stores a copy of the value of the caller."""
    _, value = _build_value("x")
    given_value = array([0.5])

    value.set({"x": given_value})
    given_value[0] = 0.25

    assert value.name_to_value["x"][0] == 0.5


def test_set_keeps_a_complex_value() -> None:
    """Check that set leaves a complex value untouched, for the complex step."""
    _, value = _build_value("x")

    value.set({"x": array([0.5 + 1j])})

    assert value.name_to_value["x"].dtype == complex128


def test_set_does_not_alias_the_given_complex_value() -> None:
    """Check that set stores a copy of a complex value of the caller."""
    _, value = _build_value("x")
    given_value = array([0.5 + 1j])

    value.set({"x": given_value})
    given_value[0] = 0.25

    assert value.name_to_value["x"][0] == 0.5 + 1j


def test_set_variable_does_not_alias_the_given_value() -> None:
    """Check that set_variable stores a copy of the value of the caller."""
    _, value = _build_value("x")
    given_value = array([0.5])

    value.set_variable("x", given_value)
    given_value[0] = 0.25

    assert value.name_to_value["x"][0] == 0.5


def test_set_keeps_wrong_size_value() -> None:
    """Check that a wrong-size value passed to set is left in place.

    It is up to the membership check to reject it with a size message,
    instead of being silently dropped as a stale value.
    """
    variables, value = _build_value("x", "y")
    _resize(variables, "x", 3)

    value.set({"x": array([0.5]), "y": array([0.5])})

    assert not value.has_value
    assert_equal(value.name_to_value["x"], array([0.5]))


def test_set_variable_keeps_wrong_size_value() -> None:
    """Check that set_variable leaves a wrong-size value to the membership check.

    The domain of the kind of the variable cannot be checked component-wise
    against a value of another size; rejecting it is up to the membership check,
    with a size message.
    """
    variables, value = _build_value("x")
    _resize(variables, "x", 3)

    value.set_variable("x", array([0.5]))

    assert_equal(value.name_to_value["x"], array([0.5]))


def test_get_as_dict_excludes_wrong_size_value() -> None:
    """Check that a partial as_dict get() drops a wrong-size stored value.

    A wrong-size value can reach the store outside of a resize (Value.set is
    permissive by design, see test_set_keeps_wrong_size_value above); get()
    must not hand it back as if it were a valid current value.
    """
    value = _build_value("x", "y")[1]
    value.set({"x": array([0.5, 0.5]), "y": array([0.5])})

    assert not value.has_value
    assert_equal(value.get(as_dict=True), {"y": array([0.5])})


def test_get_full_raises_naming_wrong_size_value(snapshot) -> None:
    """Check that the full get() names a wrong-size stored value as missing."""
    value = _build_value("x", "y")[1]
    value.set({"x": array([0.5, 0.5]), "y": array([0.5])})

    with assert_exception(KeyError, snapshot):
        value.get()


def test_get_names_raises_for_wrong_size_value(snapshot) -> None:
    """Check that a per-name get() rejects a wrong-size stored value."""
    value = _build_value("x", "y")[1]
    value.set({"x": array([0.5, 0.5]), "y": array([0.5])})

    with assert_exception(KeyError, snapshot):
        value.get(["x"], as_dict=True)


def test_initialize_missing_after_resize() -> None:
    """Check that a value invalidated by a resize reads as missing."""
    variables, value = _build_value("x")
    value.set_variable("x", array([0.32]))

    _resize(variables, "x", 3)
    value.initialize_missing()

    assert value.has_value
    assert_equal(value.name_to_value["x"], array([0.5, 0.5, 0.5]))


def test_check_value_ignores_resized_value() -> None:
    """Check that a value invalidated by a resize is not checked against bounds."""
    variables = Variables()
    variables["x"] = ContinuousVariable(size=2, lower_bound=0.0, upper_bound=1.0)
    bounds = Bounds(variables)
    value = Value(
        variables, bounds, Normalizer(variables, bounds, IntegerRounder(variables))
    )
    value.set_variable("x", array([0.5, 0.5]))

    # The new size does not broadcast against the stored one and the new bounds
    # exclude the stored value.
    variables["x"] = ContinuousVariable(size=3, lower_bound=2.0, upper_bound=3.0)

    value.check_value("x")
    assert value.name_to_value == {"x": None}


def test_to_complex_keeps_no_value(snapshot) -> None:
    """Check that casting to complex preserves the no-value markers."""
    _, value = _build_value("x", "y")
    value.set_variable("x", array([0.5]))
    value.set_variable("y", None)

    value.to_complex()

    assert value.name_to_value["y"] is None
    assert_equal(value.name_to_value["x"], array([0.5 + 0j]))
    assert not value.has_value
    with assert_exception(KeyError, snapshot):
        value.get()
