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
from numpy.testing import assert_equal

from gemseo.space._variable import DataType
from gemseo.space._variable import Variable
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
        variables[name] = Variable(
            size=1, type=DataType.FLOAT, lower_bound=0.0, upper_bound=1.0
        )
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
    variables[name] = Variable(
        size=size, type=DataType.FLOAT, lower_bound=0.0, upper_bound=1.0
    )


@pytest.fixture
def value() -> Value:
    """A Value over a single float variable with bounds [0, 1]."""
    return _build_value("x")[1]


def test_set_variable_unknown_name(value, snapshot) -> None:
    """Check that setting the value of an unknown variable raises."""
    with assert_exception(KeyError, snapshot):
        value.set_variable("unknown", array([0.5]))


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
    variables["x"] = Variable(
        size=2, type=DataType.FLOAT, lower_bound=0.0, upper_bound=1.0
    )
    bounds = Bounds(variables)
    value = Value(
        variables, bounds, Normalizer(variables, bounds, IntegerRounder(variables))
    )
    value.set_variable("x", array([0.5, 0.5]))

    # The new size does not broadcast against the stored one and the new bounds
    # exclude the stored value.
    variables["x"] = Variable(
        size=3, type=DataType.FLOAT, lower_bound=2.0, upper_bound=3.0
    )

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
