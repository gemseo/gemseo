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
"""Tests for the integer variable."""

from __future__ import annotations

import warnings

import pytest
from numpy import array
from numpy import inf
from numpy import int64
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from gemseo.space.variable import DataType
from gemseo.space.variable import IntegerVariable
from gemseo.util.testing.helper import assert_exception


def test_init_defaults() -> None:
    """Test the default values of __init__."""
    variable = IntegerVariable()
    assert variable.type == DataType.INTEGER
    assert variable.size == 1
    assert variable.lower_bound == -array([inf])
    assert variable.upper_bound == array([inf])


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("bound", [array([1.5]), array([1.5, 2.5])])
def test_bound_with_non_integer_components(side, bound, snapshot) -> None:
    """Check a bound with one or several non-integer components."""
    with assert_exception(ValidationError, snapshot):
        IntegerVariable(size=bound.size, **{f"{side}_bound": bound})


def test_component_type() -> None:
    """Check the NumPy type of the components of an integer variable."""
    assert IntegerVariable().component_type is int64


def test_cast() -> None:
    """Check that an integer variable casts to int."""
    cast = IntegerVariable(size=2).cast(array([1.6, 2.6]))
    assert cast.dtype == int64
    assert_array_equal(cast, array([1, 2]))


def test_model_copy_converts_the_update() -> None:
    """Check that a scalar bound of an update is converted and typed as expected."""
    variable = IntegerVariable(size=2, lower_bound=0, upper_bound=10)

    new_variable = variable.model_copy(update={"upper_bound": 3})

    assert isinstance(new_variable, IntegerVariable)
    assert new_variable.type == DataType.INTEGER
    assert new_variable.upper_bound.dtype == int64
    assert_array_equal(new_variable.upper_bound, array([3, 3]))


def test_find_components_outside_domain() -> None:
    """Check the components outside the domain of an integer variable."""
    variable = IntegerVariable(size=2, lower_bound=0, upper_bound=10)
    assert variable.find_components_outside_domain(array([1.0, 1.5])) == {1}


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound"),
    [
        (array([-inf, 0]), array([inf, 1])),
        (array([-inf, 0]), array([1, 1])),
        (array([1, 0]), array([inf, 1])),
    ],
)
@pytest.mark.parametrize("size", [1, 2])
def test_unbounded_integer_variable(lower_bound, upper_bound, size) -> None:
    """Check that an unbounded integer variable does not warn about its bounds."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        variable = IntegerVariable(
            size=size,
            lower_bound=lower_bound[:size],
            upper_bound=upper_bound[:size],
        )

    assert_array_equal(variable.lower_bound, lower_bound[:size])
    assert_array_equal(variable.upper_bound, upper_bound[:size])
