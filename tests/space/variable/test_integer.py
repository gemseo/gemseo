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
from numpy import float64
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
@pytest.mark.parametrize(
    ("size", "bound"),
    [
        (1, 1.5),
        (1, [1.5]),
        (1, (1.5,)),
        (1, array([1.5])),
        (2, array([1.5, 2.5])),
    ],
)
def test_bound_with_non_integer_components(side, size, bound, snapshot) -> None:
    """Check a bound with one or several non-integer components."""
    with assert_exception(ValidationError, snapshot):
        IntegerVariable(size=size, **{f"{side}_bound": bound})


@pytest.mark.parametrize("bound", [2, 2.0, [2.0], (2.0,), array([2.0]), array([2])])
def test_finite_bound_is_stored_as_integer(bound) -> None:
    """Check that a finite bound is stored as an integer, whatever its type."""
    variable = IntegerVariable(lower_bound=bound, upper_bound=10)
    assert variable.lower_bound.dtype == int64
    assert_array_equal(variable.lower_bound, array([2]))


@pytest.mark.parametrize(
    ("side", "bound"),
    [
        ("lower", [-1e30]),
        ("upper", [1e30]),
        ("upper", [2.0**63]),
        ("lower", [-(2.0**63) - 2.0**11]),
        ("upper", [0.0, 1e30]),
    ],
)
def test_bound_outside_the_range_of_an_integer(side, bound, snapshot) -> None:
    """Check that a finite bound outside the range of a 64-bit integer is rejected.

    Casting it would overflow silently to the smallest 64-bit integer
    and give the variable a domain that is not the one asked for,
    while keeping it as a floating-point number would break
    the computation of the default value, which is cast to an integer.
    """
    with assert_exception(ValidationError, snapshot):
        IntegerVariable(size=len(bound), **{f"{side}_bound": array(bound)})


def test_bound_at_the_edge_of_the_range_of_an_integer() -> None:
    """Check that the smallest 64-bit integer is an acceptable bound."""
    variable = IntegerVariable(lower_bound=array([-(2.0**63)]))
    assert variable.lower_bound.dtype == int64
    assert_array_equal(variable.lower_bound, array([-(2**63)]))
    assert_array_equal(variable.get_default_value(), array([-(2**63)]))


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_infinite_bound_is_stored_as_float(side) -> None:
    """Check that an infinite bound keeps a floating-point type.

    An integer type cannot hold an infinite component.
    """
    bound = -inf if side == "lower" else inf
    variable = IntegerVariable(**{f"{side}_bound": bound})
    assert getattr(variable, f"{side}_bound").dtype == float64


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


@pytest.mark.parametrize("enable_integer_normalization", [False, True])
def test_normalization_mask_is_read_only(
    enable_integer_normalization, snapshot
) -> None:
    """Check that the normalization mask of an integer variable is frozen."""
    variable = IntegerVariable(size=2, lower_bound=0, upper_bound=1)
    mask = variable.get_normalization_mask(enable_integer_normalization)
    assert_array_equal(mask, array([enable_integer_normalization] * 2))
    assert not mask.flags.writeable
    with assert_exception(ValueError, snapshot):
        mask[0] = not enable_integer_normalization
