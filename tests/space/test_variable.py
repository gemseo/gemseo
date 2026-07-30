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
"""Tests for the Variable class."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import atleast_1d
from numpy import inf
from numpy import int64
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from gemseo.space._variable import DataType
from gemseo.space._variable import Variable
from gemseo.util.testing.helper import assert_exception


def test_init_defaults():
    """Test the default values of __init__."""
    v = Variable()
    assert v.type == DataType.FLOAT
    assert v.size == 1
    assert v.lower_bound == -array([inf])
    assert v.upper_bound == array([inf])


@pytest.mark.parametrize(
    ("size", "type_", "lower_bound", "upper_bound"),
    [
        (1, DataType.FLOAT, 0, 0.0),
        (10, "integer", -5.0, 0.0),
        (1, "float", -5.0, inf),
        (1, "float", -inf, inf),
        (2, "float", (-4, 4), inf),
    ],
)
def test_initialization(
    size: int,
    type_: DataType,
    lower_bound: float | tuple[float, float],
    upper_bound: float,
) -> None:
    """Test the instantiation."""
    my_variable = Variable(
        size=size, type=type_, lower_bound=lower_bound, upper_bound=upper_bound
    )
    assert my_variable.size == size
    assert my_variable.type == type_
    assert (my_variable.lower_bound == atleast_1d(lower_bound)).all()
    assert (my_variable.upper_bound == atleast_1d(upper_bound)).all()


@pytest.mark.parametrize("size", [-1, 0])
def test_non_positive_size(size, snapshot) -> None:
    """Check non-positive variables size."""
    with assert_exception(ValidationError, snapshot):
        Variable(size=size)


def test_invalid_type(snapshot) -> None:
    """Check invalid variable type."""
    with assert_exception(ValidationError, snapshot):
        Variable(type="complex")


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_invalid_bound_size(side, snapshot) -> None:
    """Check invalid bound size."""
    with assert_exception(ValidationError, snapshot):
        Variable(**{f"{side}_bound": [0, 0]})


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_invalid_bound_value_scalar(side, snapshot) -> None:
    """Check invalid bound value type."""
    with assert_exception(ValidationError, snapshot):
        Variable(**{f"{side}_bound": 1j})


@pytest.mark.parametrize(
    ("size", "type_", "lower_bound", "upper_bound"),
    [(1, DataType.FLOAT, 0, -1.0), (1, "integer", 0, -1)],
)
def test_wrong_boundaries(
    size: int, type_: str, lower_bound: float, upper_bound: float, snapshot
) -> None:
    """Test the instantiation with `upper_bound` lower than `lower_bound`."""
    with assert_exception(ValueError, snapshot):
        Variable(
            size=size, type=type_, lower_bound=lower_bound, upper_bound=upper_bound
        )


@pytest.fixture
def variable() -> Variable:
    """A variable."""
    return Variable(size=1, type="float", lower_bound=0, upper_bound=1)


@pytest.mark.parametrize("bound", ["lower_bound", "upper_bound"])
def test_frozen(variable, bound, snapshot) -> None:
    """Check that a variable is immutable (bounds cannot be reassigned)."""
    with assert_exception(ValidationError, snapshot):
        setattr(variable, bound, 0)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_multidimensional_bound(side, snapshot) -> None:
    """Check a bound with more than one dimension."""
    with assert_exception(ValidationError, snapshot):
        Variable(size=2, **{f"{side}_bound": array([[1.0, 2.0]])})


def test_model_copy_without_update(variable) -> None:
    """Check that copying a variable without an update returns the variable itself."""
    assert variable.model_copy() is variable
    assert variable.model_copy(deep=True) is variable


def test_model_copy_with_inconsistent_update(snapshot) -> None:
    """Check that an update inconsistent with the bounds is rejected.

    The base implementation of `model_copy` writes the update into `__dict__` without
    validating it.
    """
    variable = Variable(size=2, lower_bound=0.0, upper_bound=1.0)
    with assert_exception(ValidationError, snapshot):
        variable.model_copy(update={"size": 5})


def test_model_copy_leaves_original_alone() -> None:
    """Check that an update returns a new variable and does not touch the original."""
    variable = Variable(size=2, lower_bound=0.0, upper_bound=1.0)

    new_variable = variable.model_copy(update={"lower_bound": array([-9.0, -9.0])})

    assert new_variable is not variable
    assert_array_equal(new_variable.lower_bound, array([-9.0, -9.0]))
    assert not new_variable.lower_bound.flags.writeable
    # The base implementation would have written the update into the original.
    assert_array_equal(variable.lower_bound, array([0.0, 0.0]))
    assert not variable.lower_bound.flags.writeable


def test_model_copy_converts_the_update() -> None:
    """Check that a scalar bound of an update is converted and typed as expected."""
    variable = Variable(size=2, type="integer", lower_bound=0, upper_bound=10)

    new_variable = variable.model_copy(update={"upper_bound": 3})

    assert new_variable.type == DataType.INTEGER
    assert new_variable.upper_bound.dtype == int64
    assert_array_equal(new_variable.upper_bound, array([3, 3]))
