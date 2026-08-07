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
"""Tests for the checking module."""

from __future__ import annotations

from math import inf
from math import nan

import pytest
from numpy import array
from numpy import dtype
from numpy import zeros

from gemseo.space._variable import DataType
from gemseo.space._variable import Variable
from gemseo.space.design._bounds import Bounds
from gemseo.space.design._checking import check
from gemseo.space.design._checking import check_addable_value
from gemseo.space.design._checking import check_membership
from gemseo.space.design._checking import check_out_array
from gemseo.space.design._variables import Variables
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def variables() -> Variables:
    """A variables with a float variable and an integer variable."""
    variables = Variables()
    variables["x"] = Variable(
        size=2, type=DataType.FLOAT, lower_bound=0.0, upper_bound=10.0
    )
    variables["y"] = Variable(
        size=1, type=DataType.INTEGER, lower_bound=0, upper_bound=5
    )
    return variables


@pytest.fixture
def bounds(variables: Variables) -> Bounds:
    """The bounds of the variables."""
    return Bounds(variables)


def test_check_addable_value_valid(variables: Variables) -> None:
    """Check that a valid value is accepted."""
    assert check_addable_value(variables, array([1.0, 2.0]), "x")


def test_check_addable_value_all_none(variables: Variables) -> None:
    """Check that an all-`None` value is accepted."""
    assert check_addable_value(variables, array([None, None]), "x")


def test_check_addable_value_2d_raises(variables: Variables, snapshot) -> None:
    """Check that a value with more than one dimension raises."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([[1.0]]), "x")


def test_check_addable_value_non_numeric(variables: Variables, snapshot) -> None:
    """Check that a non-numeric component raises."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array(["a", 1.0], dtype=object), "x")


def test_check_addable_value_several_non_numeric(
    variables: Variables, snapshot
) -> None:
    """Check that several non-numeric components raise."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array(["a", "b"], dtype=object), "x")


def test_check_addable_value_nan(variables: Variables, snapshot) -> None:
    """Check that a nan component raises."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([nan, 1.0]), "x")


def test_check_addable_value_several_nan(variables: Variables, snapshot) -> None:
    """Check that several nan components raise."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([nan, nan]), "x")


def test_check_addable_value_non_integer_for_integer_variable(
    variables: Variables, snapshot
) -> None:
    """Check that a non-integer component raises for an integer variable."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([1.5]), "y")


def test_check_addable_value_several_non_integer_for_integer_variable(
    snapshot,
) -> None:
    """Check that several non-integer components raise for an integer variable."""
    variables = Variables()
    variables["z"] = Variable(
        size=2, type=DataType.INTEGER, lower_bound=0, upper_bound=5
    )
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([1.5, 2.5]), "z")


def test_check_addable_value_infinite_for_integer_variable(
    variables: Variables,
) -> None:
    """Check that an infinite component is accepted for an integer variable."""
    assert check_addable_value(variables, array([inf]), "y")


def test_check_membership_wrong_type(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that a value that is neither an array nor a mapping raises."""
    with assert_exception(TypeError, snapshot):
        check_membership(variables, bounds, [1.0, 2.0, 3.0])


def test_check_membership_wrong_shape(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that an array whose last dimension mismatches the full size raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(variables, bounds, array([1.0, 2.0]))


def test_check_membership_array_within_bounds(
    variables: Variables, bounds: Bounds
) -> None:
    """Check that a valid full array raises nothing."""
    check_membership(variables, bounds, array([5.0, 5.0, 3.0]))


def test_check_membership_array_lower_violation(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that a full array violating a lower bound raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(variables, bounds, array([-1.0, 5.0, 3.0]))


def test_check_membership_array_upper_violation(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that a full array violating an upper bound raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(variables, bounds, array([5.0, 15.0, 3.0]))


def test_check_membership_array_2d_recursion(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that each row of a stacked array is checked, recursively."""
    full_value = array([[5.0, 5.0, 3.0], [-1.0, 5.0, 3.0]])
    with assert_exception(ValueError, snapshot):
        check_membership(variables, bounds, full_value)


def test_check_membership_array_with_reordered_names(
    variables: Variables, bounds: Bounds
) -> None:
    """Check that an array with explicit, reordered names is dispatched by name."""
    # The full value is ordered as (y, x), matching `names`.
    check_membership(variables, bounds, array([3.0, 5.0, 5.0]), names=("y", "x"))


def test_check_membership_dict_valid(variables: Variables, bounds: Bounds) -> None:
    """Check that a valid mapping raises nothing."""
    check_membership(
        variables,
        bounds,
        {"x": array([5.0, 5.0]), "y": array([3.0])},
    )


def test_check_membership_dict_wrong_size(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that a mapping value of the wrong size raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(
            variables,
            bounds,
            {"x": array([1.0, 2.0, 3.0]), "y": array([3.0])},
        )


def test_check_membership_dict_lower_bound_violation(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that a mapping component violating a lower bound raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(
            variables,
            bounds,
            {"x": array([-1.0, 5.0]), "y": array([3.0])},
        )


def test_check_membership_dict_upper_bound_violation(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that a mapping component violating an upper bound raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(
            variables,
            bounds,
            {"x": array([5.0, 15.0]), "y": array([3.0])},
        )


def test_check_membership_dict_integer_violation(
    variables: Variables, bounds: Bounds, snapshot
) -> None:
    """Check that a non-integer mapping component raises for an integer variable."""
    with assert_exception(ValueError, snapshot):
        check_membership(
            variables,
            bounds,
            {"x": array([5.0, 5.0]), "y": array([3.5])},
        )


def test_check_membership_dict_with_none_value(
    variables: Variables, bounds: Bounds
) -> None:
    """Check that a `None` mapping value is skipped without error."""
    check_membership(
        variables,
        bounds,
        {"x": array([5.0, 5.0]), "y": None},
    )


def test_check_empty_variables(snapshot) -> None:
    """Check that an empty variables raises, without calling the checker."""
    calls = []
    with assert_exception(ValueError, snapshot):
        check(Variables(), lambda: calls.append(1))
    assert calls == []


def test_check_calls_current_value_checker(variables: Variables) -> None:
    """Check that a non-empty variables calls the current-value checker once."""
    calls = []
    check(variables, lambda: calls.append(1))
    assert calls == [1]


def test_check_propagates_current_value_checker_error(
    variables: Variables,
) -> None:
    """Check that an error raised by the current-value checker propagates."""

    def _raise() -> None:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        check(variables, _raise)


def test_check_out_array_valid() -> None:
    """Check that an array of the dtype and the shape of the result is accepted."""
    assert check_out_array(zeros(3), dtype("float64"), (3,)) is None


def test_check_out_array_wrong_shape(snapshot) -> None:
    """Check the error raised when the array has not the shape of the result."""
    with assert_exception(ValueError, snapshot):
        check_out_array(zeros((3, 2)), dtype("float64"), (2,))


def test_check_out_array_wrong_dtype(snapshot) -> None:
    """Check the error raised when the array has not the dtype of the result."""
    with assert_exception(ValueError, snapshot):
        check_out_array(zeros(3), dtype("complex128"), (3,))
