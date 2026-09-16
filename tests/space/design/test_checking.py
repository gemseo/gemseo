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

from gemseo.space._design.bounds import Bounds
from gemseo.space._design.checking import check_addable_value
from gemseo.space._design.checking import check_membership
from gemseo.space._design.variables import DesignVariables
from gemseo.space.variable import ContinuousVariable
from gemseo.space.variable import DiscreteVariable
from gemseo.space.variable import IntegerVariable
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def variables() -> DesignVariables:
    """A variables with a float variable and an integer variable."""
    variables = DesignVariables()
    variables["x"] = ContinuousVariable(size=2, lower_bound=0.0, upper_bound=10.0)
    variables["y"] = IntegerVariable(size=1, lower_bound=0, upper_bound=5)
    return variables


@pytest.fixture
def bounds(variables: DesignVariables) -> Bounds:
    """The bounds of the variables."""
    return Bounds(variables)


def test_check_addable_value_valid(variables: DesignVariables) -> None:
    """Check that a valid value is accepted."""
    assert check_addable_value(variables, array([1.0, 2.0]), "x")


def test_check_addable_value_all_none(variables: DesignVariables) -> None:
    """Check that an all-`None` value is accepted."""
    assert check_addable_value(variables, array([None, None]), "x")


def test_check_addable_value_2d_raises(variables: DesignVariables, snapshot) -> None:
    """Check that a value with more than one dimension raises."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([[1.0]]), "x")


def test_check_addable_value_non_numeric(variables: DesignVariables, snapshot) -> None:
    """Check that a non-numeric component raises."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array(["a", 1.0], dtype=object), "x")


def test_check_addable_value_several_non_numeric(
    variables: DesignVariables, snapshot
) -> None:
    """Check that several non-numeric components raise."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array(["a", "b"], dtype=object), "x")


def test_check_addable_value_nan(variables: DesignVariables, snapshot) -> None:
    """Check that a nan component raises."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([nan, 1.0]), "x")


def test_check_addable_value_several_nan(variables: DesignVariables, snapshot) -> None:
    """Check that several nan components raise."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([nan, nan]), "x")


def test_check_addable_value_non_integer_for_integer_variable(
    variables: DesignVariables, snapshot
) -> None:
    """Check that a non-integer component raises for an integer variable."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([1.5]), "y")


def test_check_addable_value_several_non_integer_for_integer_variable(
    snapshot,
) -> None:
    """Check that several non-integer components raise for an integer variable."""
    variables = DesignVariables()
    variables["z"] = IntegerVariable(size=2, lower_bound=0, upper_bound=5)
    with assert_exception(ValueError, snapshot):
        check_addable_value(variables, array([1.5, 2.5]), "z")


def test_check_addable_value_infinite_for_integer_variable(
    variables: DesignVariables,
) -> None:
    """Check that an infinite component is accepted for an integer variable."""
    assert check_addable_value(variables, array([inf]), "y")


def test_check_membership_wrong_type(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that a value that is neither an array nor a mapping raises."""
    with assert_exception(TypeError, snapshot):
        check_membership(variables, bounds, [1.0, 2.0, 3.0])


def test_check_membership_wrong_shape(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that an array whose last dimension mismatches the full size raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(variables, bounds, array([1.0, 2.0]))


def test_check_membership_array_within_bounds(
    variables: DesignVariables, bounds: Bounds
) -> None:
    """Check that a valid full array raises nothing."""
    check_membership(variables, bounds, array([5.0, 5.0, 3.0]))


def test_check_membership_array_lower_violation(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that a full array violating a lower bound raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(variables, bounds, array([-1.0, 5.0, 3.0]))


def test_check_membership_array_upper_violation(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that a full array violating an upper bound raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(variables, bounds, array([5.0, 15.0, 3.0]))


def test_check_membership_array_2d_recursion(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that each row of a stacked array is checked, recursively."""
    full_value = array([[5.0, 5.0, 3.0], [-1.0, 5.0, 3.0]])
    with assert_exception(ValueError, snapshot):
        check_membership(variables, bounds, full_value)


def test_check_membership_array_with_reordered_names(
    variables: DesignVariables, bounds: Bounds
) -> None:
    """Check that an array with explicit, reordered names is dispatched by name."""
    # The full value is ordered as (y, x), matching `names`.
    check_membership(variables, bounds, array([3.0, 5.0, 5.0]), names=("y", "x"))


def test_check_membership_dict_valid(
    variables: DesignVariables, bounds: Bounds
) -> None:
    """Check that a valid mapping raises nothing."""
    check_membership(
        variables,
        bounds,
        {"x": array([5.0, 5.0]), "y": array([3.0])},
    )


def test_check_membership_dict_wrong_size(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that a mapping value of the wrong size raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(
            variables,
            bounds,
            {"x": array([1.0, 2.0, 3.0]), "y": array([3.0])},
        )


def test_check_membership_dict_lower_bound_violation(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that a mapping component violating a lower bound raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(
            variables,
            bounds,
            {"x": array([-1.0, 5.0]), "y": array([3.0])},
        )


def test_check_membership_dict_upper_bound_violation(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that a mapping component violating an upper bound raises."""
    with assert_exception(ValueError, snapshot):
        check_membership(
            variables,
            bounds,
            {"x": array([5.0, 15.0]), "y": array([3.0])},
        )


def test_check_membership_dict_integer_violation(
    variables: DesignVariables, bounds: Bounds, snapshot
) -> None:
    """Check that a non-integer mapping component raises for an integer variable."""
    with assert_exception(ValueError, snapshot):
        check_membership(
            variables,
            bounds,
            {"x": array([5.0, 5.0]), "y": array([3.5])},
        )


def test_check_membership_dict_with_none_value(
    variables: DesignVariables, bounds: Bounds
) -> None:
    """Check that a `None` mapping value is skipped without error."""
    check_membership(
        variables,
        bounds,
        {"x": array([5.0, 5.0]), "y": None},
    )


@pytest.fixture
def discrete_variables() -> DesignVariables:
    """A variables with a float variable and a discrete variable."""
    variables = DesignVariables()
    variables["x"] = ContinuousVariable(lower_bound=0.0, upper_bound=10.0)
    variables["d"] = DiscreteVariable(choices=[2.0, 5.0])
    return variables


@pytest.fixture
def discrete_bounds(discrete_variables: DesignVariables) -> Bounds:
    """The bounds of the variables including a discrete one."""
    return Bounds(discrete_variables)


def test_check_addable_value_outside_a_discrete_domain(
    discrete_variables: DesignVariables, snapshot
) -> None:
    """Check that a value that is not a choice raises."""
    with assert_exception(ValueError, snapshot):
        check_addable_value(discrete_variables, array([3.0]), "d")


@pytest.mark.parametrize("value", [{"x": array([1.0]), "d": array([3.0])}])
def test_check_membership_dict_outside_a_discrete_domain(
    discrete_variables: DesignVariables, discrete_bounds: Bounds, value, snapshot
) -> None:
    """Check that the mapping path rejects a value that is not a choice."""
    with assert_exception(ValueError, snapshot):
        check_membership(discrete_variables, discrete_bounds, value)


def test_check_membership_array_outside_a_discrete_domain(
    discrete_variables: DesignVariables, discrete_bounds: Bounds, snapshot
) -> None:
    """Check that the array path rejects a value that is not a choice.

    The value lies within the derived bounds, so the bound comparison accepts it.
    """
    with assert_exception(ValueError, snapshot):
        check_membership(discrete_variables, discrete_bounds, array([1.0, 3.0]))


def test_check_membership_array_within_a_discrete_domain(
    discrete_variables: DesignVariables, discrete_bounds: Bounds
) -> None:
    """Check that the array path accepts the choices."""
    check_membership(discrete_variables, discrete_bounds, array([1.0, 5.0]))


def test_check_membership_2d_array_with_a_discrete_variable(
    discrete_variables: DesignVariables, discrete_bounds: Bounds, snapshot
) -> None:
    """Check that the array path handles several values at once."""
    check_membership(
        discrete_variables, discrete_bounds, array([[1.0, 5.0], [2.0, 2.0]])
    )
    with assert_exception(ValueError, snapshot):
        check_membership(
            discrete_variables, discrete_bounds, array([[1.0, 5.0], [2.0, 3.0]])
        )
