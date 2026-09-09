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
"""Tests for the continuous variable."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import float64
from numpy import inf
from numpy.testing import assert_array_equal

from gemseo.space.variable import ContinuousVariable
from gemseo.space.variable import DataType
from gemseo.util.testing.helper import assert_exception


def test_init_defaults() -> None:
    """Test the default values of __init__."""
    variable = ContinuousVariable()
    assert variable.type == DataType.FLOAT
    assert variable.size == 1
    assert variable.lower_bound == -array([inf])
    assert variable.upper_bound == array([inf])


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("bound", [array([1.5]), array([1.5, 2.5])])
def test_bound_with_non_integer_components(side, bound) -> None:
    """Check that a continuous variable accepts non-integer bound components."""
    kwargs = {"lower_bound": -inf, "upper_bound": inf, f"{side}_bound": bound}
    my_variable = ContinuousVariable(size=bound.size, **kwargs)
    assert_array_equal(getattr(my_variable, f"{side}_bound"), bound)


def test_component_type() -> None:
    """Check the NumPy type of the components of a continuous variable."""
    assert ContinuousVariable().component_type is float64


def test_cast() -> None:
    """Check that a continuous variable casts to float but preserves a complex value."""
    variable = ContinuousVariable(size=2)
    assert variable.cast(array([1, 2])).dtype == float64
    complex_value = array([1.0 + 1.0j, 2.0 + 2.0j])
    cast = variable.cast(complex_value)
    assert cast.dtype == complex_value.dtype
    # The value is copied, so that the caller does not keep a hand on it.
    assert cast is not complex_value
    assert_array_equal(cast, complex_value)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected"),
    [(-inf, inf, 0.0), (-inf, 2.0, 2.0), (1.0, inf, 1.0), (1.0, 3.0, 2.0)],
)
def test_compute_default_component_value(lower_bound, upper_bound, expected) -> None:
    """Check the default value of a component."""
    assert (
        ContinuousVariable.compute_default_component_value(lower_bound, upper_bound)
        == expected
    )


def test_model_copy_leaves_original_alone(snapshot) -> None:
    """Check that an update returns a new variable and does not touch the original."""
    variable = ContinuousVariable(size=2, lower_bound=0.0, upper_bound=1.0)

    new_variable = variable.model_copy(update={"lower_bound": array([-9.0, -9.0])})

    assert new_variable is not variable
    assert_array_equal(new_variable.lower_bound, array([-9.0, -9.0]))
    assert not new_variable.lower_bound.flags.writeable
    # A variable stores read-only views of its bounds,
    # so the writeable flag cannot be re-enabled.
    with assert_exception(ValueError, snapshot):
        new_variable.lower_bound.setflags(write=True)

    # The base implementation would have written the update into the original.
    assert_array_equal(variable.lower_bound, array([0.0, 0.0]))
    assert not variable.lower_bound.flags.writeable


def test_find_components_outside_domain() -> None:
    """Check the components outside the domain of a continuous variable."""
    variable = ContinuousVariable(size=2, lower_bound=0, upper_bound=10)
    assert variable.find_components_outside_domain(array([1.0, 1.5])) == set()
