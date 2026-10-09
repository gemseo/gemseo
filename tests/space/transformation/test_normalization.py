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

"""Tests of the working space built by the normalization."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import inf
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo.space.design import DesignSpace
from gemseo.space.transformation.normalization import SpaceNormalization
from gemseo.space.variable import VariableType


@pytest.fixture
def mixed_space() -> DesignSpace:
    """A space mixing what the normalization treats differently."""
    space = DesignSpace(name="mixed")
    space.add_real_variable("continuous", lower_bound=-2.0, upper_bound=3.0, value=1.0)
    space.add_real_variable(
        "vector", size=3, lower_bound=0.0, upper_bound=10.0, value=5.0
    )
    space.add_integer_variable("integer", lower_bound=0, upper_bound=8, value=4)
    space.add_real_variable("unbounded", value=0.25)
    space.add_real_variable(
        "half",
        size=2,
        lower_bound=(-inf, 1.0),
        upper_bound=(2.0, 4.0),
        value=(0.5, 2.0),
    )
    space.add_real_variable("without_value", lower_bound=0.0, upper_bound=1.0)
    return space


def test_transform_space_builds_the_whole_working_space(mixed_space) -> None:
    """Check every trait of the working space built from a mixed space."""
    working_space = SpaceNormalization(mixed_space).working_space

    # The working space has the variables of the space it normalizes,
    # in the same order and with the same sizes,
    # and is the same kind of space, bearing the same name.
    assert list(working_space) == list(mixed_space)
    assert type(working_space) is type(mixed_space)
    assert working_space.name == mixed_space.name
    assert working_space.dimension == mixed_space.dimension
    assert [working_space.variables[name].size for name in working_space] == [
        mixed_space.variables[name].size for name in mixed_space
    ]
    # A normalized variable is continuous,
    # whatever the type it is normalized from;
    # a variable left alone keeps its own type.
    assert [working_space.variables[name].type for name in working_space] == [
        VariableType.REAL,
        VariableType.REAL,
        VariableType.INTEGER,
        VariableType.REAL,
        VariableType.REAL,
        VariableType.REAL,
    ]
    # A normalizable component lands in the unit interval;
    # a component without finite bounds and an integer one keep the bounds they have.
    assert_allclose(
        working_space.get_lower_bounds(),
        array([0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, 0.0, 0.0]),
    )
    assert_allclose(
        working_space.get_upper_bounds(),
        array([1.0, 1.0, 1.0, 1.0, 8.0, inf, 2.0, 1.0, 1.0]),
    )
    # The normalization masks follow the bounds of the working space.
    assert_equal(
        dict(working_space.name_to_normalization_mask),
        dict(mixed_space.name_to_normalization_mask),
    )
    # A variable without a value keeps none,
    # and the others hold their normalized value.
    assert not working_space.has_current_value
    assert_allclose(working_space.get_current_value(["continuous"]), array([0.6]))
    assert_allclose(working_space.get_current_value(["vector"]), array([0.5] * 3))
    assert_allclose(working_space.get_current_value(["integer"]), array([4.0]))
    assert_allclose(working_space.get_current_value(["unbounded"]), array([0.25]))
    assert_allclose(working_space.get_current_value(["half"]), array([0.5, 1.0 / 3.0]))
    with pytest.raises(KeyError):
        working_space.get_current_value(["without_value"])


def test_transform_space_leaves_the_original_space_untouched(mixed_space) -> None:
    """Check that building the working space does not disturb the space it maps."""
    lower_bounds = mixed_space.get_lower_bounds().copy()
    upper_bounds = mixed_space.get_upper_bounds().copy()
    variables = dict(mixed_space.variables)

    SpaceNormalization(mixed_space).working_space

    assert_allclose(mixed_space.get_lower_bounds(), lower_bounds)
    assert_allclose(mixed_space.get_upper_bounds(), upper_bounds)
    # The variable objects are shared with the working space
    # where the normalization leaves them alone,
    # and a variable is immutable.
    assert dict(mixed_space.variables) == variables
    assert_allclose(mixed_space.get_current_value(["continuous"]), array([1.0]))


def test_working_space_current_value_mutation_is_isolated(mixed_space) -> None:
    """Check that mutating the working space's current value spares the original.

    A driver mutates the working space's current value while it runs,
    e.g. through `set_current_variable` or by casting it to complex for a
    complex-step derivative,
    and the working space only shares what is immutable with the space
    it was built from,
    so none of that reaches the current value of the latter.
    """
    original_value = mixed_space.get_current_value(["continuous"]).copy()
    working_space = SpaceNormalization(mixed_space).working_space

    working_space.set_current_variable("continuous", array([0.1]))
    assert_allclose(mixed_space.get_current_value(["continuous"]), original_value)

    working_space.to_complex()
    assert working_space.get_current_value(["continuous"]).dtype.kind == "c"
    assert mixed_space.get_current_value(["continuous"]).dtype.kind == "f"
