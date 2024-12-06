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
"""Tests for the module design_space."""

from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.mdo.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.mdo.scalable.parametric.scalable_design_space import (
    ScalableDesignSpace,
)


@pytest.fixture(scope="module")
def default_design_space() -> DesignSpace:
    """The expected default design space."""
    design_space = DesignSpace()
    design_space.add_variable("x_0", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("x_2", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y_1", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y_2", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return design_space


@pytest.fixture(scope="module")
def custom_design_space() -> DesignSpace:
    """The expected default design space."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_0", size=2, lower_bound=0.0, upper_bound=1.0, value=array([0.3, 0.7])
    )
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=1.0, value=0.1)
    design_space.add_variable(
        "x_2", size=2, lower_bound=0.0, upper_bound=1.0, value=0.5
    )
    design_space.add_variable(
        "x_3", size=3, lower_bound=0.0, upper_bound=1.0, value=0.3
    )
    design_space.add_variable(
        "y_1", size=4, lower_bound=0.0, upper_bound=1.0, value=0.4
    )
    design_space.add_variable(
        "y_2", size=5, lower_bound=0.0, upper_bound=1.0, value=0.55
    )
    design_space.add_variable(
        "y_3", size=6, lower_bound=0.0, upper_bound=1.0, value=0.5
    )
    return design_space


def test_dtype(default_design_space) -> None:
    """Check the use of a custom NumPy data type."""
    design_space = ScalableDesignSpace()
    for name in default_design_space:
        assert design_space.variable_types[name] == "float"


def check_variable(space: DesignSpace, name: str, reference_space: DesignSpace) -> None:
    """Check a variable of a design space.

    Args:
        space: The design space.
        name: The name of the variable.
        reference_space: The design space of reference.

    Returns:
        Whether the variable is valid.
    """
    assert name in space
    assert space.get_size(name) == reference_space.get_size(name)
    assert space.get_type(name) == reference_space.get_type(name)
    assert_equal(space.get_lower_bound(name), reference_space.get_lower_bound(name))
    assert_equal(space.get_upper_bound(name), reference_space.get_upper_bound(name))
    assert_equal(
        space.get_current_value([name]), reference_space.get_current_value([name])
    )


def test_default_design_space(default_design_space) -> None:
    """Check the default configuration of the design space."""
    design_space = ScalableDesignSpace()
    assert len(default_design_space) == len(design_space)
    for name in default_design_space:
        assert design_space.variable_types[name] == "float"
        check_variable(design_space, name, default_design_space)


def test_default_design_space_with_uncertainties(default_design_space) -> None:
    """Check the default configuration of the design space with uncertainties."""
    design_space = ScalableDesignSpace(add_uncertain_variables=True)
    assert len(default_design_space) + 2 == len(design_space)
    for name in default_design_space:
        check_variable(design_space, name, default_design_space)

    assert design_space.get_size("u_1") == 1
    assert design_space.get_size("u_2") == 1


def test_custom_design_space(custom_design_space) -> None:
    """Check the custom configuration of the design space."""
    design_space = ScalableDesignSpace(
        [
            ScalableDisciplineSettings(1, 4),
            ScalableDisciplineSettings(2, 5),
            ScalableDisciplineSettings(3, 6),
        ],
        2,
        {
            "x_0": array([0.3, 0.7]),
            "x_1": array([0.1] * 1),
            "x_3": array([0.3] * 3),
            "y_1": array([0.4] * 4),
            "y_2": array([0.55] * 5),
        },
    )
    assert len(custom_design_space) == len(design_space)
    for name in custom_design_space:
        check_variable(design_space, name, custom_design_space)


def test_custom_design_space_with_uncertainties(custom_design_space) -> None:
    """Check the custom configuration of the design space with uncertainties."""
    design_space = ScalableDesignSpace(
        [
            ScalableDisciplineSettings(1, 4),
            ScalableDisciplineSettings(2, 5),
            ScalableDisciplineSettings(3, 6),
        ],
        2,
        {
            "x_0": array([0.3, 0.7]),
            "x_1": array([0.1] * 1),
            "x_3": array([0.3] * 3),
            "y_1": array([0.4] * 4),
            "y_2": array([0.55] * 5),
        },
        add_uncertain_variables=True,
    )
    assert len(custom_design_space) + 3 == len(design_space)
    for name in custom_design_space:
        check_variable(design_space, name, custom_design_space)

    assert design_space.get_size("u_1") == 4
    assert design_space.get_size("u_2") == 5
    assert design_space.get_size("u_3") == 6
