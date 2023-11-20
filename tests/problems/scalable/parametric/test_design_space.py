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
from gemseo.algos.parameter_space import RandomVariable
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.scalable.parametric.scalable_design_space import (
    ScalableDesignSpace,
)


@pytest.fixture(scope="module")
def default_design_space() -> DesignSpace:
    """The expected default design space."""
    design_space = DesignSpace()
    design_space.add_variable("x_0", l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("y_1", l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("y_2", l_b=0.0, u_b=1.0, value=0.5)
    return design_space


@pytest.fixture(scope="module")
def custom_design_space() -> DesignSpace:
    """The expected default design space."""
    design_space = DesignSpace()
    design_space.add_variable("x_0", size=2, l_b=0.0, u_b=1.0, value=array([0.3, 0.7]))
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0, value=0.1)
    design_space.add_variable("x_2", size=2, l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("x_3", size=3, l_b=0.0, u_b=1.0, value=0.3)
    design_space.add_variable("y_1", size=4, l_b=0.0, u_b=1.0, value=0.4)
    design_space.add_variable("y_2", size=5, l_b=0.0, u_b=1.0, value=0.55)
    design_space.add_variable("y_3", size=6, l_b=0.0, u_b=1.0, value=0.5)
    return design_space


def test_dtype(default_design_space):
    """Check the use of a custom NumPy data type."""
    _design_space = ScalableDesignSpace()
    for name in default_design_space:
        assert _design_space.variable_types[name] == array(["float"])


def test_default_design_space(default_design_space):
    """Check the default configuration of the design space."""
    _design_space = ScalableDesignSpace()
    assert len(default_design_space) == len(_design_space)
    for name in default_design_space:
        assert _design_space.variable_types[name] == array(["float"])
        assert default_design_space[name] == _design_space[name]


def test_default_design_space_with_uncertainties(default_design_space):
    """Check the default configuration of the design space with uncertainties."""
    _design_space = ScalableDesignSpace(add_uncertain_variables=True)
    assert len(default_design_space) + 2 == len(_design_space)
    for name in default_design_space:
        assert default_design_space[name] == _design_space[name]

    assert _design_space["u_1"] == RandomVariable("OTNormalDistribution", 1)
    assert _design_space["u_2"] == RandomVariable("OTNormalDistribution", 1)


def test_custom_design_space(custom_design_space):
    """Check the custom configuration of the design space."""
    _design_space = ScalableDesignSpace(
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
    assert len(custom_design_space) == len(_design_space)
    for name in custom_design_space:
        assert_equal(custom_design_space[name], _design_space[name])


def test_custom_design_space_with_uncertainties(custom_design_space):
    """Check the custom configuration of the design space with uncertainties."""
    _design_space = ScalableDesignSpace(
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
    assert len(custom_design_space) + 3 == len(_design_space)
    for name in custom_design_space:
        assert_equal(custom_design_space[name], _design_space[name])

    assert _design_space["u_1"] == RandomVariable("OTNormalDistribution", 4)
    assert _design_space["u_2"] == RandomVariable("OTNormalDistribution", 5)
    assert _design_space["u_3"] == RandomVariable("OTNormalDistribution", 6)
