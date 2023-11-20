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

from numpy import array
from numpy.testing import assert_equal

from gemseo.problems.scalable.parametric.core.scalable_design_space import (
    ScalableDesignSpace,
)
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.scalable.parametric.core.variable import Variable


def test_default():
    """Check the default configuration of the design space."""
    variables = ScalableDesignSpace().variables
    assert variables == [
        Variable(name, 1, array([0.0]), array([1.0]), array([0.5]))
        for name in ["x_0", "x_1", "x_2", "y_1", "y_2"]
    ]


def test_custom_sizes():
    """Check the design spaces with custom variable sizes."""
    assert_equal(
        ScalableDesignSpace(
            [ScalableDisciplineSettings(2, 4)],
            d_0=3,
            names_to_default_values={"x_0": array([0.3, 0.3, 0.3])},
        ).variables,
        [
            Variable(
                "x_0", 3, array([0.0] * 3), array([1.0] * 3), array([0.3, 0.3, 0.3])
            ),
            Variable("x_1", 2, array([0.0] * 2), array([1.0] * 2), array([0.5] * 2)),
            Variable("y_1", 4, array([0.0] * 4), array([1.0] * 4), array([0.5] * 4)),
        ],
    )
