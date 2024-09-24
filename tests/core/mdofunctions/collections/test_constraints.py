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
"""Tests for Constraints."""

from __future__ import annotations

import pytest

from gemseo.algos.constraint_tolerances import ConstraintTolerances
from gemseo.algos.design_space import DesignSpace
from gemseo.core.mdo_functions.collections.constraints import Constraints
from gemseo.core.mdo_functions.mdo_function import MDOFunction


@pytest.fixture
def constraints(problem) -> Constraints:
    """A set of constraints."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1.0, upper_bound=1.0)
    constraints = Constraints(design_space, ConstraintTolerances())
    constraints.append(
        MDOFunction(lambda x: x, "c1", f_type=MDOFunction.FunctionType.EQ)
    )
    constraints.append(
        MDOFunction(lambda x: x, "c2", f_type=MDOFunction.FunctionType.EQ)
    )
    return constraints


@pytest.mark.parametrize(
    ("values", "expected_number"),
    [
        ({"c1": 0.0}, 0),
        ({"c1": 0.0, "c2": 0.0}, 0),
        ({"c1": 1.0}, 1),
        ({"c1": 1.0, "c2": 0.0}, 1),
        ({"c1": 1.0, "c2": 1.0}, 2),
    ],
)
def test_get_number_of_unsatisfied_constraints(constraints, values, expected_number):
    """Check get_number_of_unsatisfied_constraints."""
    assert constraints.get_number_of_unsatisfied_constraints(values) == expected_number
