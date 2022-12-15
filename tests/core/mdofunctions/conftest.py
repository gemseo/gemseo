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
"""Tests for the functions."""
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction


@pytest.fixture
def problem_with_identity() -> OptimizationProblem:
    """An optimization problem whose objective is the identity function."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f", special_repr="Identity")
    return problem
