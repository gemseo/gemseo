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
from __future__ import annotations

import re

import pytest
from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.preprocessed_functions.norm_function import NormFunction
from gemseo.core.mdofunctions.mdo_function import MDOFunction


def test_user_gradient_option_without_jacobian():
    """Check the exception raised when calling a missing Jacobian function."""
    design_space = DesignSpace()
    design_space.add_variable("x")

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f")
    function = NormFunction(problem.objective, design_space)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Selected user gradient but function f has no Jacobian function."
        ),
    ):
        function.jac(array([1]))
