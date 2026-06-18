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

from typing import TYPE_CHECKING

import pytest
from numpy import array

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.reliability.problem import ReliabilityProblem

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.typing import RealArray


@pytest.fixture(scope="module")
def f() -> Callable[[RealArray], RealArray]:
    def _f(u: RealArray) -> RealArray:
        """The identity function.

        Args:
            u: The input value.

        Returns:
            The output value.
        """
        return u

    return _f


def j(u: RealArray) -> RealArray:
    """The Jacobian function of the identity function.

    Args:
        u: The input value.

    Returns:
        The Jacobian value.
    """
    return array([[1.0]])


@pytest.fixture(params=[False, True])
def function(request, f) -> ArrayFunction:
    """The ArrayFunction wrapping the identity function."""
    return ArrayFunction(f, name="y", jac=j if request.param else None)


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """The uncertainty space defined by the random variable u ~ U([0,1])."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("u", OTUniformDistribution_Settings())
    return parameter_space


@pytest.fixture(scope="module")
def unbounded_uncertain_space() -> ParameterSpace:
    """The uncertainty space defined by the random variable u ~ N(0,1)."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("u", OTNormalDistribution_Settings())
    return parameter_space


@pytest.fixture
def problem(uncertain_space, function) -> ReliabilityProblem:
    """A simple reliability analysis problem."""
    problem = ReliabilityProblem(uncertain_space)
    f = problem.get_event_variables(function)
    problem.add_event(f > 0.75, event_name="a")
    return problem
