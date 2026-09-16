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

import pytest

from gemseo.problem.uncertainty.ishigami.ishigami_function import IshigamiFunction
from gemseo.problem.uncertainty.ishigami.ishigami_problem import IshigamiProblem
from gemseo.problem.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.problem.uncertainty.util import UniformDistribution
from gemseo.uncertainty.distribution.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distribution.scipy.joint import SPJointDistribution


def test_ishigami_problem() -> None:
    """Check the Ishigami problem."""
    problem = IshigamiProblem()
    input_space = problem.input_space
    assert isinstance(input_space, IshigamiSpace)
    functions = problem.functions
    assert len(functions) == 1
    assert isinstance(functions[0], IshigamiFunction)
    for random_variable in input_space.variables.values():
        assert isinstance(random_variable.distribution, SPJointDistribution)


@pytest.mark.parametrize(
    "uniform_distribution_name",
    [UniformDistribution.OPENTURNS, "OTUniformDistribution"],
)
def test_ishigami_problem_openturns(uniform_distribution_name) -> None:
    """Check the Ishigami problem using OpenTURNS."""
    problem = IshigamiProblem(uniform_distribution_name)
    for random_variable in problem.input_space.variables.values():
        assert isinstance(random_variable.distribution, OTJointDistribution)
