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

from gemseo.problems.uncertainty.ishigami.ishigami_function import IshigamiFunction
from gemseo.problems.uncertainty.ishigami.ishigami_problem import IshigamiProblem
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.problems.uncertainty.utils import UniformDistribution
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.scipy.joint import SPJointDistribution


def test_ishigami_problem() -> None:
    """Check the Ishigami problem."""
    problem = IshigamiProblem()
    uncertain_space = problem.design_space
    assert isinstance(uncertain_space, IshigamiSpace)
    assert isinstance(problem.objective, IshigamiFunction)
    assert isinstance(uncertain_space.distribution, SPJointDistribution)


@pytest.mark.parametrize(
    "uniform_distribution_name",
    [UniformDistribution.OPENTURNS, "OTUniformDistribution"],
)
def test_ishigami_problem_openturns(uniform_distribution_name) -> None:
    """Check the Ishigami problem using OpenTURNS."""
    problem = IshigamiProblem(uniform_distribution_name)
    assert isinstance(problem.design_space.distribution, OTJointDistribution)
