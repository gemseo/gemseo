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
"""A problem connecting the Ishigami function with its uncertain space."""

from __future__ import annotations

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.problems.uncertainty.ishigami.ishigami_function import IshigamiFunction
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.problems.uncertainty.utils import UniformDistribution


class IshigamiProblem(OptimizationProblem):
    """A problem connecting the Ishigami function with its uncertain space."""

    def __init__(
        self,
        uniform_distribution_name: UniformDistribution = UniformDistribution.SCIPY,
    ) -> None:
        """
        Args:
            uniform_distribution_name: The name of the class
                implementing the uniform distribution.
        """  # noqa: D205, D212
        super().__init__(IshigamiSpace(uniform_distribution_name))
        self.objective = IshigamiFunction()
