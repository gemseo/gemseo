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
"""The base DOE algorithm using the OpenTURNS library."""

from __future__ import annotations

from typing import Final

from openturns import ComposedDistribution
from openturns import Uniform

from gemseo.algos.doe._base_doe import BaseDOE


class BaseOTDOE(BaseDOE):
    """The base DOE algorithm using the OpenTURNS library."""

    _STANDARD_UNIFORM_DISTRIBUTION: Final[Uniform] = Uniform(0, 1)
    r"""The uniform distribution over the interval :math:`[0,1]`"""

    @classmethod
    def _get_uniform_distribution(
        cls,
        dimension: int,
    ) -> ComposedDistribution:
        """Return the uniform distribution over the unit hypercube.

        Args:
            dimension: The dimension of the uniform distribution.

        Returns:
            The uniform distribution over the unit hypercube.
        """
        return ComposedDistribution([cls._STANDARD_UNIFORM_DISTRIBUTION] * dimension)
