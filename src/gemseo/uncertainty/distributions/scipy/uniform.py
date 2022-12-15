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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Class to create a uniform distribution from the SciPy library.

This class inherits from :class:`.SPDistribution`.
"""
from __future__ import annotations

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution


class SPUniformDistribution(SPDistribution):
    """Create a uniform distribution.

    Example:
        >>> from gemseo.uncertainty.distributions.scipy.uniform import (
        ...     SPUniformDistribution
        ... )
        >>> distribution = SPUniformDistribution('x', -1, 1)
        >>> print(distribution)
        uniform(lower=-1, upper=1)
    """

    def __init__(
        self,
        variable: str,
        minimum: float = 0.0,
        maximum: float = 1.0,
        dimension: int = 1,
    ) -> None:
        """
        Args:
            variable: The name of the uniform random variable.
            minimum: The minimum of the uniform random variable.
            maximum: The maximum of the uniform random variable.
            dimension: The dimension of the uniform random variable.
        """  # noqa: D205,D212,D415
        parameters = {"loc": minimum, "scale": maximum - minimum}
        standard_parameters = {self._LOWER: minimum, self._UPPER: maximum}
        super().__init__(
            variable, "uniform", parameters, dimension, standard_parameters
        )
