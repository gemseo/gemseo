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
"""Class to create a normal distribution from the SciPy library.

This class inherits from :class:`.SPDistribution`.
"""
from __future__ import annotations

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution


class SPNormalDistribution(SPDistribution):
    """Create a normal distribution.

    Example:
        >>> from gemseo.uncertainty.distributions.scipy.normal import (
        ...     SPNormalDistribution
        ... )
        >>> distribution = SPNormalDistribution('x', -1, 2)
        >>> print(distribution)
        norm(mu=-1, sigma=2)
    """

    def __init__(
        self,
        variable: str,
        mu: float = 0.0,
        sigma: float = 1.0,
        dimension: int = 1,
    ) -> None:
        """
        Args:
            variable: The name of the normal random variable.
            mu: The mean of the normal random variable.
            sigma: The standard deviation of the normal random variable.
            dimension: The dimension of the normal random variable.
        """  # noqa: D205,D212,D415
        standard_parameters = {self._MU: mu, self._SIGMA: sigma}
        parameters = {"loc": mu, "scale": sigma}
        super().__init__(variable, "norm", parameters, dimension, standard_parameters)
