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
"""Class to create an exponential distribution from the SciPy library.

This class inherits from :class:`.SPDistribution`.
"""
from __future__ import annotations

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution


class SPExponentialDistribution(SPDistribution):
    """Create an exponential distribution.

    Example:
        >>> from gemseo.uncertainty.distributions.scipy.exponential import (
        ...     SPExponentialDistribution
        ... )
        >>> distribution = SPExponentialDistribution('x', 2, 3)
        >>> print(distribution)
        expon(loc=3, scale=0.5)
    """

    def __init__(
        self,
        variable: str,
        rate: float = 1.0,
        loc: float = 0.0,
        dimension: int = 1,
    ) -> None:
        """
        Args:
            variable: The name of the exponential random variable.
            rate: The rate of the exponential random variable.
            loc: The location of the exponential random variable.
            dimension: The dimension of the exponential random variable.
        """  # noqa: D205,D212,D415
        parameters = {"loc": loc, "scale": 1 / float(rate)}
        super().__init__(variable, "expon", parameters, dimension)
