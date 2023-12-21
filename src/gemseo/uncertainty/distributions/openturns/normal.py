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
"""The OpenTURNS-based normal distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTNormalDistribution(OTDistribution):
    """The OpenTURNS-based normal distribution.

    Examples:
        >>> from gemseo.uncertainty.distributions.openturns.normal import (
        ...     OTNormalDistribution
        >>> )
        >>> distribution = OTNormalDistribution("x", -1, 2)
        >>> print(distribution)
        Normal(mu=-1, sigma=2)
    """

    def __init__(
        self,
        variable: str = OTDistribution.DEFAULT_VARIABLE_NAME,
        mu: float = 0.0,
        sigma: float = 1.0,
        dimension: int = 1,
        transformation: str | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            mu: The mean of the normal random variable.
            sigma: The standard deviation
                of the normal random variable.
        """  # noqa: D205,D212,D415
        super().__init__(
            variable,
            "Normal",
            (mu, sigma),
            dimension,
            {self._MU: mu, self._SIGMA: sigma},
            transformation,
            lower_bound,
            upper_bound,
            threshold,
        )
