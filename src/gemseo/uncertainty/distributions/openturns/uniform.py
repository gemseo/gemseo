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
"""The OpenTURNS-based uniform distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTUniformDistribution(OTDistribution):
    """The OpenTURNS-based uniform distribution.

    Examples:
        >>> from gemseo.uncertainty.distributions.openturns.uniform import (
        ...     OTUniformDistribution
        >>> )
        >>> distribution = OTUniformDistribution("x", -1, 1)
        >>> print(distribution)
        Uniform(lower=-1, upper=1)
    """

    def __init__(
        self,
        variable: str = OTDistribution.DEFAULT_VARIABLE_NAME,
        minimum: float = 0.0,
        maximum: float = 1.0,
        dimension: int = 1,
        transformation: str | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            minimum: The minimum of the uniform random variable.
            maximum: The maximum of the uniform random variable.
        """  # noqa: D205,D212,D415
        super().__init__(
            variable,
            "Uniform",
            (minimum, maximum),
            dimension,
            {self._LOWER: minimum, self._UPPER: maximum},
            transformation,
            lower_bound,
            upper_bound,
            threshold,
        )
