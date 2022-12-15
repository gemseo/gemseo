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
"""Class to create a uniform distribution from the OpenTURNS library.

This class inherits from :class:`.OTDistribution`.
"""
from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTUniformDistribution(OTDistribution):
    """Create a uniform distribution.

    Example:
        >>> from gemseo.uncertainty.distributions.openturns.uniform import (
        ...     OTUniformDistribution
        >>> )
        >>> distribution = OTUniformDistribution('x', -1, 1)
        >>> print(distribution)
        Uniform(lower=-1, upper=1)
    """

    def __init__(
        self,
        variable: str,
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
            variable: The name of the uniform random variable.
            minimum: The minimum of the uniform random variable.
            maximum: The maximum of the uniform random variable.
            dimension: The dimension of the uniform random variable.
            transformation: A transformation
                applied to the random variable,
                e.g. 'sin(x)'. If None, no transformation.
            lower_bound: A lower bound to truncate the distribution.
                If None, no lower truncation.
            upper_bound: An upper bound to truncate the distribution.
                If None, no upper truncation.
            threshold: A threshold in [0,1].
        """  # noqa: D205,D212,D415
        standard_parameters = {self._LOWER: minimum, self._UPPER: maximum}
        super().__init__(
            variable,
            "Uniform",
            (minimum, maximum),
            dimension,
            standard_parameters,
            transformation,
            lower_bound,
            upper_bound,
            threshold,
        )
