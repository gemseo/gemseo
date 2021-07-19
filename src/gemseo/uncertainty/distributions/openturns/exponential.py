# -*- coding: utf-8 -*-
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

"""Class to create an exponential distribution from the OpenTURNS library.

This class inherits from :class:`.OTDistribution`.
"""

from __future__ import division, unicode_literals

from typing import Optional

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTExponentialDistribution(OTDistribution):
    """Create an exponential distribution.

    Example:
        >>> from gemseo.uncertainty.distributions.openturns.exponential import (
        ...     OTExponentialDistribution
        ... )
        >>> distribution = OTExponentialDistribution('x', 2, 3)
        >>> print(distribution)
        Exponential(loc=3, rate=2)
    """

    def __init__(
        self,
        variable,  # type: str
        rate=1.0,  # type: float
        loc=0.0,  # type: float
        dimension=1,  # type: int
        transformation=None,  # type: Optional[str]
        lower_bound=None,  # type: Optional[float]
        upper_bound=None,  # type: Optional[float]
        threshold=0.5,  # type: float
    ):  # noqa: D205,D212,D415
        # type: (...) -> None
        """
        Args:
            variable: The name of the exponential random variable.
            rate: The rate of the exponential random variable.
            loc: The location of the exponential random variable.
            dimension: The dimension of the exponential random variable.
            transformation: A transformation
                applied to the random variable,
                e.g. 'sin(x)'. If None, no transformation.
            lower_bound: A lower bound to truncate the distribution.
                If None, no lower truncation.
            upper_bound: An upper bound to truncate the distribution.
                If None, no upper truncation.
            threshold: A threshold in [0,1].
        """
        standard_parameters = {self._RATE: rate, self._LOC: loc}
        super(OTExponentialDistribution, self).__init__(
            variable,
            "Exponential",
            (rate, loc),
            dimension,
            standard_parameters,
            transformation,
            lower_bound,
            upper_bound,
            threshold,
        )
