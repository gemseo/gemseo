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
"""Class to create a triangular distribution from the OpenTURNS library.

This class inherits from :class:`.OTDistribution`.
"""
from __future__ import division
from __future__ import unicode_literals

from typing import Optional

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTTriangularDistribution(OTDistribution):
    """Create a triangular distribution.

    Example:
        >>> from gemseo.uncertainty.distributions.openturns.triangular import (
        ...     OTTriangularDistribution
        ... )
        >>> distribution = OTTriangularDistribution('x', -1, 0, 1)
        >>> print(distribution)
        Triangular(lower=-1, mode=0, upper=1)
    """

    def __init__(
        self,
        variable,  # type: str
        minimum=0.0,  # type: float
        mode=0.5,  # type: float
        maximum=1.0,  # type: float
        dimension=1,  # type: int
        transformation=None,  # type: Optional[str]
        lower_bound=None,  # type: Optional[float]
        upper_bound=None,  # type: Optional[float]
        threshold=0.5,  # type: float
    ):  # noqa: D205,D212,D415
        # type: (...) -> None
        """
        Args:
            variable: The name of the triangular random variable.
            minimum: The minimum of the triangular random variable.
            mode: The mode of the triangular random variable.
            maximum: The maximum of the random variable.
            dimension: The dimension of the triangular random variable.
            transformation: A transformation
                applied to the random variable,
                e.g. 'sin(x)'. If None, no transformation.
            lower_bound: A lower bound to truncate the distribution.
                If None, no lower truncation.
            upper_bound: An upper bound to truncate the distribution.
                If None, no upper truncation.
            threshold: A threshold in [0,1].
        """
        standard_parameters = {
            self._LOWER: minimum,
            self._MODE: mode,
            self._UPPER: maximum,
        }
        super(OTTriangularDistribution, self).__init__(
            variable,
            "Triangular",
            (minimum, mode, maximum),
            dimension,
            standard_parameters,
            transformation,
            lower_bound,
            upper_bound,
            threshold,
        )
