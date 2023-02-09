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
"""Computation of tolerance intervals from a data-fitted log-normal distribution."""
from __future__ import annotations

from numpy import exp

from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    Bounds,
)
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)
from gemseo.uncertainty.statistics.tolerance_interval.normal import (
    NormalToleranceInterval,
)


class LogNormalToleranceInterval(NormalToleranceInterval):
    """Computation of tolerance intervals from a data-fitted log-normal distribution.

    The formulae come from the R library *tolerance* [1]_.

    .. [1] Derek S. Young,
           *tolerance: An R Package for Estimating Tolerance Intervals*,
           Journal of Statistical Software, 36(5), 2010
    """

    def __init__(
        self,
        size: int,
        mean: float,
        std: float,
        location: float,
    ) -> None:
        """
        Args:
            mean: The estimation of the mean of the natural logarithm
                of a log-normal distributed random variable.
            std: The estimation of the standard deviation of the natural logarithm
                of a log-normal distributed random variable.
            location: The estimation of the location of the log-normal distributed.
        """  # noqa: D205 D212 D415
        super().__init__(size, mean, std)
        self.__location = location

    def compute(  # noqa: D102
        self,
        coverage: float,
        confidence: float = 0.95,
        side: ToleranceIntervalSide = ToleranceIntervalSide.BOTH,
    ) -> Bounds:
        lower, upper = super().compute(coverage, confidence, side)
        return Bounds(exp(lower) + self.__location, exp(upper) + self.__location)
