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
"""Computation of tolerance intervals from a data-fitted exponential distribution."""
from __future__ import annotations

import openturns as ot

from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceInterval,
)


class ExponentialToleranceInterval(ToleranceInterval):
    """Computation of tolerance intervals from a data-fitted exponential distribution.

    The formulae come from the R library *tolerance* [1]_.

    .. [1] Derek S. Young,
           *tolerance: An R Package for Estimating Tolerance Intervals*,
           Journal of Statistical Software, 36(5), 2010
    """

    def __init__(
        self,
        size: int,
        rate: float,
        location: float,
    ) -> None:
        """
        Args:
            rate: The estimation of the rate of the exponential distribution.
            location: The estimation of the location of the exponential distribution.
        """  # noqa: D205 D212 D415
        super().__init__(size)
        self.__rate = rate
        self.__location = location

    def _compute_lower_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        k_1 = 1 - (coverage**size / alpha) ** (1.0 / (size - 1))
        return self.__location + k_1 / self.__rate

    def _compute_upper_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        chi2_num = ot.ChiSquare(2).computeQuantile(coverage)[0]
        chi2_den = ot.ChiSquare(2 * size - 2).computeQuantile(coverage)[0]
        k_2 = size * chi2_num / chi2_den
        return self.__location + k_2 / self.__rate
