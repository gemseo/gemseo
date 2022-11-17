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
"""Computation of tolerance intervals from a data-fitted uniform distribution."""
from __future__ import annotations

from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceInterval,
)


class UniformToleranceInterval(ToleranceInterval):
    """Computation of tolerance intervals from a data-fitted uniform distribution.

    The formulae come from the R library *tolerance* [1]_.

    .. [1] Derek S. Young,
           *tolerance: An R Package for Estimating Tolerance Intervals*,
           Journal of Statistical Software, 36(5), 2010
    """

    def __init__(
        self,
        size: int,
        minimum: float,
        maximum: float,
    ) -> None:
        """
        Args:
            minimum: The estimation of the lower bound of the uniform distribution.
            maximum: The estimation of the upper bound of the uniform distribution.
        """  # noqa: D205 D212 D415
        super().__init__(size)
        self.__minimum = minimum
        self.__maximum = maximum

    def _compute_lower_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        return self.__compute_exponential_bound(1 - coverage, 1 - alpha, size)

    def _compute_upper_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        return self.__compute_exponential_bound(coverage, alpha, size)

    def __compute_exponential_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        """Compute a bound of the tolerance interval for a uniform distribution.

        Args:
            coverage: A minimum percentage of belonging to the TI.
            alpha: ``1-alpha`` is the level of confidence in [0,1].
            size: The number of samples.

        Returns:
            The bound of the tolerance interval.
        """
        coefficient = coverage / alpha ** (1.0 / size)
        return (self.__maximum - self.__minimum) * coefficient + self.__minimum
