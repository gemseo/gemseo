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
"""Computation of tolerance intervals from a data-fitted Weibull distribution."""
from __future__ import annotations

import openturns as ot
from numpy import exp
from numpy import log

from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceInterval,
)


class WeibullToleranceInterval(ToleranceInterval):
    """Computation of tolerance intervals from a data-fitted Weibull distribution.

    The formulae come from the R library *tolerance* [1]_.

    .. [1] Derek S. Young,
           *tolerance: An R Package for Estimating Tolerance Intervals*,
           Journal of Statistical Software, 36(5), 2010
    """

    def __init__(
        self,
        size: int,
        scale: float,
        shape: float,
        location: float,
    ) -> None:
        """
        Args:
            scale: The estimation of the scale of the Weibull distribution.
            shape: The estimation of the shape of the Weibull distribution.
            location: The estimation of the location of the Weibull distribution.
        """  # noqa: D205 D212 D415
        super().__init__(size)
        self.__scale = scale
        self.__shape = shape
        self.__location = location

    @staticmethod
    def __lambda_function(
        value: float,
    ) -> float:
        """Compute the natural logarithm of the opposite natural logarithm.

        Args:
            value: The value to be transformed.

        Returns:
            The transformed value.
        """
        return log(-log(value))

    def _compute_lower_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        xi_ = log(self.__scale)
        delta = 1.0 / self.__shape
        offset = -(size**0.5) * self.__lambda_function(coverage)
        student = ot.Student(size - 1, offset, 1.0)
        bound = xi_ - delta * student.computeQuantile(1 - alpha)[0] / (size - 1) ** 0.5
        return exp(bound) + self.__location

    def _compute_upper_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        xi_ = log(self.__scale)
        delta = 1.0 / self.__shape
        offset = -(size**0.5) * self.__lambda_function(1 - coverage)
        student = ot.Student(size - 1, offset, 1.0)
        bound = xi_ - delta * student.computeQuantile(alpha)[0] / (size - 1) ** 0.5
        return exp(bound) + self.__location


class WeibullMinToleranceInterval(WeibullToleranceInterval):
    """Computation of tolerance intervals from a data-fitted Weibull distribution.

    The formulae come from the R library *tolerance* [1]_.
    """
