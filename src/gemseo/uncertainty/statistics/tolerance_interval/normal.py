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
"""Computation of tolerance intervals from a data-fitted normal distribution."""
from __future__ import annotations

import openturns as ot
from numpy import array
from numpy import inf

from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    Bounds,
)
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceInterval,
)
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)


class NormalToleranceInterval(ToleranceInterval):
    """Computation of tolerance intervals from a data-fitted normal distribution.

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
    ) -> None:
        """
        Args:
            mean: The estimation of the mean of the normal distribution.
            std: The estimation of the standard deviation of the normal distribution.
        """  # noqa: D205 D212 D415
        super().__init__(size)
        self.__mean = mean
        self.__std = std

    def _compute(
        self,
        coverage: float,
        alpha: float,
        size: int,
        side: ToleranceIntervalSide,
    ) -> Bounds:
        if side in [
            ToleranceIntervalSide.UPPER,
            ToleranceIntervalSide.LOWER,
        ]:
            offset = ot.Normal().computeQuantile(coverage)[0] * size**0.5
            student = ot.Student(size - 1, offset, 1.0)
            student_quantile = student.computeQuantile(1 - alpha)[0]
            tolerance_factor = student_quantile / size**0.5

            if side == ToleranceIntervalSide.UPPER:
                return Bounds(
                    array([-inf]), array([self.__mean + tolerance_factor * self.__std])
                )
            else:
                return Bounds(
                    array([self.__mean - tolerance_factor * self.__std]), array([inf])
                )

        elif side == ToleranceIntervalSide.BOTH:
            z_p = ot.Normal().computeQuantile((1 + coverage) / 2.0)[0]
            u_term = (1 + 1.0 / size) ** 0.5 * z_p
            chi_square = ot.ChiSquare(size - 1)
            v_term = ((size - 1) / chi_square.computeQuantile(alpha)[0]) ** 0.5
            w_term = (
                1
                + (size - 3 - chi_square.computeQuantile(alpha)[0])
                / (2 * (size + 1) ** 2)
            ) ** 0.05
            tolerance_factor = u_term * v_term * w_term
            return Bounds(
                array([self.__mean - tolerance_factor * self.__std]),
                array([self.__mean + tolerance_factor * self.__std]),
            )

        else:
            raise ValueError("The type of tolerance interval is incorrect.")
