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
#        :author: Clément Laboulfie
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Computation of tolerance intervals empirically."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import argmax
from numpy import argmin
from numpy import linspace
from numpy import sort
from scipy.stats import binom

from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    _BaseToleranceInterval,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class EmpiricalToleranceInterval(_BaseToleranceInterval):
    """Computation of empirical tolerance intervals."""

    __sorted_samples: RealArray
    """The samples sorted in ascending order."""

    def __init__(self, data: RealArray) -> None:
        """
        Args:
            data: The samples from which the tolerance interval is estimated,
            shaped as ``(n_samples,)``.
        """  # noqa: D205 D212 D415
        self._size = len(data)
        self.__sorted_samples = sort(data)

    def _compute_lower_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        # Meeker et al., p. 88, 5.3.2
        indices = linspace(0, size, num=size + 1)
        return self.__sorted_samples[
            max(
                argmin((binom.cdf(size - indices, size, coverage) - (1 - alpha)) > 0)
                - 1,
                0,
            )
        ]

    def _compute_upper_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        # Meeker et al., p. 88, 5.3.2
        indices = linspace(1, size + 1, num=size + 1)
        return self.__sorted_samples[
            min(
                argmax((binom.cdf(indices - 1, size, coverage) - (1 - alpha)) > 0), size
            )
        ]

    def _compute_bounds(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> tuple[float, float]:
        # Meeker et al., p. 87, 5.3.1
        nu = size - binom.ppf(1 - alpha, size, coverage)
        if nu % 2:
            # nu is odd
            nu_1 = nu / 2 - 1 / 2
            nu_2 = nu_1 + 1
        else:
            # nu is even
            nu_1 = nu_2 = nu / 2

        lower = max(int(nu_1), 0)
        upper = min(int(size - nu_2 + 1), size)

        return self.__sorted_samples[lower], self.__sorted_samples[upper]
