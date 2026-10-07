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
"""An adapter exposing the results of an OpenTURNS Sobol' indices algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array

if TYPE_CHECKING:
    from openturns import SobolIndicesAlgorithmImplementation

    from gemseo.util.typing import RealArray


class OTSobolIndicesEstimator:
    """An adapter exposing the results of an OpenTURNS Sobol' indices algorithm."""

    algo: SobolIndicesAlgorithmImplementation
    """The OpenTURNS Sobol' indices algorithm."""

    def __init__(self, algo: SobolIndicesAlgorithmImplementation) -> None:
        """
        Args:
            algo: The configured OpenTURNS Sobol' indices algorithm.
        """  # noqa: D205, D212, D415
        self.algo = algo

    @property
    def first_order_indices(self) -> RealArray:
        """The first-order Sobol' indices, shaped as `(dimension,)`."""
        return array(self.algo.getFirstOrderIndices())

    @property
    def second_order_indices(self) -> RealArray:
        """The second-order Sobol' indices, shaped as `(dimension, dimension)`."""
        return array(self.algo.getSecondOrderIndices())

    @property
    def total_order_indices(self) -> RealArray:
        """The total-order Sobol' indices, shaped as `(dimension,)`."""
        return array(self.algo.getTotalOrderIndices())

    @property
    def first_order_interval(self) -> tuple[RealArray, RealArray]:
        """The lower and upper bounds of the first-order indices."""
        interval = self.algo.getFirstOrderIndicesInterval()
        return array(interval.getLowerBound()), array(interval.getUpperBound())

    @property
    def total_order_interval(self) -> tuple[RealArray, RealArray]:
        """The lower and upper bounds of the total-order indices."""
        interval = self.algo.getTotalOrderIndicesInterval()
        return array(interval.getLowerBound()), array(interval.getUpperBound())
