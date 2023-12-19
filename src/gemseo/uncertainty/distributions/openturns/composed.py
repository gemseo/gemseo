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
"""The OpenTURNS-based joint probability distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

import openturns as ots

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution

from numpy import array
from numpy import ndarray

from gemseo.uncertainty.distributions.composed import ComposedDistribution


class OTComposedDistribution(ComposedDistribution):
    """The OpenTURNS-based joint probability distribution."""

    def __init__(  # noqa: D107
        self,
        distributions: Sequence[OTDistribution],
        copula: ots.Distribution | None = None,
        variable: str = "",
    ) -> None:
        super().__init__(distributions, copula=copula, variable=variable)
        marginals = [
            marginal
            for distribution in distributions
            for marginal in distribution.marginals
        ]
        if copula is None:
            copula = ots.IndependentCopula(len(marginals))
        self.distribution = ots.ComposedDistribution(marginals, copula)
        self._mapping = {}
        index = 0
        for distribution_index, distribution in enumerate(distributions):
            for marginal_index in range(distribution.dimension):
                self._mapping[index] = (distribution_index, marginal_index)
                index += 1

        self._set_bounds(distributions)

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
    ) -> ndarray:
        return array(self.distribution.getSample(n_samples))

    def compute_cdf(  # noqa: D102
        self,
        vector: Iterable[float],
    ) -> ndarray:
        tmp = []
        for index, value in enumerate(vector):
            id1 = self._mapping[index][0]
            id2 = self._mapping[index][1]
            value = ots.Point([value])
            tmp.append(self.marginals[id1].marginals[id2].computeCDF(value))
        return array(tmp)

    def compute_inverse_cdf(  # noqa: D102
        self,
        vector: ndarray,
    ) -> Iterable[float]:
        tmp = []
        for index, value in enumerate(vector):
            id1 = self._mapping[index][0]
            id2 = self._mapping[index][1]
            tmp.append(self.marginals[id1].marginals[id2].computeQuantile(value)[0])
        return array(tmp)

    def _pdf(  # noqa: D102
        self,
        index: int,
    ) -> Callable:
        id1 = self._mapping[index][0]
        id2 = self._mapping[index][1]

        def pdf(
            point: float,
        ) -> float:
            """Probability Density Function (PDF).

            Args:
                point: An evaluation point.

            Returns:
                The PDF value at the evaluation point.
            """
            return self.marginals[id1].marginals[id2].computePDF(point)

        return pdf

    def _cdf(  # noqa: D102
        self,
        index: int,
    ) -> Callable:
        id1 = self._mapping[index][0]
        id2 = self._mapping[index][1]

        def cdf(
            level: float,
        ) -> float:
            """Cumulative Density Function (CDF).

            Args:
                level: A probability level.

            Returns:
                The CDF value for the probability level.
            """
            return self.marginals[id1].marginals[id2].computeCDF(level)

        return cdf
