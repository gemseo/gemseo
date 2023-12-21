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
"""The SciPy-based joint probability distribution.

:class:`.SPComposedDistribution` is a :class:`.ComposedDistribution`
based on the `SciPy <https://docs.scipy.org/doc/scipy/tutorial/stats.html>`_ library.

.. warning::

   For the moment,
   there is no copula that can be used with :class:`.SPComposedDistribution`;
   if you want to introduce dependency between random variables,
   please consider :class:`.OTComposedDistribution`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution

from numpy import array
from numpy import ndarray

from gemseo.uncertainty.distributions.composed import ComposedDistribution


class SPComposedDistribution(ComposedDistribution):
    """The SciPy-based joint probability distribution."""

    def __init__(  # noqa: D107
        self,
        distributions: Sequence[SPDistribution],
        copula: None = None,
        variable: str = "",
    ) -> None:
        """
        Raises:
            NotImplementedError: When the copula is not ``None``.
        """  # noqa: D205 D212 D415
        if copula is not None:
            raise NotImplementedError(
                "There is not copula distribution yet for SciPy-based distributions."
            )

        super().__init__(distributions, copula=copula, variable=variable)
        self.distribution = distributions
        self._mapping = {}
        index = 0
        for marginal_index, marginal in enumerate(self.distribution):
            for submarginal_index in range(marginal.dimension):
                self._mapping[index] = (marginal_index, submarginal_index)
                index += 1
        self._set_bounds(distributions)

    def compute_cdf(  # noqa: D102
        self,
        vector: Iterable[float],
    ) -> ndarray:
        tmp = []
        for index, value in enumerate(vector):
            id1 = self._mapping[index][0]
            id2 = self._mapping[index][1]
            tmp.append(self.distribution[id1].marginals[id2].cdf(value))
        return array(tmp)

    def compute_inverse_cdf(  # noqa: D102
        self,
        vector: Iterable[float],
    ) -> ndarray:
        tmp = []
        for index, value in enumerate(vector):
            id1 = self._mapping[index][0]
            id2 = self._mapping[index][1]
            tmp.append(self.distribution[id1].marginals[id2].ppf(value))
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
            return self.distribution[id1].marginals[id2].pdf(point)

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
            return self.distribution[id1].marginals[id2].cdf(level)

        return cdf
