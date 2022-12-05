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
"""Class to create a joint probability distribution from the SciPy library.

The :class:`.SPComposedDistribution` class is a concrete class
inheriting from :class:`.ComposedDistribution` which is an abstract one.
SP stands for `scipy <https://docs.scipy.org/doc/scipy/reference/
tutorial/stats.html>`_ which is the library it relies on.

This class inherits from :class:`.SPDistribution`.
It builds a composed probability distribution
related to given random variables from a list of :class:`.SPDistribution` objects
implementing the probability distributions of these variables
based on the SciPy library and from a copula name.

.. note::

   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative density functions.
   `See more <https://en.wikipedia.org/wiki/Copula_(probability_theory)>`__.
"""
from __future__ import annotations

from typing import Callable
from typing import Iterable
from typing import Sequence
from typing import TYPE_CHECKING

from gemseo.utils.base_enum import BaseEnum
from gemseo.utils.base_enum import get_names

if TYPE_CHECKING:
    from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution

from numpy import array, ndarray

from gemseo.uncertainty.distributions.composed import ComposedDistribution


class SPComposedDistribution(ComposedDistribution):
    """Scipy composed distribution."""

    class CopulaModel(BaseEnum):
        """A copula model."""

        independent_copula = None

    # TODO: API: remove this attribute in the next major release.
    AVAILABLE_COPULA_MODELS = get_names(CopulaModel)

    def __init__(  # noqa: D107
        self,
        distributions: Sequence[SPDistribution],
        copula: CopulaModel | str = CopulaModel.independent_copula,
        variable: str = "",
    ) -> None:
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
