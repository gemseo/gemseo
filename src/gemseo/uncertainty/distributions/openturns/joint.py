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

from openturns import ComposedDistribution
from openturns import Distribution
from openturns import IndependentCopula

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping
    from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution

from numpy import array

from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution


class OTJointDistribution(BaseJointDistribution):
    """The OpenTURNS-based joint probability distribution."""

    def __init__(  # noqa: D107
        self,
        distributions: Sequence[OTDistribution],
        copula: Distribution | None = None,
    ) -> None:
        super().__init__(distributions, copula=copula)

    def _create_distribution(
        self,
        distribution_name: str,
        parameters: StrKeyMapping,
        copula: Distribution | None,
        distributions: Sequence[OTDistribution],
    ) -> None:
        """
        Args:
            copula: The copula modelling the dependency structure.
                If empty, use an independent copula.
            distributions: The marginal distributions.
        """  # noqa: D205 D212
        if copula is None:
            copula = IndependentCopula(len(distributions))
        self.distribution = ComposedDistribution(
            [distribution.distribution for distribution in distributions], copula
        )
        self._set_bounds(distributions)

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
    ) -> RealArray:
        return array(self.distribution.getSample(n_samples))

    def compute_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        # We cast the values to float
        # because computeCDF does not support numpy.int32.
        return array([
            marginal.distribution.computeCDF(float(value_))
            for value_, marginal in zip(value, self.marginals)
        ])

    def compute_inverse_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        return array([
            marginal.distribution.computeQuantile(value_)[0]
            for value_, marginal in zip(value, self.marginals)
        ])
