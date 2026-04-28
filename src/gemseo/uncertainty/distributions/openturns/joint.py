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
from typing import ClassVar

from openturns import BlockIndependentCopula
from openturns import DistributionImplementation
from openturns import IndependentCopula
from openturns import MarginalDistribution

from gemseo.uncertainty.distributions.openturns.joint_settings import (
    OTJointDistribution_Settings,
)
from gemseo.utils.compatibility.openturns import JointDistribution
from gemseo.utils.string_tools import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import RealArray

from numpy import array

from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution


class OTJointDistribution(BaseJointDistribution):
    """The OpenTURNS-based joint probability distribution."""

    settings_class: ClassVar[type[OTJointDistribution_Settings]] = (
        OTJointDistribution_Settings
    )

    def __repr__(self) -> str:
        if len(self._settings.marginal_settings) == 1:
            return super().__repr__()

        return (
            f"{self.__class__.__name__}"
            f"({pretty_repr(self.marginals, sort=False)}; "
            f"{self.distribution.getCopula()})"
        )

    def _create_distribution(self, settings: OTJointDistribution_Settings) -> None:
        copula = settings.copula
        if copula == ():
            copula = IndependentCopula(len(self.marginals))
        elif not isinstance(copula, DistributionImplementation):
            copula = self.__create_block_independent_copula(copula)

        self.distribution = JointDistribution(
            [marginal.distribution for marginal in self.marginals],
            copula,
        )
        self._set_bounds(self.marginals)

    def __create_block_independent_copula(
        self, blocks: Iterable[tuple[tuple[int, ...], DistributionImplementation]]
    ) -> MarginalDistribution:
        """Create a block-independent copula.

        Args:
            blocks: A collection of independent blocks
                defined by random variables and copulas.

        Returns:
            The block-independent copula.
        """
        dimension = self.dimension
        remaining_indices = tuple(
            set(range(dimension))
            - {index for indices, _ in blocks for index in indices}
        )
        extended_blocks = list(blocks)
        if remaining_indices:
            extended_blocks.append((
                remaining_indices,
                IndependentCopula(len(remaining_indices)),
            ))

        permutations = []
        for indices, _ in extended_blocks:
            permutations.extend(indices)

        inverse_permutations = [0] * dimension
        for index, original_index in enumerate(permutations):
            inverse_permutations[original_index] = index

        copula = BlockIndependentCopula([copula for _, copula in extended_blocks])
        return MarginalDistribution(copula, inverse_permutations)

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
    ) -> RealArray:
        # We cast the value to int
        # because getSample does not support numpy.int_.
        return array(self.distribution.getSample(int(n_samples)))

    def compute_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        # We cast the values to float
        # because computeCDF does not support numpy.int32.
        return array([
            marginal.distribution.computeCDF(float(value_))
            for value_, marginal in zip(value, self.marginals, strict=False)
        ])

    def compute_inverse_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        return array([
            marginal.distribution.computeQuantile(value_)[0]
            for value_, marginal in zip(value, self.marginals, strict=False)
        ])
