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

from openturns import IndependentCopula

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

    Settings: ClassVar[type[OTJointDistribution_Settings]] = (
        OTJointDistribution_Settings
    )

    def __init__(self, settings: OTJointDistribution_Settings) -> None:  # noqa: D107
        super().__init__(settings)
        if len(settings.marginal_settings) > 1:
            name = (
                "IndependentCopula"
                if settings.copula is None
                else settings.copula.__class__.__name__
            )
            self._get_string_representation = (
                f"{self.__class__.__name__}"
                f"({pretty_repr(self.marginals, sort=False)}; "
                f"{name})"
            )

    def _create_distribution(self, settings: OTJointDistribution_Settings) -> None:
        if settings.parameters[0].copula is None:
            copula = IndependentCopula(len(self.marginals))
        else:
            copula = settings.parameters[0].copula
        self.distribution = JointDistribution(
            [marginal.distribution for marginal in self.marginals],
            copula,
        )
        self._set_bounds(self.marginals)

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
