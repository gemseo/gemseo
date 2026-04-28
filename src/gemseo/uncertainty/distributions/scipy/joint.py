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

[SPJointDistribution][gemseo.uncertainty.distributions.scipy.joint.SPJointDistribution]
is a
[BaseJointDistribution][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution]
based on the [SciPy](https://docs.scipy.org/doc/scipy/tutorial/stats.html) library.

Warning:
   For the moment,
   there is no copula that can be used with
   [SPJointDistribution][gemseo.uncertainty.distributions.scipy.joint.SPJointDistribution];
   if you want to introduce dependency between random variables,
   please consider
   [OTJointDistribution][gemseo.uncertainty.distributions.openturns.joint.OTJointDistribution].
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.uncertainty.distributions.scipy.joint_settings import (
    SPJointDistribution_Settings,
)
from gemseo.utils.string_tools import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import RealArray

from numpy import array

from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution


class SPJointDistribution(BaseJointDistribution):
    """The SciPy-based joint probability distribution."""

    settings_class: ClassVar[type[SPJointDistribution_Settings]] = (
        SPJointDistribution_Settings
    )

    def __repr__(self) -> str:
        if len(self._settings.marginal_settings) == 1:
            return super().__repr__()

        return (
            f"{self.__class__.__name__}"
            f"({pretty_repr(self.marginals, sort=False)}; "
            f"IndependentCopula)"
        )

    def _create_distribution(self, settings: SPJointDistribution_Settings) -> None:
        self.distribution = self.marginals
        self._set_bounds(self.marginals)

    def compute_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        return array([
            marginal.distribution.cdf(value_)
            for value_, marginal in zip(value, self.marginals, strict=False)
        ])

    def compute_inverse_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        return array([
            marginal.distribution.ppf(value_)
            for value_, marginal in zip(value, self.marginals, strict=False)
        ])
