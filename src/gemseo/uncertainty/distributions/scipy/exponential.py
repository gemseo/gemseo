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
"""The SciPy-based exponential distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.base_settings.exponential_settings import _LOC
from gemseo.uncertainty.distributions.base_settings.exponential_settings import _RATE
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.exponential_settings import (
    SPExponentialDistribution_Settings,
)


class SPExponentialDistribution(SPDistribution):
    """The SciPy-based exponential distribution."""

    Settings = SPExponentialDistribution_Settings

    def __init__(
        self,
        rate: float = _RATE,
        loc: float = _LOC,
        settings: SPExponentialDistribution_Settings | None = None,
    ) -> None:
        """
        Args:
            rate: The rate of the exponential random variable.
            loc: The location of the exponential random variable.
        """  # noqa: D205,D212,D415
        if settings is None:
            settings = SPExponentialDistribution_Settings(rate=rate, loc=loc)
        super().__init__(
            interfaced_distribution="expon",
            parameters={"loc": settings.loc, "scale": 1 / settings.rate},
            standard_parameters={self._RATE: settings.rate, self._LOC: settings.loc},
        )
