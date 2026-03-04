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
"""The SciPy-based normal distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.normal_settings import (
    SPNormalDistribution_Settings,
)
from gemseo.utils.pydantic import create_model


class SPNormalDistribution(SPDistribution):
    """The SciPy-based normal distribution."""

    settings_class = SPNormalDistribution_Settings

    def __init__(self, settings: SPNormalDistribution_Settings | None = None) -> None:  # noqa: D107
        settings = create_model(SPNormalDistribution_Settings, settings_model=settings)
        super().__init__(
            SPDistribution_Settings(
                interfaced_distribution="norm",
                parameters={"loc": settings.mu, "scale": settings.sigma},
                standard_parameters={
                    self._MU: settings.mu,
                    self._SIGMA: settings.sigma,
                },
            )
        )
