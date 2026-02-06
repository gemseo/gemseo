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
"""The OpenTURNS-based normal distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)


class OTNormalDistribution(OTDistribution):
    """The OpenTURNS-based normal distribution."""

    settings_class = OTNormalDistribution_Settings

    def __init__(self, settings: OTNormalDistribution_Settings | None = None) -> None:  # noqa: D107
        if settings is None:
            settings = OTNormalDistribution_Settings()
        super().__init__(
            OTDistribution_Settings(
                interfaced_distribution="Normal",
                parameters=(settings.mu, settings.sigma),
                standard_parameters={
                    self._MU: settings.mu,
                    self._SIGMA: settings.sigma,
                },
                transformation=settings.transformation,
                lower_bound=settings.lower_bound,
                upper_bound=settings.upper_bound,
                threshold=settings.threshold,
            )
        )
