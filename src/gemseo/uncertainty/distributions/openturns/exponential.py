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
"""The OpenTURNS-based exponential distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.exponential_settings import (
    OTExponentialDistribution_Settings,
)


class OTExponentialDistribution(OTDistribution):
    """The OpenTURNS-based exponential distribution."""

    settings_class = OTExponentialDistribution_Settings

    def __init__(  # noqa: D107
        self, settings: OTExponentialDistribution_Settings | None = None
    ) -> None:
        if settings is None:
            settings = OTExponentialDistribution_Settings()
        super().__init__(
            OTDistribution_Settings(
                interfaced_distribution="Exponential",
                parameters=(settings.rate, settings.loc),
                standard_parameters={
                    self._RATE: settings.rate,
                    self._LOC: settings.loc,
                },
                transformation=settings.transformation,
                lower_bound=settings.lower_bound,
                upper_bound=settings.upper_bound,
                threshold=settings.threshold,
            )
        )
