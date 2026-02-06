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
"""The OpenTURNS-based triangular distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.triangular_settings import (
    OTTriangularDistribution_Settings,
)


class OTTriangularDistribution(OTDistribution):
    """The OpenTURNS-based triangular distribution."""

    settings_class = OTTriangularDistribution_Settings

    def __init__(  # noqa: D107
        self, settings: OTTriangularDistribution_Settings | None = None
    ) -> None:
        if settings is None:
            settings = OTTriangularDistribution_Settings()
        super().__init__(
            OTDistribution_Settings(
                interfaced_distribution="Triangular",
                parameters=(settings.minimum, settings.mode, settings.maximum),
                standard_parameters={
                    self._LOWER: settings.minimum,
                    self._MODE: settings.mode,
                    self._UPPER: settings.maximum,
                },
                transformation=settings.transformation,
                lower_bound=settings.lower_bound,
                upper_bound=settings.upper_bound,
                threshold=settings.threshold,
            )
        )
