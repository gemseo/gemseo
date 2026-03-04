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
"""The SciPy-based triangular distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.triangular_settings import (
    SPTriangularDistribution_Settings,
)
from gemseo.utils.pydantic import create_model


class SPTriangularDistribution(SPDistribution):
    """The SciPy-based triangular distribution."""

    settings_class = SPTriangularDistribution_Settings

    def __init__(  # noqa: D107
        self, settings: SPTriangularDistribution_Settings | None = None
    ) -> None:
        settings = create_model(
            SPTriangularDistribution_Settings, settings_model=settings
        )
        super().__init__(
            SPDistribution_Settings(
                interfaced_distribution="triang",
                parameters={
                    "loc": settings.minimum,
                    "scale": settings.maximum - settings.minimum,
                    "c": (settings.mode - settings.minimum)
                    / float(settings.maximum - settings.minimum),
                },
                standard_parameters={
                    self._LOWER: settings.minimum,
                    self._MODE: settings.mode,
                    self._UPPER: settings.maximum,
                },
            )
        )
