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

from gemseo.uncertainty.distributions.base_settings.triangular_settings import _MAXIMUM
from gemseo.uncertainty.distributions.base_settings.triangular_settings import _MINIMUM
from gemseo.uncertainty.distributions.base_settings.triangular_settings import _MODE
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.triangular_settings import (
    SPTriangularDistribution_Settings,
)


class SPTriangularDistribution(SPDistribution):
    """The SciPy-based triangular distribution."""

    Settings = SPTriangularDistribution_Settings

    def __init__(
        self,
        minimum: float = _MINIMUM,
        mode: float = _MODE,
        maximum: float = _MAXIMUM,
        settings: SPTriangularDistribution_Settings | None = None,
    ) -> None:
        """
        Args:
            minimum: The minimum of the triangular random variable.
            mode: The mode of the triangular random variable.
            maximum: The maximum of the triangular random variable.
        """  # noqa: D205,D212,D415
        if settings is None:
            settings = SPTriangularDistribution_Settings(
                minimum=minimum, maximum=maximum, mode=mode
            )
        super().__init__(
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
