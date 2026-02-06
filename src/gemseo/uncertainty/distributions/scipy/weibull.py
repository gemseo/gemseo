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
"""The SciPy-based Weibull distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.weibull_settings import (
    SPWeibullDistribution_Settings,
)


class SPWeibullDistribution(SPDistribution):
    """The SciPy-based Weibull distribution."""

    settings_class = SPWeibullDistribution_Settings

    def __init__(self, settings: SPWeibullDistribution_Settings | None = None) -> None:  # noqa: D107
        if settings is None:
            settings = SPWeibullDistribution_Settings()
        super().__init__(
            SPDistribution_Settings(
                interfaced_distribution=(
                    "weibull_min" if settings.use_weibull_min else "weibull_max"
                ),
                parameters={
                    "loc": settings.location,
                    "scale": settings.scale,
                    "c": settings.shape,
                },
                standard_parameters={
                    self._LOCATION: settings.location,
                    self._SCALE: settings.scale,
                    self._SHAPE: settings.shape,
                },
            )
        )
