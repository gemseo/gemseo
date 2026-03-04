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
"""The OpenTURNS-based Weibull distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.weibull_settings import (
    OTWeibullDistribution_Settings,
)
from gemseo.utils.pydantic import create_model


class OTWeibullDistribution(OTDistribution):
    """The OpenTURNS-based Weibull distribution."""

    settings_class = OTWeibullDistribution_Settings

    def __init__(self, settings: OTWeibullDistribution_Settings | None = None) -> None:  # noqa: D107
        settings = create_model(OTWeibullDistribution_Settings, settings_model=settings)
        super().__init__(
            OTDistribution_Settings(
                interfaced_distribution="WeibullMin"
                if settings.use_weibull_min
                else "WeibullMax",
                parameters=(settings.scale, settings.shape, settings.location),
                standard_parameters={
                    self._LOCATION: settings.location,
                    self._SCALE: settings.scale,
                    self._SHAPE: settings.shape,
                },
                transformation=settings.transformation,
                lower_bound=settings.lower_bound,
                upper_bound=settings.upper_bound,
                threshold=settings.threshold,
            )
        )
