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
"""The OpenTURNS-based Beta distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.beta_settings import (
    OTBetaDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.utils.pydantic import create_model


class OTBetaDistribution(OTDistribution):
    """The OpenTURNS-based Beta distribution."""

    settings_class = OTBetaDistribution_Settings

    def __init__(self, settings: OTBetaDistribution_Settings | None = None) -> None:  # noqa: D107
        settings = create_model(OTBetaDistribution_Settings, settings_model=settings)
        super().__init__(
            OTDistribution_Settings(
                interfaced_distribution="Beta",
                parameters=(
                    settings.alpha,
                    settings.beta,
                    settings.minimum,
                    settings.maximum,
                ),
                standard_parameters={
                    self._LOWER: settings.minimum,
                    self._UPPER: settings.maximum,
                    self._ALPHA: settings.alpha,
                    self._BETA: settings.beta,
                },
                transformation=settings.transformation,
                lower_bound=settings.lower_bound,
                upper_bound=settings.upper_bound,
                threshold=settings.threshold,
            )
        )
