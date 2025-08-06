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

from gemseo.uncertainty.distributions.base_settings.beta_settings import _ALPHA
from gemseo.uncertainty.distributions.base_settings.beta_settings import _BETA
from gemseo.uncertainty.distributions.base_settings.beta_settings import _MAXIMUM
from gemseo.uncertainty.distributions.base_settings.beta_settings import _MINIMUM
from gemseo.uncertainty.distributions.openturns.beta_settings import (
    OTBetaDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    _LOWER_BOUND,
)
from gemseo.uncertainty.distributions.openturns.distribution_settings import _THRESHOLD
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    _TRANSFORMATION,
)
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    _UPPER_BOUND,
)


class OTBetaDistribution(OTDistribution):
    """The OpenTURNS-based Beta distribution."""

    Settings = OTBetaDistribution_Settings

    def __init__(
        self,
        alpha: float = _ALPHA,
        beta: float = _BETA,
        minimum: float = _MINIMUM,
        maximum: float = _MAXIMUM,
        transformation: str = _TRANSFORMATION,
        lower_bound: float | None = _LOWER_BOUND,
        upper_bound: float | None = _UPPER_BOUND,
        threshold: float = _THRESHOLD,
        settings: OTBetaDistribution_Settings | None = None,
    ) -> None:
        """
        Args:
            alpha: The first shape parameter of the beta random variable.
            beta: The second shape parameter of the beta random variable.
            minimum: The minimum of the beta random variable.
            maximum: The maximum of the beta random variable.
        """  # noqa: D205,D212,D415
        if settings is None:
            settings = OTBetaDistribution_Settings(
                alpha=alpha,
                beta=beta,
                minimum=minimum,
                maximum=maximum,
                transformation=transformation,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                threshold=threshold,
            )
        super().__init__(
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
