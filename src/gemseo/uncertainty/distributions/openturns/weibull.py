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

from gemseo.uncertainty.distributions.base_settings.weibull_settings import _LOCATION
from gemseo.uncertainty.distributions.base_settings.weibull_settings import _SCALE
from gemseo.uncertainty.distributions.base_settings.weibull_settings import _SHAPE
from gemseo.uncertainty.distributions.base_settings.weibull_settings import (
    _USE_WEIBULL_MIN,
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
from gemseo.uncertainty.distributions.openturns.weibull_settings import (
    OTWeibullDistribution_Settings,
)


class OTWeibullDistribution(OTDistribution):
    """The OpenTURNS-based Weibull distribution."""

    Settings = OTWeibullDistribution_Settings

    def __init__(
        self,
        location: float = _LOCATION,
        scale: float = _SCALE,
        shape: float = _SHAPE,
        use_weibull_min: bool = _USE_WEIBULL_MIN,
        transformation: str = _TRANSFORMATION,
        lower_bound: float | None = _LOWER_BOUND,
        upper_bound: float | None = _UPPER_BOUND,
        threshold: float = _THRESHOLD,
        settings: OTWeibullDistribution_Settings | None = None,
    ) -> None:
        r"""
        Args:
            location: The location parameter :math:`\gamma` of the Weibull distribution.
            scale: The scale parameter of the Weibull distribution.
            shape: The shape parameter of the Weibull distribution.
            use_weibull_min: Whether to use
                the Weibull minimum extreme value distribution
                (the support of the random variable is :math:`[\gamma,+\infty[`)
                or the Weibull maximum extreme value distribution
                (the support of the random variable is :math:`]-\infty[,\gamma]`).
        """  # noqa: D205,D212,D415
        if settings is None:
            settings = OTWeibullDistribution_Settings(
                location=location,
                scale=scale,
                shape=shape,
                use_weibull_min=use_weibull_min,
                transformation=transformation,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                threshold=threshold,
            )
        super().__init__(
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
