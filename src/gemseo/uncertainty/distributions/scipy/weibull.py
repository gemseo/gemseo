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

from gemseo.uncertainty.distributions.base_settings.weibull_settings import _LOCATION
from gemseo.uncertainty.distributions.base_settings.weibull_settings import _SCALE
from gemseo.uncertainty.distributions.base_settings.weibull_settings import _SHAPE
from gemseo.uncertainty.distributions.base_settings.weibull_settings import (
    _USE_WEIBULL_MIN,
)
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.weibull_settings import (
    SPWeibullDistribution_Settings,
)


class SPWeibullDistribution(SPDistribution):
    """The SciPy-based Weibull distribution."""

    Settings = SPWeibullDistribution_Settings

    def __init__(
        self,
        location: float = _LOCATION,
        scale: float = _SCALE,
        shape: float = _SHAPE,
        use_weibull_min: bool = _USE_WEIBULL_MIN,
        settings: SPWeibullDistribution_Settings | None = None,
    ) -> None:
        r"""
        Args:
            location: The location parameter of the Weibull distribution.
            scale: The scale parameter of the Weibull distribution.
            shape: The shape parameter of the Weibull distribution.
            use_weibull_min: Whether to use
                the Weibull minimum extreme value distribution
                (the support of the random variable is :math:`[\gamma,+\infty[`)
                or the Weibull maximum extreme value distribution
                (the support of the random variable is :math:`]-\infty[,\gamma]`).
        """  # noqa: D205,D212,D415
        if settings is None:
            settings = SPWeibullDistribution_Settings(
                location=location,
                scale=scale,
                shape=shape,
                use_weibull_min=use_weibull_min,
            )
        super().__init__(
            interfaced_distribution="weibull_min"
            if settings.use_weibull_min
            else "weibull_max",
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
