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


class OTWeibullDistribution(OTDistribution):
    """The OpenTURNS-based Weibull distribution."""

    def __init__(
        self,
        location: float = 0.0,
        scale: float = 1.0,
        shape: float = 1.0,
        use_weibull_min: bool = True,
        transformation: str = "",
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        threshold: float = 0.5,
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
        super().__init__(
            interfaced_distribution="WeibullMin" if use_weibull_min else "WeibullMax",
            parameters=(scale, shape, location),
            standard_parameters={
                self._LOCATION: location,
                self._SCALE: scale,
                self._SHAPE: shape,
            },
            transformation=transformation,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            threshold=threshold,
        )
