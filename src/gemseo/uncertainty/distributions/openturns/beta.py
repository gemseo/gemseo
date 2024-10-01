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

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTBetaDistribution(OTDistribution):
    """The OpenTURNS-based Beta distribution."""

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 2.0,
        minimum: float = 0.0,
        maximum: float = 1.0,
        transformation: str = "",
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            alpha: The first shape parameter of the beta random variable.
            beta: The second shape parameter of the beta random variable.
            minimum: The minimum of the beta random variable.
            maximum: The maximum of the beta random variable.
        """  # noqa: D205,D212,D415
        super().__init__(
            interfaced_distribution="Beta",
            parameters=(
                alpha,
                beta,
                minimum,
                maximum,
            ),
            standard_parameters={
                self._LOWER: minimum,
                self._UPPER: maximum,
                self._ALPHA: alpha,
                self._BETA: beta,
            },
            transformation=transformation,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            threshold=threshold,
        )
