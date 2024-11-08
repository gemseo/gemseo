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
"""The OpenTURNS-based exponential distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution


class OTExponentialDistribution(OTDistribution):
    """The OpenTURNS-based exponential distribution."""

    def __init__(
        self,
        rate: float = 1.0,
        loc: float = 0.0,
        transformation: str = "",
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            rate: The rate of the exponential random variable.
            loc: The location of the exponential random variable.
        """  # noqa: D205,D212,D415
        super().__init__(
            interfaced_distribution="Exponential",
            parameters=(rate, loc),
            standard_parameters={self._RATE: rate, self._LOC: loc},
            transformation=transformation,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            threshold=threshold,
        )
