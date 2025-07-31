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

from gemseo.uncertainty.distributions.base_settings.exponential_settings import _LOC
from gemseo.uncertainty.distributions.base_settings.exponential_settings import _RATE
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
from gemseo.uncertainty.distributions.openturns.exponential_settings import (
    OTExponentialDistribution_Settings,
)


class OTExponentialDistribution(OTDistribution):
    """The OpenTURNS-based exponential distribution."""

    Settings = OTExponentialDistribution_Settings

    def __init__(
        self,
        rate: float = _RATE,
        loc: float = _LOC,
        transformation: str = _TRANSFORMATION,
        lower_bound: float | None = _LOWER_BOUND,
        upper_bound: float | None = _UPPER_BOUND,
        threshold: float = _THRESHOLD,
        settings: OTExponentialDistribution_Settings | None = None,
    ) -> None:
        """
        Args:
            rate: The rate of the exponential random variable.
            loc: The location of the exponential random variable.
        """  # noqa: D205,D212,D415
        if settings is None:
            settings = OTExponentialDistribution_Settings(
                rate=rate,
                loc=loc,
                transformation=transformation,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                threshold=threshold,
            )
        super().__init__(
            interfaced_distribution="Exponential",
            parameters=(settings.rate, settings.loc),
            standard_parameters={self._RATE: settings.rate, self._LOC: settings.loc},
            transformation=settings.transformation,
            lower_bound=settings.lower_bound,
            upper_bound=settings.upper_bound,
            threshold=settings.threshold,
        )
