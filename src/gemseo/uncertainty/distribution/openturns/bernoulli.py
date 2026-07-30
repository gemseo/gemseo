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
"""The OpenTURNS-based Bernoulli distribution."""

from __future__ import annotations

from gemseo.uncertainty.distribution.openturns.bernoulli_settings import (
    OTBernoulliDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.distribution import OTDistribution
from gemseo.uncertainty.distribution.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.util.pydantic import create_model


class OTBernoulliDistribution(OTDistribution):
    """The OpenTURNS-based Bernoulli distribution.

    This distribution is defined by the probability of a binary event occurring.
    """

    settings_class = OTBernoulliDistribution_Settings

    def __init__(  # noqa: D107
        self, settings: OTBernoulliDistribution_Settings | None = None
    ) -> None:
        settings = create_model(
            OTBernoulliDistribution_Settings, settings_model=settings
        )
        super().__init__(
            OTDistribution_Settings(
                interfaced_distribution="Bernoulli",
                parameters=(settings.p,),
                standard_parameters={"p": settings.p},
                transformation=settings.transformation,
                lower_bound=settings.lower_bound,
                upper_bound=settings.upper_bound,
                threshold=settings.threshold,
            )
        )
