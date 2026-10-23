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
"""The OpenTURNS-based finite discrete distribution."""

from __future__ import annotations

from copy import copy

from gemseo.uncertainty.distribution.openturns.distribution import OTDistribution
from gemseo.uncertainty.distribution.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.finite_discrete_settings import (
    OTFiniteDiscreteDistribution_Settings,
)
from gemseo.util.pydantic import create_model


class OTFiniteDiscreteDistribution(OTDistribution):
    """The OpenTURNS-based finite discrete distribution.

    This distribution is defined by the weights associated with the possible values.
    """

    settings_class = OTFiniteDiscreteDistribution_Settings

    def __init__(  # noqa: D107
        self, settings: OTFiniteDiscreteDistribution_Settings | None = None
    ) -> None:
        settings = create_model(
            OTFiniteDiscreteDistribution_Settings, settings_model=settings
        )
        value_to_weight = settings.value_to_weight
        values = tuple(value_to_weight.keys())
        weights = tuple(value_to_weight.values())
        value_to_weight_str = (
            str(value_to_weight)
            if len(values[0]) > 1
            else str({k[0]: v for k, v in value_to_weight.items()})
        )
        super().__init__(
            OTDistribution_Settings(
                interfaced_distribution="FiniteDiscreteDistribution",
                parameters=(values, weights),
                standard_parameters={"value_to_weight": value_to_weight_str},
                transformation=settings.transformation,
                lower_bound=settings.lower_bound,
                upper_bound=settings.upper_bound,
                threshold=settings.threshold,
            )
        )

    def __deepcopy__(self, memo):
        # TODO: remove after the openturns 1.28 release and add a compatibility layer.
        return copy(self)
