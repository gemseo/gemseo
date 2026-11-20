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
"""The OpenTURNS-based maximum-entropy distribution."""

from __future__ import annotations

from math import inf
from math import isfinite
from typing import TYPE_CHECKING

import openturns

from gemseo.uncertainty.distribution._maximum_entropy.family import MaximumEntropyFamily
from gemseo.uncertainty.distribution._maximum_entropy.parameter import (
    MaximumEntropyParameter,
)
from gemseo.uncertainty.distribution.core.base_maximum_entropy import (
    BaseMaximumEntropyDistribution,
)
from gemseo.uncertainty.distribution.openturns.distribution import OTDistribution
from gemseo.uncertainty.distribution.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.maximum_entropy_settings import (
    OTMaximumEntropyDistribution_Settings,
)

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping


class OTMaximumEntropyDistribution(BaseMaximumEntropyDistribution, OTDistribution):
    """The OpenTURNS-based maximum-entropy distribution.

    Instead of a distribution name and its parameters,
    this distribution is defined by what is known about the random variable,
    e.g. a minimum, a maximum, a mean and a standard deviation,
    and uses the probability distribution
    that maximizes the Shannon entropy for this information,
    i.e. the least committal one with respect to the missing information.
    """

    settings_class = OTMaximumEntropyDistribution_Settings

    def __init__(  # noqa: D107
        self, settings: OTMaximumEntropyDistribution_Settings | None = None
    ) -> None:
        super().__init__(settings)

    def _compute_truncated_normal_representation_error(
        self,
        parameters: StrKeyMapping,
        settings: OTMaximumEntropyDistribution_Settings,
    ) -> float:
        mu = parameters[MaximumEntropyParameter.MU]
        sigma = parameters[MaximumEntropyParameter.SIGMA]
        minimum = parameters[MaximumEntropyParameter.MINIMUM]
        maximum = parameters[MaximumEntropyParameter.MAXIMUM]
        normal = openturns.Normal(mu, sigma)
        try:
            # Under a strong truncation,
            # OpenTURNS rejects a truncation interval
            # beyond the range it computes for the distribution.
            if minimum is not None and maximum is not None:
                truncated = openturns.TruncatedDistribution(
                    normal, openturns.Interval(minimum, maximum)
                )
            elif minimum is not None:
                truncated = openturns.TruncatedDistribution(
                    normal, minimum, openturns.TruncatedDistribution.LOWER
                )
            else:
                truncated = openturns.TruncatedDistribution(
                    normal, maximum, openturns.TruncatedDistribution.UPPER
                )
            error = max(
                abs(truncated.getMean()[0] - settings.mu),
                abs(truncated.getStandardDeviation()[0] - settings.sigma),
            )
        except (TypeError, RuntimeError):
            return inf

        return error if isfinite(error) else inf

    def _create_maximum_entropy_settings(
        self,
        family: MaximumEntropyFamily,
        parameters: StrKeyMapping,
        settings: OTMaximumEntropyDistribution_Settings,
    ) -> OTDistribution_Settings:
        standard_parameters = self._create_standard_parameters(family, settings)
        if family == MaximumEntropyFamily.UNIFORM:
            return OTDistribution_Settings(
                interfaced_distribution="Uniform",
                parameters=(
                    parameters[MaximumEntropyParameter.MINIMUM],
                    parameters[MaximumEntropyParameter.MAXIMUM],
                ),
                standard_parameters=standard_parameters,
            )

        if family == MaximumEntropyFamily.NORMAL:
            return OTDistribution_Settings(
                interfaced_distribution="Normal",
                parameters=(
                    parameters[MaximumEntropyParameter.MU],
                    parameters[MaximumEntropyParameter.SIGMA],
                ),
                standard_parameters=standard_parameters,
            )

        if family == MaximumEntropyFamily.EXPONENTIAL:
            return OTDistribution_Settings(
                interfaced_distribution="Exponential",
                parameters=(
                    parameters[MaximumEntropyParameter.RATE],
                    parameters[MaximumEntropyParameter.MINIMUM],
                ),
                standard_parameters=standard_parameters,
            )

        if family == MaximumEntropyFamily.REFLECTED_EXPONENTIAL:
            # The exponential distribution starting at the opposite of the maximum
            # is reflected to obtain a distribution ending at the maximum.
            return OTDistribution_Settings(
                interfaced_distribution="Exponential",
                parameters=(
                    parameters[MaximumEntropyParameter.RATE],
                    -parameters[MaximumEntropyParameter.MAXIMUM],
                ),
                standard_parameters=standard_parameters,
                transformation="-x",
            )

        if family == MaximumEntropyFamily.TILTED_UNIFORM:
            tilt = parameters[MaximumEntropyParameter.TILT]
            minimum = parameters[MaximumEntropyParameter.MINIMUM]
            maximum = parameters[MaximumEntropyParameter.MAXIMUM]
            if tilt < 0.0:
                # The probability density function decreases,
                # as that of an exponential distribution truncated to the support.
                return OTDistribution_Settings(
                    interfaced_distribution="Exponential",
                    parameters=(-tilt, minimum),
                    standard_parameters=standard_parameters,
                    upper_bound=maximum,
                )

            # The probability density function increases;
            # a reflected exponential distribution is truncated to the support.
            # The truncation bounds apply to the reflected random variable.
            return OTDistribution_Settings(
                interfaced_distribution="Exponential",
                parameters=(tilt, -maximum),
                standard_parameters=standard_parameters,
                transformation="-x",
                lower_bound=minimum,
            )

        # A normal distribution is truncated to the support.
        # The OpenTURNS TruncatedNormal distribution is not used,
        # as it silently returns a wrong distribution for an infinite bound.
        # TODO: use TruncatedNormal distribution once the OpenTURNS bug has been fixed.
        # See the issue: https://github.com/openturns/openturns/issues/3275.
        return OTDistribution_Settings(
            interfaced_distribution="Normal",
            parameters=(
                parameters[MaximumEntropyParameter.MU],
                parameters[MaximumEntropyParameter.SIGMA],
            ),
            standard_parameters=standard_parameters,
            lower_bound=parameters[MaximumEntropyParameter.MINIMUM],
            upper_bound=parameters[MaximumEntropyParameter.MAXIMUM],
        )
