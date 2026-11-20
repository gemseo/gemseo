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
"""The SciPy-based maximum-entropy distribution."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING
from warnings import catch_warnings
from warnings import simplefilter

from numpy import inf
from scipy.stats import truncnorm

from gemseo.uncertainty.distribution._maximum_entropy.family import MaximumEntropyFamily
from gemseo.uncertainty.distribution._maximum_entropy.parameter import (
    MaximumEntropyParameter,
)
from gemseo.uncertainty.distribution.core.base_maximum_entropy import (
    BaseMaximumEntropyDistribution,
)
from gemseo.uncertainty.distribution.scipy.distribution import SPDistribution
from gemseo.uncertainty.distribution.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distribution.scipy.maximum_entropy_settings import (
    SPMaximumEntropyDistribution_Settings,
)

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping


class SPMaximumEntropyDistribution(BaseMaximumEntropyDistribution, SPDistribution):
    """The SciPy-based maximum-entropy distribution.

    Instead of a distribution name and its parameters,
    this distribution is defined by what is known about the random variable,
    e.g. a minimum, a maximum, a mean and a standard deviation,
    and uses the probability distribution
    that maximizes the Shannon entropy for this information,
    i.e. the least committal one with respect to the missing information.
    """

    settings_class = SPMaximumEntropyDistribution_Settings

    def __init__(  # noqa: D107
        self, settings: SPMaximumEntropyDistribution_Settings | None = None
    ) -> None:
        super().__init__(settings)

    @staticmethod
    def _is_family_available(
        family: MaximumEntropyFamily,
        parameters: StrKeyMapping,
    ) -> bool:
        # SciPy implements neither the reflected exponential distribution
        # nor the tilted uniform distributions whose density function increases.
        if family == MaximumEntropyFamily.REFLECTED_EXPONENTIAL:
            return False

        return not (
            family == MaximumEntropyFamily.TILTED_UNIFORM
            and parameters[MaximumEntropyParameter.TILT] > 0.0
        )

    def _compute_truncated_normal_representation_error(
        self,
        parameters: StrKeyMapping,
        settings: SPMaximumEntropyDistribution_Settings,
    ) -> float:
        mu = parameters[MaximumEntropyParameter.MU]
        sigma = parameters[MaximumEntropyParameter.SIGMA]
        minimum = parameters[MaximumEntropyParameter.MINIMUM]
        maximum = parameters[MaximumEntropyParameter.MAXIMUM]
        distribution = truncnorm(
            -inf if minimum is None else (minimum - mu) / sigma,
            inf if maximum is None else (maximum - mu) / sigma,
            loc=mu,
            scale=sigma,
        )
        # Under a strong truncation,
        # SciPy computes the moments with overflows and invalid operations
        # whose warnings are silenced, as nan moments map to an infinite error.
        with catch_warnings():
            simplefilter("ignore", RuntimeWarning)
            error = max(
                abs(distribution.mean() - settings.mu),
                abs(distribution.std() - settings.sigma),
            )
        return error if isfinite(error) else inf

    def _create_maximum_entropy_settings(
        self,
        family: MaximumEntropyFamily,
        parameters: StrKeyMapping,
        settings: SPMaximumEntropyDistribution_Settings,
    ) -> SPDistribution_Settings:
        standard_parameters = self._create_standard_parameters(family, settings)
        if family == MaximumEntropyFamily.UNIFORM:
            minimum = parameters[MaximumEntropyParameter.MINIMUM]
            return SPDistribution_Settings(
                interfaced_distribution="uniform",
                parameters={
                    "loc": minimum,
                    "scale": parameters[MaximumEntropyParameter.MAXIMUM] - minimum,
                },
                standard_parameters=standard_parameters,
            )

        if family == MaximumEntropyFamily.NORMAL:
            return SPDistribution_Settings(
                interfaced_distribution="norm",
                parameters={
                    "loc": parameters[MaximumEntropyParameter.MU],
                    "scale": parameters[MaximumEntropyParameter.SIGMA],
                },
                standard_parameters=standard_parameters,
            )

        if family == MaximumEntropyFamily.EXPONENTIAL:
            return SPDistribution_Settings(
                interfaced_distribution="expon",
                parameters={
                    "loc": parameters[MaximumEntropyParameter.MINIMUM],
                    "scale": 1.0 / parameters[MaximumEntropyParameter.RATE],
                },
                standard_parameters=standard_parameters,
            )

        if family == MaximumEntropyFamily.TILTED_UNIFORM:
            # The probability density function decreases,
            # as that of an exponential distribution truncated to the support.
            tilt = parameters[MaximumEntropyParameter.TILT]
            scale = -1.0 / tilt
            minimum = parameters[MaximumEntropyParameter.MINIMUM]
            maximum = parameters[MaximumEntropyParameter.MAXIMUM]
            return SPDistribution_Settings(
                interfaced_distribution="truncexpon",
                parameters={
                    "b": (maximum - minimum) / scale,
                    "loc": minimum,
                    "scale": scale,
                },
                standard_parameters=standard_parameters,
            )

        # A normal distribution is truncated to the support,
        # with bounds standardized as SciPy expects.
        mu = parameters[MaximumEntropyParameter.MU]
        sigma = parameters[MaximumEntropyParameter.SIGMA]
        minimum = parameters[MaximumEntropyParameter.MINIMUM]
        maximum = parameters[MaximumEntropyParameter.MAXIMUM]
        return SPDistribution_Settings(
            interfaced_distribution="truncnorm",
            parameters={
                "a": -inf if minimum is None else (minimum - mu) / sigma,
                "b": inf if maximum is None else (maximum - mu) / sigma,
                "loc": mu,
                "scale": sigma,
            },
            standard_parameters=standard_parameters,
        )
