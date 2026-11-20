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
"""The selection of the distribution maximizing the entropy.

Given a state of information about a random variable,
this module recognizes which probability distribution family maximizes the entropy,
computes the parameters of this distribution
and, for every family but the truncated normal one,
its mean and standard deviation.
"""

from __future__ import annotations

from math import isclose
from math import sqrt
from typing import TYPE_CHECKING

from gemseo.uncertainty.distribution._maximum_entropy.family import MaximumEntropyFamily
from gemseo.uncertainty.distribution._maximum_entropy.parameter import (
    MaximumEntropyParameter,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_mean,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_standard_deviation,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_tilt,
)
from gemseo.uncertainty.distribution._maximum_entropy.truncated_normal import (
    compute_truncated_normal_maximum_standard_deviation,
)
from gemseo.uncertainty.distribution._maximum_entropy.truncated_normal import (
    compute_truncated_normal_parameters,
)
from gemseo.util.constant import epsilon

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.uncertainty.distribution.core.base_univariate_settings.maximum_entropy_numerical_settings import (  # noqa: E501
        MaximumEntropyNumericalSettings,
    )


def _find_support_and_mean_family(
    minimum: float | None,
    maximum: float | None,
    mu: float,
    *,
    settings: MaximumEntropyNumericalSettings,
) -> MaximumEntropyFamily:
    """Find the family maximizing the entropy for a support and a mean only.

    This family is used both when the standard deviation is unknown
    and when it is that of this distribution,
    into which the truncated normal one degenerates.

    Args:
        minimum: The minimum of the random variable.
            If `None`, the variable is assumed to be unbounded on the left.
        maximum: The maximum of the random variable.
            If `None`, the variable is assumed to be unbounded on the right.
        mu: The mean of the random variable.
        settings: The settings of the numerical methods
            computing the distribution parameters.

    Returns:
        The family of the distribution maximizing the entropy.
    """
    if maximum is None:
        # The mean and the lower bound are given.
        return MaximumEntropyFamily.EXPONENTIAL

    if minimum is None:
        # The mean and the upper bound are given
        return MaximumEntropyFamily.REFLECTED_EXPONENTIAL

    if compute_tilted_uniform_tilt(minimum, maximum, mu, settings=settings) == 0.0:
        # With a mean at the middle of the support,
        # the tilt is zero and the tilted uniform distribution is the uniform one.
        return MaximumEntropyFamily.UNIFORM

    return MaximumEntropyFamily.TILTED_UNIFORM


def find_maximum_entropy_family(
    minimum: float | None,
    maximum: float | None,
    mu: float | None,
    sigma: float | None,
    *,
    settings: MaximumEntropyNumericalSettings,
) -> MaximumEntropyFamily:
    """Find the distribution family maximizing the entropy for a state of information.

    The state of information is assumed to be one of the eight combinations
    for which a probability distribution maximizes the entropy,
    as checked by
    [BaseMaximumEntropyDistributionSettings][gemseo.uncertainty.distribution.core.base_univariate_settings.maximum_entropy.BaseMaximumEntropyDistributionSettings].

    Args:
        minimum: The minimum of the random variable.
            If `None`, the variable is assumed to be unbounded on the left.
        maximum: The maximum of the random variable.
            If `None`, the variable is assumed to be unbounded on the right.
        mu: The mean of the random variable.
            If `None`, this information is not available.
        sigma: The standard deviation of the random variable.
            If `None`, this information is not available.
        settings: The settings of the numerical methods
            computing the distribution parameters.

    Returns:
        The family of the distribution maximizing the entropy.

    Raises:
        ValueError: When the standard deviation is greater than that
            of the distribution maximizing the entropy
            for the support and the mean only.
    """
    if mu is None:
        # Only the bounds are given.
        return MaximumEntropyFamily.UNIFORM

    if minimum is None and maximum is None:
        # Only the mean and the standard deviation are given.
        return MaximumEntropyFamily.NORMAL

    if sigma is None:
        # The mean is given with one bound or two.
        return _find_support_and_mean_family(minimum, maximum, mu, settings=settings)

    # The mean, the standard deviation and one bound or two are given.
    # The truncated normal distributions cannot be more dispersed
    # than the distribution maximizing the entropy for the support and the mean only.
    maximum_sigma = compute_truncated_normal_maximum_standard_deviation(
        minimum, maximum, mu, settings=settings
    )
    safety_factor = 100.0
    if minimum is None or maximum is None:
        # With a single bound,
        # the maximum standard deviation is the numerical distance
        # from the mean to that bound,
        # so the slack only absorbs the roundoffs of this numerical distance
        # and of the standard deviation computed by the user.
        slack = safety_factor * epsilon
    else:
        # With two bounds,
        # the maximum standard deviation is that of a tilted uniform distribution
        # and so inherits the solver error through the tilt,
        # which the map from the tilt to the standard deviation contracts.
        slack = safety_factor * settings.solver_tolerance

    if isclose(sigma, maximum_sigma, rel_tol=slack):
        # The standard deviation is almost equal to the standard deviation
        # of the distribution maximizing the entropy for the support and the mean,
        # into which the truncated normal distribution degenerates;
        # the standard deviation is thus no longer informative
        # and this distribution is used.
        return _find_support_and_mean_family(minimum, maximum, mu, settings=settings)

    if sigma > maximum_sigma:
        # The standard deviation is greater than that of the distribution
        # maximizing the entropy for the support and the mean only,
        # so no truncated normal distribution can deliver it;
        # why depends on the support and is explained by the messages below.
        if minimum is None or maximum is None:
            # On a semi-infinite support,
            # the density maximizing the entropy for the first two moments
            # is not integrable beyond this standard deviation;
            # this is the existence criterion of Dowson and Wragg (1973),
            # cited in compute_truncated_normal_maximum_standard_deviation.
            msg = (
                "No probability distribution maximizes the entropy "
                f"for this information: the standard deviation {sigma} is too large "
                f"for this support and this mean; it must not be greater "
                f"than {maximum_sigma}."
            )
        else:
            # This ceiling is the Bhatia-Davis inequality,
            # *A better bound on the variance*,
            # The American Mathematical Monthly, 107(4):353-357, 2000,
            # doi:10.1080/00029890.2000.12005203,
            # whose equality case is the distribution
            # concentrated at the bounds of the support.
            # The square root is computed by sqrt rather than by an exponentiation,
            # as the IEEE-754 square root is correctly rounded on every platform,
            # whereas a power is not, so this bound is the same everywhere.
            largest_sigma = sqrt((mu - minimum) * (maximum - mu))
            # On a bounded support, the largest standard deviation
            # of a truncated normal distribution comes from the tilt solver,
            # whose last digits depend on the mathematical library of the platform;
            # it is reported with the digits that the solver tolerance guarantees.
            rounded_maximum_sigma = f"{maximum_sigma:.6g}"
            if sigma >= largest_sigma:
                # On a bounded support,
                # no distribution with this mean has a standard deviation
                # greater than that of the distribution
                # concentrated at the bounds of the support.
                msg = (
                    "No probability distribution maximizes the entropy "
                    f"for this information: the standard deviation {sigma} "
                    "is too large for this support and this mean; "
                    f"it must be less than {largest_sigma}, "
                    "which only the distribution concentrated at the bounds reaches, "
                    f"and not greater than about {rounded_maximum_sigma} "
                    "for this distribution to be a truncated normal one."
                )
            else:
                # On a bounded support,
                # the density maximizing the entropy for the first two moments
                # exists for any achievable standard deviation
                # but is log-convex beyond this one,
                # i.e. U-shaped, or monotone and convex,
                # whereas the truncated normal densities are log-concave.
                msg = (
                    "The probability distribution maximizing the entropy "
                    "for this information has a log-convex probability density "
                    "function, which no truncated normal distribution has: "
                    f"the standard deviation {sigma} must not be greater "
                    f"than about {rounded_maximum_sigma}."
                )

        raise ValueError(msg)

    return MaximumEntropyFamily.TRUNCATED_NORMAL


def compute_maximum_entropy_parameters(
    minimum: float | None,
    maximum: float | None,
    mu: float | None,
    sigma: float | None,
    *,
    settings: MaximumEntropyNumericalSettings,
) -> tuple[MaximumEntropyFamily, dict[str, float | None]]:
    """Compute the parameters of the distribution maximizing the entropy.

    These parameters are those of the distribution itself,
    independently of the library implementing it.

    Args:
        minimum: The minimum of the random variable.
            If `None`, the variable is assumed to be unbounded on the left.
        maximum: The maximum of the random variable.
            If `None`, the variable is assumed to be unbounded on the right.
        mu: The mean of the random variable.
            If `None`, this information is not available.
        sigma: The standard deviation of the random variable.
            If `None`, this information is not available.
        settings: The settings of the numerical methods
            computing the distribution parameters.

    Returns:
        The family of the distribution maximizing the entropy
        and the parameters of this distribution.

    Raises:
        ValueError: When the standard deviation is greater than that
            of the distribution maximizing the entropy
            for the support and the mean only,
            or when the parameters of the truncated normal distribution
            cannot be computed.
    """
    family = find_maximum_entropy_family(minimum, maximum, mu, sigma, settings=settings)
    if family == MaximumEntropyFamily.UNIFORM:
        return family, {
            MaximumEntropyParameter.MINIMUM: minimum,
            MaximumEntropyParameter.MAXIMUM: maximum,
        }

    if family == MaximumEntropyFamily.NORMAL:
        return family, {
            MaximumEntropyParameter.MU: mu,
            MaximumEntropyParameter.SIGMA: sigma,
        }

    if family == MaximumEntropyFamily.EXPONENTIAL:
        return family, {
            MaximumEntropyParameter.RATE: 1.0 / (mu - minimum),
            MaximumEntropyParameter.MINIMUM: minimum,
        }

    if family == MaximumEntropyFamily.REFLECTED_EXPONENTIAL:
        return family, {
            MaximumEntropyParameter.RATE: 1.0 / (maximum - mu),
            MaximumEntropyParameter.MAXIMUM: maximum,
        }

    if family == MaximumEntropyFamily.TILTED_UNIFORM:
        return family, {
            MaximumEntropyParameter.TILT: compute_tilted_uniform_tilt(
                minimum, maximum, mu, settings=settings
            ),
            MaximumEntropyParameter.MINIMUM: minimum,
            MaximumEntropyParameter.MAXIMUM: maximum,
        }

    # The mean and the standard deviation are those of the truncated random variable,
    # whereas the parameters of a truncated normal distribution
    # are those of the normal distribution before truncation.
    normal_mu, normal_sigma = compute_truncated_normal_parameters(
        minimum, maximum, mu, sigma, settings=settings
    )
    return family, {
        MaximumEntropyParameter.MU: normal_mu,
        MaximumEntropyParameter.SIGMA: normal_sigma,
        MaximumEntropyParameter.MINIMUM: minimum,
        MaximumEntropyParameter.MAXIMUM: maximum,
    }


def compute_maximum_entropy_moments(
    family: MaximumEntropyFamily,
    parameters: Mapping[str, float | None],
    *,
    settings: MaximumEntropyNumericalSettings,
) -> tuple[float, float]:
    """Compute the mean and the standard deviation of a maximum-entropy distribution.

    These moments have closed-form expressions for every family
    but the truncated normal one,
    whose moments are prescribed to the solver computing its parameters
    and are thus already known to its caller.

    Args:
        family: The family of the distribution;
            it cannot be the truncated normal one.
        parameters: The parameters of the distribution.
        settings: The settings of the numerical methods
            computing the distribution parameters.

    Returns:
        The mean and the standard deviation of the distribution.

    Raises:
        ValueError: When the family is the truncated normal one.
    """
    if family == MaximumEntropyFamily.UNIFORM:
        minimum = parameters[MaximumEntropyParameter.MINIMUM]
        maximum = parameters[MaximumEntropyParameter.MAXIMUM]
        return (minimum + maximum) / 2.0, (maximum - minimum) / 12.0**0.5

    if family == MaximumEntropyFamily.NORMAL:
        return (
            parameters[MaximumEntropyParameter.MU],
            parameters[MaximumEntropyParameter.SIGMA],
        )

    if family == MaximumEntropyFamily.EXPONENTIAL:
        scale = 1.0 / parameters[MaximumEntropyParameter.RATE]
        return parameters[MaximumEntropyParameter.MINIMUM] + scale, scale

    if family == MaximumEntropyFamily.REFLECTED_EXPONENTIAL:
        scale = 1.0 / parameters[MaximumEntropyParameter.RATE]
        return parameters[MaximumEntropyParameter.MAXIMUM] - scale, scale

    if family == MaximumEntropyFamily.TILTED_UNIFORM:
        tilt = parameters[MaximumEntropyParameter.TILT]
        minimum = parameters[MaximumEntropyParameter.MINIMUM]
        maximum = parameters[MaximumEntropyParameter.MAXIMUM]
        small_tilt = settings.small_tilt
        large_tilt = settings.large_tilt
        return (
            compute_tilted_uniform_mean(
                tilt, minimum, maximum, small_tilt=small_tilt, large_tilt=large_tilt
            ),
            compute_tilted_uniform_standard_deviation(
                tilt, minimum, maximum, small_tilt=small_tilt, large_tilt=large_tilt
            ),
        )

    msg = "The truncated normal distribution has no closed-form moments."
    raise ValueError(msg)
