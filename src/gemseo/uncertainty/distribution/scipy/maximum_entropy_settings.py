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
"""The settings of a SciPy-based maximum-entropy distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from pydantic import model_validator

from gemseo.uncertainty.distribution._maximum_entropy.family import MaximumEntropyFamily
from gemseo.uncertainty.distribution._maximum_entropy.selection import (
    find_maximum_entropy_family,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_tilt,
)
from gemseo.uncertainty.distribution.core.base_univariate_settings.maximum_entropy import (  # noqa: E501
    BaseMaximumEntropyDistributionSettings,
)
from gemseo.uncertainty.distribution.scipy.base_settings import (
    BaseSPDistributionSettings,
)

if TYPE_CHECKING:
    from typing import Self

    from gemseo.uncertainty.distribution.core.base_univariate_settings.maximum_entropy_numerical_settings import (  # noqa: E501
        MaximumEntropyNumericalSettings,
    )

_scipy_unavailable: Final[str] = (
    "SciPy does not implement the {} "
    "that maximizes the entropy for this information; "
    "use 'OTMaximumEntropyDistribution' instead."
)
"""The template of the error when SciPy cannot build the requested distribution."""


def _check_scipy_availability(
    minimum: float | None,
    maximum: float | None,
    mu: float | None,
    sigma: float | None,
    *,
    settings: MaximumEntropyNumericalSettings,
) -> None:
    """Check that SciPy implements the distribution maximizing the entropy.

    SciPy implements neither the reflected exponential distribution
    nor the tilted uniform distributions whose density function increases.

    Args:
        minimum: The minimum of the random variable.
            If `None`, this information is not available.
        maximum: The maximum of the random variable.
            If `None`, this information is not available.
        mu: The mean of the random variable.
            If `None`, this information is not available.
        sigma: The standard deviation of the random variable.
            If `None`, this information is not available.
        settings: The settings of the numerical methods
            computing the distribution parameters.

    Raises:
        ValueError: When SciPy does not implement this distribution.
    """
    family = find_maximum_entropy_family(minimum, maximum, mu, sigma, settings=settings)
    if family == MaximumEntropyFamily.REFLECTED_EXPONENTIAL:
        msg = _scipy_unavailable.format(f"{family} distribution")
        raise ValueError(msg)

    if (
        family == MaximumEntropyFamily.TILTED_UNIFORM
        and compute_tilted_uniform_tilt(minimum, maximum, mu, settings=settings) > 0.0
    ):
        msg = _scipy_unavailable.format(
            f"{family} distribution with an increasing probability density function"
        )
        raise ValueError(msg)


class SPMaximumEntropyDistribution_Settings(  # noqa: N801
    BaseMaximumEntropyDistributionSettings, BaseSPDistributionSettings
):
    """The settings of a SciPy-based maximum-entropy distribution.

    This class implements the distributions
    that can maximize the entropy for the information expressed by these settings,
    except the reflected exponential distribution,
    resulting from a maximum and a mean,
    and the tilted uniform distributions
    resulting from a minimum, a maximum and a mean
    that is greater than the middle of the support.
    Use
    [OTMaximumEntropyDistribution_Settings][gemseo.uncertainty.distribution.openturns.maximum_entropy_settings.OTMaximumEntropyDistribution_Settings]
    for these two cases.
    """

    @model_validator(mode="after")
    def __validate_scipy_availability(self) -> Self:
        _check_scipy_availability(
            self.minimum,
            self.maximum,
            self.mu,
            self.sigma,
            settings=self.numerical_settings,
        )
        return self
