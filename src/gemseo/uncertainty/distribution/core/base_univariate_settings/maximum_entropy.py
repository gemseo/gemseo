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
"""The base settings of a maximum-entropy distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from pydantic import Field
from pydantic import PositiveFloat
from pydantic import model_validator

from gemseo.uncertainty.distribution._maximum_entropy.selection import (
    find_maximum_entropy_family,
)
from gemseo.uncertainty.distribution.core.base_univariate_settings.maximum_entropy_numerical_settings import (  # noqa: E501
    MaximumEntropyNumericalSettings,
)
from gemseo.util.pydantic import BaseSettings

if TYPE_CHECKING:
    from typing import Self

_no_knowledge: Final[str] = (
    "no information was given; "
    "for example, you can give a minimum and a maximum, "
    "or a mean and a standard deviation"
)
"""The reason why no distribution is built when no information is given."""
_one_bound_only: Final[str] = (
    "the entropy of a distribution with a semi-infinite support is unbounded; "
    "for example, you can give the mean, or the other bound"
)
"""The reason why no distribution is built when only one bound is given."""
_mean_only: Final[str] = (
    "the entropy of a distribution with an infinite support is unbounded; "
    "for example, you can give the standard deviation, or a bound"
)
"""The reason why no distribution is built when only the mean is given."""
_sigma_only: Final[str] = (
    "every normal distribution with this standard deviation maximizes the entropy, "
    "whatever its mean; for example, you can give the mean"
)
"""The reason why the distribution is not unique for a standard deviation only."""
_sigma_without_mu: Final[str] = (
    "no distribution maximizes the entropy "
    "for a semi-infinite support and a standard deviation, "
    "as moving the mean away from the bound always increases the entropy, "
    "up to that of the normal distribution with this standard deviation, "
    "which is never reached; "
    "for example, you can give the mean"
)
"""The reason why no distribution exists for a standard deviation and one bound."""
_sigma_without_mu_bounded: Final[str] = (
    "a distribution maximizing the entropy for a bounded support "
    "and a standard deviation exists, "
    "but the constraint defined by the standard deviation "
    "is not linear when the mean is unknown "
    "and solving this problem is not supported; "
    "for example, you can give the mean"
)
"""The reason why no distribution is built for a standard deviation and two bounds."""

_presence_to_reason: Final[dict[tuple[bool, bool, bool, bool], str]] = {
    # (minimum, maximum, mu, sigma) is given or not.
    # The eight other combinations lead to a maximum-entropy distribution.
    (False, False, False, False): _no_knowledge,
    (True, False, False, False): _one_bound_only,
    (False, True, False, False): _one_bound_only,
    (False, False, True, False): _mean_only,
    (False, False, False, True): _sigma_only,
    (True, False, False, True): _sigma_without_mu,
    (False, True, False, True): _sigma_without_mu,
    (True, True, False, True): _sigma_without_mu_bounded,
}
"""The reason why no distribution is built, for each combination of given information.

The combination is whether (minimum, maximum, mu, sigma) is given.
"""


class BaseMaximumEntropyDistributionSettings(BaseSettings):
    """The base settings of a maximum-entropy distribution.

    These settings express what an expert knows about a random variable,
    instead of the name and the parameters of a probability distribution.
    State the information you have, leave the rest as `None`,
    and the probability distribution that is the least committal
    with respect to the missing information will be used,
    namely the distribution maximizing the
    [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)),
    according to the maximum-entropy principle (Jaynes, 1957),
    whose distributions are catalogued in Kapur (1989).

    The information can be a `minimum`, a `maximum`,
    a mean `mu` and a standard deviation `sigma`,
    from which eight combinations lead to a probability distribution:

    | Information                   | Distribution                               |
    |-------------------------------|--------------------------------------------|
    | `minimum` and `maximum`       | uniform                                    |
    | `mu` and `sigma`              | normal                                     |
    | `minimum` and `mu`            | exponential, shifted to start at `minimum` |
    | `maximum` and `mu`            | exponential, reflected to end at `maximum` |
    | `minimum`, `maximum` and `mu` | uniform, tilted to have the mean `mu`      |
    | `minimum`, `mu` and `sigma`   | normal, truncated from below               |
    | `maximum`, `mu` and `sigma`   | normal, truncated from above               |
    | The four of them              | normal, truncated from below and above     |

    When `sigma` is given with one bound or two,
    it cannot exceed the standard deviation of the distribution
    maximizing the entropy for the support and the mean only,
    namely the exponential, reflected exponential or tilted uniform distribution
    of the rows above.
    The truncated normal distribution degenerates into that distribution
    when `sigma` is equal to this maximum,
    to within a slack derived from the solver tolerance,
    and a greater `sigma` raises a `ValueError`:
    with a single bound,
    no distribution maximizes the entropy beyond this maximum (Dowson and Wragg, 1973);
    with two bounds,
    one does as long as `sigma` remains achievable on this support with this mean,
    i.e. below that of the distribution concentrated at the bounds
    (Bhatia and Davis, 2000),
    but its probability density function is log-convex,
    i.e. U-shaped, or monotone and convex,
    hence not the one of a truncated normal distribution,
    whose density is log-concave.

    !!! quote "References"

        Rajendra Bhatia and Chandler Davis.
        A better bound on the variance.
        The American Mathematical Monthly, 107(4):353-357, 2000.
        [doi:10.1080/00029890.2000.12005203](https://doi.org/10.1080/00029890.2000.12005203).

        David C. Dowson and Alan Wragg.
        Maximum-entropy distributions having prescribed first and second moments.
        IEEE Transactions on Information Theory, 19(5):689-693, 1973.
        [doi:10.1109/TIT.1973.1055060](https://doi.org/10.1109/TIT.1973.1055060).

        Edwin T. Jaynes.
        Information theory and statistical mechanics.
        Physical Review, 106(4):620-630, 1957.

        Jagat N. Kapur.
        Maximum entropy models in science and engineering.
        John Wiley & Sons, 1989.
    """

    minimum: float | None = Field(
        default=None,
        description="""The smallest value that the random variable can take.

If `None`, the random variable is assumed to be unbounded on the left.""",
    )

    maximum: float | None = Field(
        default=None,
        description="""The greatest value that the random variable can take.

If `None`, the random variable is assumed to be unbounded on the right.""",
    )

    mu: float | None = Field(
        default=None,
        description="""The mean of the random variable.

If `None`, the mean is unknown.""",
    )

    sigma: PositiveFloat | None = Field(
        default=None,
        description="""The standard deviation of the random variable.

If `None`, the standard deviation is unknown.""",
    )

    numerical_settings: MaximumEntropyNumericalSettings = Field(
        default_factory=MaximumEntropyNumericalSettings,
        description=MaximumEntropyNumericalSettings.__doc__,
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        minimum = self.minimum
        maximum = self.maximum
        mu = self.mu
        sigma = self.sigma
        reason = _presence_to_reason.get((
            minimum is not None,
            maximum is not None,
            mu is not None,
            sigma is not None,
        ))
        if reason is not None:
            msg = (
                "No maximum-entropy distribution can be created "
                f"from this information: {reason}."
            )
            raise ValueError(msg)

        if minimum is not None and maximum is not None and maximum <= minimum:
            msg = (
                f"The maximum {maximum} of the random variable "
                f"must be greater than its minimum {minimum}."
            )
            raise ValueError(msg)

        if mu is not None:
            if minimum is not None and mu <= minimum:
                msg = (
                    f"The mean {mu} of the random variable "
                    f"must be greater than its minimum {minimum}."
                )
                raise ValueError(msg)

            if maximum is not None and mu >= maximum:
                msg = (
                    f"The mean {mu} of the random variable "
                    f"must be less than its maximum {maximum}."
                )
                raise ValueError(msg)

        find_maximum_entropy_family(
            minimum, maximum, mu, sigma, settings=self.numerical_settings
        )
        return self
