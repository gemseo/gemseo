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
r"""Joint probability distribution.

The base class
[BaseJointDistribution][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution]
implements the concept of [joint probability distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution).

The joint probability distribution of a set of random variables
is the probability distribution of the random vector
consisting of these random variables.

It takes into account
both the marginal probability distributions of these random variables
and their dependency structure.

A
[BaseJointDistribution][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution]
is defined
from a list of
[BaseDistribution][gemseo.uncertainty.distributions.base_distribution.BaseDistribution]
objects
defining the marginals of the random variables
and a copula defining the dependency structure between them.

Note:
   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative distribution functions.
   [See more](https://en.wikipedia.org/wiki/Copula_(probability_theory)).

By definition, a joint probability distribution is a probability distribution.
Therefore,
[BaseJointDistribution][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution]
inherits from the abstract class
[BaseDistribution][gemseo.uncertainty.distributions.base_distribution.BaseDistribution].

The
[BaseJointDistribution][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution]
of a list of given uncertain variables is built
from a list of
[BaseDistribution][gemseo.uncertainty.distributions.base_distribution.BaseDistribution]
objects
implementing the probability distributions of these variables
and from a copula.

Because
[BaseJointDistribution][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution]
inherits from
[BaseDistribution][gemseo.uncertainty.distributions.base_distribution.BaseDistribution],
we can easily get statistics, such as
[mean][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution.mean] and
[standard_deviation][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution.standard_deviation].
We can also get the numerical
[range][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution.range]
and the mathematical
[support][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution.support].

Note:
    We call mathematical *support* the set of values that the random variable
    can take in theory, e.g. $]-\infty,+\infty[$ for a Gaussian variable,
    and numerical *range* the set of values that it can take in practice,
    taking into account the values rounded to zero double precision.
    Both support and range are described in terms of lower and upper bounds

We can also evaluate the cumulative density function
([compute_cdf()][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution.compute_cdf])
for the different marginals of the random variable,
as well as the inverse cumulative density function
([compute_inverse_cdf()][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution.compute_inverse_cdf]).

Lastly, we can compute realizations of the random variable
by means of the
[compute_samples()][gemseo.uncertainty.distributions.base_joint.BaseJointDistribution.compute_samples]
method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import array
from numpy import column_stack

from gemseo.typing import RealArray
from gemseo.typing import StrKeyMapping
from gemseo.uncertainty.distributions.base_distribution import BaseDistribution
from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseGenericDistributionSettings,
)
from gemseo.uncertainty.distributions.factory import DISTRIBUTION_FACTORY

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.uncertainty.distributions.base_joint_settings import (
        BaseJointDistributionSettings,
    )


class BaseJointDistribution(BaseDistribution[RealArray, StrKeyMapping, Any]):
    r"""The base class for joint probability distributions.

    The joint probability distribution of a random vector $U=(U_1,\ldots,U_d)$
    is characterized by
    the marginal probability distributions
    of $U_1$, $U_1$, ... and $U_d$
    and a copula
    used to describe the dependence between these $d$ random variables.
    """

    Settings: ClassVar[type[BaseJointDistributionSettings]]

    __marginals: list[BaseDistribution]
    """The marginal probability distributions."""

    def __init__(self, settings: BaseJointDistributionSettings) -> None:
        """
        Args:
            settings: The settings of the probability distribution.
        """  # noqa: D205,D212,D415
        self.__marginals = [
            DISTRIBUTION_FACTORY.create_from_settings(settings)
            for settings in settings.marginal_settings
        ]
        super().__init__(
            BaseGenericDistributionSettings(
                interfaced_distribution="Joint", parameters=(settings,)
            )
        )
        self._get_string_representation = repr(self.__marginals[0])

    @property
    def marginals(self) -> Sequence[BaseDistribution]:
        """The marginal distributions."""
        return self.__marginals

    @property
    def dimension(self) -> int:
        """The dimension of the uncertain space."""
        return len(self.__marginals)

    def _set_bounds(
        self,
        distributions: Iterable[BaseDistribution],
    ) -> None:
        """Set the mathematical and numerical bounds (= support and range).

        Args:
            distributions: The distributions.
        """
        self.math_lower_bound = array([
            distribution.math_lower_bound for distribution in distributions
        ])
        self.math_upper_bound = array([
            distribution.math_upper_bound for distribution in distributions
        ])
        self.num_lower_bound = array([
            distribution.num_lower_bound for distribution in distributions
        ])
        self.num_upper_bound = array([
            distribution.num_upper_bound for distribution in distributions
        ])

    @property
    def range(self) -> RealArray:  # noqa: D102
        return column_stack([self.num_lower_bound, self.num_upper_bound])

    @property
    def support(self) -> RealArray:  # noqa: D102
        return column_stack([self.math_lower_bound, self.math_upper_bound])

    @property
    def mean(self) -> RealArray:  # noqa: D102
        return array([marginal.mean for marginal in self.marginals])

    @property
    def standard_deviation(self) -> RealArray:  # noqa: D102
        return array([marginal.standard_deviation for marginal in self.marginals])

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
    ) -> RealArray:
        return column_stack([
            marginal.compute_samples(n_samples) for marginal in self.marginals
        ])
