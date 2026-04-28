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
"""The interface to OpenTURNS-based probability distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import openturns
from numpy import array
from numpy import inf
from openturns import CompositeDistribution
from openturns import DistributionImplementation
from openturns import Interval
from openturns import SymbolicFunction
from openturns import TruncatedDistribution

from gemseo.uncertainty.distributions.base_distribution import BaseDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.scalar_distribution_mixin import (
    ScalarDistributionMixin,
)

if TYPE_CHECKING:
    from openturns import Distribution

    from gemseo.typing import RealArray


class OTDistribution(
    BaseDistribution[float, tuple[Any, ...], DistributionImplementation],
    ScalarDistributionMixin,
):
    """An OpenTURNS-based probability distribution.

    Warning:
        The probability distribution parameters must be provided
        according to the signature of the OpenTURNS classes.
        [Access the OpenTURNS documentation](http://openturns.github.io/openturns/latest/user_manual/probabilistic_modelling.html).
    """

    settings_class = OTDistribution_Settings

    JOINT_DISTRIBUTION_CLASS: ClassVar[type[OTJointDistribution]] = OTJointDistribution

    _WEBSITE: ClassVar[str] = (
        "http://openturns.github.io/openturns/latest/user_manual/"
        "probabilistic_modelling.html"
    )

    def _create_distribution(self, settings: OTDistribution_Settings) -> None:
        distribution = self._create_distribution_from_module(openturns, settings)
        self.__set_bounds(distribution)
        if settings.transformation:
            distribution = self.__transform_distribution(distribution, settings)

        self.__set_bounds(distribution)
        if settings.lower_bound is not None or settings.upper_bound is not None:
            distribution = self.__truncate_distribution(distribution, settings)
        self.__set_bounds(distribution)
        self.distribution = distribution

    def compute_samples(self, n_samples: int = 1) -> RealArray:  # noqa: D102
        # We cast the value to int
        # because getSample does not support numpy.int_.
        return array(self.distribution.getSample(int(n_samples))).ravel()

    def compute_cdf(self, value: float) -> float:  # noqa: D102
        # We cast the value to float
        # because computeCDF does not support numpy.int32.
        return self.distribution.computeCDF(float(value))

    def compute_inverse_cdf(self, value: float) -> float:  # noqa: D102
        return self.distribution.computeQuantile(value)[0]

    def _pdf(self, value: float) -> float:
        return self.distribution.computePDF(value)

    def _cdf(self, level: float) -> float:
        return self.distribution.computeCDF(level)

    @property
    def mean(self) -> float:  # noqa: D102
        return self.distribution.getMean()[0]

    @property
    def standard_deviation(self) -> float:  # noqa: D102
        return self.distribution.getStandardDeviation()[0]

    def __transform_distribution(
        self, distribution: Distribution, settings: OTDistribution_Settings
    ) -> CompositeDistribution:
        """Apply a transformation to a distribution.

        Examples of transformations: -, +, *, **, sin, exp, log, ...

        Args:
            distribution: The original distribution.
            settings: The settings of the distribution.

        Returns:
            The transformed distribution.
        """
        transformation = settings.transformation.replace(" ", "")
        symbolic_function = SymbolicFunction(
            [self.DEFAULT_VARIABLE_NAME], [transformation]
        )
        self._transformation = transformation.replace(
            self.DEFAULT_VARIABLE_NAME, f"({self._transformation})"
        )
        return CompositeDistribution(symbolic_function, distribution)

    def __truncate_distribution(
        self, distribution: Distribution, settings: OTDistribution_Settings
    ) -> TruncatedDistribution:
        """Truncate the distribution of a random variable.

        Args:
            distribution: The original distribution.
            settings: The settings of the distribution.

        Returns:
            The transformed distributions.
        """
        self._transformation = f"Trunc({self._transformation})"
        if settings.lower_bound is None:
            if settings.upper_bound > self.math_upper_bound:
                msg = "upper_bound is greater than the current upper bound."
                raise ValueError(msg)
            args = (
                distribution,
                settings.upper_bound,
                TruncatedDistribution.UPPER,
                settings.threshold,
            )
        elif settings.upper_bound is None:
            if settings.lower_bound < self.math_lower_bound:
                msg = "lower_bound is less than the current lower bound."
                raise ValueError(msg)
            args = (
                distribution,
                settings.lower_bound,
                TruncatedDistribution.LOWER,
                settings.threshold,
            )
        else:
            if settings.lower_bound < self.math_lower_bound:
                msg = "lower_bound is less than the current lower bound."
                raise ValueError(msg)
            if settings.upper_bound > self.math_upper_bound:
                msg = "upper_bound is greater than the current upper bound."
                raise ValueError(msg)
            args = (
                distribution,
                Interval(settings.lower_bound, settings.upper_bound),
                settings.threshold,
            )

        return TruncatedDistribution(*args)

    def __set_bounds(self, distribution: Distribution) -> None:
        """Set the mathematical and numerical bounds (= support and range).

        Args:
            distribution: The original probability distribution.
        """
        range_ = distribution.getRange()
        lower_bound = range_.getLowerBound()[0]
        upper_bound = range_.getUpperBound()[0]
        self.num_lower_bound = lower_bound
        self.num_upper_bound = upper_bound
        if not range_.getFiniteLowerBound()[0]:
            lower_bound = -inf
        if not range_.getFiniteUpperBound()[0]:
            upper_bound = inf
        self.math_lower_bound = lower_bound
        self.math_upper_bound = upper_bound
