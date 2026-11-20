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

from re import escape
from re import sub
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

import openturns
from numpy import array
from numpy import inf
from openturns import CompositeDistribution
from openturns import DistributionImplementation
from openturns import Interval
from openturns import SymbolicFunction
from openturns import TruncatedDistribution

from gemseo.uncertainty.distribution.core.base_univariate import (
    BaseUnivariateDistribution,
)
from gemseo.uncertainty.distribution.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.joint import OTJointDistribution

if TYPE_CHECKING:
    from openturns import Distribution

    from gemseo.util.typing import RealArray

_relative_slack: Final[float] = 1e-9
"""The relative distance beyond the support within which a bound is not rejected.

The support of a transformed distribution is known
up to the rounding error of a numerical range or of a symbolic computation,
which this slack tolerates when a truncation bound is compared with it.
It is relative to the bound of the support that the comparison is made with.
"""


def _compute_slack(bound: float) -> float:
    """Compute the slack of a comparison with a bound of the support.

    Args:
        bound: The bound of the support.

    Returns:
        The distance beyond the bound of the support
        within which a truncation bound is not rejected.
    """
    return _relative_slack * max(1.0, abs(bound))


class OTDistribution(
    BaseUnivariateDistribution[float, tuple[Any, ...], DistributionImplementation],
):
    """An OpenTURNS-based probability distribution.

    Warning:
        The probability distribution parameters must be provided
        according to the signature of the OpenTURNS classes.
        [Access the OpenTURNS documentation](http://openturns.github.io/openturns/latest/user_manual/probabilistic_modeling.html).
    """

    settings_class = OTDistribution_Settings

    joint_distribution_class: ClassVar[type[OTJointDistribution]] = OTJointDistribution

    _website: ClassVar[str] = (
        "http://openturns.github.io/openturns/latest/user_manual/probabilistic_modeling.html"
    )

    def _create_distribution(self, settings: OTDistribution_Settings) -> None:
        # Create the original distribution, before transformation and truncation.
        distribution = self._create_distribution_from_module(openturns, settings)
        # Set the range and support of the original distribution.
        self.__set_range_and_support(distribution)

        # OpenTURNS estimates the range of a CompositeDistribution numerically
        # and reports the ends of the support as finite,
        # regardless of the transformation and the support before transformation.
        # This is a bug documented
        # in the issue: https://github.com/openturns/openturns/issues/3274.
        # TODO: Remove support management once this OpenTURNS bug has been fixed.
        # To get round this bug,
        # the transformation is analyzed symbolically using sympy:
        # an end of the support after transformation is infinite
        # where sympy proves that the transformation diverges
        # or cannot analyze it near a singular point,
        # and is otherwise the end of the range that OpenTURNS estimates.
        # On a side that is unbounded before transformation,
        # this estimate is flagged unknown,
        # so that comparing a truncation bound with it does not reject valid bounds.
        # The support is initialized as the set of all real numbers.
        math_lower_bound = -inf
        math_upper_bound = inf
        # Both ends of the support are known without transformation.
        lower_bound_is_known = upper_bound_is_known = True
        if settings.transformation:
            # The spaces are removed to normalize the transformation,
            # e.g. so that "2*x" and "2 * x" share the support computed below;
            # this normalized form is the one reported by the transformation property.
            transformation = settings.transformation.replace(" ", "")
            math_lower_bound_before_transformation = self.math_lower_bound
            math_upper_bound_before_transformation = self.math_upper_bound
            distribution = self.__transform_distribution(distribution, transformation)
            self.__set_range_and_support(distribution)
            if transformation == f"-{self.default_variable_name}":
                # The reflection is special-cased:
                # it is used by OTMaximumEntropyDistribution,
                # which must work without the optional dependency sympy.
                # Its image is the reflected support, with both ends known.
                support = (
                    -math_upper_bound_before_transformation,
                    -math_lower_bound_before_transformation,
                    True,
                    True,
                )
            else:
                # This sympy-dependent import is local,
                # so that sympy which is optional is only imported when needed.
                from gemseo.uncertainty.distribution.openturns._transformed_support import (  # noqa: E501
                    compute_transformed_support,
                )

                support = compute_transformed_support(
                    transformation,
                    math_lower_bound_before_transformation,
                    math_upper_bound_before_transformation,
                )

            lower_bound, upper_bound, lower_bound_is_known, upper_bound_is_known = (
                support
            )
            # An end that the computation does not provide
            # is the end of the range that OpenTURNS estimates,
            # which the call to __set_range_and_support above has already stored.
            if lower_bound is not None:
                self.math_lower_bound = lower_bound
                # The range is estimated numerically,
                # so it can exceed the support by a rounding error.
                self.num_lower_bound = max(self.num_lower_bound, lower_bound)

            if upper_bound is not None:
                self.math_upper_bound = upper_bound
                self.num_upper_bound = min(self.num_upper_bound, upper_bound)

            math_lower_bound = self.math_lower_bound
            math_upper_bound = self.math_upper_bound

        if settings.lower_bound is not None or settings.upper_bound is not None:
            distribution = self.__truncate_distribution(
                distribution,
                settings,
                lower_bound_is_known,
                upper_bound_is_known,
            )
            self.__set_range_and_support(distribution)
            if settings.transformation:
                # The range of the truncated composite distribution is numerical too;
                # the mathematical bounds are the bounds established above,
                # narrowed by the truncation bounds.
                # A truncation bound is not used as is,
                # as it can exceed the support by the slack of its check.
                if settings.lower_bound is not None:
                    math_lower_bound = max(math_lower_bound, settings.lower_bound)

                if settings.upper_bound is not None:
                    math_upper_bound = min(math_upper_bound, settings.upper_bound)

                self.math_lower_bound = math_lower_bound
                self.math_upper_bound = math_upper_bound

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
        self, distribution: Distribution, transformation: str
    ) -> CompositeDistribution:
        """Apply a transformation to a distribution.

        Examples of transformations: -, +, *, **, sin, exp, log, ...

        Args:
            distribution: The original distribution.
            transformation: The transformation, without any space.

        Returns:
            The transformed distribution.
        """
        # Only the occurrences of the variable are substituted,
        # and not those inside a name including it, e.g. the "x" of "exp".
        self._transformation = sub(
            rf"\b{escape(self.default_variable_name)}\b",
            lambda _: f"({self._transformation})",
            transformation,
        )
        return CompositeDistribution(
            SymbolicFunction([self.default_variable_name], [transformation]),
            distribution,
        )

    def __truncate_distribution(
        self,
        distribution: Distribution,
        settings: OTDistribution_Settings,
        check_lower_bound: bool = True,
        check_upper_bound: bool = True,
    ) -> TruncatedDistribution:
        """Truncate the distribution of a random variable.

        Args:
            distribution: The original distribution.
            settings: The settings of the distribution.
            check_lower_bound: Whether to check that the lower truncation bound
                is not below the support of the distribution.
                This check must be skipped
                when the lower end of the support is not known.
            check_upper_bound: Whether to check that the upper truncation bound
                is not above the support of the distribution.
                This check must be skipped
                when the upper end of the support is not known.

        Returns:
            The truncated distribution.
        """
        self._transformation = f"Trunc({self._transformation})"
        if (
            check_lower_bound
            and settings.lower_bound is not None
            and settings.lower_bound
            < self.math_lower_bound - _compute_slack(self.math_lower_bound)
        ):
            msg = (
                f"The lower bound {settings.lower_bound} of the truncation "
                f"is less than the lower bound {self.math_lower_bound:.10g} "
                "of the support."
            )
            raise ValueError(msg)

        if (
            check_upper_bound
            and settings.upper_bound is not None
            and settings.upper_bound
            > self.math_upper_bound + _compute_slack(self.math_upper_bound)
        ):
            msg = (
                f"The upper bound {settings.upper_bound} of the truncation "
                f"is greater than the upper bound {self.math_upper_bound:.10g} "
                "of the support."
            )
            raise ValueError(msg)

        if settings.lower_bound is None:
            args = (
                distribution,
                settings.upper_bound,
                TruncatedDistribution.UPPER,
                settings.threshold,
            )
        elif settings.upper_bound is None:
            args = (
                distribution,
                settings.lower_bound,
                TruncatedDistribution.LOWER,
                settings.threshold,
            )
        else:
            args = (
                distribution,
                Interval(settings.lower_bound, settings.upper_bound),
                settings.threshold,
            )

        return TruncatedDistribution(*args)

    def __set_range_and_support(self, distribution: Distribution) -> None:
        """Set the range (numerical domain) and support (mathematical domain).

        The lower (resp. upper) bound of the support is
        the lower (resp. upper) bound of the range when the bound is finite
        and negative (resp. positive) infinity otherwise.

        Args:
            distribution: The original probability distribution.
        """
        range_ = distribution.getRange()
        self.num_lower_bound = range_.getLowerBound()[0]
        self.num_upper_bound = range_.getUpperBound()[0]
        if range_.getFiniteLowerBound()[0]:
            self.math_lower_bound = self.num_lower_bound
        else:
            self.math_lower_bound = -inf

        if range_.getFiniteUpperBound()[0]:
            self.math_upper_bound = self.num_upper_bound
        else:
            self.math_upper_bound = inf
