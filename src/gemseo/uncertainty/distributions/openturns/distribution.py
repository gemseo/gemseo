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
from openturns import Distribution
from openturns import DistributionImplementation
from openturns import Interval
from openturns import SymbolicFunction
from openturns import TruncatedDistribution

from gemseo.uncertainty.distributions.base_distribution import BaseDistribution
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.scalar_distribution_mixin import (
    ScalarDistributionMixin,
)
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from gemseo.typing import RealArray
    from gemseo.uncertainty.distributions.base_distribution import (
        StandardParametersType,
    )
    from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution


class OTDistribution(
    BaseDistribution[float, tuple[Any, ...], DistributionImplementation],
    ScalarDistributionMixin,
):
    """An OpenTURNS-based probability distribution.

    .. warning::

        The probability distribution parameters must be provided
        according to the signature of the OpenTURNS classes.
        `Access the OpenTURNS documentation
        <http://openturns.github.io/openturns/latest/user_manual/
        probabilistic_modelling.html>`_.

    Examples:
        >>> from gemseo.uncertainty.distributions.openturns.distribution import (
        ...     OTDistribution,
        ... )
        >>> distribution = OTDistribution("Exponential", (3, 2))
        >>> print(distribution)
        Exponential(3, 2)
    """

    JOINT_DISTRIBUTION_CLASS: ClassVar[type[BaseJointDistribution]] = (
        OTJointDistribution
    )

    _WEBSITE: ClassVar[str] = (
        "http://openturns.github.io/openturns/latest/user_manual/"
        "probabilistic_modelling.html"
    )

    def __init__(
        self,
        interfaced_distribution: str = "Uniform",
        parameters: tuple[Any, ...] = (),
        standard_parameters: StandardParametersType = READ_ONLY_EMPTY_DICT,
        transformation: str = "",
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        r"""
        Args:
            standard_parameters: The parameters of the probability distribution
                used for string representation only
                (use ``parameters`` for computation).
                If empty, use ``parameters`` instead.
                For instance,
                let us consider the interfaced OpenTURNS distribution ``"Dirac"``.
                Then,
                the string representation of
                ``OTDistribution("x", "Dirac", (1,), 1, {"loc": 1})``
                is ``"Dirac(loc=1)"``
                while the string representation of
                ``OTDistribution("x", "Dirac", (1,))``
                is ``"Dirac(1)"``.
            transformation: A transformation applied
                to the random variable,
                e.g. :math:`\sin(x)`. If empty, no transformation.
            lower_bound: A lower bound to truncate the probability distribution.
                If ``None``, no lower truncation.
            upper_bound: An upper bound to truncate the probability distribution.
                If ``None``, no upper truncation.
            threshold: A threshold in [0,1]
                (`see OpenTURNS documentation
                <http://openturns.github.io/openturns/latest/user_manual/
                _generated/openturns.TruncatedDistribution.html>`_).
        """  # noqa: D205,D212,D415
        super().__init__(
            interfaced_distribution,
            parameters or (),
            standard_parameters=standard_parameters,
            transformation=transformation,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            threshold=threshold,
        )

    def _create_distribution(
        self,
        distribution_name: str,
        parameters: tuple[Any, ...],
        transformation: str,
        lower_bound: float | None,
        upper_bound: float | None,
        threshold: float,
    ) -> None:
        r"""
        Args:
            transformation: A transformation applied
                to the random variable,
                e.g. :math:`\sin(x)`. If ``None``, no transformation.
            lower_bound: A lower bound to truncate the probability distribution.
                If ``None``, no lower truncation.
            upper_bound: An upper bound to truncate the probability distribution.
                If ``None``, no upper truncation.
            threshold: A threshold in [0,1].
        """  # noqa: D205, D212
        distribution = self._create_distribution_from_module(
            openturns, distribution_name, parameters
        )

        self.__set_bounds(distribution)
        if transformation:
            distribution = self.__transform_distribution(distribution, transformation)

        self.__set_bounds(distribution)
        if lower_bound is not None or upper_bound is not None:
            distribution = self.__truncate_distribution(
                distribution, lower_bound, upper_bound, threshold
            )

        self.__set_bounds(distribution)
        self.distribution = distribution

    def compute_samples(self, n_samples: int = 1) -> RealArray:  # noqa: D102
        return array(self.distribution.getSample(n_samples)).ravel()

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
            transformation: A transformation applied to the random variable,
                e.g. 'sin(x)'. If ``None``, no transformation.

        Returns:
            The transformed distribution.
        """
        transformation = transformation.replace(" ", "")
        symbolic_function = SymbolicFunction(
            [self.DEFAULT_VARIABLE_NAME], [transformation]
        )
        self.transformation = transformation.replace(
            self.DEFAULT_VARIABLE_NAME, f"({self.transformation})"
        )
        return CompositeDistribution(symbolic_function, distribution)

    def __truncate_distribution(
        self,
        distribution: Distribution,
        lower_bound: float | None,
        upper_bound: float | None,
        threshold: float = 0.5,
    ) -> TruncatedDistribution:
        """Truncate the distribution of a random variable.

        Args:
            distribution: The original distribution.
            lower_bound: A lower bound to truncate the probability distributions.
                If ``None``, no lower truncation.
            upper_bound: An upper bound to truncate the probability distributions.
                If ``None``, no upper truncation.
            threshold: A threshold in [0,1].

        Returns:
            The transformed distributions.
        """
        self.transformation = f"Trunc({self.transformation})"
        if lower_bound is None:
            if upper_bound > self.math_upper_bound:
                msg = "u_b is greater than the current upper bound."
                raise ValueError(msg)
            args = (
                distribution,
                upper_bound,
                TruncatedDistribution.UPPER,
                threshold,
            )
        elif upper_bound is None:
            if lower_bound < self.math_lower_bound:
                msg = "l_b is less than the current lower bound."
                raise ValueError(msg)
            args = (
                distribution,
                lower_bound,
                TruncatedDistribution.LOWER,
                threshold,
            )
        else:
            if lower_bound < self.math_lower_bound:
                msg = "l_b is less than the current lower bound."
                raise ValueError(msg)
            if upper_bound > self.math_upper_bound:
                msg = "u_b is greater than the current upper bound."
                raise ValueError(msg)
            args = (distribution, Interval(lower_bound, upper_bound), threshold)

        return TruncatedDistribution(*args)

    def __set_bounds(self, distribution: Distribution) -> None:
        """Set the mathematical and numerical bounds (= support and range).

        Args:
            distribution: The original probability distribution.
        """
        _range = distribution.getRange()
        lower_bound = _range.getLowerBound()[0]
        upper_bound = _range.getUpperBound()[0]
        self.num_lower_bound = lower_bound
        self.num_upper_bound = upper_bound
        if not _range.getFiniteLowerBound()[0]:
            lower_bound = -inf
        if not _range.getFiniteUpperBound()[0]:
            upper_bound = inf
        self.math_lower_bound = lower_bound
        self.math_upper_bound = upper_bound
