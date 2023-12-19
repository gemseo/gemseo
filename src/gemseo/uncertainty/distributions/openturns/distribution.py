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
"""The interface to OpenTURNS-based probability distributions.

The :class:`.OTDistribution` class is a concrete class
inheriting from :class:`.Distribution` which is an abstract one.
OT stands for `OpenTURNS <https://openturns.github.io/www/>`_
which is the library it relies on.

The :class:`.OTDistribution` of a given uncertain variable is built
from mandatory arguments:

- a variable name,
- a probability distribution name recognized by OpenTURNS,
- a set of parameters provided as a tuple
  of positional arguments filled in the order
  specified by the OpenTURNS constructor of this probability distribution.

.. warning::

    The probability distribution parameters must be provided according to the signature
    of the openTURNS classes. `Access the openTURNS documentation
    <http://openturns.github.io/openturns/latest/user_manual/
    probabilistic_modelling.html>`_.

The constructor has also optional arguments:

- a variable dimension (default: 1),
- a standard representation of these parameters
  (default: use the parameters provided in the tuple),
- a transformation of the variable (default: no transformation),
- lower and upper bounds for truncation (default: no truncation),
- a threshold for the OpenTURNS truncation tool
  (`more details <http://openturns.github.io/openturns/latest/user_manual/
  _generated/openturns.TruncatedDistribution.html>`_).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

import openturns as ot
from numpy import array
from numpy import inf
from numpy import ndarray

from gemseo.uncertainty.distributions.distribution import Distribution
from gemseo.uncertainty.distributions.distribution import ParametersType
from gemseo.uncertainty.distributions.distribution import StandardParametersType
from gemseo.uncertainty.distributions.openturns.composed import OTComposedDistribution
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.uncertainty.distributions.composed import ComposedDistribution

OT_WEBSITE = (
    "http://openturns.github.io/openturns/latest/user_manual/"
    "probabilistic_modelling.html"
)

LOGGER = logging.getLogger(__name__)


class OTDistribution(Distribution):
    """An OpenTURNS-based probability distribution.

    Create a probability distribution for an uncertain variable
    from its dimension and probability distribution name and properties.

    Examples:
        >>> from gemseo.uncertainty.distributions.openturns.distribution import (
        ...     OTDistribution,
        ... )
        >>> distribution = OTDistribution("x", "Exponential", (3, 2))
        >>> print(distribution)
        Exponential(3, 2)
    """

    COMPOSED_DISTRIBUTION_CLASS: ClassVar[type[ComposedDistribution] | None] = (
        OTComposedDistribution
    )

    marginals: list[ot.Distribution]
    distribution: ot.ComposedDistribution

    def __init__(
        self,
        variable: str = Distribution.DEFAULT_VARIABLE_NAME,
        interfaced_distribution: str = "Uniform",
        parameters: tuple[Any] = (),
        dimension: int = 1,
        standard_parameters: StandardParametersType | None = None,
        transformation: str | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        r"""
        Args:
            standard_parameters: The parameters of the probability distribution
                used for string representation only
                (use ``parameters`` for computation).
                If ``None``, use ``parameters`` instead.
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
                e.g. :math:`\sin(x)`. If ``None``, no transformation.
            lower_bound: A lower bound to truncate the probability distribution.
                If ``None``, no lower truncation.
            upper_bound: An upper bound to truncate the probability distribution.
                If ``None``, no upper truncation.
            threshold: A threshold in [0,1].
        """  # noqa: D205,D212,D415
        super().__init__(
            variable,
            interfaced_distribution,
            parameters,
            dimension,
            standard_parameters,
        )
        self.marginals = self.__create_distributions(
            self.distribution_name,
            self.parameters,
            transformation,
            lower_bound,
            upper_bound,
            threshold,
        )
        self.distribution = ot.ComposedDistribution(self.marginals)
        msg = MultiLineString()
        msg.indent()
        msg.add("Mathematical support: {}", self.support)
        msg.add("Numerical range: {}", self.range)
        msg.add("Transformation: {}", self.transformation)
        LOGGER.debug("%s", msg)

    def _get_empty_parameter_set(self) -> tuple:
        """Return an empty parameter set."""
        return ()

    def compute_samples(self, n_samples: int = 1) -> ndarray:  # noqa: D102
        return array(self.distribution.getSample(n_samples))

    def compute_cdf(self, vector: Iterable[float]) -> ndarray:  # noqa: D102
        return array([
            self.marginals[index].computeCDF(ot.Point([value]))
            for index, value in enumerate(vector)
        ])

    def compute_inverse_cdf(self, vector: Iterable[float]) -> ndarray:  # noqa: D102
        return array([
            self.marginals[index].computeQuantile(value)[0]
            for index, value in enumerate(vector)
        ])

    def _pdf(self, index: int) -> Callable:  # noqa: D102
        def pdf(point: float) -> float:
            """Probability Density Function (PDF).

            Args:
                point: An evaluation point.

            Returns:
                The PDF value at the evaluation point.
            """
            return self.marginals[index].computePDF(point)

        return pdf

    def _cdf(self, index: int) -> Callable:  # noqa: D102
        def cdf(level: float) -> float:
            """Cumulative Density Function (CDF).

            Args:
                level: A probability level.

            Returns:
                The CDF value for the probability level.
            """
            return self.marginals[index].computeCDF(level)

        return cdf

    @property
    def mean(self) -> ndarray:  # noqa: D102
        return array(self.distribution.getMean())

    @property
    def standard_deviation(self) -> ndarray:  # noqa: D102
        return array(self.distribution.getStandardDeviation())

    def __create_distributions(
        self,
        distribution: str,
        parameters: ParametersType,
        transformation: str,
        lower_bound: float,
        upper_bound: float,
        threshold: float,
    ) -> list[ot.Distribution]:
        """Instantiate an OpenTURNS distribution for each random variable component.

        Args:
            distribution: The name of the probability distribution.
            parameters: The parameters of the probability distribution.
            transformation: A transformation applied to the random variable,
                e.g. 'sin(x)'. If ``None``, no transformation.
            lower_bound: A lower truncation bound.
                If ``None``, no lower truncation.
            upper_bound: An upper truncation bound.
                If ``None``, no upper truncation.
            threshold: A threshold value in [0,1].

        Returns:
            The marginal OpenTURNS distributions.
        """
        try:
            create_distribution = getattr(ot, distribution)
        except AttributeError:
            raise ValueError(
                f"{distribution} is an unknown OpenTURNS distribution."
            ) from None

        try:
            distributions = [create_distribution(*parameters)] * self.dimension
        except BaseException:
            raise ValueError(
                f"Arguments are wrong in {distribution}({pretty_str(parameters)}); "
                f"more details on: {OT_WEBSITE}."
            ) from None

        self.__set_bounds(distributions)
        if transformation is not None:
            distributions = self.__transform_marginal_distributions(
                distributions, transformation
            )

        self.__set_bounds(distributions)
        if lower_bound is not None or upper_bound is not None:
            distributions = self.__truncate_marginal_distributions(
                distributions, lower_bound, upper_bound, threshold
            )

        self.__set_bounds(distributions)
        return distributions

    def __transform_marginal_distributions(
        self, marginals: Iterable[ot.Distribution], transformation: str
    ) -> list[ot.CompositeDistribution]:
        """Apply the standard transformations on the marginals.

        Examples of transformations: -, +, *, **, sin, exp, log, ...

        Args:
            marginals: The marginal distributions.
            transformation: A transformation applied to the random variable,
                e.g. 'sin(x)'. If ``None``, no transformation.

        Returns:
            The transformed marginal distributions.
        """
        transformation = transformation.replace(" ", "")
        symbolic_function = ot.SymbolicFunction([self.variable_name], [transformation])
        self.transformation = transformation.replace(
            self.variable_name, f"({self.transformation})"
        )
        return [
            ot.CompositeDistribution(symbolic_function, marginal)
            for marginal in marginals
        ]

    def __truncate_marginal_distributions(
        self,
        distributions: Iterable[ot.Distribution],
        lower_bound: float,
        upper_bound: float,
        threshold: float = 0.5,
    ) -> list[ot.TruncatedDistribution]:
        """Truncate the distribution of a random variable.

        Args:
            distributions: The distributions.
            lower_bound: A lower bound to truncate the probability distributions.
                If ``None``, no lower truncation.
            upper_bound: An upper bound to truncate the probability distributions.
                If ``None``, no upper truncation.
            threshold: A threshold in [0,1].

        Returns:
            The transformed distributions.
        """
        self.transformation = f"Trunc({self.transformation})"
        return [
            self.__truncate_distribution(
                distribution, index, lower_bound, upper_bound, threshold
            )
            for index, distribution in enumerate(distributions)
        ]

    def __truncate_distribution(
        self,
        distribution: ot.Distribution,
        index: int,
        lower_bound: float,
        upper_bound: float,
        threshold: float = 0.5,
    ) -> ot.TruncatedDistribution:
        """Truncate a probability distribution with lower bound, upper bound or both.

        Args:
            distribution: The probability distribution.
            index: A random variable component.
            lower_bound: A lower bound to truncate the probability distribution.
                If ``None``, no lower truncation.
            upper_bound: An upper bound to truncate the probability distribution.
                If ``None``, no upper truncation.
            threshold: A threshold in [0,1].

        Returns:
            The truncated probability distributions.
        """
        if lower_bound is None:
            LOGGER.debug(
                "Truncate distribution of component %s above %s.", index, upper_bound
            )
            if upper_bound > self.math_upper_bound[index]:
                raise ValueError("u_b is greater than the current upper bound.")
            args = (
                distribution,
                upper_bound,
                ot.TruncatedDistribution.UPPER,
                threshold,
            )
        elif upper_bound is None:
            LOGGER.debug(
                "Truncate distribution of component %s below %s.", index, lower_bound
            )
            if lower_bound < self.math_lower_bound[index]:
                raise ValueError("l_b is less than the current lower bound.")
            args = (
                distribution,
                lower_bound,
                ot.TruncatedDistribution.LOWER,
                threshold,
            )
        else:
            LOGGER.debug(
                "Truncate distribution of component %s below %s and above %s.",
                index,
                lower_bound,
                upper_bound,
            )
            if lower_bound < self.math_lower_bound[index]:
                raise ValueError("l_b is less than the current lower bound.")
            if upper_bound > self.math_upper_bound[index]:
                raise ValueError("u_b is greater than the current upper bound.")
            args = (distribution, ot.Interval(lower_bound, upper_bound), threshold)

        return ot.TruncatedDistribution(*args)

    def __set_bounds(self, distributions: Iterable[ot.Distribution]) -> None:
        """Set the mathematical and numerical bounds (= support and range).

        Args:
            distributions: The probability distributions.
        """
        self.math_lower_bound = []
        self.math_upper_bound = []
        self.num_lower_bound = []
        self.num_upper_bound = []
        for distribution in distributions:
            dist_range = distribution.getRange()
            l_b = dist_range.getLowerBound()[0]
            u_b = dist_range.getUpperBound()[0]
            self.num_lower_bound.append(l_b)
            self.num_upper_bound.append(u_b)
            if not dist_range.getFiniteLowerBound()[0]:
                l_b = -inf
            if not dist_range.getFiniteUpperBound()[0]:
                u_b = inf
            self.math_lower_bound.append(l_b)
            self.math_upper_bound.append(u_b)
        self.math_lower_bound = array(self.math_lower_bound)
        self.math_upper_bound = array(self.math_upper_bound)
        self.num_lower_bound = array(self.num_lower_bound)
        self.num_upper_bound = array(self.num_upper_bound)
