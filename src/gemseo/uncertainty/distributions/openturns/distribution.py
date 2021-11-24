# -*- coding: utf-8 -*-
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

"""Class to create a probability distribution from the OpenTURNS library.

The :class:`.OTDistribution` class is a concrete class
inheriting from :class:`.Distribution` which is an abstract one.
OT stands for `OpenTURNS <http://www.openturns.org/>`_
which is the library it relies on.

The :class:`.OTDistribution` of a given uncertain variable is built
from mandatory arguments:

- a variable name,
- a distribution name recognized by OpenTURNS,
- a set of parameters provided as a tuple
  of positional arguments filled in the order
  specified by the OpenTURNS constructor of this distribution.

.. warning::

    The distribution parameters must be provided according to the signature
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

from __future__ import division, unicode_literals

import logging
from typing import Callable, Iterable, List, Optional

import openturns as ots
from numpy import array, inf, ndarray

from gemseo.uncertainty.distributions.distribution import (
    Distribution,
    ParametersType,
    StandardParametersType,
)
from gemseo.uncertainty.distributions.openturns.composed import OTComposedDistribution
from gemseo.utils.string_tools import MultiLineString

OT_WEBSITE = (
    "http://openturns.github.io/openturns/latest/user_manual/"
    "probabilistic_modelling.html"
)

LOGGER = logging.getLogger(__name__)


class OTDistribution(Distribution):
    """OpenTURNS probability distribution.

    Create a probability distribution for an uncertain variable
    from its dimension and distribution names and properties.

    Example:
        >>> from gemseo.uncertainty.distributions.openturns.distribution import (
        ...     OTDistribution
        ... )
        >>> distribution = OTDistribution('x', 'Exponential', (3, 2))
        >>> print(distribution)
        Exponential(3, 2)
    """

    _COMPOSED_DISTRIBUTION = OTComposedDistribution

    def __init__(
        self,
        variable,  # type: str
        interfaced_distribution,  # type: str
        parameters,  # type: ParametersType
        dimension=1,  # type: int
        standard_parameters=None,  # type: Optional[StandardParametersType]
        transformation=None,  # type: Optional[str]
        lower_bound=None,  # type: Optional[float]
        upper_bound=None,  # type: Optional[float]
        threshold=0.5,  # type: float
    ):  # noqa: D205,D212,D415
        # type: (...) -> None
        """
        Args:
            variable: The name of the random variable.
            interfaced_distribution: The name of the probability distribution,
                typically the name of a class wrapped from an external library,
                such as 'Normal' for OpenTURNS or 'norm' for SciPy.
            parameters: The parameters of the probability distribution.
            dimension: The dimension of the random variable.
            standard_parameters: The standard representation
                of the parameters of the probability distribution.
            transformation: A transformation applied
                to the random variable,
                e.g. 'sin(x)'. If None, no transformation.
            lower_bound: A lower bound to truncate the distribution.
                If None, no lower truncation.
            upper_bound: An upper bound to truncate the distribution.
                If None, no upper truncation.
            threshold: A threshold in [0,1].
        """
        super(OTDistribution, self).__init__(
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
        self.distribution = ots.ComposedDistribution(self.marginals)
        msg = MultiLineString()
        msg.indent()
        msg.add("Mathematical support: {}", self.support)
        msg.add("Numerical range: {}", self.range)
        msg.add("Transformation: {}", self.transformation)
        LOGGER.debug("%s", msg)

    def compute_samples(
        self,
        n_samples=1,  # type: int
    ):  # noqa: D102
        # type: (...) -> ndarray
        sample = array(self.distribution.getSample(n_samples))
        return sample

    def compute_cdf(
        self,
        vector,  # type: Iterable[float]
    ):  # noqa: D102
        # type: (...) -> ndarray
        return array(
            [
                self.marginals[index].computeCDF(ots.Point([value]))
                for index, value in enumerate(vector)
            ]
        )

    def compute_inverse_cdf(
        self,
        vector,  # type: Iterable[float]
    ):  # noqa: D102
        # type: (...) -> ndarray
        return array(
            [
                self.marginals[index].computeQuantile(value)[0]
                for index, value in enumerate(vector)
            ]
        )

    def _pdf(
        self,
        index,  # type: int
    ):  # noqa: D102
        # type: (...) -> Callable
        def pdf(
            point,  # type: float
        ):
            # type: (...) -> float
            """Probability Density Function (PDF).

            Args:
                point: An evaluation point.

            Returns:
                The PDF value at the evaluation point.
            """
            return self.marginals[index].computePDF(point)

        return pdf

    def _cdf(
        self,
        index,  # type: int
    ):  # noqa: D102
        # type: (...) -> Callable
        def cdf(
            level,  # type: float
        ):
            # type: (...) -> float
            """Cumulative Density Function (CDF).

            Args:
                level: A probability level.

            Returns:
                The CDF value for the probability level.
            """
            return self.marginals[index].computeCDF(level)

        return cdf

    @property
    def mean(self):  # noqa: D102
        # type: (...) -> ndarray
        return array(self.distribution.getMean())

    @property
    def standard_deviation(self):  # noqa: D102
        # type: (...) -> ndarray
        return array(self.distribution.getStandardDeviation())

    def __create_distributions(
        self,
        distribution,  # type: str
        parameters,  # type:ParametersType
        transformation,  # type: str
        lower_bound,  # type:float
        upper_bound,  # type:float
        threshold,  # type:float
    ):  # type: (...) -> List[ots.Distribution]
        """Instantiate an OpenTURNS distribution for each random variable component.

        Args:
            distribution: The name of the distribution.
            parameters: The parameters of the distribution.
            transformation: A transformation applied to the random variable,
                e.g. 'sin(x)'. If None, no transformation.
            lower_bound: A lower truncation bound.
                If None, no lower truncation.
            upper_bound: An upper truncation bound.
                If None, no upper truncation.
            threshold: A threshold value in [0,1].

        Returns:
            The marginal OpenTURNS distributions.
        """
        try:
            ot_dist = getattr(ots, distribution)
        except Exception:
            raise ValueError(
                "{} is an unknown OpenTURNS distribution.".format(distribution)
            )
        try:
            ot_dist = [ot_dist(*parameters)] * self.dimension
        except Exception:
            args = ", ".join([str(val) for val in parameters])
            raise ValueError(
                "Arguments are wrong in {}({}); "
                "more details on: {}.".format(distribution, args, OT_WEBSITE)
            )
        self.__set_bounds(ot_dist)
        if transformation is not None:
            ot_dist = self.__transform_marginal_dist(ot_dist, transformation)
        self.__set_bounds(ot_dist)
        if lower_bound is not None or upper_bound is not None:
            ot_dist = self.__truncate_marginal_dist(
                ot_dist, lower_bound, upper_bound, threshold
            )
        self.__set_bounds(ot_dist)
        return ot_dist

    def __transform_marginal_dist(
        self,
        marginals,  # type: Iterable[ots.Distribution],
        transformation,  # type: str
    ):  # type: (...) -> List[ots.Distribution]
        """Apply the standard transformations on the marginals.

        Examples of transformations: -, +, *, **, sin, exp, log, ...

        Args:
            marginals: The marginal distributions.
            transformation: A transformation applied to the random variable,
                e.g. 'sin(x)'. If None, no transformation.

        Returns:
            The transformed marginal distributions.
        """
        variable_name = self.variable_name
        transformation = transformation.replace(" ", "")
        symbolic_function = ots.SymbolicFunction([self.variable_name], [transformation])
        marginals = [
            ots.CompositeDistribution(symbolic_function, marginal)
            for marginal in marginals
        ]
        prev = self.transformation
        transformation = transformation.replace(variable_name, "({})".format(prev))
        self.transformation = transformation
        return marginals

    def __truncate_marginal_dist(
        self,
        distributions,  # type: Iterable[ots.Distribution]
        lower_bound,  # type: float
        upper_bound,  # type: float
        threshold=0.5,  # type: float
    ):  # type: (...) -> List[ots.Distribution]
        """Truncate the distribution of a random variable.

        Args:
            distributions: The distributions.
            lower_bound: A lower bound to truncate the distributions.
                If None, no lower truncation.
            upper_bound: An upper bound to truncate the distributions.
                If None, no upper truncation.
            threshold: A threshold in [0,1].

        Returns:
            The transformed distributions.
        """
        marginals = [
            self.__truncate_distribution(
                dist, index, lower_bound, upper_bound, threshold
            )
            for index, dist in enumerate(distributions)
        ]
        prev = self.transformation
        self.transformation = "Trunc({})".format(prev)
        return marginals

    def __truncate_distribution(
        self,
        distributions,  # type: Iterable[ots.Distribution]
        index,  # type: int
        lower_bound,  # type: float
        upper_bound,  # type: float
        threshold=0.5,  # type: float
    ):  # type: (...) -> List[ots.Distribution]
        """Truncate a distribution with lower bound, upper bound or both.

        Args:
            distributions: The distributions.
            index: A random variable component.
            lower_bound: A lower bound to truncate the distribution.
                If None, no lower truncation.
            upper_bound: An upper bound to truncate the distribution.
                If None, no upper truncation.
            threshold: A threshold in [0,1].

        Returns:
            The truncated distributions.
        """
        if lower_bound is None:
            LOGGER.debug(
                "Truncate distribution of component %s above %s.", index, upper_bound
            )
            upper = ots.TruncatedDistribution.UPPER
            current_u_b = self.math_upper_bound[index]
            if upper_bound > current_u_b:
                raise ValueError("u_b is greater than the current upper bound.")
            distributions = ots.TruncatedDistribution(
                distributions, upper_bound, upper, threshold
            )
        elif upper_bound is None:
            LOGGER.debug(
                "Truncate distribution of component %s below %s.", index, lower_bound
            )
            lower = ots.TruncatedDistribution.LOWER
            current_l_b = self.math_lower_bound[index]
            if lower_bound < current_l_b:
                raise ValueError("l_b is lower than the current lower bound.")
            distributions = ots.TruncatedDistribution(
                distributions, lower_bound, lower, threshold
            )
        else:
            LOGGER.debug(
                "Truncate distribution of component %s below %s and above %s.",
                index,
                lower_bound,
                upper_bound,
            )
            current_l_b = self.math_lower_bound[index]
            current_u_b = self.math_upper_bound[index]
            if lower_bound < current_l_b:
                raise ValueError("l_b is lower than the current lower bound.")
            if upper_bound > current_u_b:
                raise ValueError("u_b is greater than the current upper bound.")
            distributions = ots.TruncatedDistribution(
                distributions, lower_bound, upper_bound, threshold
            )
        return distributions

    def __set_bounds(
        self, distributions  # type: Iterable[ots.Distribution]
    ):  # type: (...) -> None
        """Set the mathematical and numerical bounds (= support and range).

        Args:
            distributions: The distributions.
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
