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
"""Class to create a probability distribution from the SciPy library.

The :class:`.SPDistribution` class is a concrete class
inheriting from :class:`.Distribution` which is an abstract one.
SP stands for `scipy <https://docs.scipy.org/doc/scipy/reference/
tutorial/stats.html>`_ which is the library it relies on.

The :class:`.SPDistribution` of a given uncertain variable is built
from mandatory arguments:

- a variable name,
- a distribution name recognized by SciPy,
- a set of parameters provided as a dictionary
  of keyword arguments named as the arguments of the scipy constructor
  of this distribution.

.. warning::

    The distribution parameters must be provided according to the signature
    of the scipy classes. `Access the scipy documentation
    <https://docs.scipy.org/doc/scipy/reference/stats.html>`_.

The constructor has also optional arguments:

- a variable dimension (default: 1),
- a standard representation of these parameters
  (default: use :code:`parameters`).
"""
from __future__ import annotations

import logging
from typing import Callable
from typing import Iterable
from typing import Mapping

import scipy
import scipy.stats as sp_stats
from numpy import array
from numpy import ndarray
from numpy import vstack

from gemseo.uncertainty.distributions.distribution import Distribution
from gemseo.uncertainty.distributions.distribution import ParametersType
from gemseo.uncertainty.distributions.distribution import StandardParametersType
from gemseo.uncertainty.distributions.scipy.composed import SPComposedDistribution
from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)

SP_WEBSITE = "https://docs.scipy.org/doc/scipy/reference/stats.html"


class SPDistribution(Distribution):
    """SciPy probability distribution.

    Create a probability distribution for an uncertain variable
    from its dimension and distribution names and properties.

    .. seealso::

        :class:`.SPExponentialDistribution`
        :class:`.SPNormalDistribution`
        :class:`.SPTriangularDistribution`
        :class:`.SPUniformDistribution`

    Example:
        >>> from gemseo.uncertainty.distributions.scipy.distribution import (
        ...    SPDistribution
        ... )
        >>> distribution = SPDistribution('x', 'expon', {'loc': 3, 'scale': 1/2.})
        >>> print(distribution)
        expon(loc=3, scale=0.5)
    """

    _COMPOSED_DISTRIBUTION = SPComposedDistribution

    def __init__(
        self,
        variable: str,
        interfaced_distribution: str,
        parameters: ParametersType,
        dimension: int = 1,
        standard_parameters: StandardParametersType | None = None,
    ) -> None:
        """
        Args:
            variable: The name of the random variable.
            interfaced_distribution: The name of the probability distribution,
                typically the name of a class wrapped from an external library,
                such as 'Normal' for OpenTURNS or 'norm' for SciPy.
            parameters: The parameters of the probability distribution.
            dimension: The dimension of the random variable.
            standard_parameters (dict, optional): The standard representation
                of the parameters of the probability distribution.
        """  # noqa: D205,D212,D415
        super().__init__(
            variable,
            interfaced_distribution,
            parameters,
            dimension,
            standard_parameters,
        )
        self.marginals = self.__create_distributions(
            self.distribution_name, self.parameters
        )
        math_support = str(self.support)
        num_range = str(self.range)
        msg = MultiLineString()
        msg.indent()
        msg.add("Mathematical support: {}", math_support)
        msg.add("Numerical range: {}", num_range)
        LOGGER.debug("%s", msg)

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
    ) -> ndarray:
        return vstack([marginal.rvs(n_samples) for marginal in self.marginals]).T

    def compute_cdf(  # noqa: D102
        self,
        vector: Iterable[float],
    ) -> ndarray:
        return array(
            [self.marginals[index].cdf(value) for index, value in enumerate(vector)]
        )

    def compute_inverse_cdf(  # noqa: D102
        self,
        vector: Iterable[float],
    ) -> ndarray:
        return array(
            [self.marginals[index].ppf(value) for index, value in enumerate(vector)]
        )

    @property
    def mean(self) -> ndarray:  # noqa: D102
        mean = [marginal.mean() for marginal in self.marginals]
        return array(mean)

    @property
    def standard_deviation(self) -> ndarray:  # noqa: D102
        std = [marginal.std() for marginal in self.marginals]
        return array(std)

    def __create_distributions(
        self,
        distribution: str,
        parameters: Mapping[str, int | float],
    ) -> list[scipy.stats.rv_continuous]:
        """Instantiate a SciPy distribution for each random variable component.

        Args:
            distribution: The name of the distribution.
            parameters: The parameters of the distribution.

        Returns:
            The marginal SciPy distributions.
        """
        try:
            sp_dist = getattr(sp_stats, distribution)
        except Exception:
            raise ValueError(f"{distribution} is an unknown scipy distribution.")
        try:
            sp_dist(**parameters)
        except Exception:
            raise ValueError(
                "{} does not take these arguments; "
                "more details on: {}.".format(distribution, SP_WEBSITE)
            )
        sp_dist = [sp_dist(**parameters)] * self.dimension
        self.__set_bounds(sp_dist)
        return sp_dist

    def __set_bounds(
        self,
        distributions: Iterable[scipy.stats.rv_continuous],
        extrema_level: float = 1e-12,
    ) -> None:
        """Set mathematical and numerical bounds (= support and range).

        Args:
            distributions: The distributions.
            extrema_level: A quantile level corresponding to the lower
                bound of the numerical random variable range.
        """
        self.math_lower_bound = []
        self.math_upper_bound = []
        self.num_lower_bound = []
        self.num_upper_bound = []
        for distribution in distributions:
            dist_range = distribution.interval(1.0)
            self.math_lower_bound.append(dist_range[0])
            self.math_upper_bound.append(dist_range[1])
            l_b = distribution.ppf(extrema_level)
            u_b = distribution.ppf(1 - extrema_level)
            self.num_lower_bound.append(l_b)
            self.num_upper_bound.append(u_b)
        self.math_lower_bound = array(self.math_lower_bound)
        self.math_upper_bound = array(self.math_upper_bound)
        self.num_lower_bound = array(self.num_lower_bound)
        self.num_upper_bound = array(self.num_upper_bound)

    def _pdf(  # noqa: D102
        self,
        index: int,
    ) -> Callable:
        def pdf(
            point: float,
        ) -> float:
            """Probability Density Function (PDF).

            Args:
                point: An evaluation point.

            Returns:
                The PDF value at the evaluation point.
            """
            return self.marginals[index].pdf(point)

        return pdf

    def _cdf(  # noqa: D102
        self,
        index: int,
    ) -> Callable:
        def cdf(
            level: float,
        ) -> float:
            """Cumulative Density Function (CDF).

            Args:
                level: A probability level.

            Returns:
                The CDF value for the probability level.
            """
            return self.marginals[index].cdf(level)

        return cdf
