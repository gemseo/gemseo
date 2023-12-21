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
"""The interface to SciPy-based probability distributions.

The :class:`.SPDistribution` class is a concrete class
inheriting from :class:`.Distribution` which is an abstract one.
SP stands for `scipy <https://docs.scipy.org/doc/scipy/tutorial/stats.html>`_
which is the library it relies on.

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
  (default: use ``parameters``).
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

import scipy
import scipy.stats as sp_stats
from numpy import array
from numpy import ndarray
from numpy import vstack

from gemseo.uncertainty.distributions.distribution import Distribution
from gemseo.uncertainty.distributions.distribution import StandardParametersType
from gemseo.uncertainty.distributions.scipy.composed import SPComposedDistribution
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy.random import RandomState

    from gemseo.uncertainty.distributions.composed import ComposedDistribution

LOGGER = logging.getLogger(__name__)

SP_WEBSITE = "https://docs.scipy.org/doc/scipy/reference/stats.html"


class SPDistribution(Distribution):
    """A SciPy-based probability distribution.

    Create a probability distribution for an uncertain variable
    from its dimension and distribution names and properties.

    .. seealso::

        :class:`.SPExponentialDistribution`
        :class:`.SPNormalDistribution`
        :class:`.SPTriangularDistribution`
        :class:`.SPUniformDistribution`

    Examples:
        >>> from gemseo.uncertainty.distributions.scipy.distribution import (
        ...     SPDistribution,
        ... )
        >>> distribution = SPDistribution("x", "expon", {"loc": 3, "scale": 1 / 2.0})
        >>> print(distribution)
        expon(loc=3, scale=0.5)
    """

    COMPOSED_DISTRIBUTION_CLASS: ClassVar[type[ComposedDistribution] | None] = (
        SPComposedDistribution
    )

    def __init__(  # noqa: D107
        self,
        variable: str = Distribution.DEFAULT_VARIABLE_NAME,
        interfaced_distribution: str = "uniform",
        parameters: Mapping[str, Any] = MappingProxyType({}),
        dimension: int = 1,
        standard_parameters: StandardParametersType | None = None,
    ) -> None:
        """
        Args:
            standard_parameters: The parameters of the probability distribution
                used for string representation only
                (use ``parameters`` for computation).
                If ``None``, use ``parameters`` instead.
                For instance,
                let us consider the interfaced SciPy distribution ``"uniform"``.
                Then,
                the string representation of
                ``SPDistribution("x", "uniform", parameters, 1, {"min": 1, "max": 3})``
                with ``parameters={"loc": 1, "scale": 2}``
                is ``"uniform(max=3, min=1)"``
                while the string representation of
                ``SPDistribution("x", "uniform", parameters)``
                is ``"uniform(loc=1, scale=2)"``.
        """  # noqa: D205 D212 D415
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
        msg = MultiLineString()
        msg.indent()
        msg.add("Mathematical support: {}", self.support)
        msg.add("Numerical range: {}", self.range)
        LOGGER.debug("%s", msg)

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
        random_state: None | int | Generator | RandomState = None,
    ) -> ndarray:
        """
        Args:
            random_state: The SciPy random state.
        """  # noqa: D205, D212
        return vstack([m.rvs(n_samples, random_state) for m in self.marginals]).T

    def compute_cdf(  # noqa: D102
        self,
        vector: Iterable[float],
    ) -> ndarray:
        return array([
            self.marginals[index].cdf(value) for index, value in enumerate(vector)
        ])

    def compute_inverse_cdf(  # noqa: D102
        self,
        vector: Iterable[float],
    ) -> ndarray:
        return array([
            self.marginals[index].ppf(value) for index, value in enumerate(vector)
        ])

    @property
    def mean(self) -> ndarray:  # noqa: D102
        return array([marginal.mean() for marginal in self.marginals])

    @property
    def standard_deviation(self) -> ndarray:  # noqa: D102
        return array([marginal.std() for marginal in self.marginals])

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
            create_distribution = getattr(sp_stats, distribution)
        except BaseException:
            raise ValueError(
                f"{distribution} is an unknown scipy distribution."
            ) from None

        try:
            parameters = parameters or {}
            create_distribution(**parameters)
            distributions = [create_distribution(**parameters)] * self.dimension
        except BaseException:
            raise ValueError(
                f"Arguments are wrong in {distribution}({pretty_str(parameters)}); "
                f"more details on: {SP_WEBSITE}."
            ) from None

        self.__set_bounds(distributions)
        return distributions

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
