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
r"""Abstract classes defining the concept of joint probability distribution.

Overview
--------

The abstract :class:`.ComposedDistribution` class
implements the concept of `joint probability distribution
<https://en.wikipedia.org/wiki/Joint_probability_distribution>`_,
which is a mathematical function giving the probabilities of occurrence
of different possible outcomes of several random variables for an experiment.
In the style of OpenTURNS, a :class:`.ComposedDistribution` is defined
from a list of :class:`.Distribution` instances
defining the marginals of the random variables
and a copula defining the dependence structure between them.

.. note::

   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative density functions.
   `See more <https://en.wikipedia.org/wiki/Copula_(probability_theory)>`_.

By definition, a joint probability distribution is a probability distribution
Therefore, :class:`.ComposedDistribution` inherits
from the abstract class :class:`.Distribution`.

Construction
------------

The :class:`.ComposedDistribution` of a list of given uncertain variables is built
from a list of :class:`.Distribution` objects
implementing the probability distributions of these variables
and from a copula name.

Capabilities
------------

Because :class:`.ComposedDistribution` inherits from :class:`.Distribution`,
we can easily get statistics, such as :attr:`.ComposedDistribution.mean`,
:attr:`.ComposedDistribution.standard_deviation`.
We can also get the numerical :attr:`.ComposedDistribution.range` and
mathematical :attr:`.ComposedDistribution.support`.

.. note::

    We call mathematical *support* the set of values that the random variable
    can take in theory, e.g. :math:`]-\infty,+\infty[` for a Gaussian variable,
    and numerical *range* the set of values that it can take in practice,
    taking into account the values rounded to zero double precision.
    Both support and range are described in terms of lower and upper bounds

We can also evaluate the cumulative density function
(:meth:`.ComposedDistribution.compute_cdf`)
for the different marginals of the random variable,
as well as the inverse cumulative density function
(:meth:`.ComposedDistribution.compute_inverse_cdf`). We can plot them,
either for a given marginal (:meth:`.ComposedDistribution.plot`)
or for all marginals (:meth:`.ComposedDistribution.plot_all`).

Lastly, we can compute realizations of the random variable
by means of the :meth:`.ComposedDistribution.compute_samples` method.
"""
from __future__ import annotations

import logging
from typing import ClassVar
from typing import Iterable
from typing import Sequence

from numpy import array
from numpy import concatenate
from numpy import ndarray

from gemseo.uncertainty.distributions.distribution import Distribution
from gemseo.utils.base_enum import BaseEnum
from gemseo.utils.base_enum import get_names
from gemseo.utils.python_compatibility import Final
from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)


class ComposedDistribution(Distribution):
    """Composed distribution."""

    class CopulaModel(BaseEnum):
        """A copula model."""

        independent_copula = "independent_copula"

    # TODO: API: remove this attribute in the next major release.
    INDEPENDENT_COPULA: Final[str] = CopulaModel.independent_copula.value
    """The name of the independent copula."""

    # TODO: API: remove this attribute in the next major release.
    AVAILABLE_COPULA_MODELS: ClassVar[list[str]] = get_names(CopulaModel)
    """The names of the models defining copulas."""

    _COMPOSED = "Composed"

    def __init__(
        self,
        distributions: Sequence[Distribution],
        copula: CopulaModel | str = CopulaModel.independent_copula,
        variable: str = "",
    ) -> None:
        """
        Args:
            distributions: The distributions.
            copula: A copula model.
            variable: The name of the variable, if any;
                otherwise,
                concatenate the names of the random variables
                defined by ``distributions``.
        """  # noqa: D205,D212,D415
        dimension = sum(distribution.dimension for distribution in distributions)
        self._marginal_variables = [
            distribution.variable_name for distribution in distributions
        ]
        super().__init__(
            variable or "_".join(self._marginal_variables),
            self._COMPOSED,
            (copula,),
            dimension,
        )
        self.marginals = distributions
        msg = MultiLineString()
        msg.indent()
        msg.add("Marginals:")
        msg.indent()
        for distribution in distributions:
            msg.add(
                "{}({}): {}",
                distribution.variable_name,
                distribution.dimension,
                distribution,
            )
        LOGGER.debug("%s", msg)

    def _set_bounds(
        self,
        distributions: Iterable[Distribution],
    ) -> None:
        """Set the mathematical and numerical bounds (= support and range).

        Args:
            distributions: The distributions.
        """
        self.math_lower_bound = array([])
        self.math_upper_bound = array([])
        self.num_lower_bound = array([])
        self.num_upper_bound = array([])
        for dist in distributions:
            self.math_lower_bound = concatenate(
                (self.math_lower_bound, dist.math_lower_bound)
            )
            self.num_lower_bound = concatenate(
                (self.num_lower_bound, dist.num_lower_bound)
            )
            self.math_upper_bound = concatenate(
                (self.math_upper_bound, dist.math_upper_bound)
            )
            self.num_upper_bound = concatenate(
                (self.num_upper_bound, dist.num_upper_bound)
            )

    @property
    def mean(self) -> ndarray:  # noqa: D102
        mean = [marginal.mean for marginal in self.marginals]
        return array(mean).flatten()

    @property
    def standard_deviation(self) -> ndarray:  # noqa: D102
        std = [marginal.standard_deviation for marginal in self.marginals]
        return array(std).flatten()

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
    ) -> ndarray:
        sample = self.marginals[0].compute_samples(n_samples)
        for marginal in self.marginals[1:]:
            sample = concatenate((sample, marginal.compute_samples(n_samples)), axis=1)
        return sample
