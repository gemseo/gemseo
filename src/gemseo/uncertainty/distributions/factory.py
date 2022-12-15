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
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Module containing a factory to create an instance of :class:`.Distribution`."""
from __future__ import annotations

from typing import Optional
from typing import Sequence
from typing import Union

from gemseo.core.factory import Factory
from gemseo.uncertainty.distributions.composed import ComposedDistribution
from gemseo.uncertainty.distributions.distribution import Distribution
from gemseo.uncertainty.distributions.distribution import ParametersType
from gemseo.uncertainty.distributions.distribution import StandardParametersType
from gemseo.utils.string_tools import pretty_str

DistributionParametersType = Union[
    int, ParametersType, Optional[StandardParametersType], float
]


class DistributionFactory:
    """Factory to build instances of :class:`.Distribution`.

    At initialization, this factory scans the following modules
    to search for subclasses of this class:

    - the modules located in ``gemseo.uncertainty.distributions`` and its sub-packages,
    - the modules referenced in the ``GEMSEO_PATH,``
    - the modules referenced in the ``PYTHONPATH`` and starting with ``gemseo_``.

    Then, it can check if a class is present or return the list of available classes.

    Lastly, it can create an instance of a class.

    Examples:
        >>> from gemseo.uncertainty.distributions.factory import DistributionFactory
        >>> factory = DistributionFactory()
        >>> factory.is_available("OTNormalDistribution")
        True
        >>> factory.available_distributions[-3:]
        ['SPNormalDistribution', 'SPTriangularDistribution', 'SPUniformDistribution']
        >>> distribution = factory.create("OTNormalDistribution", "x")
        >>> print(distribution)
        Normal(mu=0.0, sigma=1.0)
    """

    def __init__(self) -> None:  # noqa: D107
        self.factory = Factory(Distribution, ("gemseo.uncertainty.distributions",))

    def create_marginal_distribution(
        self,
        distribution_name: str,
        variable: str,
        **parameters: DistributionParametersType,
    ) -> Distribution:
        """Create a marginal probability distribution for a given random variable.

        Args:
            distribution_name: The name of a class defining a distribution.
            variable: The name of the random variable.
            **parameters: The parameters of the distribution.

        Returns:
            The marginal probability distribution.
        """
        return self.factory.create(distribution_name, variable=variable, **parameters)

    create = create_marginal_distribution

    def create_composed_distribution(
        self,
        distributions: Sequence[Distribution],
        copula_name: str = ComposedDistribution.CopulaModel.independent_copula.value,
        variable: str = "",
    ) -> ComposedDistribution:
        """Create a composed probability distribution from marginal ones.

        Args:
            distributions: The marginal distributions.
            copula_name: The name of the copula.
            variable: The name of the variable, if any;
                otherwise,
                concatenate the names of the random variables
                defined by ``distributions``.

        Returns:
            The composed probability distribution.
        """
        identifiers = {dist.__class__.__name__[0:2] for dist in distributions}
        if len(identifiers) > 1:
            raise ValueError(
                "A composed probability distribution cannot mix distributions "
                f"with different identifiers; got {pretty_str(identifiers)}."
            )

        return self.factory.create(
            f"{next(iter(identifiers))}ComposedDistribution",
            distributions=distributions,
            copula=copula_name,
            variable=variable,
        )

    @property
    def available_distributions(self) -> list[str]:
        """The available probability distributions."""
        return self.factory.classes

    def is_available(self, distribution_name: str) -> bool:
        """Check the availability of a probability distribution.

        Args:
            distribution_name: The name of a class defining a distribution.

        Returns:
            The availability of the distribution.
        """
        return self.factory.is_available(distribution_name)
