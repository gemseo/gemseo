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
"""A factory of probability distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.base_factory import BaseFactory
from gemseo.uncertainty.distributions.base_distribution import BaseDistribution
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution


class DistributionFactory(BaseFactory):
    """A factory of probability distributions."""

    _CLASS = BaseDistribution
    _MODULE_NAMES = ("gemseo.uncertainty.distributions",)

    def create_marginal_distribution(
        self,
        distribution_name: str,
        **parameters: Any,
    ) -> BaseDistribution:
        """Create a marginal probability distribution for a given random variable.

        Args:
            distribution_name: The name of a class defining a distribution.
            **parameters: The parameters of the distribution.

        Returns:
            The marginal probability distribution.
        """
        return super().create(distribution_name, **parameters)

    create = create_marginal_distribution

    def create_joint_distribution(
        self,
        distributions: Sequence[BaseDistribution],
        copula: Any = None,
    ) -> BaseJointDistribution:
        """Create a joint probability distribution from marginal ones.

        Args:
            distributions: The marginal distributions.
            copula: A copula distribution
                defining the dependency structure between random variables;
                if ``None``, consider an independent copula.

        Returns:
            The joint probability distribution.
        """
        identifiers = {dist.__class__.__name__[0:2] for dist in distributions}
        if len(identifiers) > 1:
            msg = (
                "A joint probability distribution cannot mix distributions "
                f"with different identifiers; got {pretty_str(identifiers)}."
            )
            raise ValueError(msg)

        return super().create(
            f"{next(iter(identifiers))}JointDistribution",
            distributions=distributions,
            copula=copula,
        )
