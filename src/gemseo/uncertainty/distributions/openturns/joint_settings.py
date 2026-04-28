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
"""Settings for the OpenTURNS-based joint probability distributions."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import TYPE_CHECKING

from openturns import DistributionImplementation
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import model_validator

from gemseo.uncertainty.distributions.base_joint_settings import (
    BaseJointDistributionSettings,
)
from gemseo.uncertainty.distributions.base_settings import (  # noqa: TC001
    BaseDistributionSettings,
)
from gemseo.uncertainty.distributions.openturns.base_settings import (
    BaseOTDistributionSettings,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class OTJointDistribution_Settings(  # noqa: N801
    BaseJointDistributionSettings, BaseOTDistributionSettings
):
    r"""The settings of a OpenTURNS-based joint probability distribution.

    An OpenTURNS-based joint probability distribution relies on copulas.
    Using copulas is common practice
    to model the dependency structure
    between the components $U_1,\ldots,U_d$ of the random vector $U=(U_1,\ldots,U_d)$.

    Let $F_i$ be the marginal cumulative density function (CDF) of $U_i$
    and $F$ the joint CDF of $U$.
    The Sklar's theorem states that
    F is the combination of $F_1,\ldots,F_d$
    and a function $C: [0,1]^d \to [0,1]$, called copula $C$,
    verifying $F(\mathbf{u})=C(F_1(u_1),\ldots,F_d(u_d))$.
    When $U_1,\ldots,U_d$ are continuous random variables,
    this function is unique and defined
    by $C(\boldsymbol{\xi})=F(F_1^{(-1)}(\xi_1),\ldots,F_d^{(-1)}(\xi_d))$
    where $\xi_1,\ldots,\xi_d$ are independent and uniformly distributed in $[0,1]$.\\

    When the random variables $U_1,\ldots,U_d$ are independent by block,
    the copula $C$ can be defined as a product of copulas, i.e.
    \[
    C(\boldsymbol{\xi}) =
    C_1(\boldsymbol{\xi}^{[1]})
    \times C_2(\boldsymbol{\xi}^{[2]})
    \times \ldots
    \times C_K(\boldsymbol{\xi}^{[K]})
    \]
    where

    - $\boldsymbol{xi}^{[k]}=(\xi_{n_1+\ldots+n_{k-1}+1},\ldots,\xi_{n_1+\ldots+n_k})$,
    - the random vectors $\boldsymbol{\xi}^{[1]},\ldots,\boldsymbol{\xi}^{[K]}$
      are independent,
    - $C_k$ is the copula of $\boldsymbol{u}^{[k]}$.
    """

    marginal_settings: Sequence[BaseDistributionSettings] = Field(
        description="The OpenTURNS-based marginal probability distributions."
    )

    copula: (
        DistributionImplementation
        | tuple[tuple[tuple[NonNegativeInt, ...], DistributionImplementation], ...]
    ) = Field(
        default=(),
        description="Either a copula distribution defining "
        "the dependency structure between the components of the random vector, "
        "or a collection of pairs of components and copula distributions"
        "such that the components between pairs are independent of each other; "
        "if empty, the components are independent.",
    )

    @model_validator(mode="after")
    def __validate_copula(self) -> Self:
        """Validate the copula.

        Raises:
            ValueError: When the dimension of the copula is not consistent
                with the number of marginals,
                when the dimension of a sub-copula is not consistent
                with the number of components,
                when a component has more than one copula
                or when the value of a component is wrong.

        """
        if self.copula == ():
            return self

        dimension = len(self.marginal_settings)
        if isinstance(self.copula, DistributionImplementation):
            if (copula_dimension := self.copula.getDimension()) != dimension:
                msg = (
                    "The dimension of the copula "
                    f"must be equal to the number of marginals ({dimension}); "
                    f"got {copula_dimension}."
                )
                raise ValueError(msg)

            return self

        for i, (indices, copula) in enumerate(self.copula):
            if (sub_dimension := copula.getDimension()) != (n_indices := len(indices)):
                msg = (
                    f"The dimension of the block-independent copula at position {i} "
                    f"must be equal to the number of components ({n_indices}); "
                    f"got {sub_dimension}."
                )
                raise ValueError(msg)

        if not 1 <= sum(len(indices) for indices, _ in self.copula) <= dimension:
            msg = (
                "The sum of the dimensions of the block-independent copulas must "
                f"be less than or equal to the number of marginals ({dimension})."
            )
            raise ValueError(msg)

        all_indices = [index for indices, _ in self.copula for index in indices]
        if len(set(all_indices)) != len(all_indices):
            msg = "A component cannot have more than one copula."
            raise ValueError(msg)

        if any(not 0 <= index < dimension for index in all_indices):
            msg = (
                "The components must be in the range [0, n_marginals - 1], "
                f"i.e. [0, {dimension - 1}]."
            )
            raise ValueError(msg)

        return self
