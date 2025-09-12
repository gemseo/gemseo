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

"""Base class for OpenTURNS orthonormal multivariate bases."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from numpy import diag
from numpy import vstack
from openturns import DistributionTransformation
from openturns import FixedStrategy
from openturns import LinearEnumerateFunction

from gemseo.mlearning._basis.base_basis import BaseBasis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openturns.typ import Indices

    from gemseo.typing import RealArray
    from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution


class BaseOTBasis(BaseBasis):
    """The base class for OpenTURNS orthonormal multivariate bases."""

    __enumerate_function: LinearEnumerateFunction
    """The OpenTURNS linear enumerate function."""

    __transformation: DistributionTransformation
    """The OpenTURNS distribution transformation to pre-process input data."""

    def __init__(self, distribution: OTJointDistribution, degree: int) -> None:
        super().__init__(distribution, degree)
        input_dimension = distribution.dimension
        enumerate_function = LinearEnumerateFunction(input_dimension)
        full_basis = self._create_full_basis(input_dimension, enumerate_function)
        strategy = FixedStrategy(
            full_basis,
            enumerate_function.getBasisSizeFromTotalDegree(degree),
        )
        strategy.computeInitialBasis()
        self._basis_functions = strategy.getPsi()
        self._basis_function_gradients = [
            basis_function.gradient for basis_function in self._basis_functions
        ]
        self.__enumerate_function = LinearEnumerateFunction(distribution.dimension)
        self.__transformation = DistributionTransformation(
            self._distribution.distribution, full_basis.getMeasure()
        )

    def get_multi_index(self, index: int) -> Indices:
        return self.__enumerate_function(index)

    def get_index(self, multi_index: Sequence[int]) -> int:
        return self.__enumerate_function.inverse(multi_index)

    def _evaluate_transformation(self, input_data: RealArray) -> RealArray:
        return self.__transformation(input_data)

    def _differentiate_transformation(self, input_data: RealArray) -> RealArray:
        return vstack([diag(self.__transformation.gradient(x)) for x in input_data])

    @staticmethod
    @abstractmethod
    def _create_full_basis(
        input_dimension: int, enumerate_function: LinearEnumerateFunction
    ) -> Any:
        """Create the full orthonormal basis.

        Args:
            input_dimension: The dimension of the input space.
            enumerate_function: The enumerate function.

        Returns:
            The full orthonormal basis.
        """
