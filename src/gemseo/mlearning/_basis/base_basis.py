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

"""The base class for orthonormal multivariate bases."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from numpy import array
from numpy import hstack
from numpy import newaxis
from numpy import stack

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from gemseo.typing import RealArray
    from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution


class BaseBasis(metaclass=ABCGoogleDocstringInheritanceMeta):
    """The base class for orthonormal multivariate bases."""

    _basis_functions: Sequence[Callable[[RealArray], RealArray]]
    """The basis functions."""

    _basis_function_gradients: Sequence[Callable[[RealArray], RealArray]]
    """The gradients of the basis functions."""

    _distribution: BaseJointDistribution
    """The joint probability distribution."""

    @abstractmethod
    def __init__(self, distribution: BaseJointDistribution, degree: int) -> None:
        """
        Args:
            distribution: The joint probability distribution.
            degree: The total degree of the basis.
        """  # noqa: D205, D212
        self._distribution = distribution

    @property
    def basis_functions(self) -> Sequence[Callable[[RealArray], RealArray]]:
        """The basis functions."""
        return self._basis_functions

    @abstractmethod
    def get_multi_index(self, index: int) -> Any:
        """Return the multi-index of a basis function.

        Args:
            index: The index of the basis function.

        Returns:
            The multi-index of the basis function.
        """

    @abstractmethod
    def get_index(self, multi_index: Sequence[int]) -> int:
        """Return the index of a basis function.

        Args:
            multi_index: The multi-index of the basis function.

        Returns:
            The index of the basis function.
        """

    @abstractmethod
    def _evaluate_transformation(self, input_data: RealArray) -> RealArray:
        """Evaluate the transformation from the input space to the standard space.

        Args:
            input_data: The input data.

        Returns:
            The input data in the standard space.
        """

    @abstractmethod
    def _differentiate_transformation(self, input_data: RealArray) -> RealArray:
        """Differentiate the transformation from the input space to the standard space.

        Args:
            input_data: The input data.

        Returns:
            The diagonal of the Jacobian matrix of the transformation.
        """

    def compute_output_data(self, input_data: RealArray) -> RealArray:
        """Evaluate the basis functions at different input points.

        Args:
            input_data: The input points, shaped as ``(n_samples, input_dimension)``.

        Returns:
            The output data, shaped as ``(n_samples, n_basis_functions)``.
        """
        transformed_input_data = self._evaluate_transformation(input_data)
        return hstack([
            array(basis_function(transformed_input_data))
            for basis_function in self._basis_functions
        ])

    def compute_jacobian_data(self, input_data: RealArray) -> RealArray:
        """Evaluate the derivatives of the basis functions at different input points.

        Args:
            input_data: The input points, shaped as ``(n_samples, input_dimension)``.

        Returns:
            The Jacobian data,
            shaped as ``(n_samples, input_dimension, n_basis_functions)``.
        """
        transformed_input_data = self._evaluate_transformation(input_data)
        transformed_input_data_jac = self._differentiate_transformation(input_data)
        return transformed_input_data_jac[:, :, newaxis] * stack(
            [
                hstack([
                    array(basis_function_gradient(x))
                    for basis_function_gradient in self._basis_function_gradients
                ])
                for x in transformed_input_data
            ],
        )
