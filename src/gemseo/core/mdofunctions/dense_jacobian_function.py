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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Alexandre Scotto Di Perrotolo
#    OTHER AUTHORS - MACROSCOPIC CHANGES
#        :author: Matthias De Lozzo
"""A MDOFunction wrapper casting Jacobians as dense NumPy arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_function import OutputType
from gemseo.utils.compatibility.scipy import sparse_classes

if TYPE_CHECKING:
    from numpy import ndarray


class DenseJacobianFunction(MDOFunction):
    """A wrapper of :class:`.MDOFunction` casting Jacobians as dense NumPy arrays."""

    __original_function: MDOFunction
    """The wrapped function."""

    __evaluate_original_function: Callable[[ArrayType], OutputType]
    """The wrapped function evaluation callable."""

    def __init__(
        self,
        original_function: MDOFunction,
    ) -> None:
        """
        Args:
            original_function: The original function which is wrapped.
        """  # noqa: D205, D212, D415
        self.__original_function = original_function
        self.__evaluate_original_function = self.__original_function.evaluate

        super().__init__(
            self.__evaluate_original_function,
            name=original_function.name,
            jac=self._jac_to_wrap,
            f_type=original_function.f_type,
            expr=original_function.expr,
            input_names=original_function.input_names,
            dim=original_function.dim,
            output_names=original_function.output_names,
            special_repr=original_function.special_repr,
            original_name=original_function.original_name,
        )

    def _jac_to_wrap(
        self,
        x_vect: ndarray,
    ) -> ndarray:
        """Evaluate the gradient of the original function.

        Args:
            x_vect: An input vector.

        Returns:
            The value of the gradient of the original function at this input vector.

        Raises:
            ValueError: If the original function does not provide a Jacobian matrix.
        """
        if not self.__original_function.has_jac:
            raise ValueError(
                f"Selected user gradient, but function {self.__original_function} "
                "has no Jacobian matrix."
            )

        original_jacobian = self.__original_function.jac(x_vect)
        if isinstance(original_jacobian, sparse_classes):
            return original_jacobian.todense()

        return original_jacobian
