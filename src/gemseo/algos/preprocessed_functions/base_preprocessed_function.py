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
"""A function preprocessing another function."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Callable

from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.utils.compatibility.scipy import ArrayType


class BasePreprocessedFunction(MDOFunction):
    """A function preprocessing another function."""

    _original_func: Callable[[ArrayType], ArrayType]
    """The original function to compute the output."""

    _original_jac: Callable[[ArrayType], ArrayType]
    """The original function to compute the Jacobian."""

    __has_jac: bool
    """Whether the original function has an implemented Jacobian function."""

    def __init__(self, function: MDOFunction) -> None:
        """
        Args:
            function: The original function.
        """  # noqa: D205, D212, D415
        self._original_func = function.evaluate
        self._original_jac = function.jac
        self.__has_jac = function.has_jac
        super().__init__(
            self._compute_output,
            name=function.name,
            jac=self._check_then_compute_jacobian,
            f_type=function.f_type,
            expr=function.expr,
            input_names=function.input_names,
            dim=function.dim,
            output_names=function.output_names,
            special_repr=function.special_repr,
            original_name=function.original_name,
        )

    @abstractmethod
    def _compute_output(self, x_vect: ndarray) -> ndarray:
        """The function computing the output.

        Args:
            x_vect: The input value.

        Returns:
            The output value.
        """

    @abstractmethod
    def _compute_jacobian(self, x_vect: ndarray) -> ndarray:
        """The function computing the Jacobian.

        Args:
            x_vect: The input value.

        Returns:
            The Jacobian value.
        """

    def _check_then_compute_jacobian(self, x_vect: ndarray) -> ndarray:
        """The function computing the Jacobian after checking it is possible.

        Args:
            x_vect: The input value.

        Returns:
            The Jacobian value.

        Raises:
            ValueError: When the original function has no Jacobian function.
        """
        if not self.__has_jac:
            msg = (
                "Selected user gradient "
                f"but function {self.name} has no Jacobian function."
            )
            raise ValueError(msg)

        return self._compute_jacobian(x_vect)
