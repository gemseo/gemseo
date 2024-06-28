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
"""A function with integer inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo.algos.preprocessed_functions.base_preprocessed_function import (
    BasePreprocessedFunction,
)

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.core.mdofunctions.mdo_function import MDOFunction


class IntFunction(BasePreprocessedFunction):
    """A function with integer inputs."""

    __round_vect: Callable[[ndarray, bool], ndarray]
    """A function to round the integer components of a real vector."""

    def __init__(
        self, function: MDOFunction, round_vect: Callable[[ndarray, bool], ndarray]
    ) -> None:
        """
        Args:
            round_vect: A function to round the integer components of a real vector.
        """  # noqa: D205, D212, D415
        super().__init__(function)
        self.__round_vect = round_vect
        self.expects_normalized_inputs = function.expects_normalized_inputs

    def _compute_output(self, x_vect: ndarray) -> ndarray:
        return self._original_func(self.__round_vect(x_vect))

    def _compute_jacobian(self, x_vect: ndarray) -> ndarray:
        return self._original_jac(self.__round_vect(x_vect))
