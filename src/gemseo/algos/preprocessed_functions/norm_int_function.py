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
"""A function with normalized inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo.algos.preprocessed_functions.base_preprocessed_function import (
    BasePreprocessedFunction,
)
from gemseo.algos.preprocessed_functions.norm_function_mixin import NormFunctionMixin

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.utils.compatibility.scipy import ArrayType


class NormIntFunction(NormFunctionMixin, BasePreprocessedFunction):
    """A function with normalized inputs."""

    __normalize_grad: Callable[[ArrayType], ArrayType]
    """Normalize an unnormalized gradient."""

    __unnormalize_vect: Callable[[ArrayType, bool, bool, ArrayType | None], ArrayType]
    """Unnormalize a normalized vector of the design space."""

    __round_vect: Callable[[ndarray, bool], ndarray]
    """Round the vector where variables are of integer type."""

    def __init__(self, function: MDOFunction, design_space: DesignSpace) -> None:
        """
        Args:
            design_space: The design space on which to evaluate the function.
        """  # noqa: D205, D212, D415
        super().__init__(function)
        self.__unnormalize_vect = design_space.unnormalize_vect
        self.__normalize_grad = design_space.normalize_grad
        self.__round_vect = design_space.round_vect

    def _compute_output(self, x_vect: ndarray) -> ndarray:
        return self._original_func(self.__round_vect(self.__unnormalize_vect(x_vect)))

    def _compute_jacobian(self, x_vect: ndarray) -> ndarray:
        return self.__normalize_grad(
            self._original_jac(self.__round_vect(self.__unnormalize_vect(x_vect)))
        )
