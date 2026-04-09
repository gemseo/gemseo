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
"""A function mapping another one from some input components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import empty

from gemseo.core.functions.array_function import ArrayFunction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

    from gemseo.typing import NumberArray


class FunctionRestriction(ArrayFunction):
    """Take an [ArrayFunction][gemseo.core.functions.array_function.ArrayFunction] and apply a given restriction to its inputs."""  # noqa: E501

    def __init__(
        self,
        frozen_indexes: ndarray[int],
        frozen_values: NumberArray,
        input_dim: int,
        function: ArrayFunction,
        name: str = "",
        f_type: ArrayFunction.FunctionType = ArrayFunction.FunctionType.NONE,
        expr: str = "",
        input_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            frozen_indexes: The indexes of the inputs that will be frozen
            frozen_values: The values of the inputs that will be frozen.
            input_dim: The dimension of input space of the function before restriction.
            name: The name of the function after restriction.
                If empty,
                create a default name
                based on the name of the current function
                and on the argument `input_names`.
            function: The original function.

        Raises:
            ValueError: If the `frozen_indexes` and the `frozen_values` arrays do
                not have the same shape.
        """  # noqa: D205, D212, D415
        # Check the shapes of the passed arrays
        if frozen_indexes.shape != frozen_values.shape:
            msg = "Arrays of frozen indexes and values must have same shape"
            raise ValueError(msg)

        self.__frozen_indexes = frozen_indexes
        self.__frozen_values = frozen_values
        self.__input_dim = input_dim
        self.__array_function = function
        self.__name = name
        self.__f_type = f_type
        self.__expr = expr
        self.__input_names = input_names

        self._active_indexes = array([
            i for i in range(self.__input_dim) if i not in self.__frozen_indexes
        ])

        # Build the name of the restriction
        if self.__name is None and self.__input_names is not None:
            self.__name = "{}_wrt_{}".format(
                self.__array_function.name, "_".join(self.__input_names)
            )
        elif name is None:
            self.__name = f"{self.__array_function.name}_restriction"

        if self.__array_function.has_jac:
            jac = self._jac_to_wrap
        else:
            jac = self.__array_function.jac

        super().__init__(
            self._func_to_wrap,
            self.__name,
            self.__f_type,
            expr=self.__expr,
            input_names=self.__input_names,
            jac=jac,
            dim=self.__array_function.dim,
            output_names=self.__array_function.output_names,
            force_real=self.__array_function.force_real,
            original_name=function.original_name,
        )

    def __extend_subvect(self, x_subvect: NumberArray) -> NumberArray:
        """Extend an input vector of the restriction with the frozen values.

        Args:
            x_subvect: The values of the inputs of the restriction.

        Returns:
            The extended input vector.
        """
        x_vect = empty(self.__input_dim)
        x_vect[self._active_indexes] = x_subvect
        x_vect[self.__frozen_indexes] = self.__frozen_values
        return x_vect

    def _func_to_wrap(self, x_subvect: NumberArray) -> NumberArray:
        """Evaluate the restriction.

        Args:
            x_subvect: The value of the inputs of the restriction.

        Returns:
            The value of the outputs of the restriction.
        """
        return self.__array_function.func(self.__extend_subvect(x_subvect))

    def _jac_to_wrap(self, x_subvect: NumberArray) -> NumberArray:
        """Compute the Jacobian matrix of the restriction.

        Args:
            x_subvect: The value of the inputs of the restriction.

        Returns:
            The Jacobian matrix of the restriction.
        """
        return self.__array_function.jac(self.__extend_subvect(x_subvect))[
            ..., self._active_indexes
        ]
