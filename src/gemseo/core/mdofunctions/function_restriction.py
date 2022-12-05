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

from typing import Sequence

from numpy import array
from numpy import empty
from numpy import ndarray

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction


class FunctionRestriction(MDOFunction):
    """Take an :class:`.MDOFunction` and apply a given restriction to its inputs."""

    def __init__(
        self,
        frozen_indexes: ndarray[int],
        frozen_values: ArrayType,
        input_dim: int,
        mdo_function: MDOFunction,
        name: str | None = None,
        f_type: str | None = None,
        expr: str | None = None,
        args: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            frozen_indexes: The indexes of the inputs that will be frozen
            frozen_values: The values of the inputs that will be frozen.
            input_dim: The dimension of input space of the function before restriction.
            name: The name of the function after restriction.
                If ``None``,
                create a default name
                based on the name of the current function
                and on the argument `args`.
            mdo_function: The function to restrict.
            f_type: The type of the function after restriction.
                If ``None``, the function will have no type.
            expr: The expression of the function after restriction.
                If ``None``, the function will have no expression.
            args: The names of the inputs of the function after restriction.
                If ``None``, the inputs of the function will have no names.

        Raises:
            ValueError: If the `frozen_indexes` and the `frozen_values` arrays do
                not have the same shape.
        """  # noqa: D205, D212, D415
        # Check the shapes of the passed arrays
        if frozen_indexes.shape != frozen_values.shape:
            raise ValueError("Arrays of frozen indexes and values must have same shape")

        self.__frozen_indexes = frozen_indexes
        self.__frozen_values = frozen_values
        self.__input_dim = input_dim
        self.__mdo_function = mdo_function
        self.__name = name
        self.__f_type = f_type
        self.__expr = expr
        self.__args = args

        self._active_indexes = array(
            [i for i in range(self.__input_dim) if i not in self.__frozen_indexes]
        )

        # Build the name of the restriction
        if self.__name is None and self.__args is not None:
            self.__name = "{}_wrt_{}".format(
                self.__mdo_function.name, "_".join(self.__args)
            )
        elif name is None:
            self.__name = f"{self.__mdo_function.name}_restriction"

        if self.__mdo_function.has_jac():
            jac = self._jac_to_wrap
        else:
            jac = self.__mdo_function.jac

        super().__init__(
            self._func_to_wrap,
            self.__name,
            self.__f_type,
            expr=self.__expr,
            args=self.__args,
            jac=jac,
            dim=self.__mdo_function.dim,
            outvars=self.__mdo_function.outvars,
            force_real=self.__mdo_function.force_real,
        )

    def __extend_subvect(self, x_subvect: ArrayType) -> ArrayType:
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

    def _func_to_wrap(self, x_subvect: ArrayType) -> ArrayType:
        """Evaluate the restriction.

        Args:
            x_subvect: The value of the inputs of the restriction.

        Returns:
            The value of the outputs of the restriction.
        """
        return self.__mdo_function.evaluate(self.__extend_subvect(x_subvect))

    def _jac_to_wrap(self, x_subvect: ArrayType) -> ArrayType:
        """Compute the Jacobian matrix of the restriction.

        Args:
            x_subvect: The value of the inputs of the restriction.

        Returns:
            The Jacobian matrix of the restriction.
        """
        return self.__mdo_function.jac(self.__extend_subvect(x_subvect))[
            ..., self._active_indexes
        ]
