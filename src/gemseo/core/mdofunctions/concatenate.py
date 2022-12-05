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
"""A function evaluating other functions and concatenating their outputs."""
from __future__ import annotations

from typing import Iterable

from numpy import atleast_1d
from numpy import atleast_2d
from numpy import concatenate
from numpy import vstack

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction


class Concatenate(MDOFunction):
    """Wrap the concatenation of a set of functions."""

    def __init__(
        self, functions: Iterable[MDOFunction], name: str, f_type: str | None = None
    ) -> None:
        """
        Args:
            functions: The functions to be concatenated.
            name: The name of the concatenation function.
            f_type: The type of the concatenation function.
                If ``None``, the function will have no type.
        """  # noqa: D205, D212, D415
        self.__functions = functions
        func_output_names = [func.outvars for func in self.__functions]
        if [] in func_output_names:
            output_names = []
        else:
            output_names = [
                output_name
                for output_names in func_output_names
                for output_name in output_names
            ]

        super().__init__(
            self._func_to_wrap,
            name,
            f_type,
            self._jac_to_wrap,
            dim=sum(func.dim for func in self.__functions),
            outvars=output_names,
        )

    def _func_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Concatenate the values of the outputs of the functions.

        Args:
            x_vect: The value of the inputs of the functions.

        Returns:
            The concatenation of the values of the outputs of the functions.
        """
        return concatenate([atleast_1d(func(x_vect)) for func in self.__functions])

    def _jac_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Concatenate the outputs of the Jacobian functions.

        Args:
            x_vect: The value of the inputs of the Jacobian functions.

        Returns:
            The concatenation of the outputs of the Jacobian functions.
        """
        return vstack([atleast_2d(func.jac(x_vect)) for func in self.__functions])
