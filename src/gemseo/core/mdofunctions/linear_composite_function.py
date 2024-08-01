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
"""Linear composite function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class LinearCompositeFunction(MDOFunction):
    r"""Linear composite function.

    Given a matrix :math:`A`, a function :math:`f` and an input vector :math:`x`,
    the linear composite function outputs :math:`f(Ax)`.
    """

    _function: MDOFunction
    r"""The function :math:`f`."""

    _matrix: RealArray
    r"""The matrix :math:`A`."""

    def __init__(
        self,
        function: MDOFunction,
        matrix: RealArray,
    ) -> None:
        r"""
        Args:
            function: The function :math:`f`.
            matrix: The matrix :math:`A`.
        """  # noqa: D205, D212, D415
        self._function = function
        self._matrix = matrix
        input_names = function.input_names
        if len(input_names) == 1:
            x = function.input_names[0]
        else:
            x = f"({pretty_str(function.input_names)})'"

        super().__init__(
            self._restricted_function,
            f"[{function.name} o A]",
            jac=self._restricted_jac,
            f_type=function.f_type,
            expr=f"{function.name}(A.{x})",
            input_names=function.input_names,
            dim=function.dim,
            output_names=function.output_names,
        )

    def _restricted_function(self, x_vect: RealArray) -> RealArray:
        """Wrap the provided function in order to be given to the optimizer.

        Args:
            x_vect: The design variable values.

        Returns:
            The evaluation of the function at x_vect.
        """
        return self._function.evaluate(self._matrix.dot(x_vect))

    def _restricted_jac(self, x_vect: RealArray) -> RealArray:
        """Wrap the provided Jacobian in order to be given to the optimizer.

        Args:
            x_vect: The design variable values.

        Returns:
            The evaluation of the function at x_vect.
        """
        return self._matrix.T.dot(self._function.jac(self._matrix.dot(x_vect)))
