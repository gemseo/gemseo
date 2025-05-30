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
"""Function restricted to a subset of input components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import delete
from numpy import insert

from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.typing import IntegerArray
    from gemseo.typing import RealArray


class RestrictedFunction(MDOFunction):
    """Function restricted to a subset of input components.

    The rest of the input component are fixed.
    """

    _function: MDOFunction
    """The original function to restrict."""

    _restriction_indices: Sequence[int]
    """The indices array of the input vector to fix."""

    restriction_values: RealArray
    """The values of the input vector at the restriction indices."""

    def __init__(
        self,
        function: MDOFunction,
        restriction_indices: IntegerArray,
        restriction_values: RealArray,
    ) -> None:
        """
        Args:
            function: The original function to restrict.
            restriction_indices: The indices array of the input vector to fix.
            restriction_values: The values of the input vector
                at the restriction indices.

        Raises:
            ValueError: If the shape of the restriction values is not consistent
                with the shape of the restriction indices.
        """  # noqa: D205, D212, D415
        if restriction_indices.shape != restriction_values.shape:
            msg = "Inconsistent shapes for restriction values and indices."
            raise ValueError(msg)
        self.restriction_values = restriction_values
        self._restriction_indices = restriction_indices
        self._function = function
        super().__init__(
            self._func_to_wrap,
            f"{function.name}_restr",
            jac=self._jac_to_wrap,
            f_type=function.f_type,
            expr=function.expr,
            input_names=function.input_names,
            dim=function.dim,
            output_names=function.output_names,
            original_name=function.original_name,
        )

    def _func_to_wrap(self, x_vect: RealArray) -> RealArray:
        """Wrap the provided function in order to be given to the optimizer.

        Args:
            x_vect: The design variables values.

        Returns:
            The evaluation of the function at x_vect.
        """
        x_full = insert(x_vect, self._restriction_indices, self.restriction_values)
        return self._function.evaluate(x_full)

    def _jac_to_wrap(self, x_vect: RealArray) -> RealArray:
        """Wrap the provided Jacobian in order to be given to the optimizer.

        Args:
            x_vect: The design variables values.

        Returns:
            The evaluation of the Jacobian at x_vect.
        """
        x_full = insert(x_vect, self._restriction_indices, self.restriction_values)
        jac = self._function.jac(x_full)
        return delete(jac, self._restriction_indices, axis=0)
