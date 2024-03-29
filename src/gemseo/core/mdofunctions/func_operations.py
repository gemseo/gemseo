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
"""The functional operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import delete
from numpy import insert

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.typing import RealArray


class RestrictedFunction(MDOFunction):
    """Restrict an MDOFunction to a subset of its input vector.

    Fixes the rest of the indices.
    """

    def __init__(
        self,
        orig_function: MDOFunction,
        restriction_indices: Sequence[int],
        restriction_values: RealArray,
    ) -> None:
        """
        Args:
            orig_function: The original function to restrict.
            restriction_indices: The indices array of the input vector to fix.
            restriction_values: The values of the input vector at the indices,
                'restriction_indices' are set to 'restriction_values'.

        Raises:
            ValueError: If the shape of the restriction values is not consistent
                with the shape of the restriction indices.
        """  # noqa: D205, D212, D415
        if restriction_indices.shape != restriction_values.shape:
            msg = "Inconsistent shapes for restriction values and indices."
            raise ValueError(msg)
        self.restriction_values = restriction_values
        self._restriction_indices = restriction_indices
        self._orig_function = orig_function
        super().__init__(
            self._func_to_wrap,
            f"{orig_function.name}_restr",
            jac=self._jac_to_wrap,
            f_type=orig_function.f_type,
            expr=orig_function.expr,
            input_names=orig_function.input_names,
            dim=orig_function.dim,
            output_names=orig_function.output_names,
            original_name=orig_function.original_name,
        )

    def _func_to_wrap(self, x_vect: RealArray) -> RealArray:
        """Wrap the provided function in order to be given to the optimizer.

        Args:
            x_vect: The design variables values.

        Returns:
            The evaluation of the function at x_vect.
        """
        x_full = insert(x_vect, self._restriction_indices, self.restriction_values)
        return self._orig_function(x_full)

    def _jac_to_wrap(self, x_vect: RealArray) -> RealArray:
        """Wrap the provided Jacobian in order to be given to the optimizer.

        Args:
            x_vect: The design variables values.

        Returns:
            The evaluation of the Jacobian at x_vect.
        """
        x_full = insert(x_vect, self._restriction_indices, self.restriction_values)
        jac = self._orig_function.jac(x_full)
        return delete(jac, self._restriction_indices, axis=0)


# TODO: API: move to a specific module
class LinearComposition(MDOFunction):
    r"""Linear composite function.

    Given a matrix :math:`A`, a function :math:`f` and an input vector :math:`x`,
    the linear composite function outputs :math:`f(Ax)`.
    """

    # TODO: API: rename orig_function to function.
    # TODO: API: rename interp_operator to matrix.
    def __init__(
        self,
        orig_function: MDOFunction,
        interp_operator: RealArray,
    ) -> None:
        r"""
        Args:
            orig_function: The function :math:`f`.
            interp_operator: The matrix :math:`A`.
        """  # noqa: D205, D212, D415
        self._orig_function = orig_function
        self._interp_operator = interp_operator
        self._orig_function = orig_function
        # TODO: API: Rename function name to "[f o A]"
        input_names = orig_function.input_names
        if len(input_names) == 1:
            x = orig_function.input_names[0]
        else:
            x = f"({pretty_str(orig_function.input_names)})'"
        super().__init__(
            self._restricted_function,
            str(orig_function.name) + "_comp",
            jac=self._restricted_jac,
            f_type=orig_function.f_type,
            expr=f"{orig_function.name}(A.{x})",
            input_names=orig_function.input_names,
            dim=orig_function.dim,
            output_names=orig_function.output_names,
        )

    def _restricted_function(self, x_vect: RealArray) -> RealArray:
        """Wrap the provided function in order to be given to the optimizer.

        Args:
            x_vect: The design variable values.

        Returns:
            The evaluation of the function at x_vect.
        """
        x_full = self._interp_operator.dot(x_vect)
        return self._orig_function(x_full)

    def _restricted_jac(self, x_vect: RealArray) -> RealArray:
        """Wrap the provided Jacobian in order to be given to the optimizer.

        Args:
            x_vect: The design variable values.

        Returns:
            The evaluation of the function at x_vect.
        """
        x_full = self._interp_operator.dot(x_vect)
        jac = self._orig_function.jac(x_full)
        return self._interp_operator.T.dot(jac)
