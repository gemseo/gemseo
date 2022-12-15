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

from numpy import delete
from numpy import insert
from numpy import ndarray

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction


class RestrictedFunction(MDOFunction):
    """Restrict an MDOFunction to a subset of its input vector.

    Fixes the rest of the indices.
    """

    def __init__(
        self,
        orig_function: MDOFunction,
        restriction_indices: ndarray,
        restriction_values: ndarray,
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
        if not restriction_indices.shape == restriction_values.shape:
            raise ValueError("Inconsistent shapes for restriction values and indices.")
        self.restriction_values = restriction_values
        self._restriction_indices = restriction_indices
        self._orig_function = orig_function
        super().__init__(
            self._func_to_wrap,
            f"{orig_function.name}_restr",
            jac=self._jac_to_wrap,
            f_type=orig_function.f_type,
            expr=orig_function.expr,
            args=orig_function.args,
            dim=orig_function.dim,
            outvars=orig_function.outvars,
        )

    def _func_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Wrap the provided function in order to be given to the optimizer.

        Args:
            x_vect: The design variables values.

        Returns:
            The evaluation of the function at x_vect.
        """
        x_full = insert(x_vect, self._restriction_indices, self.restriction_values)
        return self._orig_function(x_full)

    def _jac_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Wrap the provided Jacobian in order to be given to the optimizer.

        Args:
            x_vect: The design variables values.

        Returns:
            The evaluation of the Jacobian at x_vect.
        """
        x_full = insert(x_vect, self._restriction_indices, self.restriction_values)
        jac = self._orig_function.jac(x_full)
        jac = delete(jac, self._restriction_indices, axis=0)
        return jac


class LinearComposition(MDOFunction):
    """Compose a function with a linear operator defined by a matrix.

    Compute orig_f(Mat.dot(x)).
    """

    def __init__(
        self,
        orig_function: MDOFunction,
        interp_operator: ndarray,
    ):
        """
        Args:
            orig_function: The original function to be restricted.
            interp_operator: The operator matrix, the output of the
                function will be f(interp_operator.dot(x)).
        """  # noqa: D205, D212, D415
        self._orig_function = orig_function
        self._interp_operator = interp_operator
        self._orig_function = orig_function
        super().__init__(
            self._restricted_function,
            str(orig_function.name) + "_comp",
            jac=self._restricted_jac,
            f_type=orig_function.f_type,
            expr="Mat*" + str(orig_function.expr),
            args=orig_function.args,
            dim=orig_function.dim,
            outvars=orig_function.outvars,
        )

    def _restricted_function(self, x_vect: ndarray) -> MDOFunction:
        """Wrap the provided function in order to be given to the optimizer.

        Args:
            x_vect: The design variable values.

        Returns:
            The evaluation of the function at x_vect.
        """
        x_full = self._interp_operator.dot(x_vect)
        return self._orig_function(x_full)

    def _restricted_jac(self, x_vect: ndarray) -> MDOFunction.jac:
        """Wrap the provided Jacobian in order to be given to the optimizer.

        Args:
            x_vect: The design variable values.

        Returns:
            The evaluation of the function at x_vect.
        """
        x_full = self._interp_operator.dot(x_vect)
        jac = self._orig_function.jac(x_full)
        return self._interp_operator.T.dot(jac)
