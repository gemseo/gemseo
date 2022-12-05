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
"""A function computing the convex linearization of another one."""
from __future__ import annotations

from numpy import absolute
from numpy import atleast_2d
from numpy import matmul
from numpy import multiply
from numpy import ndarray
from numpy import ones_like
from numpy import where
from numpy import zeros_like

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction


class ConvexLinearApprox(MDOFunction):
    """Wrap a convex linearization of the function."""

    def __init__(
        self,
        x_vect: ArrayType,
        mdo_function: MDOFunction,
        approx_indexes: ndarray[bool] | None = None,
        sign_threshold: float = 1e-9,
    ) -> None:
        """
        Args:
            x_vect: The input vector at which to build the convex linearization.
            mdo_function: The function to approximate.
            approx_indexes: A boolean mask
                specifying w.r.t. which inputs the function should be approximated.
                If ``None``, consider all the inputs.
            sign_threshold: The threshold for the sign of the derivatives.

        Raises:
            ValueError: If the length of boolean array
                and the number of inputs of the functions are inconsistent.
            AttributeError: If the function does not have a Jacobian function.
        """  # noqa: D205, D212, D415
        self.__x_vect = x_vect
        self.__mdo_function = mdo_function
        self.__approx_indexes = approx_indexes
        self.__sign_threshold = sign_threshold

        # Check the approximation indexes
        if self.__approx_indexes is None:
            self.__approx_indexes = ones_like(x_vect, dtype=bool)
        elif (
            self.__approx_indexes.shape != self.__x_vect.shape
            or self.__approx_indexes.dtype != "bool"
        ):
            raise ValueError(
                "The approximation array must be an array of booleans with "
                "the same shape as the function argument."
            )

        # Get the function Jacobian matrix
        if not self.__mdo_function.has_jac():
            raise AttributeError(
                "Function Jacobian unavailable for convex linearization."
            )

        jac = atleast_2d(self.__mdo_function.jac(x_vect))

        # Build the coefficients matrices
        coeffs = jac[:, self.__approx_indexes]
        self.__direct_coeffs = where(coeffs > self.__sign_threshold, coeffs, 0.0)
        self.__recipr_coeffs = multiply(
            -where(-coeffs > self.__sign_threshold, coeffs, 0.0),
            self.__x_vect[self.__approx_indexes] ** 2,
        )

        super().__init__(
            self._func_to_wrap,
            f"{self.__mdo_function.name}_convex_lin",
            self.__mdo_function.f_type,
            self._jac_to_wrap,
            dim=self.__mdo_function.dim,
            outvars=self.__mdo_function.outvars,
            force_real=self.__mdo_function.force_real,
        )

    def __get_steps(self, x_new: ArrayType) -> tuple[ArrayType, ArrayType]:
        """Return the steps on the direct and reciprocal variables.

        Args:
            x_new: The value of the inputs of the convex linearization function.

        Returns:
            Both the step on the direct variables
            and the step on the reciprocal variables.
        """
        step = x_new[self.__approx_indexes] - self.__x_vect[self.__approx_indexes]
        inv_step = zeros_like(step)
        nonzero_indexes = (absolute(step) > self.__sign_threshold).nonzero()
        inv_step[nonzero_indexes] = 1.0 / step[nonzero_indexes]
        return step, inv_step

    def _func_to_wrap(self, x_new: ArrayType) -> ArrayType:
        """Return the value of the convex linearization function.

        Args:
            x_new: The value of the inputs of the convex linearization function.

        Returns:
            The value of the outputs of the convex linearization function.
        """
        merged_vect = where(self.__approx_indexes, self.__x_vect, x_new)
        step, inv_step = self.__get_steps(x_new)
        value = (
            self.__mdo_function.evaluate(merged_vect)
            + matmul(self.__direct_coeffs, step)
            + matmul(self.__recipr_coeffs, inv_step)
        )
        if self.__mdo_function._dim == 1:
            return value[0]
        return value

    def _jac_to_wrap(self, x_new: ArrayType) -> ArrayType:
        """Return the Jacobian matrix of the convex linearization function.

        Args:
            x_new: The value of the inputs of the convex linearization function.

        Returns:
            The Jacobian matrix of the convex linearization function.
        """
        merged_vect = where(self.__approx_indexes, self.__x_vect, x_new)
        value = atleast_2d(self.__mdo_function.jac(merged_vect))
        _, inv_step = self.__get_steps(x_new)
        value[:, self.__approx_indexes] = self.__direct_coeffs + multiply(
            self.__recipr_coeffs, -(inv_step**2)
        )
        if self.__mdo_function._dim == 1:
            value = value[0, :]
        return value
