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
"""A quadratic function defined from coefficients and offset matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import ndarray
from numpy import zeros
from numpy import zeros_like

from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.mdo_function import OutputType
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.typing import NumberArray


class MDOQuadraticFunction(MDOFunction):
    r"""Scalar-valued quadratic multivariate function defined by.

    * a *square* matrix :math:`A` of second-order coefficients
      :math:`(a_{ij})_{\substack{i = 1, \dots n \\ j = 1, \dots n}}`
    * a vector :math:`b` of first-order coefficients :math:`(b_i)_{i = 1, \dots n}`
    * and a scalar zero-order coefficient :math:`c`

    .. math::

        f(x)
        =
        c
        +
        \sum_{i = 1}^n b_i \, x_i
        +
        \sum_{i = 1}^n \sum_{j = 1}^n a_{ij} \, x_i \, x_j.
    """

    def __init__(
        self,
        quad_coeffs: NumberArray,
        name: str,
        f_type: MDOFunction.FunctionType = MDOFunction.FunctionType.NONE,
        input_names: Sequence[str] = (),
        linear_coeffs: NumberArray | None = None,
        value_at_zero: OutputType = 0.0,
    ) -> None:
        """
        Args:
            quad_coeffs: The second-order coefficients.
            linear_coeffs: The first-order coefficients.
                If ``None``, the first-order coefficients will be zero.
            value_at_zero: The zero-order coefficient.
        """  # noqa: D205, D212, D415
        self._input_dim = 0
        self._quad_coeffs = array([])
        self.quad_coeffs = quad_coeffs  # sets the input dimension
        self._linear_part = MDOLinearFunction(zeros(self._input_dim), f"{name}_lin")
        new_input_names = self.generate_input_names(self._input_dim, input_names)

        # Build the first-order term
        if linear_coeffs is not None and linear_coeffs.size:
            self._linear_part.coefficients = linear_coeffs

        self._value_at_zero = value_at_zero

        super().__init__(
            self._func_to_wrap,
            name,
            f_type,
            self._jac_to_wrap,
            self.__build_expression(
                self._quad_coeffs,
                new_input_names,
                self.linear_coeffs,
                self._value_at_zero,
            ),
            input_names=new_input_names,
            dim=1,
        )

    def _func_to_wrap(self, x_vect: NumberArray) -> NumberArray:
        """Compute the output of the quadratic function.

        Args:
            x_vect: The value of the inputs of the quadratic function.

        Returns:
            The value of the quadratic function.
        """
        return (
            x_vect.T @ (self._quad_coeffs @ x_vect)
            + self._linear_part.func(x_vect)
            + self._value_at_zero
        )

    def _jac_to_wrap(self, x_vect: NumberArray) -> NumberArray:
        """Compute the gradient of the quadratic function.

        Args:
            x_vect: The value of the inputs of the quadratic function.

        Returns:
            The value of the gradient of the quadratic function.
        """
        return (
            self._quad_coeffs + self._quad_coeffs.T
        ) @ x_vect + self._linear_part.jac(x_vect)

    @property
    def quad_coeffs(self) -> NumberArray:
        """The second-order coefficients of the function.

        Raises:
            ValueError: If the coefficients are not passed
                as a 2-dimensional square ``ndarray``.
        """
        return self._quad_coeffs

    @quad_coeffs.setter
    def quad_coeffs(self, coefficients: NumberArray) -> None:
        # Check the second-order coefficients
        if (
            not isinstance(coefficients, ndarray)
            or len(coefficients.shape) != 2
            or coefficients.shape[0] != coefficients.shape[1]
        ):
            msg = (
                "Quadratic coefficients must be passed as a 2-dimensional "
                "square ndarray."
            )
            raise ValueError(msg)
        self._quad_coeffs = coefficients
        self._input_dim = self._quad_coeffs.shape[0]

    @property
    def linear_coeffs(self) -> NumberArray:
        """The first-order coefficients of the function.

        Raises:
            ValueError: If the number of first-order coefficients is not consistent
                with the dimension of the input space.
        """
        return self._linear_part.coefficients

    @linear_coeffs.setter
    def linear_coeffs(self, coefficients: NumberArray) -> None:
        if coefficients.size != self._input_dim:
            msg = (
                "The number of first-order coefficients must be equal "
                "to the input dimension."
            )
            raise ValueError(msg)
        self._linear_part.coefficients = coefficients

    @classmethod
    def __build_expression(
        cls,
        quad_coeffs: NumberArray,
        input_names: Sequence[str],
        linear_coeffs: NumberArray,
        value_at_zero: OutputType,
    ) -> str:
        """Build the expression of the quadratic function.

        Args:
            quad_coeffs: The second-order coefficients.
            input_names: The names of the inputs of the function.
            linear_coeffs: The first-order coefficients.
                If ``None``, the first-order coefficients will be zero.
            value_at_zero: The zero-order coefficient.

        Returns:
            The expression of the quadratic function.
        """
        transpose_str = "'"
        expr = ""
        for index, line in enumerate(quad_coeffs):
            arg = input_names[index]
            # Second-order expression
            line = quad_coeffs[index, :].tolist()
            expr += f"[{arg}]"
            expr += transpose_str if index == 0 else " "
            quad_coeffs_str = (cls.COEFF_FORMAT_ND.format(val) for val in line)
            expr += "[{}]".format(" ".join(quad_coeffs_str))
            expr += f"[{arg}]"
            # First-order expression
            if (
                linear_coeffs is not None
                and (linear_coeffs != zeros_like(linear_coeffs)).any()
            ):
                expr += " + " if index == 0 else "   "
                expr += f"[{cls.COEFF_FORMAT_ND.format(linear_coeffs[0, index])}]"
                expr += transpose_str if index == 0 else " "
                expr += f"[{arg}]"
            # Zero-order expression
            if value_at_zero != 0.0 and index == 0:
                sign_str = "+" if value_at_zero > 0.0 else "-"
                expr += (" {} " + cls.COEFF_FORMAT_ND).format(
                    sign_str, abs(value_at_zero)
                )
            if index < quad_coeffs.shape[0] - 1:
                expr += "\n"
        return expr
