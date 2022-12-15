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
"""A linear function defined from coefficients and offset matrices."""
from __future__ import annotations

from numbers import Number
from typing import Any
from typing import Sequence

from numpy import array
from numpy import atleast_2d
from numpy import matmul
from numpy import ndarray

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_function import OutputType


class MDOLinearFunction(MDOFunction):
    r"""Linear multivariate function defined by.

    * a matrix :math:`A` of first-order coefficients
      :math:`(a_{ij})_{\substack{i = 1, \dots m \\ j = 1, \dots n}}`
    * and a vector :math:`b` of zero-order coefficients :math:`(b_i)_{i = 1, \dots m}`

    .. math::
        F(x)
        =
        Ax + b
        =
        \begin{bmatrix}
            a_{11} & \cdots & a_{1n} \\
            \vdots & \ddots & \vdots \\
            a_{m1} & \cdots & a_{mn}
        \end{bmatrix}
        \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix}
        +
        \begin{bmatrix} b_1 \\ \vdots \\ b_m \end{bmatrix}.
    """

    def __init__(
        self,
        coefficients: ArrayType,
        name: str,
        f_type: str | None = None,
        args: Sequence[str] | None = None,
        value_at_zero: OutputType = 0.0,
        output_names: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            coefficients: The coefficients :math:`A` of the linear function.
            name: The name of the linear function.
            f_type: The type of the linear function among
                :attr:`.MDOFunction.AVAILABLE_TYPES`.
                If ``None``, the linear function will have no type.
            args: The names of the inputs of the linear function.
                If ``None``, the inputs of the linear function will have no names.
            value_at_zero: The value :math:`b` of the linear function output at zero.
            output_names: The names of the outputs of the function.
                If ``None``, the outputs of the function will have no names.
        """  # noqa: D205, D212, D415
        # Format the passed coefficients and value at zero
        self.coefficients = coefficients
        output_dim, input_dim = self._coefficients.shape
        self.value_at_zero = value_at_zero

        # Generate the arguments strings
        new_args = self.__class__.generate_args(input_dim, args)

        # Generate the expression string
        if output_dim == 1:
            expr = self._generate_1d_expr(new_args)
        else:
            expr = self._generate_nd_expr(new_args)

        super().__init__(
            self._func_to_wrap,
            name,
            f_type=f_type,
            jac=self._jac_to_wrap,
            expr=expr,
            args=new_args,
            dim=output_dim,
            outvars=output_names,
        )

    def _func_to_wrap(self, x_vect: ArrayType) -> OutputType:
        """Return the linear combination with an offset.

        :math:`sum_{i=1}^n a_i * x_i + b`

        Args:
            x_vect: The design variables values.
        """
        value = matmul(self._coefficients, x_vect) + self._value_at_zero
        if value.size == 1:
            value = value[0]
        return value

    def _jac_to_wrap(self, _: Any) -> ArrayType:
        """Set and return the coefficients.

        If the function is scalar, the gradient of the function is returned as a
        1d-array. If the function is vectorial, the Jacobian of the function is
        returned as a 2d-array.

        Args:
            _: This argument is not used.
        """
        if self._coefficients.shape[0] == 1:
            return self._coefficients[0, :]
        return self._coefficients

    @property
    def coefficients(self) -> ArrayType:
        """The coefficients of the linear function.

        This is the matrix :math:`A` in the expression :math:`y=Ax+b`.

        Raises:
            ValueError: If the coefficients are not passed
                as a 1-dimensional or 2-dimensional ndarray.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients: Number | ArrayType) -> None:
        if isinstance(coefficients, Number):
            self._coefficients = atleast_2d(coefficients)
        elif isinstance(coefficients, ndarray) and coefficients.ndim == 2:
            self._coefficients = coefficients
        elif isinstance(coefficients, ndarray) and coefficients.ndim == 1:
            self._coefficients = coefficients.reshape((1, -1))
        else:
            raise ValueError(
                "Coefficients must be passed as a 2-dimensional "
                "or a 1-dimensional ndarray."
            )

    @property
    def value_at_zero(self) -> ArrayType:
        """The value of the function at zero.

        This is the vector :math:`b` in the expression :math:`y=Ax+b`.

        Raises:
            ValueError: If the value at zero is neither a ndarray nor a number.
        """
        return self._value_at_zero

    @value_at_zero.setter
    def value_at_zero(self, value_at_zero: OutputType) -> None:
        output_dim = self.coefficients.shape[0]  # N.B. the coefficients must be set
        if isinstance(value_at_zero, ndarray) and value_at_zero.size == output_dim:
            self._value_at_zero = value_at_zero.reshape(output_dim)
        elif isinstance(value_at_zero, Number):
            self._value_at_zero = array([value_at_zero] * output_dim)
        else:
            raise ValueError("Value at zero must be an ndarray or a number.")

    def _generate_1d_expr(self, args: Sequence[str]) -> str:
        """Generate the literal expression of the linear function in scalar form.

        Args:
            args: The names of the inputs of the function.

        Returns:
            The literal expression of the linear function in scalar form.
        """
        pattern = self.COEFF_FORMAT_1D
        strings = []
        # Build the expression of the linear combination
        first_non_zero_index = -1
        for index, coefficient in enumerate(self._coefficients[0, :]):
            if coefficient != 0.0:
                if first_non_zero_index == -1:
                    first_non_zero_index = index
                # Add the monomial sign
                if index == first_non_zero_index and coefficient < 0.0:
                    # The first nonzero coefficient is negative.
                    strings.append("-")  # unary minus
                elif index != first_non_zero_index and coefficient < 0.0:
                    strings.append(" - ")
                elif index != first_non_zero_index and coefficient > 0.0:
                    strings.append(" + ")
                # Add the coefficient value
                if abs(coefficient) != 1.0:
                    strings.append(f"{pattern.format(abs(coefficient))}*")
                # Add argument string
                strings.append(args[index])

        # Add the offset expression
        value_at_zero = pattern.format(self._value_at_zero[0])
        if first_non_zero_index == -1:
            # Constant function
            strings.append(value_at_zero)
        elif self._value_at_zero > 0.0:
            strings.append(f" + {value_at_zero}")
        elif self._value_at_zero < 0.0:
            strings.append(f" - {value_at_zero}")

        return "".join(strings)

    def _generate_nd_expr(self, args: Sequence[str]) -> str:
        """Generate the literal expression of the linear function in matrix form.

        Args:
            args: The names of the inputs of the function.

        Returns:
            The literal expression of the linear function in matrix form.
        """
        max_args_len = max(len(arg) for arg in args)
        out_dim, in_dim = self._coefficients.shape
        strings = []
        for i in range(max(out_dim, in_dim)):
            if i > 0:
                strings.append("\n")
            # matrix line
            if i < out_dim:
                coefficients = (
                    self.COEFF_FORMAT_ND.format(coefficient)
                    for coefficient in self._coefficients[i, :]
                )
                strings.append(f"[{' '.join(coefficients)}]")
            else:
                strings.append(" " + " ".join([" " * 3] * in_dim) + " ")
            # vector line
            strings.append(f"[{args[i]}]" if i < in_dim else " " * (max_args_len + 2))
            # sign
            strings.append(" + " if i == 0 else "   ")
            # value at zero
            if i < out_dim:
                strings.append(
                    f"[{self.COEFF_FORMAT_ND.format(self._value_at_zero[i])}]"
                )
        return "".join(strings)

    def __neg__(self) -> MDOLinearFunction:  # noqa:D102
        return self.__class__(
            -self._coefficients,
            f"-{self.name}",
            self.f_type,
            self.args,
            -self._value_at_zero,
        )

    def offset(self, value: OutputType) -> MDOLinearFunction:  # noqa:D102
        return self.__class__(
            self._coefficients,
            self.name,
            self.f_type,
            self.args,
            self._value_at_zero + value,
        )

    def restrict(
        self, frozen_indexes: ndarray[int], frozen_values: ArrayType
    ) -> MDOLinearFunction:
        """Build a restriction of the linear function.

        Args:
            frozen_indexes: The indexes of the inputs that will be frozen.
            frozen_values: The values of the inputs that will be frozen.

        Returns:
            The restriction of the linear function.

        Raises:
            ValueError: If the frozen indexes and values have different shapes.
        """
        if frozen_indexes.shape != frozen_values.shape:
            raise ValueError(
                "Arrays of frozen indexes and values must have same shape."
            )

        frozen_coefficients = self.coefficients[:, frozen_indexes]
        active_indexes = array(
            [
                index
                for index in range(self.coefficients.shape[1])
                if index not in frozen_indexes
            ]
        )
        return self.__class__(
            self.coefficients[:, active_indexes],
            f"{self.name}_restriction",
            args=[self.args[i] for i in active_indexes],
            value_at_zero=(
                matmul(frozen_coefficients, frozen_values) + self._value_at_zero
            ),
        )
