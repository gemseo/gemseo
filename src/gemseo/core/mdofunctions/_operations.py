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
"""Some helpers for simple operations with functions."""
from __future__ import annotations

from abc import abstractmethod
from numbers import Number
from operator import mul
from operator import truediv
from typing import TYPE_CHECKING

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import add as _add
from numpy import ndarray
from numpy import subtract as _subtract

if TYPE_CHECKING:
    from gemseo.core.mdofunctions.mdo_function import ArrayType
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.core.mdofunctions.mdo_function import OperatorType
    from gemseo.core.mdofunctions.mdo_function import OutputType


class _OperationFunctionMaker(metaclass=GoogleDocstringInheritanceMeta):
    """A helper to create a function applying an operation to another function."""

    def __init__(
        self,
        cls: type[MDOFunction],
        first_operand: MDOFunction,
        second_operand: MDOFunction | Number,
        operator: OperatorType,
        operator_repr: str,
    ) -> None:
        """
        Args:
            cls: The type of :class:`.MDOFunction`.
            first_operand: The other function or number.
            second_operand: The operator as a function pointer.
            operator: The operator.
            operator_repr: The representation of the operator.

        Raises:
            TypeError: When the second operand is
                neither an :class:`.MDOFunction` nor a ``Number``.
        """  # noqa: D205, D212, D415
        f_type = ""
        expr = ""
        args = None
        jac = None
        self._first_operand = first_operand
        self._second_operand = second_operand
        self._second_operand_is_number = isinstance(second_operand, (Number, ndarray))
        self._second_operand_is_func = isinstance(second_operand, cls)
        self._operator = operator
        self._operator_repr = operator_repr
        if not self._second_operand_is_number and not self._second_operand_is_func:
            raise TypeError(
                f"Unsupported {operator_repr} operator "
                f"for MDOFunction and {type(self._second_operand)}."
            )

        if self._second_operand_is_func:
            self._second_operand_expr = self._second_operand.expr
            self._second_operand_name = self._second_operand.name
        else:
            self._second_operand_expr = str(self._second_operand)
            self._second_operand_name = self._second_operand_expr

        if self._second_operand_is_func:
            if self._first_operand.has_jac() and self._second_operand.has_jac():
                jac = self._compute_operation_jacobian

            if self._first_operand.has_expr() and self._second_operand.has_expr():
                expr = self._compute_expr()

            if self._first_operand.has_args() and self._second_operand.has_args():
                args = sorted(
                    list(set(self._first_operand.args + self._second_operand.args))
                )

            if self._first_operand.has_f_type():
                f_type = self._first_operand.f_type
            elif self._second_operand.has_f_type():
                f_type = self._second_operand.f_type

        else:
            args = self._first_operand.args
            f_type = self._first_operand.f_type
            if self._first_operand.has_expr():
                expr = self._compute_expr()

            if self._first_operand.has_jac():
                jac = self._compute_operation_jacobian

        self.function = cls(
            self._compute_operation,
            self._compute_name(),
            f_type=f_type,
            jac=jac,
            expr=expr,
            args=args,
            dim=self._first_operand.dim,
            outvars=self._first_operand.outvars,
        )

    def _compute_expr(self) -> str:
        """Compute the string expression of the function.

        Returns:
            The string expression of the function.
        """
        return (
            self._first_operand.expr + self._operator_repr + self._second_operand_expr
        )

    def _compute_name(self) -> str:
        """Compute the name of the function.

        Returns:
            The name of the function.
        """
        return (
            self._first_operand.name + self._operator_repr + self._second_operand_name
        )

    def _compute_operation(self, input_value: ArrayType) -> OutputType:
        """Compute the result of the operation..

        Args:
            input_value: The input value.

        Returns:
            The result of the operation.
        """
        second_operand = self._second_operand
        if self._second_operand_is_func:
            second_operand = second_operand(input_value)

        return self._operator(self._first_operand(input_value), second_operand)

    @abstractmethod
    def _compute_operation_jacobian(self, input_value: ArrayType) -> OutputType:
        """Compute the Jacobian of the operation..

        Args:
            input_value: The input value.

        Returns:
            The Jacobian of the operation.
        """
        ...


class _AdditionFunctionMaker(_OperationFunctionMaker):
    """A helper to create a function summing a function with a constant or a function.

    If the function operands have a Jacobian, the function will support automatic
    differentiation.
    """

    def __init__(
        self,
        cls: type[MDOFunction],
        first_operand: MDOFunction,
        second_operand: MDOFunction | Number,
        inverse: bool = False,
    ) -> None:
        """
        Args:
            inverse: Whether to apply the inverse operation, i.e. subtraction.
        """  # noqa: D205, D212, D415
        super().__init__(
            cls,
            first_operand,
            second_operand,
            _subtract if inverse else _add,
            "-" if inverse else "+",
        )

    def _compute_operation_jacobian(self, input_value: ArrayType) -> ArrayType:
        if self._second_operand_is_number:
            return self._first_operand._jac(input_value)

        return self._operator(
            self._first_operand._jac(input_value),
            self._second_operand._jac(input_value),
        )


class _MultiplicationFunctionMaker(_OperationFunctionMaker):
    """A helper to create a function multiplying a function by a number or a function.

    If the function operands have a Jacobian, the function will support automatic
    differentiation.
    """

    def __init__(
        self,
        cls: type[MDOFunction],
        first_operand: MDOFunction,
        second_operand: MDOFunction | Number,
        inverse: bool = False,
    ) -> None:
        """
        Args:
            inverse: Whether to apply the inverse operation, i.e. subtraction.
        """  # noqa: D205, D212, D415
        super().__init__(
            cls,
            first_operand,
            second_operand,
            truediv if inverse else mul,
            "/" if inverse else "*",
        )

    def _compute_expr(self) -> str:
        if self._second_operand_is_number and self._operator == mul:
            return (
                self._second_operand_expr
                + self._operator_repr
                + self._first_operand.expr
            )
        else:
            return super()._compute_expr()

    def _compute_name(self) -> str:
        if self._second_operand_is_number and self._operator == mul:
            return (
                self._second_operand_name
                + self._operator_repr
                + self._first_operand.name
            )
        else:
            return super()._compute_name()

    def _compute_operation_jacobian(self, input_value: ArrayType) -> ArrayType:
        if self._second_operand_is_number:
            return self._operator(
                self._first_operand._jac(input_value), self._second_operand
            )

        first_func = self._first_operand(input_value)
        second_func = self._second_operand(input_value)
        first_jac = self._first_operand._jac(input_value)
        second_jac = self._second_operand._jac(input_value)

        if self._operator == mul:
            return first_jac * second_func + second_jac * first_func

        return (first_jac * second_func - second_jac * first_func) / second_jac**2
