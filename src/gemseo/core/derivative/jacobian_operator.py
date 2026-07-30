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
"""Abstraction of Jacobian as linear operators."""

from __future__ import annotations

import logging
from copy import copy
from typing import TYPE_CHECKING
from typing import Any

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import dtype
from numpy import eye
from numpy import promote_types
from scipy.sparse.linalg import LinearOperator

from gemseo.util.compatibility.scipy import array_classes

if TYPE_CHECKING:
    from typing import TypeAlias

    from gemseo.util.typing import RealArray
    from gemseo.util.typing import SparseOrDenseRealArray

    _OperandType: TypeAlias = "JacobianOperator | SparseOrDenseRealArray"

LOGGER = logging.getLogger(__name__)


class JacobianOperator(LinearOperator, metaclass=GoogleDocstringInheritanceMeta):  # type: ignore[misc] # missing typing
    """The Jacobian of a discipline as linear operator."""

    def __init__(self, dtype: dtype[Any], shape: tuple[int, ...]) -> None:
        """
        Args:
            dtype: The data type of the Jacobian.
            shape: The shape of the Jacobian.
        """  # noqa: D205 D212 D415
        super().__init__(dtype, shape)

    def __add__(self, other: _OperandType) -> JacobianOperator:
        return _SumOperation(self, _convert_to_operator(other))

    def __radd__(self, other: _OperandType) -> JacobianOperator:
        return _SumOperation(_convert_to_operator(other), self)

    def __sub__(self, other: _OperandType) -> JacobianOperator:
        return _SubOperation(self, _convert_to_operator(other))

    def __rsub__(self, other: _OperandType) -> JacobianOperator:
        return _SubOperation(_convert_to_operator(other), self)

    def __matmul__(self, other: _OperandType) -> JacobianOperator:
        return _ComposedOperation(self, _convert_to_operator(other))

    def __rmatmul__(self, other: _OperandType) -> JacobianOperator:
        return _ComposedOperation(_convert_to_operator(other), self)

    @property
    def real(self) -> JacobianOperator:
        """The real casting of the Jacobian operator output."""
        return _RealJacobianOperator(self)

    def copy(self) -> JacobianOperator:
        """Create a shallow copy of the Jacobian operator.

        Returns:
            A shallow copy of the Jacobian operator.
        """
        return copy(self)

    @property
    def T(self) -> JacobianOperator:  # noqa: N802
        """The transpose of the Jacobian operator.

        Returns:
            The transpose of the Jacobian operator.
        """
        return _AdjointJacobianOperator(self)

    def shift_identity(self) -> JacobianOperator:
        """Subtract the identity from the Jacobian operator.

        Returns:
            The Jacobian operator shifted by minus the identity.
        """
        return self - _IdentityOperator(self.shape[0])

    def get_matrix_representation(self) -> RealArray:
        """Compute the matrix representation of the Jacobian.

        Returns:
            The matrix representation of the Jacobian.
        """
        LOGGER.info(
            "The Jacobian is given as a linear operator. Performing the assembly "
            "required to apply it to the identity which is not performant."
        )

        return self.dot(eye(self.shape[1]))  # type: ignore[no-any-return]


def _convert_to_operator(operand: _OperandType) -> JacobianOperator:
    """Wrap an operand into a Jacobian operator if it is an array.

    Args:
        operand: The Jacobian operator or the dense or sparse array.

    Returns:
        The operand as a Jacobian operator.
    """
    if isinstance(operand, array_classes):
        return _ArrayOperator(operand)
    return operand


class _ArrayOperator(JacobianOperator):
    """A Jacobian operator wrapping a dense or sparse array."""

    def __init__(self, array: SparseOrDenseRealArray) -> None:
        """
        Args:
            array: The dense or sparse array.
        """  # noqa: D205 D212 D415
        super().__init__(array.dtype, array.shape)  # type: ignore[attr-defined]
        self.__array = array

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The product of the operator with the vector.
        """  # noqa: D205 D212
        return self.__array @ x  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the adjoint of the operator to.

        Returns:
            The product of the adjoint of the operator with the vector.
        """  # noqa: D205 D212
        return self.__array.T @ x  # type:ignore[no-any-return]


class _RealJacobianOperator(JacobianOperator):
    """A Jacobian operator that casts its output to real."""

    def __init__(self, operator: JacobianOperator) -> None:
        """
        Args:
            operator: The Jacobian operator to cast to real.
        """  # noqa: D205 D212 D415
        super().__init__(operator.dtype, operator.shape)

        self.__operator = operator

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The real part of the product of the operator with the vector.
        """  # noqa: D205 D212
        return self.__operator.matvec(x).real  # type: ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the adjoint of the operator to.

        Returns:
            The real part of the product of the adjoint of the operator with the
            vector.
        """  # noqa: D205 D212
        return self.__operator.rmatvec(x).real  # type: ignore[no-any-return]


class _AdjointJacobianOperator(JacobianOperator):
    """A Jacobian operator that handles adjoints."""

    def __init__(self, operator: JacobianOperator) -> None:
        """
        Args:
            operator: The Jacobian operator to take the adjoint of.
        """  # noqa: D205 D212 D415
        super().__init__(operator.dtype, operator.shape[::-1])
        self.__operator = operator

    @property
    def T(self) -> JacobianOperator:  # noqa: N802
        """The transpose of the Jacobian operator.

        Returns:
            The original Jacobian operator.
        """
        return self.__operator

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The product of the adjoint of the original operator with the vector.
        """  # noqa: D205 D212
        return self.__operator.rmatvec(x)  # type: ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the adjoint of the operator to.

        Returns:
            The product of the original operator with the vector.
        """  # noqa: D205 D212
        return self.__operator.matvec(x)  # type: ignore[no-any-return]


class _IdentityOperator(JacobianOperator):
    """A Jacobian operator that represents the identity operator."""

    def __init__(self, size: int) -> None:
        """
        Args:
            size: The size of the identity matrix.
        """  # noqa: D205 D212 D415
        super().__init__(dtype(float), (size, size))

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The vector itself.
        """  # noqa: D205 D212
        return x

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the adjoint of the operator to.

        Returns:
            The vector itself.
        """  # noqa: D205 D212
        return x


class _BaseOperation(JacobianOperator):
    """A base class to handle operations on 2 Jacobian operators."""

    _operand_1: JacobianOperator
    """The first operand."""

    _operand_2: JacobianOperator
    """The second operand."""

    def __init__(
        self, operand_1: JacobianOperator, operand_2: JacobianOperator
    ) -> None:
        """
        Args:
            operand_1: The first operand.
            operand_2: The second operand.
        """  # noqa: D205 D212 D415
        super().__init__(
            promote_types(operand_1.dtype, operand_2.dtype),
            (operand_1.shape[0], operand_2.shape[1]),
        )
        self._operand_1 = operand_1
        self._operand_2 = operand_2


class _SumOperation(_BaseOperation):
    """A Jacobian operator that handles the sum of 2 Jacobian operators."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The sum of the products of the operands with the vector.
        """  # noqa: D205 D212
        return self._operand_1.matvec(x) + self._operand_2.matvec(x)  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the adjoint of the operator to.

        Returns:
            The sum of the products of the adjoints of the operands with the vector.
        """  # noqa: D205 D212
        return self._operand_1.rmatvec(x) + self._operand_2.rmatvec(x)  # type:ignore[no-any-return]


class _SubOperation(_BaseOperation):
    """A Jacobian operator that handles the subtraction of 2 Jacobian operators."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The difference of the products of the operands with the vector.
        """  # noqa: D205 D212
        return self._operand_1.matvec(x) - self._operand_2.matvec(x)  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the adjoint of the operator to.

        Returns:
            The difference of the products of the adjoints of the operands with the
            vector.
        """  # noqa: D205 D212
        return self._operand_1.rmatvec(x) - self._operand_2.rmatvec(x)  # type:ignore[no-any-return]


class _ComposedOperation(_BaseOperation):
    """A Jacobian operator that composes 2 Jacobian operators."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The product of the composition of the operands with the vector.
        """  # noqa: D205 D212
        return self._operand_1.matvec(self._operand_2.matvec(x))  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the adjoint of the operator to.

        Returns:
            The product of the composition of the adjoints of the operands with the
            vector.
        """  # noqa: D205 D212
        return self._operand_2.rmatvec(self._operand_1.rmatvec(x))  # type:ignore[no-any-return]
