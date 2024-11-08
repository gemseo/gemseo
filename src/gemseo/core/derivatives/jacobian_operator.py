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
from typing import Generic
from typing import TypeVar

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import dtype
from numpy import eye
from scipy.sparse.linalg import LinearOperator

from gemseo.typing import SparseOrDenseRealArray
from gemseo.utils.compatibility.scipy import array_classes

if TYPE_CHECKING:
    from gemseo.typing import RealArray

_OperandT = TypeVar("_OperandT", "JacobianOperator", SparseOrDenseRealArray)

LOGGER = logging.getLogger(__name__)


class JacobianOperator(LinearOperator, metaclass=GoogleDocstringInheritanceMeta):  # type: ignore[misc] # missing typing
    """The Jacobian of a discipline as linear operator."""

    def __init__(
        self,
        dtype: dtype[Any],
        shape: tuple[int, ...],
    ) -> None:
        """
        Args:
            dtype: The data type of the Jacobian.
            shape: The shape of the Jacobian.
        """  # noqa: D205 D212 D415
        super().__init__(dtype, shape)

    def __add__(self, other: _OperandT) -> _BaseOperation[_OperandT]:
        if isinstance(other, array_classes):
            return _SumOperationWithArray(self, other)
        return _SumOperation(self, other)

    def __sub__(self, other: _OperandT) -> _BaseOperation[_OperandT]:
        if isinstance(other, array_classes):
            return _SubOperationWithArray(self, other)
        return _SubOperation(self, other)

    def __matmul__(
        self,
        other: _OperandT,
    ) -> _BaseComposedOperation[JacobianOperator, _OperandT]:
        if isinstance(other, array_classes):
            return _ComposedOperationOperatorArray(self, other)
        return _ComposedOperationOperatorOperator(self, other)

    def __rmatmul__(
        self,
        other: _OperandT,
    ) -> _BaseComposedOperation[_OperandT, JacobianOperator]:
        if isinstance(other, array_classes):
            return _ComposedOperationArrayOperator(other, self)
        return _ComposedOperationOperatorOperator(other, self)

    @property
    def real(self) -> _RealJacobianOperator:
        """The real casting of the Jacobian operator output."""
        return _RealJacobianOperator(self)

    def copy(self) -> JacobianOperator:
        """Create a shallow copy of the Jacobian operator.

        Returns:
            A shallow copy of the Jacobian operator.
        """
        return copy(self)

    @property
    def T(self) -> _AdjointJacobianOperator:  # noqa: N802
        """The transpose of the Jacobian operator.

        Returns:
            The transpose of the Jacobian operator.
        """
        return _AdjointJacobianOperator(self)

    def shift_identity(self) -> _SubOperation:
        """Subtract the identity to the Jacobian operator.

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


class _RealJacobianOperator(JacobianOperator):
    """A jacobian operator that casts to real."""

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
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self.__operator.matvec(x).real  # type: ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self.__operator.rmatvec(x).real  # type: ignore[no-any-return]


class _AdjointJacobianOperator(JacobianOperator):
    """A jacobian operator that handles adjoints."""

    def __init__(self, operator: JacobianOperator) -> None:
        """
        Args:
            operator: The Jacobian operator to take the adjoint of.
        """  # noqa: D205 D212 D415
        super().__init__(operator.dtype, operator.shape[::-1])
        self.__operator = operator

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self.__operator.rmatvec(x)  # type: ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self.__operator.matvec(x)  # type: ignore[no-any-return]


class _IdentityOperator(JacobianOperator):
    """A jacobian operator that represents the identity operator."""

    def __init__(self, size: int) -> None:
        """
        Args:
            size: The size of the identity matrix.
        """  # noqa: D205 D212 D415
        super().__init__(dtype(float), (size, size))

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return x

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return x


class _BaseOperation(JacobianOperator, Generic[_OperandT]):
    """A base class to handle operations on 2 jacobian operators."""

    _operand_1: JacobianOperator
    """The first operand."""

    _operand_2: _OperandT
    """The second operand."""

    def __init__(
        self,
        operand_1: JacobianOperator,
        operand_2: _OperandT,
    ) -> None:
        """
        Args:
            operand_1: The first operand.
            operand_2: The second operand.
        """  # noqa: D205 D212 D415
        super().__init__(operand_1.dtype, operand_1.shape)
        self._operand_1 = operand_1
        self._operand_2 = operand_2


class _SumOperation(_BaseOperation[JacobianOperator]):
    """A jacobian operator that handles the sum of 2 jacobian operators."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.matvec(x) + self._operand_2.matvec(x)  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.rmatvec(x) + self._operand_2.rmatvec(x)  # type:ignore[no-any-return]


class _SumOperationWithArray(_BaseOperation[SparseOrDenseRealArray]):
    """A jacobian operator that handles the sum operation with a standard jacobian."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.matvec(x) + self._operand_2 @ x  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.rmatvec(x) + self._operand_2.T @ x  # type:ignore[no-any-return]


class _SubOperation(_BaseOperation[JacobianOperator]):
    """A jacobian operator that handles the subtraction of 2 jacobian operators."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.matvec(x) - self._operand_2.matvec(x)  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.rmatvec(x) - self._operand_2.rmatvec(x)  # type:ignore[no-any-return]


class _SubOperationWithArray(_BaseOperation[SparseOrDenseRealArray]):
    """A jacobian operator that handles the subtraction with a standard jacobian."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.matvec(x) - self._operand_2 @ x  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.rmatvec(x) - self._operand_2.T @ x  # type:ignore[no-any-return]


# TODO: How to avoid duplication? (TypeAlias does not work)
Operand1T = TypeVar("Operand1T", JacobianOperator, SparseOrDenseRealArray)
Operand2T = TypeVar("Operand2T", JacobianOperator, SparseOrDenseRealArray)


class _BaseComposedOperation(JacobianOperator, Generic[Operand1T, Operand2T]):
    """A base class for jacobian operators composition."""

    _operand_1: Operand1T
    """The first operand."""

    _operand_2: Operand2T
    """The second operand."""

    def __init__(
        self,
        operand_1: Operand1T,
        operand_2: Operand2T,
    ) -> None:
        """
        Args:
            operand_1: The first operand of the composition.
            operand_2: The second operand of the composition.
        """  # noqa: D205 D212 D415
        super().__init__(operand_1.dtype, (operand_1.shape[0], operand_2.shape[1]))
        self._operand_1 = operand_1
        self._operand_2 = operand_2


class _ComposedOperationArrayOperator(
    _BaseComposedOperation[SparseOrDenseRealArray, JacobianOperator]
):
    """A jacobian operator that handles a left composition with a standard jacobian."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1 @ self._operand_2.matvec(x)  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_2.rmatvec(self._operand_1.T @ x)  # type:ignore[no-any-return]


class _ComposedOperationOperatorOperator(
    _BaseComposedOperation[JacobianOperator, JacobianOperator]
):
    """A jacobian operator that compose another jacobian operator."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.matvec(self._operand_2.matvec(x))  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_2.rmatvec(self._operand_1.rmatvec(x))  # type:ignore[no-any-return]


class _ComposedOperationOperatorArray(
    _BaseComposedOperation[JacobianOperator, SparseOrDenseRealArray]
):
    """A jacobian operator that handles a right composition with a standard jacobian."""

    def _matvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_1.matvec(self._operand_2 @ x)  # type:ignore[no-any-return]

    def _rmatvec(self, x: RealArray) -> RealArray:
        """
        Args:
            x: The vector to apply the transpose of ∂f/∂v to.
        """  # noqa: D205 D212
        return self._operand_2.T @ self._operand_1.rmatvec(x)  # type:ignore[no-any-return]
