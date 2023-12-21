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

from numpy import dtype
from numpy import eye
from numpy import ndarray
from scipy.sparse.linalg import LinearOperator

from gemseo.utils.compatibility.scipy import ArrayType
from gemseo.utils.compatibility.scipy import array_classes
from gemseo.utils.metaclasses import GoogleDocstringInheritanceMeta

LOGGER = logging.getLogger(__name__)


class JacobianOperator(LinearOperator, metaclass=GoogleDocstringInheritanceMeta):
    """The Jacobian of a discipline as linear operator."""

    def __init__(
        self,
        dtype: dtype,
        shape: tuple[int, ...],
    ):
        """
        Args:
            dtype: The data type of the Jacobian.
            shape: The shape of the Jacobian.
        """  # noqa: D205 D212 D415
        super().__init__(dtype, shape)

    def __add__(self, other: JacobianOperator | ArrayType) -> _SumJacobianOperator:
        """
        Raises:
            TypeError: if the operand has type different from JacobianOperator, NumPy
                ndarray or SciPy spmatrix.
        """  # noqa: D205 D212 D415
        if not isinstance(other, (JacobianOperator, array_classes)):
            raise TypeError(
                f"Adding a JacobianOperator with {type(other)} is not supported."
            )

        return _SumJacobianOperator(self, other)

    def __sub__(self, other: JacobianOperator | ArrayType) -> _SubJacobianOperator:
        """
        Raises:
            TypeError: if the operand has type different from JacobianOperator, NumPy
                ndarray or SciPy spmatrix.
        """  # noqa: D205 D212 D415
        if not isinstance(other, (JacobianOperator, array_classes)):
            raise TypeError(
                f"Substracting a JacobianOperator with {type(other)} is not supported."
            )

        return _SubJacobianOperator(self, other)

    def __matmul__(
        self, other: JacobianOperator | ArrayType
    ) -> _ComposedJacobianOperator:
        """
        Raises:
            TypeError: if the operand has type different from JacobianOperator, NumPy
                ndarray or SciPy spmatrix.
        """  # noqa: D205 D212 D415
        if not isinstance(other, (JacobianOperator, array_classes)):
            raise TypeError(
                f"Multiplying a JacobianOperator with {type(other)} is not supported."
            )

        return _ComposedJacobianOperator(self, other)

    def __rmatmul__(
        self, other: JacobianOperator | ArrayType
    ) -> _ComposedJacobianOperator:
        """
        Raises:
            TypeError: if the operand has type different from JacobianOperator, NumPy
                ndarray or SciPy spmatrix.
        """  # noqa: D205 D212 D415
        if not isinstance(other, (JacobianOperator, array_classes)):
            raise TypeError(
                f"Multiplying a JacobianOperator with {type(other)} is not supported."
            )

        return _ComposedJacobianOperator(other, self)

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

    def shift_identity(self) -> _SubJacobianOperator:
        """Substract the identity to the Jacobian operator.

        Returns:
            The Jacobian operator shifted by minus the identity.
        """
        return self - _IdentityOperator(self.shape[0])

    def get_matrix_representation(self) -> ndarray:
        """Compute the matrix representation of the Jacobian.

        Returns:
            The matrix representation of the Jacobian.
        """
        LOGGER.info(
            "The Jacobian is given as a linear operator. Performing the assembly "
            "required to apply it to the identity which is not performant."
        )

        return self.dot(eye(self.shape[1]))


class _RealJacobianOperator(JacobianOperator):
    def __init__(self, operator: JacobianOperator):
        """
        Args:
            operator: The Jacobian operator to cast to real.
        """  # noqa: D205 D212 D415
        super().__init__(operator.dtype, operator.shape)

        self.__operator = operator

    def _matvec(self, x: ndarray) -> ndarray:
        return self.__operator.matvec(x).real

    def _rmatvec(self, x: ndarray) -> ndarray:
        return self.__operator.rmatvec(x).real


class _AdjointJacobianOperator(JacobianOperator):
    def __init__(self, operator: JacobianOperator):
        """
        Args:
            operator: The Jacobian operator to take the adjoint of.
        """  # noqa: D205 D212 D415
        super().__init__(operator.dtype, operator.shape[::-1])

        self.__operator = operator

    def _matvec(self, x: ndarray) -> ndarray:
        return self.__operator.rmatvec(x)

    def _rmatvec(self, x: ndarray) -> ndarray:
        return self.__operator.matvec(x)


class _IdentityOperator(JacobianOperator):
    def __init__(self, size: int):
        """
        Args:
            operator: The size of the identity.
        """  # noqa: D205 D212 D415
        super().__init__(dtype(float), (size, size))

    def _matvec(self, x: ndarray) -> ndarray:
        return x

    def _rmatvec(self, x: ndarray) -> ndarray:
        return x


class _SumJacobianOperator(JacobianOperator):
    def __init__(
        self,
        operand_1: JacobianOperator,
        operand_2: JacobianOperator | ArrayType,
    ):
        """
        Args:
            operand_1: First operand of the summation.
            operand_2: Second operand of the summation.
        """  # noqa: D205 D212 D415
        super().__init__(operand_1.dtype, operand_1.shape)

        self.__operand_1 = operand_1
        self.__operand_2 = operand_2

        self.__array_like = isinstance(operand_2, array_classes)

    def _matvec(self, x: ndarray) -> ndarray:
        if self.__array_like:
            return self.__operand_1.matvec(x) + self.__operand_2 @ x
        return self.__operand_1.matvec(x) + self.__operand_2.matvec(x)

    def _rmatvec(self, x: ndarray) -> ndarray:
        if self.__array_like:
            return self.__operand_1.rmatvec(x) + self.__operand_2.T @ x
        return self.__operand_1.rmatvec(x) + self.__operand_2.rmatvec(x)


class _SubJacobianOperator(JacobianOperator):
    def __init__(
        self,
        operand_1: JacobianOperator,
        operand_2: JacobianOperator | ArrayType,
    ):
        """
        Args:
            operand_1: First operand of the substraction.
            operand_2: Second operand of the substraction.
        """  # noqa: D205 D212 D415
        super().__init__(operand_1.dtype, operand_1.shape)

        self.__operand_1 = operand_1
        self.__operand_2 = operand_2

        self.__array_like = isinstance(operand_2, array_classes)

    def _matvec(self, x: ndarray) -> ndarray:
        if self.__array_like:
            return self.__operand_1.matvec(x) - self.__operand_2 @ x
        return self.__operand_1.matvec(x) - self.__operand_2.matvec(x)

    def _rmatvec(self, x: ndarray) -> ndarray:
        if self.__array_like:
            return self.__operand_1.rmatvec(x) - self.__operand_2.T @ x
        return self.__operand_1.rmatvec(x) - self.__operand_2.rmatvec(x)


class _ComposedJacobianOperator(JacobianOperator):
    def __init__(
        self,
        operand_1: JacobianOperator | ArrayType,
        operand_2: JacobianOperator | ArrayType,
    ):
        """
        Args:
            operand_1: First operand of the composition.
            operand_2: Second operand of the composition.
        """  # noqa: D205 D212 D415
        super().__init__(operand_1.dtype, (operand_1.shape[0], operand_2.shape[1]))

        self.__operand_1 = operand_1
        self.__operand_2 = operand_2

        self.__array_like_1 = isinstance(operand_1, array_classes)
        self.__array_like_2 = isinstance(operand_2, array_classes)

    def _matvec(self, x: ndarray) -> ndarray:
        x = self.__operand_2 @ x if self.__array_like_2 else self.__operand_2.matvec(x)

        if self.__array_like_1:
            return self.__operand_1 @ x
        return self.__operand_1.matvec(x)

    def _rmatvec(self, x: ndarray) -> ndarray:
        if self.__array_like_1:
            x = self.__operand_1.T @ x
        else:
            x = self.__operand_1.rmatvec(x)

        if self.__array_like_2:
            return self.__operand_2.T @ x
        return self.__operand_2.rmatvec(x)
