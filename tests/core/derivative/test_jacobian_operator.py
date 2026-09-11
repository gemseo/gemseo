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
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import complex128
from numpy import dtype
from numpy import eye
from numpy.random import default_rng
from scipy.sparse import rand
from scipy.sparse import random_array

from gemseo.core.derivative.jacobian_operator import JacobianOperator
from gemseo.util.seeder import seed

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.util.typing import SparseOrDenseRealArray

rng = default_rng(seed)

rectangular_shape = (10, 5)

array_factories = {
    "dense": lambda shape: rng.normal(size=shape),
    "spmatrix": lambda shape: rand(*shape, density=0.25, rng=rng),
    "sparray": lambda shape: random_array(shape, density=0.25, rng=rng),
}


class MatrixJacobianOperator(JacobianOperator):
    """A Jacobian operator wrapping an explicit matrix."""

    matrix: ndarray
    """The matrix defining the linear operator."""

    def __init__(self, matrix: ndarray) -> None:
        """
        Args:
            matrix: The matrix defining the linear operator.
        """  # noqa: D205 D212 D415
        super().__init__(matrix.dtype, matrix.shape)
        self.matrix = matrix

    def _matvec(self, x: ndarray) -> ndarray:
        """
        Args:
            x: The vector to apply the operator to.

        Returns:
            The matrix-vector product.
        """  # noqa: D205 D212
        return self.matrix @ x

    def _rmatvec(self, x: ndarray) -> ndarray:
        """
        Args:
            x: The vector to apply the adjoint operator to.

        Returns:
            The transposed matrix-vector product.
        """  # noqa: D205 D212
        return self.matrix.T @ x


@pytest.fixture(params=array_factories.values(), ids=list(array_factories))
def array(request) -> SparseOrDenseRealArray:
    """Generate a random NumPy array or SciPy sparse array.

    Returns:
        A random array of rectangular shape.
    """
    return request.param(rectangular_shape)


@pytest.fixture
def jacobian_operator() -> MatrixJacobianOperator:
    """Generate a Jacobian operator wrapping a random rectangular matrix.

    Returns:
        The Jacobian operator.
    """
    return MatrixJacobianOperator(rng.normal(size=rectangular_shape))


def assert_equivalent_to_matrix(
    operator: JacobianOperator, matrix: SparseOrDenseRealArray
) -> None:
    """Check that a Jacobian operator and its adjoint behave as a given matrix.

    Args:
        operator: The Jacobian operator.
        matrix: The reference matrix.
    """
    assert isinstance(operator, JacobianOperator)
    assert allclose(operator.get_matrix_representation(), matrix, atol=1e-12)
    assert allclose(operator.T.get_matrix_representation(), matrix.T, atol=1e-12)


def test_unimplemented_products() -> None:
    """Tests errors raised when the matrix-vector products are not implemented."""
    jacobian = JacobianOperator(dtype(float), rectangular_shape)

    with pytest.raises(RecursionError):
        jacobian.matvec(rng.normal(size=rectangular_shape[1]))

    with pytest.raises(NotImplementedError):
        jacobian.rmatvec(rng.normal(size=rectangular_shape[0]))


def test_matvec(jacobian_operator) -> None:
    """Tests the matrix-vector products of the operator and its adjoint."""
    m, n = jacobian_operator.shape
    x, y = rng.normal(size=n), rng.normal(size=m)

    assert (jacobian_operator.dot(x) == jacobian_operator.matrix @ x).all()
    assert (jacobian_operator.T.dot(y) == jacobian_operator.matrix.T @ y).all()


def test_copy(jacobian_operator) -> None:
    """Tests the copying."""
    m, n = jacobian_operator.shape
    x, y = rng.normal(size=n), rng.normal(size=m)

    jacobian_copy = jacobian_operator.copy()

    assert jacobian_copy is not jacobian_operator
    assert (jacobian_copy.dot(x) == jacobian_operator.dot(x)).all()
    assert (jacobian_copy.T.dot(y) == jacobian_operator.T.dot(y)).all()


def test_transpose(jacobian_operator) -> None:
    """Tests the transposition."""
    jacobian_transposed = jacobian_operator.T

    assert jacobian_transposed.shape == jacobian_operator.shape[::-1]
    assert jacobian_transposed.T is jacobian_operator
    assert_equivalent_to_matrix(jacobian_transposed, jacobian_operator.matrix.T)


def test_shift_identity() -> None:
    """Tests the shifting by minus the identity."""
    jacobian_operator = MatrixJacobianOperator(rng.normal(size=(5, 5)))

    assert_equivalent_to_matrix(
        jacobian_operator.shift_identity(),
        jacobian_operator.matrix - eye(5),
    )


def test_real() -> None:
    """Tests the real casting of the Jacobian operator output."""
    matrix = rng.normal(size=rectangular_shape) + 1j * rng.normal(
        size=rectangular_shape
    )
    jacobian_operator = MatrixJacobianOperator(matrix)

    assert_equivalent_to_matrix(jacobian_operator.real, matrix.real)


def test_dtype_promotion(jacobian_operator) -> None:
    """Tests the data type promotion of operations between Jacobian operators."""
    complex_operator = MatrixJacobianOperator(
        rng.normal(size=rectangular_shape).astype(complex128)
    )

    assert (jacobian_operator + complex_operator).dtype == complex128
    assert (jacobian_operator - complex_operator).dtype == complex128
    assert (jacobian_operator @ complex_operator.T).dtype == complex128


def test_matrix_representation(jacobian_operator) -> None:
    """Tests the computation of matrix representation."""
    matrix_representation = jacobian_operator.get_matrix_representation()

    assert (matrix_representation == jacobian_operator.matrix).all()


def test_operator_plus_array(jacobian_operator, array) -> None:
    """Tests the addition of an array to a Jacobian operator."""
    assert_equivalent_to_matrix(
        jacobian_operator + array, jacobian_operator.matrix + array
    )


def test_array_plus_operator(jacobian_operator, array) -> None:
    """Tests the addition of a Jacobian operator to an array."""
    assert_equivalent_to_matrix(
        array + jacobian_operator, array + jacobian_operator.matrix
    )


def test_operator_minus_array(jacobian_operator, array) -> None:
    """Tests the subtraction of an array from a Jacobian operator."""
    assert_equivalent_to_matrix(
        jacobian_operator - array, jacobian_operator.matrix - array
    )


def test_array_minus_operator(jacobian_operator, array) -> None:
    """Tests the subtraction of a Jacobian operator from an array."""
    assert_equivalent_to_matrix(
        array - jacobian_operator, array - jacobian_operator.matrix
    )


def test_operator_matmul_array(jacobian_operator, array) -> None:
    """Tests the composition of a Jacobian operator with an array."""
    assert_equivalent_to_matrix(
        jacobian_operator @ array.T, jacobian_operator.matrix @ array.T
    )


def test_array_matmul_operator(jacobian_operator, array) -> None:
    """Tests the composition of an array with a Jacobian operator."""
    assert_equivalent_to_matrix(
        array.T @ jacobian_operator, array.T @ jacobian_operator.matrix
    )


def test_operator_plus_operator(jacobian_operator) -> None:
    """Tests the addition of two Jacobian operators."""
    assert_equivalent_to_matrix(
        jacobian_operator + jacobian_operator,
        jacobian_operator.matrix + jacobian_operator.matrix,
    )


def test_operator_minus_operator(jacobian_operator) -> None:
    """Tests the subtraction of two Jacobian operators."""
    assert_equivalent_to_matrix(
        jacobian_operator - jacobian_operator,
        jacobian_operator.matrix - jacobian_operator.matrix,
    )


def test_operator_matmul_operator(jacobian_operator) -> None:
    """Tests the composition of two Jacobian operators."""
    assert_equivalent_to_matrix(
        jacobian_operator @ jacobian_operator.T,
        jacobian_operator.matrix @ jacobian_operator.matrix.T,
    )
