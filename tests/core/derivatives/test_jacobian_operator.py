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

import pytest
from numpy import allclose
from numpy import ndarray
from numpy.random import default_rng
from scipy.sparse import rand

from gemseo.core.derivatives.jacobian_operator import JacobianOperator

RNG = default_rng()

RECTANGULAR_SHAPE = (10, 5)
SQUARE_SHAPE = (5, 5)


@pytest.fixture(scope="module")
def square_jacobian() -> tuple[ndarray, JacobianOperator]:
    """Generate a square Jacobian operator from a NumPy array.

    Returns:
        The NumPy array and the JacobianOperator wrapping it.
    """
    matrix = RNG.normal(size=SQUARE_SHAPE)
    operator = JacobianOperator(
        dtype=matrix.dtype,
        shape=matrix.shape,
    )

    def matvec(x):
        return matrix @ x

    def rmatvec(x):
        return matrix.T @ x

    operator._matvec = matvec
    operator._rmatvec = rmatvec

    return matrix, operator


@pytest.fixture(scope="module")
def rectangular_jacobian() -> tuple[ndarray, JacobianOperator]:
    """Generate a rectangular Jacobian operator from a NumPy array.

    Returns:
        The NumPy array and the JacobianOperator wrapping it.
    """
    matrix = RNG.normal(size=RECTANGULAR_SHAPE)
    operator = JacobianOperator(
        dtype=matrix.dtype,
        shape=matrix.shape,
    )

    def matvec(x):
        return matrix @ x

    def rmatvec(x):
        return matrix.T @ x

    operator._matvec = matvec
    operator._rmatvec = rmatvec

    return matrix, operator


def test_matvec():
    """Tests the matrix-vector product."""
    matrix = RNG.normal(size=RECTANGULAR_SHAPE)

    m, n = matrix.shape
    x, y = RNG.normal(size=n), RNG.normal(size=m)

    jacobian = JacobianOperator(
        dtype=matrix.dtype,
        shape=matrix.shape,
    )

    with pytest.raises(RecursionError):
        jacobian.matvec(x)

    with pytest.raises(NotImplementedError):
        jacobian.rmatvec(y)

    jacobian._matvec = lambda x: matrix @ x
    jacobian._rmatvec = lambda x: matrix.T @ x

    assert (jacobian.dot(x) == matrix.dot(x)).all()
    assert (jacobian.T.dot(y) == matrix.T.dot(y)).all()


def test_copy(rectangular_jacobian):
    """Tests the copying."""
    _, jacobian = rectangular_jacobian

    m, n = jacobian.shape
    x, y = RNG.normal(size=n), RNG.normal(size=m)

    jacobian_copy = jacobian.copy()

    assert id(jacobian_copy) != id(jacobian)
    assert (jacobian_copy.dot(x) == jacobian.dot(x)).all()
    assert (jacobian_copy.T.dot(y) == jacobian.T.dot(y)).all()


def test_transpose(rectangular_jacobian):
    """Tests the transposition."""
    _, jacobian = rectangular_jacobian

    m, n = jacobian.shape
    x, y = RNG.normal(size=n), RNG.normal(size=m)

    jacobian_transposed = jacobian.T

    assert jacobian_transposed.shape == (n, m)
    assert (jacobian_transposed.dot(y) == jacobian.T.dot(y)).all()
    assert (jacobian_transposed.T.dot(x) == jacobian.dot(x)).all()


def test_shift_identity(square_jacobian):
    """Tests the shifting."""
    _, jacobian = square_jacobian

    m, _ = jacobian.shape
    x = RNG.normal(size=m)

    jacobian_shifted = jacobian.shift_identity()

    assert (jacobian_shifted.dot(x) == (jacobian.dot(x) - x)).all()


def test_matrix_representation(rectangular_jacobian):
    """Tests the computation of matrix representation."""
    matrix, jacobian = rectangular_jacobian

    jacobian_matrix = jacobian.get_matrix_representation()

    assert (jacobian_matrix == matrix).all()


@pytest.mark.parametrize(
    "matrix",
    [RNG.normal(size=RECTANGULAR_SHAPE), rand(*RECTANGULAR_SHAPE, density=0.25)],
)
def test_algebra_with_arrays(matrix, rectangular_jacobian):
    """Tests the algebraic operations with array-like objects."""
    jacobian_matrix, jacobian_operator = rectangular_jacobian

    # Addition with NumPy array or SciPy sparse matrix
    result = jacobian_operator + matrix
    assert isinstance(result, JacobianOperator)
    assert allclose(
        result.get_matrix_representation(), jacobian_matrix + matrix, atol=1e-12
    )
    assert allclose(
        result.T.get_matrix_representation(), jacobian_matrix.T + matrix.T, atol=1e-12
    )

    # Substraction with NumPy array or SciPy sparse matrix
    result = jacobian_operator - matrix
    assert isinstance(result, JacobianOperator)
    assert allclose(
        result.get_matrix_representation(), jacobian_matrix - matrix, atol=1e-12
    )
    assert allclose(
        result.T.get_matrix_representation(), jacobian_matrix.T - matrix.T, atol=1e-12
    )

    # Left composition with NumPy array or SciPy sparse matrix
    result = jacobian_operator @ matrix.T
    assert isinstance(result, JacobianOperator)
    assert allclose(
        result.get_matrix_representation(), jacobian_matrix @ matrix.T, atol=1e-12
    )
    assert allclose(
        result.T.get_matrix_representation(), matrix @ jacobian_matrix.T, atol=1e-12
    )

    # Right composition with NumPy array or SciPy sparse matrix
    result = jacobian_operator.__rmatmul__(matrix.T)
    assert isinstance(result, JacobianOperator)
    assert allclose(
        result.get_matrix_representation(), matrix.T @ jacobian_matrix, atol=1e-12
    )
    assert allclose(
        result.T.get_matrix_representation(), jacobian_matrix.T @ matrix, atol=1e-12
    )


def test_algebra_between_jacobian_operators(rectangular_jacobian):
    """Tests the algebraic operations between JacobianOperators."""
    jacobian_matrix, jacobian_operator = rectangular_jacobian

    # Addition between JacobianOperator
    result = jacobian_operator + jacobian_operator
    result_mat = jacobian_matrix + jacobian_matrix
    assert isinstance(result, JacobianOperator)
    assert allclose(result.get_matrix_representation(), result_mat, atol=1e-12)
    assert allclose(result.T.get_matrix_representation(), result_mat.T, atol=1e-12)

    # Substraction between JacobianOperator
    result = jacobian_operator - jacobian_operator
    result_mat = jacobian_matrix - jacobian_matrix
    assert isinstance(result, JacobianOperator)
    assert allclose(result.get_matrix_representation(), result_mat, atol=1e-12)
    assert allclose(result.T.get_matrix_representation(), result_mat.T, atol=1e-12)

    # Composition between JacobianOperator
    result = jacobian_operator @ jacobian_operator.T
    result_mat = jacobian_matrix @ jacobian_matrix.T
    assert isinstance(result, JacobianOperator)
    assert allclose(result.get_matrix_representation(), result_mat, atol=1e-12)
    assert allclose(result.T.get_matrix_representation(), result_mat.T, atol=1e-12)


def test_algebra_with_not_supported_type(rectangular_jacobian):
    """Tests the algebraic operations with non supported objects."""
    _, jacobian_operator = rectangular_jacobian

    a = 1.0

    with pytest.raises(
        TypeError,
        match=(f"Adding a JacobianOperator with {type(a)} is not supported."),
    ):
        _ = jacobian_operator + a

    with pytest.raises(
        TypeError,
        match=(f"Substracting a JacobianOperator with {type(a)} is not supported."),
    ):
        _ = jacobian_operator - a

    with pytest.raises(
        TypeError,
        match=(f"Multiplying a JacobianOperator with {type(a)} is not supported."),
    ):
        _ = jacobian_operator @ a

    with pytest.raises(
        TypeError,
        match=(f"Multiplying a JacobianOperator with {type(a)} is not supported."),
    ):
        _ = jacobian_operator.__rmatmul__(a)
