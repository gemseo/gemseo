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
from numpy import array
from numpy import asarray
from numpy import matrix as np_matrix
from numpy.testing import assert_allclose
from scipy.sparse import block_diag
from scipy.sparse import csr_matrix

from gemseo.core.function._blocks import assemble_blocks
from gemseo.core.function._blocks import iter_blocks
from gemseo.core.function._blocks import read_blocks
from gemseo.core.function._blocks import transform_blocks
from gemseo.util._compatibility.scipy import sparse_classes

BLOCKS = (
    array([[1.0, 2.0], [3.0, 4.0]]),
    array([[5.0, 6.0], [7.0, 8.0]]),
    array([[9.0, 10.0], [11.0, 12.0]]),
)
"""The diagonal blocks of the Jacobian of three samples, of two outputs each."""


def to_dense(matrix):
    """Return the dense array of a matrix, whatever carries it.

    Args:
        matrix: The matrix.

    Returns:
        The dense array of the matrix.
    """
    return asarray(matrix.todense() if isinstance(matrix, sparse_classes) else matrix)


@pytest.fixture(params=["ndarray", "matrix", "csr", "coo"])
def matrix(request):
    """The block diagonal matrix of the samples, in the formats a function returns.

    A COO matrix is one no block can be read from,
    and a `numpy.matrix` is what a densified legacy sparse matrix is.
    """
    block_diagonal = block_diag(BLOCKS, format="csr")
    if request.param == "ndarray":
        return block_diagonal.toarray()

    if request.param == "matrix":
        return block_diagonal.todense()

    return block_diagonal.asformat(request.param)


@pytest.mark.parametrize("densify", [lambda m: m.toarray(), lambda m: m.todense()])
def test_read_blocks(densify) -> None:
    """Check that the diagonal blocks of a dense matrix are read at once.

    A `numpy.matrix`,
    which is what a densified legacy sparse matrix is,
    collapses the result of that indexing
    unless it is read through a base array view.
    """
    matrix = densify(block_diag(BLOCKS, format="csr"))
    assert_allclose(read_blocks(matrix, 3, 2, 2), array(BLOCKS))


def test_assemble_blocks_keeps_the_type_of_the_matrix() -> None:
    """Check that a matrix rebuilt from blocks is carried as the original one."""
    dense = block_diag(BLOCKS).toarray()
    assert isinstance(assemble_blocks(array(BLOCKS), dense), type(dense))
    assert isinstance(assemble_blocks(array(BLOCKS), np_matrix(dense)), np_matrix)
    assert_allclose(to_dense(assemble_blocks(array(BLOCKS), dense)), dense)


def test_iter_blocks(matrix) -> None:
    """Check that the blocks are cut one by one whatever carries the matrix."""
    blocks = list(iter_blocks(matrix, 3, 2, 2))
    assert len(blocks) == 3
    for block, expected in zip(blocks, BLOCKS, strict=True):
        assert_allclose(to_dense(block), expected)


def test_transform_blocks(matrix) -> None:
    """Check that a map of a single Jacobian is applied to every block."""
    transformed = transform_blocks(matrix, 3, 2, lambda blocks: blocks * 2.0)
    assert_allclose(to_dense(transformed), to_dense(matrix) * 2.0)


def test_transform_blocks_keeps_the_carrier(matrix) -> None:
    """Check that the transformed matrix is carried as the original one."""
    transformed = transform_blocks(matrix, 3, 2, lambda blocks: blocks * 2.0)
    assert isinstance(transformed, type(matrix))
    if isinstance(matrix, sparse_classes):
        assert transformed.format == matrix.format


def test_transform_blocks_of_a_single_sample() -> None:
    """Check that a matrix of one sample is a single block."""
    block = array([[1.0, 2.0]])
    transformed = transform_blocks(block, 1, 2, lambda blocks: blocks * 3.0)
    assert_allclose(transformed, array([[3.0, 6.0]]))


def test_transform_blocks_ignores_what_lies_outside_the_blocks() -> None:
    """Check that only the diagonal blocks of a dense matrix are transformed."""
    matrix = block_diag(BLOCKS).toarray()
    matrix[0, -1] = 1.0
    transformed = transform_blocks(matrix, 3, 2, lambda blocks: blocks * 1.0)
    # The samples of a vectorized evaluation are independent,
    # so anything outside the diagonal blocks is a structural zero
    # and says nothing.
    assert transformed[0, -1] == 0.0


def test_transform_blocks_scales_every_sample_alike() -> None:
    """Check that a map scaling the columns reaches every block, not only the first.

    This is the defect the block diagonal matrix used to hide:
    a map indexing the columns of the Jacobian of one point
    lands on the first sample only.
    """
    matrix = csr_matrix(block_diag(BLOCKS))
    transformed = transform_blocks(matrix, 3, 2, lambda blocks: blocks.multiply([1, 2]))
    for block, expected in zip(iter_blocks(transformed, 3, 2, 2), BLOCKS, strict=True):
        assert_allclose(to_dense(block), expected * array([1.0, 2.0]))


def test_transform_blocks_of_an_identity_map(matrix) -> None:
    """Check that a map changing nothing costs neither a read nor a copy."""
    assert transform_blocks(matrix, 3, 2, lambda blocks: blocks) is matrix
