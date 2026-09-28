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

"""The diagonal blocks of the Jacobian of a vectorized evaluation.

A vectorized evaluation asks a function for a matrix of samples
and gets the block diagonal matrix of their Jacobians back.
Only the diagonal blocks say anything,
the rest of the matrix being structural zeros,
and each of them belongs to one sample.
Both halves of the evaluation of a problem read them:
the evaluation half to store one Jacobian per sample
and to name the sample a NaN belongs to,
the adaptation half to map each of them to the coordinates an algorithm works on.

The blocks are read in a single indexing operation
rather than one slice per sample,
because this runs on the per-evaluation path of a vectorized DOE:
what it costs grows with the number of samples,
not with the area of the matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy import arange
from numpy import asarray
from numpy import asmatrix
from numpy import diff
from numpy import matrix as np_matrix
from numpy import ones
from numpy import repeat
from numpy import zeros
from scipy.sparse import csr_matrix

from gemseo.util._compatibility.scipy import sparse_classes

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from gemseo.util.typing import NumberArray

_unsliceable_sparse_formats: Final[frozenset[str]] = frozenset({"bsr", "coo", "dia"})
"""The sparse formats a block cannot be read from.

These are meant for assembling a matrix rather than for reading one,
so `scipy.sparse` either refuses the indexing of such a matrix outright,
as it does for a COO or a DIA one,
or leaves it unimplemented,
as it does for a BSR one.
"""


def get_block_indices(
    n_samples: int, output_dimension: int, input_dimension: int
) -> tuple[NumberArray, NumberArray]:
    """Return the indices of the diagonal blocks of a block diagonal matrix.

    The two arrays broadcast against each other,
    so indexing a matrix with them yields the blocks in a single operation
    and assigning through them writes them back.

    Args:
        n_samples: The number of samples, hence of diagonal blocks.
        output_dimension: The number of rows of a block.
        input_dimension: The number of columns of a block.

    Returns:
        The row indices, of shape `(n_samples, output_dimension, 1)`,
        and the column indices, of shape `(n_samples, 1, input_dimension)`.
    """
    rows = arange(n_samples * output_dimension).reshape(n_samples, output_dimension, 1)
    columns = (
        arange(n_samples)[:, None] * input_dimension + arange(input_dimension)
    ).reshape(n_samples, 1, input_dimension)
    return rows, columns


def read_blocks(
    matrix: NumberArray, n_samples: int, output_dimension: int, input_dimension: int
) -> NumberArray:
    """Read the diagonal blocks of a dense block diagonal matrix.

    The blocks are read from a base array view of the matrix,
    because a `numpy.matrix`,
    which is what a densified legacy sparse matrix is,
    collapses the result of that indexing back to two dimensions
    and leaves no axis to reduce.
    The view costs nothing and the matrix the caller holds keeps its type.

    Args:
        matrix: The block diagonal matrix, of shape
            `(n_samples * output_dimension, n_samples * input_dimension)`.
        n_samples: The number of samples.
        output_dimension: The number of rows of a block.
        input_dimension: The number of columns of a block.

    Returns:
        The blocks, of shape `(n_samples, output_dimension, input_dimension)`.
    """
    rows, columns = get_block_indices(n_samples, output_dimension, input_dimension)
    return asarray(matrix)[rows, columns]


def assemble_blocks(blocks: NumberArray, like: NumberArray) -> NumberArray:
    """Assemble a dense block diagonal matrix from its diagonal blocks.

    Args:
        blocks: The blocks, of shape
            `(n_samples, output_dimension, input_dimension)`.
        like: The matrix the blocks were read from, whose type the result takes.

    Returns:
        The block diagonal matrix.
    """
    n_samples, output_dimension, input_dimension = blocks.shape
    rows, columns = get_block_indices(n_samples, output_dimension, input_dimension)
    matrix = zeros(
        (n_samples * output_dimension, n_samples * input_dimension), dtype=blocks.dtype
    )
    matrix[rows, columns] = blocks
    # A densified legacy sparse matrix is a `numpy.matrix`,
    # and the caller hands what it is given to an algorithm:
    # the type of the matrix is kept,
    # as the non-vectorized path keeps it.
    return asmatrix(matrix) if isinstance(like, np_matrix) else matrix


def stack_sparse_blocks(
    matrix: NumberArray, output_dimension: int, input_dimension: int
) -> csr_matrix:
    """Stack the diagonal blocks of a sparse block diagonal matrix vertically.

    Every row of the matrix belongs to one sample,
    and the columns of that sample are the ones of its block,
    so stacking the blocks is a shift of the column index of each stored value.
    That shift is read from the rows the values lie in,
    which costs one pass over the stored values
    rather than one slice per sample.

    Args:
        matrix: The block diagonal matrix, of shape
            `(n_samples * output_dimension, n_samples * input_dimension)`.
        output_dimension: The number of rows of a block.
        input_dimension: The number of columns of a block.

    Returns:
        The blocks stacked vertically, of shape
        `(n_samples * output_dimension, input_dimension)`.
    """
    csr = matrix.tocsr() if matrix.format != "csr" else matrix
    n_rows = csr.shape[0]
    samples = repeat(arange(n_rows), diff(csr.indptr)) // output_dimension
    return csr_matrix(
        (csr.data, csr.indices - samples * input_dimension, csr.indptr),
        shape=(n_rows, input_dimension),
    )


def unstack_sparse_blocks(
    stacked: csr_matrix,
    output_dimension: int,
    sparse_format: str,
) -> NumberArray:
    """Rebuild a sparse block diagonal matrix from its stacked diagonal blocks.

    This is the inverse of
    [stack_sparse_blocks][gemseo.core.function._blocks.stack_sparse_blocks].

    Args:
        stacked: The blocks stacked vertically, of shape
            `(n_samples * output_dimension, input_dimension)`.
        output_dimension: The number of rows of a block.
        sparse_format: The format of the matrix the blocks were read from,
            which the result takes.

    Returns:
        The block diagonal matrix.
    """
    csr = stacked.tocsr() if stacked.format != "csr" else stacked
    n_rows, input_dimension = csr.shape
    n_samples = n_rows // output_dimension
    samples = repeat(arange(n_rows), diff(csr.indptr)) // output_dimension
    matrix = csr_matrix(
        (csr.data, csr.indices + samples * input_dimension, csr.indptr),
        shape=(n_rows, n_samples * input_dimension),
    )
    return matrix.asformat(sparse_format)


def iter_blocks(
    matrix: NumberArray, n_samples: int, output_dimension: int, input_dimension: int
) -> Iterator[NumberArray]:
    """Iterate over the diagonal blocks of a block diagonal matrix.

    This is what a caller storing one Jacobian per sample needs,
    where [read_blocks][gemseo.core.function._blocks.read_blocks] is what a caller
    reading them all at once needs.
    A block is cut from the matrix,
    so a dense one yields views and keeps its type,
    and a sparse matrix in a format built for assembling
    rather than for reading is copied to a CSR one,
    once for the whole matrix rather than once per sample.

    Args:
        matrix: The block diagonal matrix, of shape
            `(n_samples * output_dimension, n_samples * input_dimension)`.
        n_samples: The number of samples.
        output_dimension: The number of rows of a block.
        input_dimension: The number of columns of a block.

    Yields:
        The diagonal blocks, one per sample.
    """
    if (
        isinstance(matrix, sparse_classes)
        and matrix.format in _unsliceable_sparse_formats
    ):
        matrix = matrix.tocsr()

    for index in range(n_samples):
        yield matrix[
            index * output_dimension : (index + 1) * output_dimension,
            index * input_dimension : (index + 1) * input_dimension,
        ]


def transform_blocks(
    matrix: NumberArray,
    n_samples: int,
    input_dimension: int,
    transform: Callable[[NumberArray], NumberArray],
) -> NumberArray:
    """Apply a map of a single Jacobian to every diagonal block of a matrix.

    The map is called **once**,
    on every block at a time:
    a dense matrix hands it the three-dimensional array of its blocks,
    whose last axis is the one a map scaling the columns of a Jacobian indexes,
    and a sparse one hands it the blocks stacked vertically,
    whose columns are those of a block too.
    A map written for one Jacobian therefore lands on the right axis
    without knowing anything of the samples.

    Args:
        matrix: The block diagonal matrix, of shape
            `(n_samples * output_dimension, n_samples * input_dimension)`.
        n_samples: The number of samples.
        input_dimension: The number of columns of a block.
        transform: The map of the Jacobian of a single point.

    Returns:
        The block diagonal matrix of the mapped blocks.
    """
    is_sparse = isinstance(matrix, sparse_classes)
    # A map handing back what it was given is the identity,
    # which a block diagonal matrix survives as it is:
    # a run transforming nothing,
    # which is what a driver normalizing no space asks for,
    # pays neither the reading of the blocks
    # nor the copy of the matrix they are written back to.
    # The probe is carried as the matrix is,
    # so a map is never asked for anything it would not be asked for.
    probe = ones((1, input_dimension))
    if is_sparse:
        probe = csr_matrix(probe)

    if transform(probe) is probe:
        return matrix

    output_dimension = matrix.shape[0] // n_samples
    if is_sparse:
        return unstack_sparse_blocks(
            transform(stack_sparse_blocks(matrix, output_dimension, input_dimension)),
            output_dimension,
            matrix.format,
        )

    blocks = read_blocks(matrix, n_samples, output_dimension, input_dimension)
    return assemble_blocks(transform(blocks), matrix)
