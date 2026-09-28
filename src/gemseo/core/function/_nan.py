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

"""The check stopping an evaluation on a NaN.

Both halves of the evaluation of a problem perform it:
the evaluation half,
so that a driver deriving no problem is stopped too,
and the adaptation half,
so that a point of the working coordinates is checked before it is mapped back.
"""

from __future__ import annotations

from numpy import isnan
from numpy import ndarray
from numpy import str_

from gemseo.core.function._blocks import read_blocks
from gemseo.core.problem.termination_criterion import DesvarIsNan
from gemseo.core.problem.termination_criterion import FunctionIsNan
from gemseo.util._compatibility.scipy import sparse_classes


def check_for_nan(
    value: ndarray,
    stop_if_nan: bool = True,
    function_name: str = "",
    input_value: ndarray | None = None,
) -> None:
    """Check whether an array holds a NaN value.

    String arrays are ignored,
    and so is a sparse matrix,
    such as the Jacobian of a driver declaring it supports one:
    `isnan` refuses it,
    and reading it is the business of that driver.

    Args:
        value: The array to check.
        stop_if_nan: Whether to stop when `value` holds a NaN.
        function_name: The name of the function.
            If empty, `function_name` and `input_value` are ignored.
        input_value: The point the function is evaluated at.
            `None` if and only if `function_name` is empty.

    Raises:
        DesvarIsNan: When the value is a function input holding a NaN.
        FunctionIsNan: When the value is a function output holding a NaN.
    """
    if isinstance(value, sparse_classes):
        return

    if isinstance(value, ndarray) and value.dtype.type is str_:
        return

    if stop_if_nan and isnan(value).any():
        if function_name:
            msg = (
                f"Found a NaN in the output data of the function {function_name} "
                f"evaluated at the input array {input_value}."
            )
            raise FunctionIsNan(msg)

        msg = f"Found a NaN in the input array {value}."
        raise DesvarIsNan(msg)


def check_blocks_for_nan(
    matrix: ndarray,
    input_values: ndarray,
    output_dimension: int,
    input_dimension: int,
    function_name: str,
    stop_if_nan: bool = True,
) -> None:
    """Check whether the diagonal blocks of a block diagonal matrix hold a NaN.

    This is the check of a vectorized evaluation,
    whose Jacobian is the block diagonal matrix of the samples.
    Only the diagonal blocks say anything,
    the rest of the matrix being structural zeros,
    and each of them belongs to one sample,
    so the block holding a NaN names that sample
    rather than the whole matrix,
    exactly as a non-vectorized check names the point it evaluated.

    Reading the blocks is the business of
    [read_blocks][gemseo.core.function._blocks.read_blocks],
    which the map to the coordinates of an algorithm shares.

    A sparse matrix is ignored,
    as in `check_for_nan`.

    Args:
        matrix: The block diagonal matrix, of shape
            `(n_samples * output_dimension, n_samples * input_dimension)`.
        input_values: The samples, of shape `(n_samples, input_dimension)`.
        output_dimension: The number of rows of a block.
        input_dimension: The number of columns of a block.
        function_name: The name of the function.
        stop_if_nan: Whether to stop when a block holds a NaN.

    Raises:
        FunctionIsNan: When a block holds a NaN.
    """
    if not stop_if_nan or isinstance(matrix, sparse_classes):
        return

    n_samples = len(input_values)
    blocks = read_blocks(matrix, n_samples, output_dimension, input_dimension)
    holds_nan = isnan(blocks).any(axis=(1, 2))
    if holds_nan.any():
        index = int(holds_nan.argmax())
        check_for_nan(blocks[index], True, function_name, input_values[index])
