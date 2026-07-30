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
"""NumPy utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy import dtype

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

COMPLEX128_DTYPE: Final = dtype("complex128")
"""The NumPy complex number type with double-precision imaginary and real parts."""

FLOAT64_DTYPE: Final = dtype("float64")
"""The NumPy double-precision floating-point number type."""

INT64_DTYPE: Final = dtype("int64")
"""The NumPy signed integer type with 64 bits."""


def convert_array_type(a: ndarray, dtype_: dtype, copy: bool = True) -> ndarray:
    """Convert an array to a specific type.

    Args:
        a: The original array.
        dtype_: The specific type.
        copy: Whether to return a copy when it is possible.

    Returns:
        The array converted to the specific type.
    """
    return (a.real if dtype_.kind == "c" else a).astype(dtype_, copy=copy)


def get_common_dtype(arrays: Iterable[ndarray]) -> dtype:
    """Return the common NumPy data type of a collection of arrays.

    Use the following rules by parsing the arrays:

    1. there is a complex value: return `numpy.complex128`,
    2. there are real and mixed float/int values: return `numpy.float64`,
    3. there are only integer values: return `numpy.int64`.

    Args:
        arrays: The collection of arrays.

    Returns:
        The common data type.
    """
    at_least_one_float = False
    at_least_one_integer = False
    for array_ in arrays:
        kind = array_.dtype.kind
        if kind == "c":
            return COMPLEX128_DTYPE

        if kind == "i":
            at_least_one_integer = True

        if kind == "f":
            at_least_one_float = True

    if at_least_one_float:
        return FLOAT64_DTYPE

    if at_least_one_integer:
        return INT64_DTYPE

    return FLOAT64_DTYPE
