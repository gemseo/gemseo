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
from numpy import uint8
from xxhash import xxh3_64_hexdigest

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

COMPLEX128_DTYPE: Final = dtype("complex128")
"""The NumPy complex number type with double-precision imaginary and real parts."""

FLOAT64_DTYPE: Final = dtype("float64")
"""The NumPy double-precision floating-point number type."""

INT32_DTYPE: Final = dtype("int32")
"""The NumPy signed integer type with 32 bits."""

INT64_DTYPE: Final = dtype("int64")
"""The NumPy signed integer type with 64 bits."""

UINT32_DTYPE: Final = dtype("uint32")
"""The NumPy unsigned integer type with 32 bits."""

UINT64_DTYPE: Final = dtype("uint64")
"""The NumPy unsigned integer type with 64 bits."""


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


def hash_array(array: ndarray) -> str:
    """Hash an array with the xxh3_64 algorithm.

    The array is hashed as its flat C-contiguous equivalent,
    whatever its contiguity and number of dimensions.
    The digest is computed on the raw bytes only,
    so it encodes neither the shape nor the data type;
    a caller that needs a true identity
    must compare the arrays after a digest match.

    A 32-bit integer array is hashed as its 64-bit equivalent
    so that the hash does not depend on the platform:
    the platform-dependent data types `numpy.int_` and `numpy.uint`
    resolve to 32 bits on Windows with NumPy < 2 and to 64 bits elsewhere.

    Args:
        array: The array to hash.

    Returns:
        The hexadecimal digest of the array.
    """
    if array.dtype == INT32_DTYPE:
        array = array.astype(INT64_DTYPE)
    elif array.dtype == UINT32_DTYPE:
        array = array.astype(UINT64_DTYPE)

    # xxh3_64 requires C-contiguous data and view() requires at least one dimension;
    # ravel() returns a flat C-contiguous array and copies only when required.
    return xxh3_64_hexdigest(array.ravel().view(uint8))
