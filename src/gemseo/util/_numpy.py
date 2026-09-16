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
from numpy import frombuffer
from numpy import ndarray
from numpy import uint8
from xxhash import xxh3_64_hexdigest

if TYPE_CHECKING:
    from collections.abc import Iterable

complex128_dtype: Final = dtype("complex128")
"""The NumPy complex number type with double-precision imaginary and real parts."""

float64_dtype: Final = dtype("float64")
"""The NumPy double-precision floating-point number type."""

int32_dtype: Final = dtype("int32")
"""The NumPy signed integer type with 32 bits."""

int64_dtype: Final = dtype("int64")
"""The NumPy signed integer type with 64 bits."""

uint32_dtype: Final = dtype("uint32")
"""The NumPy unsigned integer type with 32 bits."""

uint64_dtype: Final = dtype("uint64")
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


def _is_frozen(array: ndarray) -> bool:
    """Return whether an array is a view of an immutable buffer.

    Args:
        array: The array.

    Returns:
        Whether the array is a view of an immutable buffer,
        e.g. an array frozen by [freeze_array][gemseo.util._numpy.freeze_array].
    """
    base = array.base
    while isinstance(base, ndarray):
        base = base.base

    return isinstance(base, bytes)


def freeze_array(array: ndarray) -> ndarray:
    """Return a read-only copy of an array, frozen over an immutable buffer.

    Freezing an array in place would not be enough:
    NumPy re-enables the writeable flag of an array owning its data,
    so a caller could thaw the array handed out,
    or any array it aliases through its `base`,
    and mutate what the owner of the original array reads.
    The array returned is a view of an immutable `bytes` buffer instead,
    which owns the data, shares no memory with the original array
    and refuses the writeable flag to every view of it.
    An array that is such a view already needs no copy,
    as its data cannot change,
    and a view of it is returned,
    so that the caller keeps no hand on the array object an owner stores.

    NumPy still lets a caller reassign the shape, the strides and the data type
    of a read-only array, and these belong to the array object itself,
    so an owner hands out a view of the array it stores, e.g. `array.view()`,
    rather than the array itself;
    a reassignment then reaches the view of the caller only.

    Args:
        array: The array to freeze.

    Returns:
        The read-only array.
    """
    if _is_frozen(array):
        return array.view()

    # `tobytes` returns the components of the array in C order,
    # whatever its contiguity, and `reshape` restores its shape.
    return frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


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
            return complex128_dtype

        if kind == "i":
            at_least_one_integer = True

        if kind == "f":
            at_least_one_float = True

    if at_least_one_float:
        return float64_dtype

    if at_least_one_integer:
        return int64_dtype

    return float64_dtype


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
    if array.dtype == int32_dtype:
        array = array.astype(int64_dtype)
    elif array.dtype == uint32_dtype:
        array = array.astype(uint64_dtype)

    # xxh3_64 requires C-contiguous data and view() requires at least one dimension;
    # ravel() returns a flat C-contiguous array and copies only when required.
    return xxh3_64_hexdigest(array.ravel().view(uint8))
