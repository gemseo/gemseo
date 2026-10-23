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

from numpy import complex128
from numpy import dtype
from numpy import float64
from numpy import frombuffer
from numpy import ndarray
from numpy import signbit
from numpy import uint8
from xxhash import xxh3_64_hexdigest

if TYPE_CHECKING:
    from collections.abc import Iterable

string_dtype_kinds: Final[frozenset[str]] = frozenset({"U", "S"})
"""The kinds of the NumPy dtypes of string and bytes arrays."""

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


_real_kinds: Final[str] = "biuf"
"""The data type kinds of the values converted to double precision floats."""

_complex_kind: Final[str] = "c"
"""The data type kind of the values that may be kept complex."""

_text_kinds: Final[str] = "US"
"""The data type kinds of the values converted to their textual form."""

_time_kinds: Final[str] = "Mm"
"""The data type kinds of the values converted to times in nanoseconds."""

_float_kind: Final[str] = "f"
"""The data type kind of the real values that can hold a negative zero."""


def _has_negative_zero(value: ndarray) -> bool:
    """Return whether a real array holds a negative zero.

    The zeros are looked for first, as an array rarely holds any,
    and their signs are then checked with a mask rather than by selecting them,
    so that the values of the array are never copied,
    even when they are all zero.

    Args:
        value: The real array.

    Returns:
        Whether the array holds a negative zero.
    """
    zero = value == 0
    return bool(zero.any() and (zero & signbit(value)).any())


def _canonicalize(value: ndarray) -> ndarray:
    """Convert an array to its canonical form.

    Two arrays equal in value but of different data types
    have different raw bytes,
    e.g. `array([1])`, `array([1.0])` and `array([1.0 + 0.0j])`,
    as do `array([0.0])` and `array([-0.0])`,
    while [array_equal][numpy.array_equal]
    and [compare_dict_of_arrays][gemseo.util.comparison.compare_dict_of_arrays]
    consider them equal.
    Their canonical forms have the same raw bytes,
    so that [hash_array][gemseo.util._numpy.hash_array],
    which digests the raw bytes,
    hashes equal the arrays compared equal.

    A real array is converted to double precision floats
    and a complex array whose imaginary part is zero to its real part,
    so that the arrays of double precision floats,
    which are the common case,
    are not converted to another data type.

    Args:
        value: The array.

    Returns:
        The array in its canonical form.
    """
    kind = value.dtype.kind
    if kind == _complex_kind:
        if value.imag.any():
            # A complex value equals a real one only if its imaginary part is zero,
            # so only the values that are truly complex are kept complex.
            value = value.astype(complex128, copy=False)
            if _has_negative_zero(value.real) or _has_negative_zero(value.imag):
                # Adding zero turns the negative zeros into positive ones,
                # in the real part as well as in the imaginary one.
                value = value + 0j

            return value

        # The imaginary parts are all zero, so the value is converted to a real one.
        value = value.real
        kind = value.dtype.kind

    if kind in _real_kinds:
        # Real values are the common case, hence returned without being copied
        # unless they are not double precision floats or hold a negative zero.
        if value.dtype != float64:
            value = value.astype(float64)

        # Only floats can hold a negative zero; adding zero turns it into a positive.
        if kind == _float_kind and _has_negative_zero(value):
            value = value + 0.0

        return value

    if kind in _text_kinds:
        # The bytes of a text array are padded to the size of its longest item,
        # so its items are converted to their unambiguous textual form instead.
        # The items of a bytes array are decoded here rather than by astype(str),
        # which decodes them as ASCII and fails on any other byte.
        items = [
            item.decode("utf-8", "surrogateescape") if isinstance(item, bytes) else item
            for item in value.ravel().tolist()
        ]
        return frombuffer(str(items).encode("utf-8"), dtype=uint8)

    if kind in _time_kinds:
        # The stored number of a date or a duration depends on its unit,
        # so all of them are converted to nanoseconds;
        # one out of the range of the nanoseconds wraps around,
        # identically for all the units, so that equal times stay equal.
        return value.astype(f"{kind}8[ns]")

    return value


def hash_array(array: ndarray, canonicalize: bool = True) -> str:
    """Hash an array with the xxh3_64 algorithm.

    The array is hashed as its flat C-contiguous equivalent,
    whatever its contiguity and number of dimensions.
    The digest is computed on the raw bytes only,
    so it encodes neither the shape nor the data type;
    a caller that needs a true identity
    must compare the arrays after a digest match.

    By default,
    the array is hashed in its canonical form,
    regardless of its numeric data type, of the signs of its zeros,
    of the length of its text data type and of the unit of its times,
    so that two arrays equal for [array_equal][numpy.array_equal]
    have the same digest,
    e.g. `array([1])`, `array([1.0])` and `array([1.0 + 0.0j])`.

    Without this canonicalization,
    a 32-bit integer array is hashed as its 64-bit equivalent
    so that the hash does not depend on the platform:
    the platform-dependent data types `numpy.int_` and `numpy.uint`
    resolve to 32 bits on Windows with NumPy < 2 and to 64 bits elsewhere.

    Args:
        array: The array to hash.
        canonicalize: Whether to hash the array in its canonical form;
            otherwise,
            two arrays equal in value but of different data types,
            or differing by the signs of their zeros,
            e.g. `array([0.0])` and `array([-0.0])`,
            have different digests.

    Returns:
        The hexadecimal digest of the array.
    """
    if canonicalize:
        array = _canonicalize(array)
    elif array.dtype == int32_dtype:
        array = array.astype(int64_dtype)
    elif array.dtype == uint32_dtype:
        array = array.astype(uint64_dtype)

    # xxh3_64 requires C-contiguous data and view() requires at least one dimension;
    # ravel() returns a flat C-contiguous array and copies only when required.
    return xxh3_64_hexdigest(array.ravel().view(uint8))
