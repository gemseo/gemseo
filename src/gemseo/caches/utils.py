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
"""Some cache utilities."""

from __future__ import annotations

from typing import cast

from numpy import array
from numpy import ascontiguousarray
from numpy import complex128
from numpy import float64
from numpy import int32
from numpy import int64
from numpy import ndarray
from numpy import uint8
from xxhash import xxh3_64_hexdigest

from gemseo.typing import RealArray
from gemseo.typing import RealOrComplexArray
from gemseo.typing import StrKeyMapping
from gemseo.utils.platform import PLATFORM_IS_WINDOWS


def hash_data(
    data: StrKeyMapping,
) -> int:
    """Hash data using xxh3_64 from the xxhash library.

    Args:
        data: The data to hash.

    Returns:
        The hash value of the data.

    Examples:
        >>> from gemseo.caches.utils import hash_data
        >>> from numpy import array
        >>> data = {"x": array([1.0, 2.0]), "y": array([3.0])}
        >>> hash_data(data)
        13252388834746642440
        >>> hash_data(data, "x")
        4006190450215859422
    """
    names_with_hashed_values = []

    for name in sorted(data):
        value = data.get(name)
        if value is None:
            continue

        # xxh3_64 does not support int or float as input.
        if isinstance(value, ndarray):
            if value.dtype == int32 and PLATFORM_IS_WINDOWS:
                value = value.astype(int64)

            # xxh3_64 only supports C-contiguous arrays.
            if not value.flags["C_CONTIGUOUS"]:
                value = ascontiguousarray(value)
        else:
            value = array([value])

        value = value.view(uint8)

        hashed_value = xxh3_64_hexdigest(value)
        hashed_name = xxh3_64_hexdigest(bytes(name, "utf-8"))
        names_with_hashed_values.append((hashed_name, hashed_value))

    return int(xxh3_64_hexdigest(array(names_with_hashed_values)), 16)


def to_real(
    data: RealOrComplexArray,
) -> RealArray:
    """Convert a NumPy array to a float NumPy array.

    Args:
        data: The NumPy array to be converted to real.

    Returns:
        A float NumPy array.
    """
    if data.dtype == complex128:
        return array(array(data, copy=False).real, dtype=float64)

    return cast(RealArray, data)
