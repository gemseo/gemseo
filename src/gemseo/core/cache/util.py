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

from typing import TYPE_CHECKING
from typing import cast

from numpy import array
from numpy import complex128
from numpy import float64
from numpy import ndarray
from xxhash import xxh3_64_hexdigest

from gemseo.util._numpy import hash_array

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray
    from gemseo.util.typing import RealOrComplexArray
    from gemseo.util.typing import StrKeyMapping


def hash_data(
    data: StrKeyMapping,
) -> int:
    """Hash data using xxh3_64 from the xxhash library.

    Args:
        data: The data to hash.

    Returns:
        The hash value of the data.
    """
    names_with_hashed_values = []

    for name in sorted(data):
        value = data.get(name)
        if value is None:
            continue

        if not isinstance(value, ndarray):
            value = array([value])

        hashed_value = hash_array(value)
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

    return cast("RealArray", data)
