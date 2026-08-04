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
"""Test the NumPy utilities."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import int32
from numpy import int64
from numpy import uint32
from numpy import uint64

from gemseo.util._numpy import hash_array


@pytest.mark.parametrize(("dtype_32", "dtype_64"), [(int32, int64), (uint32, uint64)])
def test_hash_array_32_bit_integers(dtype_32, dtype_64) -> None:
    """Check that a 32-bit integer array is hashed as its 64-bit equivalent."""
    assert hash_array(array([1, 2], dtype=dtype_32)) == hash_array(
        array([1, 2], dtype=dtype_64)
    )


@pytest.mark.parametrize("dtype_", [int32, int64, uint32, uint64])
def test_hash_array_integer_reference_value(dtype_) -> None:
    """Check the platform-independent hash of an integer array."""
    assert hash_array(array([1, 2], dtype=dtype_)) == "07ee86c281446bef"


def test_hash_array_default_integer_dtype() -> None:
    """Check that the default integer type is hashed as 64-bit on all platforms."""
    assert hash_array(array([1, 2])) == "07ee86c281446bef"
