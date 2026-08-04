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
"""Test the hashable NumPy array."""

from __future__ import annotations

import pytest
from numpy import arange
from numpy import array
from numpy import ascontiguousarray
from numpy import asfortranarray

from gemseo.core.problem.database import Database
from gemseo.util.hashable_ndarray import HashableNdarray

NON_CONTIGUOUS_ARRAYS = {
    "strided": arange(10.0)[::3],
    "reversed": arange(5.0)[::-1],
    "transposed": arange(6.0).reshape(2, 3).T,
    "fortran_ordered": asfortranarray(arange(6.0).reshape(2, 3)),
    "column": arange(6.0).reshape(2, 3)[:, 0],
    "sub_array": arange(24.0).reshape(2, 3, 4)[:, ::2, :],
}


def test_str() -> None:
    """Tests the string representation."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert str(x_hash) == str(x_array)


def test_repr() -> None:
    """Tests the __repr__ method."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert repr(x_hash) == str(x_array)


def test_unwrap() -> None:
    """Tests HashableNdarray unwrapping."""
    x_array = array([1.0, 1.0])
    x_hash = HashableNdarray(x_array)
    assert x_hash.unwrap() is x_hash.wrapped_array
    x_hash = HashableNdarray(x_array, copy=True)
    assert x_hash.unwrap() is not x_hash.wrapped_array
    assert (x_hash.unwrap() == x_array).all()


def test_hash_reference_value() -> None:
    """Check the hash of a C-contiguous array against a reference value."""
    assert hash(HashableNdarray(array([1.0, 2.0]))) == 7607585460423544033


@pytest.mark.parametrize(
    "array_",
    [array(1.0), array([]), arange(6.0).reshape(2, 3)],
    ids=["zero_dimensional", "empty", "two_dimensional"],
)
def test_hash_special_shapes(array_) -> None:
    """Check the hash of arrays whose shape is peculiar."""
    assert hash(HashableNdarray(array_)) == hash(HashableNdarray(array_.copy()))
    assert HashableNdarray(array_) == HashableNdarray(array_.copy())


@pytest.mark.parametrize(
    "array_", NON_CONTIGUOUS_ARRAYS.values(), ids=NON_CONTIGUOUS_ARRAYS.keys()
)
def test_non_contiguous_array(array_) -> None:
    """Check that a non-contiguous array is hashed as its C-contiguous copy."""
    assert not array_.flags.c_contiguous
    x_hash = HashableNdarray(array_)
    contiguous_x_hash = HashableNdarray(ascontiguousarray(array_))
    assert hash(x_hash) == hash(contiguous_x_hash)
    assert x_hash == contiguous_x_hash


def test_non_contiguous_wrapped_array() -> None:
    """Check that a non-contiguous array is wrapped as is."""
    array_ = arange(10.0)[::3]
    x_hash = HashableNdarray(array_)
    assert x_hash.wrapped_array is array_
    assert not x_hash.wrapped_array.flags.c_contiguous
    assert not x_hash.is_copy


def test_non_contiguous_copy_wrapped_array() -> None:
    """Check that copying a non-contiguous wrapped array preserves the hash."""
    array_ = arange(10.0)[::3]
    x_hash = HashableNdarray(array_)
    initial_hash = hash(x_hash)
    x_hash.copy_wrapped_array()
    assert x_hash.is_copy
    assert x_hash.wrapped_array.flags.c_contiguous
    assert hash(x_hash) == initial_hash
    assert x_hash == HashableNdarray(array_)


def test_non_contiguous_database_key() -> None:
    """Check that a non-contiguous array can be used as a database key."""
    x_vect = arange(10.0)[::3]
    database = Database()
    database.store(x_vect, {"f": 1.0})
    assert database[x_vect] == {"f": 1.0}
    assert database.get_iteration(x_vect) == 1
    assert (database.get_x_vect(1) == x_vect).all()
