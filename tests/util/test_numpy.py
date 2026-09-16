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
from numpy import float64
from numpy import int32
from numpy import int64
from numpy import ndarray
from numpy import shares_memory
from numpy import uint32
from numpy import uint64
from numpy.testing import assert_equal

from gemseo.util._numpy import freeze_array
from gemseo.util._numpy import hash_array
from gemseo.util.testing.helper import assert_exception


def test_freeze_array() -> None:
    """Check that an array is frozen as a copy sharing no memory with the original."""
    original = array([1.0, 2.0])
    frozen = freeze_array(original)

    assert not frozen.flags.writeable
    assert_equal(frozen, original)
    # The original is left alone and the two arrays share no memory.
    assert original.flags.writeable
    original[0] = 3.0
    assert_equal(frozen, array([1.0, 2.0]))


def test_frozen_array_cannot_be_thawed_through_its_base() -> None:
    """Check that no array reachable from a frozen array can be made writeable.

    A frozen array is a view of an immutable `bytes` buffer,
    so neither it nor any array between it and that buffer owns its data,
    and NumPy refuses the writeable flag to all of them.
    """
    frozen = freeze_array(array([[1.0, 2.0], [3.0, 4.0]]))

    array_ = frozen
    while isinstance(array_, ndarray):
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            array_.setflags(write=True)

        array_ = array_.base

    # The chain of the bases ends on an immutable buffer.
    assert isinstance(array_, bytes)


def test_frozen_array_forbids_item_assignment(snapshot) -> None:
    """Check that a frozen array cannot be mutated in place."""
    with assert_exception(ValueError, snapshot):
        freeze_array(array([1.0, 2.0]))[0] = 3.0


def test_frozen_array_cannot_be_unfrozen(snapshot) -> None:
    """Check that the writeable flag of a frozen array cannot be re-enabled."""
    with assert_exception(ValueError, snapshot):
        freeze_array(array([1.0, 2.0])).setflags(write=True)


def test_freeze_array_returns_a_view_of_a_frozen_array() -> None:
    """Check that freezing a frozen array returns a view of it, not a copy.

    The data of a frozen array cannot change, so there is nothing to copy;
    a view is returned rather than the array itself,
    so that an owner storing the result shares no array object with the caller,
    whose reassignment of the shape, the strides or the data type
    of a read-only array would otherwise reach what the owner stores.
    """
    frozen = freeze_array(array([1.0, 2.0]))
    refrozen = freeze_array(frozen)

    assert refrozen is not frozen
    assert shares_memory(refrozen, frozen)
    assert not refrozen.flags.writeable
    assert_equal(refrozen, frozen)

    frozen.shape = (2, 1)
    assert refrozen.shape == (2,)


def test_freeze_array_copies_a_read_only_view_of_a_writeable_array() -> None:
    """Check that a read-only array is copied when its base is writeable.

    Such an array is not frozen: NumPy re-enables the writeable flag of its base,
    so the array is copied onto an immutable buffer like a writeable one.
    """
    original = array([1.0, 2.0])
    read_only_view = original.view()
    read_only_view.setflags(write=False)

    frozen = freeze_array(read_only_view)

    assert frozen is not read_only_view
    assert isinstance(frozen.base, ndarray)
    assert isinstance(frozen.base.base, bytes)
    original[0] = 3.0
    assert_equal(frozen, array([1.0, 2.0]))


def test_frozen_array_is_a_plain_array() -> None:
    """Check that a frozen array is a plain NumPy array.

    A subclass would leak into every array computed from a frozen one
    and into every pickle of its owner,
    and a frozen array must be hashable like any other array.
    """
    original = array([1.0, 2.0])
    frozen = freeze_array(original)

    assert type(frozen) is ndarray
    assert type(frozen.max()) is float64
    assert hash_array(frozen) == hash_array(original)


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
