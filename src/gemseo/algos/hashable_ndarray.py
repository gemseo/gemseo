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
"""A hashable NumPy array."""

from __future__ import annotations

from typing import Any

from numpy import array as np_array
from numpy import array_equal
from numpy import ndarray
from numpy import uint8
from xxhash import xxh3_64_hexdigest


class HashableNdarray:
    """HashableNdarray wrapper for ndarray objects.

    Instances of ndarray are not HashableNdarray, meaning they cannot be added to sets,
    nor used as keys in dictionaries. This is by design, ndarray objects are mutable,
    and therefore cannot reliably implement the __hash__() method.

    The HashableNdarray class allows a way around this limitation. It implements the
    required methods for HashableNdarray objects in terms of a array ndarray object.
    This can be either a copied instance (which is safer) or the original object (which
    requires the user to be careful enough not to modify it).
    """

    __array: ndarray
    """The wrapped_array array, either the original one or a copy."""

    __copy: bool
    """Whether the wrapped_array array is a copy of the original one."""

    __hash: int
    """The hash of the wrapped_array array."""

    def __init__(self, array: ndarray, copy: bool = False) -> None:
        """
        Args:
            array: The array that must be array.
            copy: Whether the array is copied.
        """  # noqa: D205, D212, D415
        self.__copy = copy
        self.__hash = int(xxh3_64_hexdigest(array.view(uint8)), 16)
        self.__array = np_array(array) if copy else array

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__) or hash(self) != hash(other):
            return False
        return array_equal(self.__array, other.__array)

    def __hash__(self) -> int:
        return self.__hash

    def __repr__(self) -> str:
        return str(self.__array)

    @property
    def wrapped_array(self) -> ndarray:
        """The wrapped_array array."""
        return self.__array

    @property
    def is_copy(self) -> bool:
        """Whether the wrapped_array array as a copy of the original one."""
        return self.__copy

    def copy_wrapped_array(self) -> None:
        """Wrap a copy of the original array if it was not yet."""
        if not self.__copy:
            self.__copy = True
            self.__array = np_array(self.__array)

    def unwrap(self) -> ndarray:
        """Return the array ndarray.

        Returns:
            The array ndarray, or a copy if the wrapper is ``copy``.
        """
        if self.__copy:
            return np_array(self.__array)

        return self.__array
