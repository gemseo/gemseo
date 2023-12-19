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
"""Base class for converting data values to NumPy arrays and vice versa."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final
from typing import Union

from numpy import array as np_array
from numpy import concatenate
from numpy import ndarray

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.core.discipline_data import Data
    from gemseo.core.grammars.base_grammar import BaseGrammar

# The following is needed since a bool is a numbers.Number.
_NUMERIC_TYPES: Final[tuple[type]] = (int, float, complex)
"""The base types for numeric values."""

ValueType = Union[int, float, complex, ndarray]


class BaseDataConverter(ABC):
    """Base class for converting data values to NumPy arrays and vice versa.

    Typically,
    data are dictionary-like object that map names to values,
    such as :class:`.DisciplineData`.

    By default,
    a data converter can handle the conversion of a data value
    that is a standard number (``int``, ``float`` and ``complex``) or a 1D NumPy array.
    Other types could be handled in derived classes.

    A data converter can also be used
    to convert a data structure ``{data_name: data_value, ...}`` to a NumPy array
    and vice versa.
    In this class,
    a data structure is noted ``data``.

    For performance reasons,
    no checking or error handling is done when calling the methods of this class
    or of its derivatives.

    .. note::

        The data converter uses a grammar,
        and in particular its mapping from data names to data types,
        to convert a NumPy array from/to a data value.

    .. warning::

        Throughout this class, _NumPy array_ is equivalent to _1D numeric NumPy array_.
    """

    _grammar: BaseGrammar
    """The grammar providing the data types used for the conversions."""

    def __init__(self, grammar: BaseGrammar) -> None:
        """
        Args:
            grammar: The grammar providing the data types used for the conversions.
        """  # noqa: D205 D212 D415
        self._grammar = grammar

    def convert_value_to_array(
        self,
        name: str,
        value: ValueType,
    ) -> ndarray:
        """Convert a data value to a NumPy array.

        Args:
            name: The data name.
            value: The data value.

        Returns:
            The NumPy array.
        """
        if isinstance(value, _NUMERIC_TYPES):
            return np_array([value])
        return value

    def convert_array_to_value(self, name: str, array: ndarray) -> ValueType:
        """Convert a NumPy array to a data value.

        Args:
            name: The data name.
            array: The NumPy array to convert.

        Returns:
            The data value.
        """
        return self._convert_array_to_value(name, array)

    @staticmethod
    def get_value_size(name: str, value: ValueType) -> int:
        """Return the size of a data value.

        The size is typically what is returned by ``ndarray.size`` or ``len(list)``.
        The size of a number is 1.

        Args:
            name: The data name.
            value: The data value to get the size from.

        Returns:
            The size.
        """
        if isinstance(value, _NUMERIC_TYPES):
            return 1
        return value.size

    def compute_names_to_slices(
        self,
        names: Iterable[str],
        data: Data,
        names_to_sizes: Mapping[str, int] = MappingProxyType({}),
    ) -> tuple[dict[str, slice], int]:
        """Compute a mapping from data names to data value slices.

        The slices are relative to a NumPy array
        concatenating the data values associated with these data names.

        Args:
            data: The data structure.
            names: The data names.
            names_to_sizes: The mapping from the data names to the data sizes.
               If empty, it will be computed.

        Returns:
            The mapping from the data names to the data slices
            of the expected concatenated NumPy array.
            and the size of this array.
        """
        names_to_slices = {}
        get_size = self.get_value_size
        get_size_from_name = names_to_sizes.get
        start = 0
        end = 0
        for name in names:
            size = get_size_from_name(name)
            if size is None:
                size = get_size(name, data[name])
            end = start + size
            names_to_slices[name] = slice(start, end)
            start = end

        return names_to_slices, end

    def compute_names_to_sizes(
        self,
        names: Iterable[str],
        data: Data,
    ) -> dict[str, int]:
        """Compute a mapping from data names to data value sizes.

        .. seealso:: :meth:`.get_value_size`.

        Args:
            data: The data structure.
            names: The data names.

        Returns:
            The mapping from the data names to the data sizes.
        """
        get_size = self.get_value_size
        return {name: get_size(name, data[name]) for name in names}

    def convert_array_to_data(
        self,
        array: ndarray,
        names_to_slices: Mapping[str, slice],
    ) -> dict[str, ValueType]:
        """Convert a NumPy array to a data structure.

        .. seealso:: :meth:`.convert_array_to_value`

        Args:
            array: The NumPy array to slice.
            names_to_slices: The mapping from the data names to the array slices.

        Returns:
            The mapping from the data names to the array slices.
        """
        to_value = self.convert_array_to_value
        return {
            name: to_value(name, array[slice_])
            for name, slice_ in names_to_slices.items()
        }

    def convert_data_to_array(
        self,
        names: Iterable[str],
        data: Data,
    ) -> ndarray:
        """Convert a part of a data structure to a NumPy array.

        .. seealso:: :meth:`.convert_value_to_array`

        Args:
            data: The data structure.
            names: The data names which values will be concatenated.

        Returns:
            The concatenated NumPy array.
        """
        if not names:
            return np_array([])
        to_array = self.convert_value_to_array
        return concatenate(tuple(to_array(name, data[name]) for name in names))

    @abstractmethod
    def _convert_array_to_value(self, name: str, array: ndarray) -> ValueType:
        """Convert back a NumPy array to a data value.

        Args:
            name: The data name.
            array: The NumPy array to convert.

        Returns:
            The data value.
        """

    @abstractmethod
    def is_numeric(self, name: str) -> bool:
        """Check that a data item can be converted to a NumPy array.

        Args:
            name: The name of the data item.

        Returns:
            Whether the data item can be converted to a NumPy array.
        """
