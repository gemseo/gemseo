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

from abc import abstractmethod
from numbers import Complex
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from numpy import array as np_array
from numpy import concatenate
from numpy import ndarray

from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.core.grammars.base_grammar import BaseGrammar
    from gemseo.typing import NumberArray
    from gemseo.typing import StrKeyMapping

    ValueType = int | float | complex | NumberArray


T = TypeVar("T", bound="BaseGrammar")


class BaseDataConverter(Generic[T], metaclass=ABCGoogleDocstringInheritanceMeta):
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

        Throughout this class, *NumPy array* is equivalent to *1D numeric NumPy array*.
    """

    _grammar: T
    """The grammar providing the data types used for the conversions."""

    _NON_ARRAY_TYPES: ClassVar[tuple[type, ...]] = (
        int,
        float,
        complex,
        Complex,
        str,
    )
    """The base types that are not arrays like."""

    _IS_CONTINUOUS_TYPES: ClassVar[tuple[type, ...]] = (
        float,
        complex,
        Complex,
        ndarray,
    )
    """The types that represent continuous data."""

    _IS_NUMERIC_TYPES: ClassVar[tuple[type, ...]] = (
        int,
        *_IS_CONTINUOUS_TYPES,
    )
    """The types that represent numeric data."""

    value_to_array_converters: ClassVar[
        dict[str, Callable[[ValueType], NumberArray]]
    ] = {}
    """The mapping from data names to functions converting a data value to an array."""

    array_to_value_converters: ClassVar[
        dict[str, Callable[[NumberArray], ValueType]]
    ] = {}
    """The mapping from data names to functions converting an array to a data value."""

    value_size_getters: ClassVar[dict[str, Callable[[ValueType], int]]] = {}
    """The mapping from data names to functions returning the size of a data value."""

    def __init__(self, grammar: T) -> None:
        """
        Args:
            grammar: The grammar providing the data types used for the conversions.
        """  # noqa: D205 D212 D415
        self._grammar = grammar

    def convert_value_to_array(
        self,
        name: str,
        value: ValueType,
    ) -> NumberArray:
        """Convert a data value to a NumPy array.

        Args:
            name: The data name.
            value: The data value.

        Returns:
            The NumPy array.
        """
        if self.value_to_array_converters and (
            converter := self.value_to_array_converters.get(name)
        ):
            return converter(value)
        if isinstance(value, ndarray):
            return value
        return np_array([value])

    def convert_array_to_value(self, name: str, array: NumberArray) -> ValueType:
        """Convert a NumPy array to a data value.

        Args:
            name: The data name.
            array: The NumPy array to convert.

        Returns:
            The data value.
        """
        if self.array_to_value_converters and (
            converter := self.array_to_value_converters.get(name)
        ):
            return converter(array)
        if self._has_type(name, self._NON_ARRAY_TYPES):
            return array[0]
        return array

    @classmethod
    def get_value_size(cls, name: str, value: ValueType) -> int:
        """Return the size of a data value.

        The size is typically what is returned by ``ndarray.size`` or ``len(list)``.
        The size of a number is 1.

        Args:
            name: The data name.
            value: The data value to get the size from.

        Returns:
            The size.
        """
        if cls.value_size_getters and (getter := cls.value_size_getters.get(name)):
            return getter(value)
        if isinstance(value, cls._NON_ARRAY_TYPES):
            return 1

        return value.size

    def compute_names_to_slices(
        self,
        names: Iterable[str],
        data: StrKeyMapping,
        names_to_sizes: Mapping[str, int] = READ_ONLY_EMPTY_DICT,
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
        self, names: Iterable[str], data: StrKeyMapping
    ) -> dict[str, int]:
        """Compute a mapping from data names to data value sizes.

        .. seealso:: :meth:`.get_value_size`.

        Args:
            names: The data names.
            data: The data structure.

        Returns:
            The mapping from the data names to the data sizes.
        """
        get_size = self.get_value_size
        return {name: get_size(name, data[name]) for name in names}

    def convert_array_to_data(
        self,
        array: NumberArray,
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
            name: to_value(name, array[..., slice_])
            for name, slice_ in names_to_slices.items()
        }

    def convert_data_to_array(
        self,
        names: Iterable[str],
        data: StrKeyMapping,
    ) -> NumberArray:
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
        return concatenate(tuple(to_array(name, data[name]) for name in names), axis=-1)

    def is_numeric(self, name: str) -> bool:
        """Return whether a grammar item is numeric.

        Args:
            name: The name of the grammar item.

        Returns:
            Whether the grammar item is numeric.
        """
        return self._has_type(name, self._IS_NUMERIC_TYPES)

    def is_continuous(self, name: str) -> bool:
        """Return whether a grammar item is continuous.

        Args:
            name: The name of the grammar item.

        Returns:
            Whether the grammar item is continuous.
        """
        return self._has_type(name, self._IS_CONTINUOUS_TYPES)

    @abstractmethod
    def _has_type(self, name: str, types: tuple[type, ...]) -> bool:
        """Return the type of grammar item has an expected type.

        Args:
            name: The name of the grammar item.
            types: The expected types.

        Returns:
            Whether the type of the grammar item is one of the expected types.
        """
