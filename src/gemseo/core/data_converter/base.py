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
from functools import cache
from numbers import Complex
from pathlib import Path
from pathlib import PurePath
from types import NoneType
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar
from typing import cast

from numpy import array as np_array
from numpy import concatenate
from numpy import generic
from numpy import ndarray

from gemseo.util._numpy import string_dtype_kinds
from gemseo.util.constant import read_only_empty_dict
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta
from gemseo.util.portable_path import ensure_path_to_array
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.core.grammar.base import BaseGrammar
    from gemseo.util.typing import BytesArray
    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import StringArray
    from gemseo.util.typing import StrKeyMapping

    ArrayType = NumberArray | StringArray | BytesArray
    ValueType = int | float | complex | str | PurePath | ArrayType

T = TypeVar("T", bound="BaseGrammar")


class BaseDataConverter(Generic[T], metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for converting data values to NumPy arrays and vice versa.

    Typically,
    data are dictionary-like object that map names to values,
    such as [DisciplineData][gemseo.core.discipline.discipline_data.DisciplineData].

    By default,
    a data converter can handle the conversion of a data value
    that is a standard number (`int`, `float` and `complex`), a string,
    a path (`pathlib.PurePath` and its subclasses) or a 1D NumPy array.
    A string or a path is converted to a one-item string NumPy array;
    a path is stored with forward slashes whatever the platform.
    Any `pathlib.PurePath` is a path, and this cannot be changed in a derived class
    (the conversion of a path does not depend on `_non_array_types`),
    so that a full cache stores a path as it hashes and compares it.
    A path that can be `None`, e.g. declared as `Path | None`, is not supported:
    the conversion of `None` gives an object NumPy array
    that [HDF5Cache][gemseo.core.cache.hdf5.HDF5Cache] cannot store.
    Other types could be handled in derived classes.

    A data converter can also be used
    to convert a data structure `{data_name: data_value, ...}` to a NumPy array
    and vice versa.
    In this class,
    a data structure is noted `data`.

    For performance reasons,
    no checking or error handling is done when calling the methods of this class
    or of its derivatives.

    Note:
        The data converter uses a grammar,
        and in particular its mapping from data names to data types,
        to convert a NumPy array from/to a data value.

    Warning:
        Throughout this class, *NumPy array* is equivalent to *1D numeric NumPy array*,
        except for a string or a path value, which is a one-item string NumPy array.
    """

    _grammar: T
    """The grammar providing the data types used for the conversions."""

    _non_array_types: ClassVar[tuple[type, ...]] = (
        int,
        float,
        complex,
        Complex,
        str,
        PurePath,
    )
    """The base types that are not arrays like."""

    _is_continuous_types: ClassVar[tuple[type, ...]] = (
        float,
        complex,
        Complex,
        ndarray,
    )
    """The types that represent continuous data."""

    _is_numeric_types: ClassVar[tuple[type, ...]] = (
        int,
        *_is_continuous_types,
    )
    """The types that represent numeric data."""

    value_to_array_converters: ClassVar[
        dict[str, Callable[[ValueType], ArrayType]]
    ] = {}
    """The mapping from data names to functions converting a data value to an array."""

    array_to_value_converters: ClassVar[
        dict[str, Callable[[ArrayType], ValueType]]
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
    ) -> ArrayType:
        """Convert a data value to a NumPy array.

        A path is converted to a one-item string array,
        with forward slashes whatever the platform.

        Args:
            name: The data name.
            value: The data value.

        Returns:
            The NumPy array.
        """
        converters = self.value_to_array_converters
        if converters and (converter := converters.get(name)):
            return converter(value)
        value = ensure_path_to_array(value)
        if isinstance(value, ndarray):
            return value
        return np_array([value])

    def convert_array_to_value(self, name: str, array: ArrayType) -> ValueType:
        """Convert a NumPy array to a data value.

        A one-item string array, or bytes array decoded with UTF-8,
        is converted to a path
        when the grammar declares only path types for this data,
        apart from `None`, e.g. `Path | None`.
        The path has the declared type
        if the grammar declares a single path type, e.g. `PureWindowsPath`,
        and is a `Path` otherwise, e.g. for `PurePosixPath | PureWindowsPath`.
        See the note below about `Path`.
        When the grammar declares a path type and another type,
        e.g. `str | Path`, `Literal["auto"] | Path` or `int | Path`,
        a one-item string array is converted to a `str`.
        When the grammar declares no path type,
        a one-item string array is converted as any other array.

        Args:
            name: The data name.
            array: The NumPy array to convert.

        Returns:
            The data value.

        Note:
            `Path` rather than `PurePath` is used when several path types are declared
            because `Path` is concrete and can be used for I/O.
            Its concrete class, `PosixPath` or `WindowsPath`,
            depends on the platform but is the same throughout a process,
            so that the values read back are consistent.
        """
        converters = self.array_to_value_converters
        if converters and (converter := converters.get(name)):
            return converter(array)
        # Checking the dtype first avoids looking for a path type in the grammar
        # when the array cannot store a path.
        kind = array.dtype.kind
        if (
            kind in string_dtype_kinds
            and array.size == 1
            and (type_ := self._get_path_type(name))
        ):
            value = str(array[0]) if kind == "U" else bytes(array[0]).decode()
            return type_(value)
        if self._has_type(name, self._non_array_types):
            item = array[0]
            # A NumPy scalar is converted to a Python scalar because most of the NumPy
            # scalar types, such as int64, do not derive from the Python type they
            # mirror, which makes the grammar validation fail downstream.
            # The item is not a NumPy scalar when the array dtype is object,
            # in that case it is already a Python object.
            if isinstance(item, generic):
                item = item.item()
            return cast("ValueType", item)
        return array

    @classmethod
    def get_value_size(cls, name: str, value: ValueType) -> int:
        """Return the size of a data value.

        The size is typically what is returned by `ndarray.size`.
        The size of a number, a string or a path is 1.

        Args:
            name: The data name.
            value: The data value to get the size from.

        Returns:
            The size.

        Raises:
            TypeError: When the value is neither a number, a string, a path
                nor an array, i.e. when it has no `size` attribute.
        """
        getters = cls.value_size_getters
        if getters and (getter := getters.get(name)):
            return getter(value)
        if isinstance(value, cls._non_array_types):
            return 1

        try:
            return cast("NumberArray", value).size
        except AttributeError as error:
            msg = (
                f"The size of the value of {name} cannot be computed "
                f"because its type {type(value).__name__} "
                "is neither a number, a string, a path nor an array."
            )
            raise TypeError(msg) from error

    def compute_name_to_slice(
        self,
        names: Iterable[str],
        data: StrKeyMapping,
        name_to_size: Mapping[str, int] = read_only_empty_dict,
    ) -> tuple[dict[str, slice], int]:
        """Compute a mapping from data names to data value slices.

        The slices are relative to a NumPy array
        concatenating the data values associated with these data names.

        Args:
            names: The data names.
            data: The data structure.
            name_to_size: The mapping from the data names to the data sizes.
                If empty, it will be computed.

        Returns:
            The mapping from the data names to the data slices
            of the expected concatenated NumPy array,
            and the size of this array.
        """
        name_to_slice = {}
        get_size = self.get_value_size
        get_size_from_name = name_to_size.get
        start = 0
        end = 0
        for name in names:
            size = get_size_from_name(name)
            if size is None:
                size = get_size(name, data[name])
            end = start + size
            name_to_slice[name] = slice(start, end)
            start = end

        return name_to_slice, end

    def compute_name_to_size(
        self, names: Iterable[str], data: StrKeyMapping
    ) -> dict[str, int]:
        """Compute a mapping from data names to data value sizes.

        See Also:
            [get_value_size()][gemseo.core.data_converter.base.BaseDataConverter.get_value_size].

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
        array: ArrayType,
        name_to_slice: Mapping[str, slice],
    ) -> dict[str, ValueType]:
        """Convert a NumPy array to a data structure.

        See Also:
            [convert_array_to_value()][gemseo.core.data_converter.base.BaseDataConverter.convert_array_to_value].

        Args:
            array: The NumPy array to slice.
            name_to_slice: The mapping from the data names to the array slices.

        Returns:
            The mapping from the data names to the data values.
        """
        to_value = self.convert_array_to_value
        return {
            name: to_value(name, array[..., slice_])
            for name, slice_ in name_to_slice.items()
        }

    def convert_data_to_array(
        self,
        names: Collection[str],
        data: StrKeyMapping,
    ) -> ArrayType:
        """Convert a part of a data structure to a NumPy array.

        See Also:
            [convert_value_to_array()][gemseo.core.data_converter.base.BaseDataConverter.convert_value_to_array].

        Args:
            data: The data structure.
            names: The data names which values will be concatenated.

        Returns:
            The concatenated NumPy array.

        Raises:
            TypeError: When the values of some data are strings or paths
                and the values of others are not.
        """
        if not names:
            return np_array([])
        to_array = self.convert_value_to_array
        arrays = tuple(to_array(name, data[name]) for name in names)
        string_names = [
            name
            for name, array in zip(names, arrays, strict=False)
            if array.dtype.kind in string_dtype_kinds
        ]
        if string_names and len(string_names) != len(names):
            other_names = sorted(set(names).difference(string_names))
            msg = (
                f"The values of {pretty_str(string_names)} are strings or paths "
                f"while those of {pretty_str(other_names)} are not; "
                "they cannot be concatenated into one NumPy array."
            )
            raise TypeError(msg)
        return cast("ArrayType", concatenate(arrays, axis=-1))

    def is_numeric(self, name: str) -> bool:
        """Return whether a grammar item is numeric.

        Args:
            name: The name of the grammar item.

        Returns:
            Whether the grammar item is numeric.
        """
        return self._has_type(name, self._is_numeric_types)

    def is_continuous(self, name: str) -> bool:
        """Return whether a grammar item is continuous.

        Args:
            name: The name of the grammar item.

        Returns:
            Whether the grammar item is continuous.
        """
        return self._has_type(name, self._is_continuous_types)

    def _get_path_type(self, name: str) -> type[PurePath | str] | None:
        """Return the type to read a string of a grammar item with a path type.

        Args:
            name: The name of the grammar item.

        Returns:
            `None` if the grammar item declares no path type,
            `str` if it declares a path type and a type that is not a path type,
            the path type if it declares a single one,
            `Path` if it declares several ones;
            `None` is not taken into account, e.g. `Path | None` gives `Path`.
        """
        return None

    @staticmethod
    def _select_path_type(*types: Any) -> type[PurePath | str] | None:
        """Select the type to read a string among types including a path type.

        The selection is cached, whatever the grammar,
        with an unbounded cache
        because the number of distinct annotations in a workflow is bounded.

        Args:
            *types: The types, which may not all be classes.

        Returns:
            `None` if there is no path type among `types`,
            `str` if there are a path type and a type that is not a path type,
            e.g. `str`, `Literal["auto"]` or a `StrEnum`,
            the path type if there is a single one,
            `Path` if there are several ones;
            `NoneType` is not taken into account, e.g. `Path | None` gives `Path`.

        Note:
            `Path` rather than `PurePath` is used when there are several path types
            because `Path` is concrete and can be used for I/O.
            Its concrete class, `PosixPath` or `WindowsPath`,
            depends on the platform but is the same throughout a process,
            so that the values read back are consistent.
        """
        try:
            return _compute_path_type(types)
        except TypeError:
            # A type with an unhashable metadata cannot be a cache key,
            # e.g. list[Annotated[str, {"k": 1}]].
            # The computation is deterministic and is not wrapped here,
            # so a TypeError raised by it, rather than by the hashing,
            # is raised again.
            return _compute_path_type.__wrapped__(types)

    @abstractmethod
    def _has_type(self, name: str, types: tuple[type, ...]) -> bool:
        """Return whether the type of a grammar item is one of the expected types.

        Args:
            name: The name of the grammar item.
            types: The expected types.

        Returns:
            Whether the type of the grammar item is one of the expected types.
        """


@cache
def _compute_path_type(types: tuple[Any, ...]) -> type[PurePath | str] | None:
    """Select the type to read a string among types including a path type.

    Args:
        types: The types, which may not all be classes.

    Returns:
        The type to read a string,
        see `BaseDataConverter._select_path_type`.
    """
    path_types = set()
    has_other_type = False
    for type_ in types:
        if type_ is NoneType:
            continue
        if isinstance(type_, type) and issubclass(type_, PurePath):
            path_types.add(type_)
        else:
            # This type could hold a str, e.g. str, Literal["auto"] or a StrEnum.
            has_other_type = True
    if not path_types:
        return None
    if has_other_type:
        return str
    if len(path_types) == 1:
        return next(iter(path_types))
    return Path
