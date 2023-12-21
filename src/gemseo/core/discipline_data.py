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
# Contributors:
# Antoine DECHAUME
"""Provide a dict-like class for storing disciplines data."""

from __future__ import annotations

from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import MutableMapping
from copy import copy
from copy import deepcopy
from pathlib import Path
from pathlib import PurePath
from typing import Any

from numpy import ndarray
from pandas import DataFrame

from gemseo.core.namespaces import NamespacesMapping
from gemseo.core.namespaces import namespaces_separator
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.portable_path import to_os_specific

Data = Mapping[str, Any]
MutableData = MutableMapping[str, Any]


class DisciplineData(
    MutableMapping[str, Any], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """A dict-like class for handling disciplines data.

    This class replaces a standard dictionary that was previously used for storing
    discipline data.
    It allows handling values bound to :class:`pandas.DataFrame`
    as if they were multiple items bound to :class:`numpy.ndarray`.
    Then, an object of this class may be used as if it was a standard dictionary
    containing :class:`numpy.ndarray`,
    which is the assumption made by the clients of the
    :class:`.MDODiscipline` subclasses.

    As compared to a standard dictionary,
    the methods of this class may hide the values bound to :class:`pandas.DataFrame`
    and instead expose the items of those later as if they belonged
    to the dictionary.

    If a dict-like object is provided when creating a :class:`.DisciplineData` object,
    its contents is shared with this latter,
    such that any changes performed via a :class:`.DisciplineData` object
    is reflected into the passed in dict-like object.

    If a :class:`.DisciplineData` is created from another one,
    their contents are shared.

    A separator, by default ``~``, is used to identify the keys of the items that
    are bound to an array inside a :class:`pandas.DataFrame`.
    Such a key is composed of the key from the shared dict-like object,
    the separator and the name of the target column of the :class:`pandas.DataFrame`.
    When such a key is queried,
    a :class:`numpy.ndarray` view of the :class:`pandas.DataFrame` column is provided.

    Printing a :class:`.DisciplineData` object prints the shared dict-like object.

    Nested dictionaries are also supported.

    .. warning:: Nested DataFrame (i.e. in a nested dictionary) are not supported.

    Examples:
        >>> from gemseo.core.discipline_data import DisciplineData
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = {
        ...     "x": 0,
        ...     "y": pd.DataFrame(data={"a": np.array([0])}),
        ... }
        >>> disc_data = DisciplineData(data)
        >>> disc_data["x"]
        0
        >>> disc_data["y"]
           a
        0  0
        >>> # DataFrame content can be accessed with the ~ separator.
        >>> disc_data["y~a"]
        array([0])
        >>> # New columns can be inserted into a DataFrame with the ~ separator.
        >>> disc_data["y~b"] = np.array([1])
        >>> data["y"]["b"]
        0    1
        Name: b, dtype: int64
        >>> # New DataFrame can be added.
        >>> disc_data["z~c"] = np.array([2])
        >>> data["z"]
           c
        0  2
        >>> type(data["z"])
        <class 'pandas.core.frame.DataFrame'>
        >>> # DataFrame's columns can be deleted.
        >>> del disc_data["z~c"]
        >>> data["z"]
        Empty DataFrame
        Columns: []
        Index: [0]
        >>> # Iterating is only done over the exposed keys.
        >>> list(disc_data)
        ['x', 'y~a', 'y~b']
        >>> # The length is consistent with the iterator.
        >>> len(disc_data)
        3
    """

    SEPARATOR = "~"
    """The character used to separate the shared dict key from the column of a pandas
    DataFrame."""

    __data: MutableData
    """The internal dict-like object."""

    __input_to_namespaced: NamespacesMapping
    """The namespace mapping for the inputs."""

    __output_to_namespaced: NamespacesMapping
    """The namespace mapping for the outputs."""

    def __init__(
        self,
        data: MutableData | None = None,
        input_to_namespaced: NamespacesMapping | None = None,
        output_to_namespaced: NamespacesMapping | None = None,
    ) -> None:
        """
        Args:
            data: A dict-like object or a :class:`.DisciplineData` object.
                If ``None``, an empty dictionary is used.
            input_to_namespaced: The mapping from input data names
                to their prefixed names.
            output_to_namespaced: The mapping from output data names
                to their prefixed names.
        """  # noqa: D205, D212, D415
        if isinstance(data, self.__class__):
            # By construction, data's keys shall have been already checked.
            # We demangle __data to keep it private because this is an implementation
            # detail.
            self.__data = getattr(data, "_DisciplineData__data")  # noqa:B009
        elif data is None:
            self.__data = {}
        else:
            if not isinstance(data, MutableMapping):
                raise TypeError(
                    f"Invalid type for data, got {type(data)},"
                    " while expected a MutableMapping."
                )
            self.__check_keys(*data)
            self.__data = data

        self.__input_to_namespaced = (
            input_to_namespaced if input_to_namespaced is not None else {}
        )
        self.__output_to_namespaced = (
            output_to_namespaced if output_to_namespaced is not None else {}
        )

    def __getitem__(self, key: str) -> Any:
        if key in self.__data:
            return self.__data[key]

        if self.SEPARATOR in key:
            df_key, column = key.split(self.SEPARATOR)
            return self.__data[df_key][column].to_numpy()

        if self.__input_to_namespaced:
            key_with_ns = self.__input_to_namespaced.get(key)
            if key_with_ns is not None:
                return self[key_with_ns]

        if self.__output_to_namespaced:
            key_with_ns = self.__output_to_namespaced.get(key)
            if key_with_ns is not None:
                return self[key_with_ns]

        raise KeyError(key)

    def __setitem__(
        self,
        key: str,
        value: Any,
    ) -> None:
        if self.SEPARATOR not in key:
            self.__data[key] = value
            return

        df_key, column = key.split(self.SEPARATOR)

        if df_key not in self.__data:
            self.__data[df_key] = DataFrame(data={column: value})
            return

        df = self.__data[df_key]

        if not isinstance(df, DataFrame):
            raise KeyError(
                f"Cannot set {key} because {df_key} is not bound to a "
                "pandas DataFrame."
            )

        self.__data[df_key][column] = value

    def __delitem__(self, key: str) -> None:
        __data = self.__data
        if key in __data:
            del __data[key]
        elif self.SEPARATOR in key:
            df_key, column = key.split(self.SEPARATOR)
            del __data[df_key][column]
            # Remove the empty DataFrame.
            if __data[df_key].size == 0:
                del __data[df_key]
        else:
            raise KeyError(key)

    def __iter__(self) -> Generator[str, None, None]:
        for key, value in self.__data.items():
            if isinstance(value, DataFrame):
                prefix = key + self.SEPARATOR
                for name in value:
                    yield prefix + name
            else:
                yield key

    def __len__(self) -> int:
        length = 0
        for value in self.__data.values():
            if isinstance(value, DataFrame):
                length += len(value.columns)
            else:
                length += 1
        return length

    def __repr__(self) -> str:
        return repr(self.__data)

    def __copy__(self) -> DisciplineData:
        copy_ = DisciplineData({})
        data = copy(self.__data)
        # Shallow copy the DataFrames such that only their data are shared.
        for k, v in data.items():
            if isinstance(v, DataFrame):
                data[k] = v.copy(deep=False)
        copy_.__data = data
        copy_.input_to_namespaced = copy(self.__input_to_namespaced)
        copy_.output_to_namespaced = copy(self.__output_to_namespaced)
        return copy_

    def __deepcopy__(self, memo: Mapping | None = None) -> DisciplineData:
        copy_ = DisciplineData({})
        data = {}
        for k, v in self.__data.items():
            if isinstance(v, DataFrame):
                data[k] = v.copy(deep=True)
            elif isinstance(v, ndarray):
                data[k] = v.copy()
            else:
                data[k] = deepcopy(v)

        copy_.__data = data
        copy_.input_to_namespaced = deepcopy(self.__input_to_namespaced)
        copy_.output_to_namespaced = deepcopy(self.__output_to_namespaced)
        return copy_

    def clear(self) -> None:  # noqa: D102
        self.__data.clear()

    def copy(
        self,
        keys: Iterable[str] = (),
        with_namespace: bool = True,
    ) -> DisciplineData:
        """Create a shallow copy.

        Args:
            keys: The names of the items to keep, if empty then keep them all.
            with_namespace: Whether to the keys are prefixed with the namespace.

        Returns:
            The shallow copy.
        """
        copy_ = self.__copy__()
        if keys:
            copy_.restrict(*keys)
        if not with_namespace:
            for k in tuple(copy_.keys()):
                copy_[k.rsplit(namespaces_separator, 1)[-1]] = copy_.pop(k)
        return copy_

    def update(
        self,
        other: Mapping[str, Any],
        exclude: Iterable[str] = (),
    ) -> None:
        """Update from another mapping but for some keys.

        Args:
            other: The data to update from.
            exclude: The keys that shall not be updated.
        """
        for key in other.keys() - exclude:
            self[key] = other[key]

    def restrict(
        self,
        *keys: str,
    ) -> None:
        """Remove all but the given keys.

        Args:
            *keys: The keys of the elements to keep.
        """
        for name in self.keys() - keys:
            del self[name]

    def __check_keys(self, *keys: str) -> None:
        """Verify that keys do not contain the separator.

        Args:
            *keys: The keys to be checked.

        Raises:
            KeyError: If a key contains the separator.
        """
        for key in keys:
            if self.SEPARATOR in key:
                msg = f"The key {key} shall not contain {self.SEPARATOR}."
                raise KeyError(msg)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Work on a copy to avoid changing self.
        state_data = state[f"_{DisciplineData.__name__}__data"].copy()
        for item_name, item_value in self.__data.items():
            if isinstance(item_value, Path):
                # This is needed to handle the case where serialization and
                # deserialization are not made on the same platform.
                state_data[item_name] = to_os_specific(item_value)
        return state

    def __setstate__(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.__dict__.update(state)
        state_data = state[f"_{DisciplineData.__name__}__data"]
        for item_name, item_value in state_data.items():
            if isinstance(item_value, PurePath):
                self.__data[item_name] = Path(item_value)
