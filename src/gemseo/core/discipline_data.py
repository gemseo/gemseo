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

from collections import abc
from typing import Any
from typing import Generator
from typing import Mapping
from typing import MutableMapping

import pandas as pd

from gemseo.core.namespaces import NamespacesMapping

Data = Mapping[str, Any]
MutableData = MutableMapping[str, Any]


class DisciplineData(abc.MutableMapping):
    """A dict-like class for handling disciplines data.

    This class replaces a standard dictionary that was previously used for storing
    discipline data.
    It allows handling values bound to :class:`pandas.DataFrame`
    as if they were multiple items bound to :class:`numpy.ndarray`.
    Then, an object of this class may be used as if it was a standard dictionary
    containing :class:`numpy.ndarray`,
    which is the assumption made by the clients of the :class:`.MDODiscipline` subclasses.

    As compared to a standard dictionary,
    the methods of this class may hide the values bound to :class:`pandas.DataFrame`
    and instead expose the items of those latter as if they belonged
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

    Examples:
        >>> from gemseo.core.discipline_data import DisciplineData
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = {
        ...         'x': 0,
        ...         'y': pd.DataFrame(data={'a': np.array([0])}),
        ...     }
        >>> disc_data = DisciplineData(data)
        >>> disc_data['x']
        0
        >>> disc_data['y']
           a
        0  0
        >>> # DataFrame content can be accessed with the ~ separator.
        >>> disc_data['y~a']
        array([0])
        >>> # New columns can be inserted into a DataFrame with the ~ separator.
        >>> disc_data['y~b'] = np.array([1])
        >>> data['y']['b']
        0    1
        Name: b, dtype: int64
        >>> # New DataFrame can be added.
        >>> disc_data['z~c'] = np.array([2])
        >>> data['z']
           c
        0  2
        >>> type(data['z'])
        <class 'pandas.core.frame.DataFrame'>
        >>> # DataFrame columns can be deleted.
        >>> del disc_data['z~c']
        >>> data['z']
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
    """The character used to separate the shared dict key from the column of a
    pandas DataFrame."""

    def __init__(
        self,
        data: MutableData,
        input_to_namespaced: NamespacesMapping = None,
        output_to_namespaced: NamespacesMapping = None,
    ) -> None:
        """
        Args:
            data: A dict-like object or a :class:`.DisciplineData` object.
            input_to_namespaced: The mapping from input data names to their prefixed names.
            output_to_namespaced: The mapping from output data names to their prefixed names.
        """  # noqa: D205, D212, D415
        if isinstance(data, self.__class__):
            # By construction, data's keys shall have been already checked.
            self.__data = getattr(data, f"_{self.__class__.__name__}__data")
        else:
            self.__data = data
            self.__check_keys(*data)

        self.__input_to_namespaced = input_to_namespaced
        self.__output_to_namespaced = output_to_namespaced

    def __getitem_with_ns(self, key: str) -> Any:
        """Return an item whose key contains a namespace prefix.

        Args:
            key: The required key.

        Returns:
            The item value, or None if the key is not present.
        """
        if key in self.__data:
            value = self.__data[key]
            if isinstance(value, MutableMapping):
                return self.__class__(value)
            else:
                return value

        if self.SEPARATOR in key:
            df_key, column = key.split(self.SEPARATOR)
            return self.__data[df_key][column].to_numpy()

    def __getitem__(self, key: str) -> Any:
        value = self.__getitem_with_ns(key)
        if value is not None:
            return value

        if self.__input_to_namespaced is not None:
            key_with_ns = self.__input_to_namespaced.get(key)
            if key_with_ns is not None:
                return self.__getitem_with_ns(key_with_ns)

        if self.__output_to_namespaced is not None:
            key_with_ns = self.__output_to_namespaced.get(key)
            if key_with_ns is not None:
                return self.__getitem_with_ns(key_with_ns)

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
            self.__data[df_key] = pd.DataFrame(data={column: value})
            return

        df = self.__data[df_key]

        if not isinstance(df, pd.DataFrame):
            raise KeyError(
                f"Cannot set {key} because {df_key} is not bound to a "
                "pandas DataFrame."
            )

        self.__data[df_key][column] = value

    def __delitem__(self, key: str) -> None:
        if key in self.__data:
            del self.__data[key]
        elif self.SEPARATOR in key:
            df_key, column = key.split(self.SEPARATOR)
            del self.__data[df_key][column]
        else:
            raise KeyError(key)

    def __iter__(self) -> Generator[str, None, None]:
        for key, value in self.__data.items():
            if isinstance(value, pd.DataFrame):
                prefix = key + self.SEPARATOR
                for name in value.keys():
                    yield prefix + name
            else:
                yield key

    def __len__(self) -> int:
        length = 0
        for value in self.__data.values():
            if isinstance(value, pd.DataFrame):
                length += len(value.columns)
            else:
                length += 1
        return length

    def __repr__(self) -> str:
        return repr(self.__data)

    def copy(self) -> DisciplineData:
        """Create a shallow copy.

        Returns:
            The shallow copy.
        """
        return self.__class__(self.__data.copy())

    def __check_keys(self, *keys: str) -> None:
        """Verify that keys do not contain the separator.

        Args:
            *keys: The keys to be checked.

        Raises:
            KeyError: If a key contains the separator.
        """
        for key in keys:
            if self.SEPARATOR in key:
                msg = f"{key} shall not contain {self.SEPARATOR}"
                raise KeyError(msg)
