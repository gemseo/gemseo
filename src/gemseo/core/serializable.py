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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Gilberto Ruiz
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The baseclass for serializable |g| objects."""
from __future__ import annotations

from abc import abstractmethod
from multiprocessing.sharedctypes import Synchronized
from typing import Any
from typing import ClassVar
from typing import Mapping

from docstring_inheritance import GoogleDocstringInheritanceMeta


class Serializable(metaclass=GoogleDocstringInheritanceMeta):
    """Base class to handle serialization of |g| objects.

    The methods ``__setstate__`` and ``__getstate__`` used by pickle to serialize and
    de-serialize objects are overloaded to handle ``Synchronized`` attributes. It is
    also possible to define the attributes that shall be ignored at serialization.

    For the attributes that are ignored at serialization, it is necessary to handle the
    way they are retrieved and recreated by overloading ``__setstate__`` and/or
    ``__getstate__`` from the subclasses.
    """

    _ATTR_NOT_TO_SERIALIZE: ClassVar[set[str]] = set()
    """The attributes that shall be skipped at serialization.

    Private attributes shall be written following name mangling conventions:
    ``_ClassName__attribute_name``. Subclasses must expand this class attribute if
    needed.
    """

    def __getstate__(self) -> dict[str, Any]:
        """Used by pickle to define what to serialize.

        Subclasses shall overload this method only if they include a new attribute
        that is both not serializable and not ``Synchronized``.

        Returns:
            The attributes to be serialized.
        """
        state = {}
        for attribute_name in self.__dict__.keys() - self._ATTR_NOT_TO_SERIALIZE:
            attribute_value = self.__dict__[attribute_name]

            if isinstance(attribute_value, Synchronized):
                # Don't serialize shared memory object,
                # this is meaningless, save the value instead.
                attribute_value = attribute_value.value

            state[attribute_name] = attribute_value

        return state

    def __setstate__(
        self,
        state: Mapping[str, Any],
    ) -> None:
        """Used by pickle to deserialize the class.

        Subclasses shall overload this method to include new attributes that are not
        serializable nor ``Synchronized`` and need to be recreated manually.

        Subclasses shall also overload ``_init_shared_attributes`` to initialize all the
        ``Synchronized`` attributes of the subclass.
        """
        # Initialize all Synchronized attributes first.
        self._init_shared_memory_attrs()
        for attribute_name, attribute_value in state.items():
            if attribute_name not in self.__dict__.keys():
                self.__dict__[attribute_name] = attribute_value
            elif isinstance(self.__dict__[attribute_name], Synchronized):
                # Set the value of Synchronized attributes instead of deserializing the
                # entire object.
                self.__dict__[attribute_name].value = attribute_value

    @abstractmethod
    def _init_shared_memory_attrs(self) -> None:
        """Initialize the shared memory attributes in multiprocessing.

        Subclasses shall overload this method to initialize all their ``Synchronized``
        attributes.
        """
