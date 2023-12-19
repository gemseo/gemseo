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
"""A basic grammar based on names and types."""

from __future__ import annotations

import collections
import logging
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import ndarray

from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.base_grammar import NamesToTypes

if TYPE_CHECKING:
    from gemseo.core.discipline_data import Data
    from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)


class SimpleGrammar(BaseGrammar):
    """A grammar only based on names and types with a dictionary-like interface.

    The grammar could be empty, in that case the data validation always pass. If the
    type bound to a name is ``None`` then the type of the corresponding data name is
    always valid.
    """

    DATA_CONVERTER_CLASS: ClassVar[str] = "SimpleGrammarDataConverter"

    __names_to_types: dict[str, type | None]
    """The mapping from element names to element types."""

    __required_names: set[str]
    """The required names."""

    def __init__(
        self,
        name: str,
        names_to_types: NamesToTypes | None = None,
        required_names: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            names_to_types: The mapping defining the data names as keys,
                and data types as values.
                If ``None``, the grammar is empty.
            required_names: The names of the required elements.
                If ``None``, all the elements are required.
            **kwargs: These arguments are not used.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        if names_to_types:
            self.update_from_types(names_to_types)
        if required_names is not None:
            self._check_name(*required_names)
            self.__required_names = set(required_names)

    def __getitem__(self, name: str) -> type:
        return self.__names_to_types[name]

    def __len__(self) -> int:
        return len(self.__names_to_types)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__names_to_types)

    def _delitem(self, name: str) -> None:  # noqa:D102
        del self.__names_to_types[name]
        if name in self.__required_names:
            self.__required_names.remove(name)

    def _copy(self, grammar: SimpleGrammar) -> None:  # noqa:D102
        grammar.__names_to_types = self.__names_to_types.copy()
        grammar.__required_names = self.__required_names.copy()

    def _rename_element(self, current_name: str, new_name: str) -> None:  # noqa: D102
        self.__names_to_types[new_name] = self.__names_to_types.pop(current_name)
        if current_name in self.__required_names:
            self.__required_names.remove(current_name)
            self.__required_names.add(new_name)

    def update(  # noqa: D102
        self,
        grammar: BaseGrammar,
        exclude_names: Iterable[str] = (),
    ) -> None:
        if not grammar:
            return
        grammar = grammar.to_simple_grammar()
        self.__update(grammar, grammar.__required_names, exclude_names)
        super().update(grammar, exclude_names)

    def update_from_names(  # noqa: D102
        self,
        names: Iterable[str],
    ) -> None:
        if not names:
            return
        self.__update(dict.fromkeys(names, ndarray), names)

    def update_from_types(
        self,
        names_to_types: NamesToTypes,
        merge: bool = False,
    ) -> None:
        """
        Notes:
            For consistency with :class:`.__init__` behavior, it is assumed that all
            of them are required.

        Raises:
            ValueError: When merge is True,
                since it is not yet supported for SimpleGrammar.
        """  # noqa: D205, D212, D415
        if merge:
            raise ValueError("Merge is not supported yet for SimpleGrammar.")
        if not names_to_types:
            return
        self.__update(names_to_types, names_to_types.keys())

    def __update(
        self,
        grammar: SimpleGrammar | Mapping[str, Any],
        required_names: Iterable[str],
        exclude_names: Iterable[str] = (),
    ) -> None:
        """Update the elements from another grammar or elements.

        When elements are provided names and types instead of a
        :class:`.BaseGrammar`,
        for consistency with :class:`.__init__` behavior
        it is assumed that all of them are required.

        Args:
            grammar: The grammar to take the elements from.
            required_names: The names of the elements of `grammar` that are required.
            exclude_names: The names of the elements that shall not be updated.

        Raises:
            ValueError: If the types are bad.
        """
        for element_name, element_type in grammar.items():
            if element_name in exclude_names:
                continue

            self.__check_type(element_name, element_type)

            # Generalize dict to support DisciplineData objects.
            if element_type is dict:
                self.__names_to_types[element_name] = collections.abc.Mapping
            else:
                self.__names_to_types[element_name] = element_type

        updated_element_names = grammar.keys() - exclude_names
        self.__required_names |= updated_element_names.intersection(required_names)
        self.__required_names -= updated_element_names.difference(required_names)

    def _clear(self) -> None:  # noqa: D102
        self.__names_to_types = {}
        self.__required_names = set()

    def _update_grammar_repr(self, repr_: MultiLineString, properties: Any) -> None:
        repr_.add(f"Type: {properties}")

    def _validate(  # noqa: D102
        self,
        data: Data,
        error_message: MultiLineString,
    ) -> bool:
        data_is_valid = True
        for element_name, element_type in self.__names_to_types.items():
            if (
                element_name in data
                and element_type is not None
                and not isinstance(data[element_name], element_type)
            ):
                error_message.add(
                    f"Bad type for {element_name}: "
                    f"{type(data[element_name])} instead of {element_type}."
                )
                data_is_valid = False
        return data_is_valid

    def is_array(  # noqa: D102
        self,
        name: str,
        numeric_only: bool = False,
    ) -> bool:
        self._check_name(name)
        if numeric_only:
            return self.data_converter.is_numeric(name)
        element_type = self.__names_to_types[name]
        if element_type is None:
            return False
        return issubclass(element_type, Collection)

    def _restrict_to(  # noqa: D102
        self,
        names: Iterable[str],
    ) -> None:
        for element_name in self.__names_to_types.keys() - names:
            del self.__names_to_types[element_name]
            if element_name in self.__required_names:
                self.__required_names.remove(element_name)

    def to_simple_grammar(self) -> SimpleGrammar:  # noqa: D102
        return self

    @property
    def required_names(self) -> set[str]:  # noqa: D102
        return self.__required_names

    @staticmethod
    def __check_type(name: str, obj: Any) -> None:
        """Check that the type of object is a valid element type.

        Raises:
            TypeError: If the object is neither a type nor ``None``.
        """
        if obj is not None and not isinstance(obj, type):
            raise TypeError(f"The element {name} must be a type or None: it is {obj}.")

    def _check_name(self, *names: str) -> None:
        for name in names:
            if name not in self.__names_to_types:
                raise KeyError(f"The name {name} is not in the grammar.")
