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
"""Most basic grammar implementation."""
from __future__ import annotations

import collections
import logging
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Mapping

from numpy import ndarray

from gemseo.core.discipline_data import Data
from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.base_grammar import NamesToTypes
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)


class SimpleGrammar(BaseGrammar):
    """A grammar only based on names and types with a dictionary-like interface.

    The grammar could be empty, in that case the data validation always pass. If the
    type bound to a name is ``None`` then the type of the corresponding data name is
    always valid.
    """

    __names_to_types: dict[str, type | None]
    """The binding from element names to element types."""

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

    def __delitem__(
        self,
        name: str,
    ) -> None:
        super().__delitem__(name)
        del self.__names_to_types[name]
        if name in self.__required_names:
            self.__required_names.remove(name)

    def __getitem__(self, name: str) -> type:
        return self.__names_to_types[name]

    def __len__(self) -> int:
        return len(self.__names_to_types)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__names_to_types)

    def _repr_required_elements(self, text: MultiLineString) -> None:
        for name, type_ in self.items():
            if name in self.__required_names:
                text.add(f"{name}: {type_.__name__}")

    def _repr_optional_elements(self, text: MultiLineString) -> None:
        for name, type_ in self.items():
            if name not in self.__required_names:
                if type_ is None:
                    type_name = None
                else:
                    type_name = type_.__name__
                text.add(f"{name}: {type_name}")
                if name in self._defaults:
                    text.indent()
                    text.add(f"default: {self._defaults[name]}")
                    text.dedent()

    def __copy__(self) -> SimpleGrammar:
        grammar = self._copy_base()
        grammar.__names_to_types = self.__names_to_types.copy()
        grammar.__required_names = self.__required_names.copy()
        return grammar

    copy = __copy__

    def update(  # noqa: D102
        self,
        grammar: BaseGrammar,
        exclude_names: Iterable[str] = (),
    ) -> None:
        grammar = grammar.to_simple_grammar()
        self.__update(grammar, grammar.__required_names, exclude_names)
        self._update_namespaces_from_grammar(grammar)
        self._defaults.update(grammar._defaults, exclude=exclude_names)

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

    def clear(self) -> None:  # noqa: D102
        super().clear()
        self.__names_to_types = {}
        self.__required_names = set()

    def validate(  # noqa: D102
        self,
        data: Data,
        raise_exception: bool = True,
    ) -> None:
        error_message = MultiLineString()

        missing_names = self.required_names - data.keys()

        if missing_names:
            error_message.add(
                "Missing required names: {}.".format(",".join(sorted(missing_names)))
            )

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

        if error_message.lines:
            LOGGER.error(error_message)
            if raise_exception:
                raise InvalidDataError(str(error_message))

    def update_from_names(  # noqa: D102
        self,
        names: Iterable[str],
    ) -> None:
        self.__update(dict.fromkeys(names, ndarray), names)

    def update_from_data(  # noqa: D102
        self,
        data: Data,
    ) -> None:
        self.update_from_types({name: type(value) for name, value in data.items()})

    def update_from_types(
        self,
        names_to_types: NamesToTypes,
        merge: bool = False,
    ) -> None:
        """
        Note:
            For consistency with :class:`.__init__` behavior, it is assumed that all
            of them are required.

        Raises:
            ValueError: When merge is True, since it is not yet supported for SimpleGrammar.
        """  # noqa: D205, D212, D415
        if merge:
            raise ValueError("Merge is not supported yet for SimpleGrammar.")
        self.__update(names_to_types, names_to_types.keys())

    def is_array(  # noqa: D102
        self,
        name: str,
        numeric_only: bool = False,
    ) -> bool:
        element_type = self.__names_to_types[name]
        if element_type is None:
            return False
        return issubclass(element_type, ndarray)
        # TODO: why only ndarray here vs array in json grammar?

    def restrict_to(  # noqa: D102
        self,
        names: Iterable[str],
    ) -> None:
        super().restrict_to(names)
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

    def rename_element(self, current_name: str, new_name: str) -> None:  # noqa: D102
        self.__names_to_types[new_name] = self.__names_to_types.pop(current_name)
        if current_name in self.__required_names:
            self.__required_names.remove(current_name)
            self.__required_names.add(new_name)
        super().rename_element(current_name, new_name)
