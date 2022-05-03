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

import logging
from collections import abc
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

from numpy import ndarray

from gemseo.core.grammars.abstract_grammar import AbstractGrammar
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)


class SimpleGrammar(AbstractGrammar):
    """A grammar based on the names and types of the elements specified by a
    dictionary."""

    def __init__(
        self,
        name: str,
        names_to_types: Mapping[str, type] | None = None,
        required_names: Mapping[str, bool] | None = None,
        **kwargs: str | Path,
    ) -> None:
        """
        Args:
            name: The grammar name.
            names_to_types: The mapping defining the data names as keys,
                and data types as values.
                If None, the grammar is empty.
            required_names: The mapping defining the required data names as keys,
                bound to whether the data name is required. If None,
                all data names are required.
        """
        super().__init__(name)
        if names_to_types is None:
            self._names_to_types = {}
        else:
            self._names_to_types = names_to_types
            self._check_types()

        self._default_callable = (
            lambda: True
        )  # Callable to be assigned to defaultdict.default_factory at init
        self._required_names = defaultdict(self._default_callable)
        self._required_names.update(self._names_to_types)

        if required_names is not None:
            self._required_names.update(required_names)

    @property
    def data_names_keyset(self) -> Iterable[str]:
        """The data names of the grammar as dict_keys."""
        return self._names_to_types.keys()

    def is_required(self, element_name: str) -> bool:

        self._required_names.default_factory = None

        try:
            return self._required_names[element_name]
        except KeyError:
            raise ValueError(f"Element {element_name} is not in the grammar.")
        finally:
            self._required_names.default_factory = self._default_callable

    def update_required_elements(self, **elements: Mapping[str, bool]) -> None:

        for element_name, element_value in elements.items():
            if element_name not in self._names_to_types:
                raise KeyError(f"Data named {element_name} is not in the grammar.")
            if not isinstance(element_value, bool):
                raise TypeError(f"Boolean is required for element {element_name}.")
        self._required_names.update(elements)

    @property
    def data_names(self) -> list[str]:
        """The names of the elements."""
        return list(self._names_to_types.keys())

    @property
    def data_types(self) -> list[type]:
        """The types of the elements."""
        return list(self._names_to_types.values())

    def _check_types(self) -> None:
        """Check that the elements names to types mapping contains only acceptable type
        specifications, ie, are a type or None.

        Raises:
            TypeError: When at least one type specification is not a type.
        """
        for obj_name, obj in self._names_to_types.items():
            if obj is not None and not isinstance(obj, type):
                raise TypeError(
                    (
                        "{} is not a type and cannot be used as a"
                        " type specification for the element named {} in the grammar {}."
                    ).format(obj, obj_name, self.name)
                )

    def get_type_from_python_type(self, python_type: type) -> type:

        if python_type == str:
            return str
        else:
            return python_type

    def update_elements(
        self,
        python_typing: bool = False,
        **elements: Mapping[str, type],
    ) -> None:

        if python_typing:
            for element_name, element_value in elements.items():
                elements[element_name] = self.get_type_from_python_type(element_value)

        # Generalize dict to support DisciplineData objects.
        for name, type_ in elements.items():
            if type_ is dict:
                elements[name] = abc.Mapping

        self._names_to_types.update(**elements)
        self._check_types()

    def load_data(
        self,
        data: Mapping[str, Any],
        raise_exception: bool = True,
    ) -> Mapping[str, Any]:
        self.check(data, raise_exception)
        return data

    def check(
        self,
        data: Mapping[str, Any],
        raise_exception: bool = True,
    ) -> None:
        """Check the consistency (name and type) of elements with the grammar.

        Args:
            data: The elements to be checked.
            raise_exception: Whether to raise an exception
                when the elements are invalid.

        Raises:
            TypeError: If a data type in the grammar is not a type.
            InvalidDataException:
                * If the passed data is not a dictionary.
                * If a name in the passed data is not in the grammar.
                * If the type of a value in the passed data does not have
                  the specified type in the grammar for the corresponding name.
        """
        failed = False
        if not isinstance(data, Mapping):
            failed = True
            LOGGER.error("Grammar data is not a mapping, in %s.", self.name)
            if raise_exception:
                raise InvalidDataException(f"Invalid data in: {self.name}.")

        error_message = MultiLineString()
        error_message.add("Invalid data in {}", self.name)
        for element_name, element_type in self._names_to_types.items():

            if element_name not in data and self._required_names[element_name]:
                failed = True
                error_message.add(
                    "Missing mandatory elements: {} in grammar {}".format(
                        element_name, self.name
                    )
                )
            elif (
                element_name in data
                and element_type is not None
                and not isinstance(data.get(element_name), element_type)
            ):
                failed = True
                error_message.add(
                    "Wrong input type for: {} in {} got {} instead of {}.".format(
                        element_name, self.name, type(data[element_name]), element_type
                    )
                )

        if failed:
            LOGGER.error(error_message)
            if raise_exception:
                raise InvalidDataException(str(error_message))

    def initialize_from_base_dict(
        self,
        typical_data_dict: Mapping[str, Any],
    ) -> None:
        self.update_elements(
            **{name: type(value) for name, value in typical_data_dict.items()}
        )

    def get_data_names(self) -> list[str]:
        return self.data_names

    def is_all_data_names_existing(
        self,
        data_names: Iterable[str],
    ) -> bool:
        get = self._names_to_types.get
        for name in data_names:
            if get(name) is None:
                return False
        return True

    def _update_field(
        self,
        data_name: str,
        data_type: type,
    ):
        """Update the grammar elements from an element name and an element type.

        If there is no element with this name,
        create it and store its type.

        Otherwise,
        update its type.

        Args:
            data_name: The name of the element.
            data_type: The type of the element.
        """
        self._names_to_types[data_name] = data_type

    def get_type_of_data_named(
        self,
        data_name: str,
    ) -> str:
        """Return the element type associated to an element name.

        Args:
            data_name: The name of the element.

        Returns:
            The type of the element associated to the passed element name.

        Raises:
            ValueError: If the name does not correspond to an element name.
        """
        if data_name not in self._names_to_types:
            raise ValueError(f"Unknown data named: {data_name}.")
        return self._names_to_types[data_name]

    def is_type_array(self, data_name: str) -> bool:
        element_type = self.get_type_of_data_named(data_name)
        return issubclass(element_type, ndarray)

    def restrict_to(
        self,
        data_names: Sequence[str],
    ) -> None:
        for element_name in self.data_names:
            if element_name not in data_names:
                del self._names_to_types[element_name]

    def remove_item(
        self,
        item_name: str,
    ) -> None:
        del self._names_to_types[item_name]

    def update_from(
        self,
        input_grammar: AbstractGrammar,
    ) -> None:
        """
        Raises:
            TypeError: If the passed grammar is not an :class:`.AbstractGrammar`.
        """
        if not isinstance(input_grammar, AbstractGrammar):
            msg = self._get_update_error_msg(self, input_grammar)
            raise TypeError(msg)

        input_grammar = input_grammar.to_simple_grammar()
        self._names_to_types.update(input_grammar._names_to_types)

    def update_from_if_not_in(
        self,
        input_grammar: AbstractGrammar,
        exclude_grammar: AbstractGrammar,
    ) -> None:
        """
        Raises:
            TypeError: If a passed grammar is not an :class:`.AbstractGrammar`.
            ValueError: If types are inconsistent between both passed grammars.
        """
        if not isinstance(input_grammar, AbstractGrammar) or not isinstance(
            exclude_grammar, AbstractGrammar
        ):
            msg = self._get_update_error_msg(self, input_grammar, exclude_grammar)
            raise TypeError(msg)

        input_grammar = input_grammar.to_simple_grammar()
        exclude_grammar = exclude_grammar.to_simple_grammar()

        for element_name, element_type in zip(
            input_grammar.data_names, input_grammar.data_types
        ):
            if exclude_grammar.is_data_name_existing(element_name):
                ex_element_type = exclude_grammar.get_type_of_data_named(element_name)
                if element_type != ex_element_type:
                    raise ValueError(
                        "Inconsistent grammar update {} != {}.".format(
                            element_type, ex_element_type
                        )
                    )
            else:
                self._names_to_types[element_name] = element_type

    def is_data_name_existing(
        self,
        data_name: str,
    ) -> bool:
        return data_name in self._names_to_types

    def clear(self) -> None:
        self._names_to_types = {}

    def to_simple_grammar(self) -> SimpleGrammar:
        return self
