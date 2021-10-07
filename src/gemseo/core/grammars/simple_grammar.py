# -*- coding: utf-8 -*-
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

import logging
from typing import Any, Iterable, List, Mapping, Union

from gemseo.core.grammars.abstract_grammar import AbstractGrammar
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.utils.py23_compat import Path

LOGGER = logging.getLogger(__name__)


class SimpleGrammar(AbstractGrammar):
    """Store the names and types of the elements as Python lists.

    Attributes:
        data_names (List[str]): The names of the elements.
        data_types (List[type]): The types of the elements,
            stored in the same order as ``data_names``.
    """

    def __init__(
        self,
        name,  # type: str
        **kwargs  # type: Union[str,Path]
    ):  # type: (...) -> None
        super(SimpleGrammar, self).__init__(name)
        self.data_names = []
        self.data_types = []

    def load_data(
        self,
        data,  # type: Mapping[str,Any]
        raise_exception=True,  # type: bool
    ):  # type: (...) -> Mapping[str,Any]
        self.check(data, raise_exception)
        return data

    def check(
        self,
        data,  # type: Mapping[str,Any]
        raise_exception=True,  # type: bool
    ):  # type: (...) -> None
        """Check the consistency (name and type) of elements with the grammar.

        Args:
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
                raise InvalidDataException("Invalid data in {}.".format(self.name))

        for element_type in self.data_types:
            if element_type is not None and not isinstance(element_type, type):
                msg = "Invalid data type in grammar {}, {} is not a type.".format(
                    self.name, element_type
                )
                raise TypeError(msg)

        for element_type, element_name in zip(self.data_types, self.data_names):
            if element_name not in data:
                failed = True
                LOGGER.error("Missing input: %s in %s.", element_name, self.name)
            elif not isinstance(data[element_name], element_type):
                failed = True
                LOGGER.error(
                    "Wrong input type for: %s in %s got %s instead of %s.",
                    element_name,
                    self.name,
                    type(data[element_name]),
                    element_type,
                )
        if failed and raise_exception:
            raise InvalidDataException("Invalid data in {}.".format(self.name))

    def initialize_from_base_dict(
        self,
        typical_data_dict,  # type: Mapping[str,Any]
    ):  # type: (...) -> None
        self.data_names = []
        self.data_types = []
        for element_name, element_value in typical_data_dict.items():
            self.data_names.append(element_name)
            self.data_types.append(type(element_value))

    def get_data_names(self):  # type: (...) -> List[str]
        return self.data_names

    def is_all_data_names_existing(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> bool
        for element_name in data_names:
            if not self.is_data_name_existing(element_name):
                return False
        return True

    def _update_field(
        self,
        data_name,  # type: str
        data_type,  # type: type
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
        if data_name in self.data_names:
            data_index = self.data_names.index(data_name)
            self.data_names[data_index] = data_name
            self.data_types[data_index] = data_type
        else:
            self.data_names.append(data_name)
            self.data_types.append(data_type)

    def get_type_of_data_named(
        self,
        data_name,  # type: str
    ):  # type: (...) -> str
        """Return the element type associated to an element name.

        Args:
            data_name: The name of the element.

        Returns:
            The type of the element associated to the passed element name.

        Raises:
            ValueError: If the name does not correspond to an element name.
        """
        if not self.is_data_name_existing(data_name):
            raise ValueError("Unknown data named: {}.".format(data_name))
        data_index = self.data_names.index(data_name)
        return self.data_types[data_index]

    def update_from(
        self,
        input_grammar,  # type: AbstractGrammar
    ):  # type: (...) -> None
        """
        Raises:
            TypeError: If the passed grammar is not an :class:`.AbstractGrammar`.
        """
        if not isinstance(input_grammar, AbstractGrammar):
            msg = self._get_update_error_msg(self, input_grammar)
            raise TypeError(msg)

        input_grammar = input_grammar.to_simple_grammar()

        for element_name, element_type in zip(
            input_grammar.data_names, input_grammar.data_types
        ):
            self._update_field(element_name, element_type)

    def update_from_if_not_in(
        self,
        input_grammar,  # type: AbstractGrammar
        exclude_grammar,  # type: AbstractGrammar
    ):  # type: (...) -> None
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
                self._update_field(element_name, element_type)

    def is_data_name_existing(
        self,
        data_name,  # type: str
    ):  # type: (...) -> bool
        return data_name in self.data_names

    def clear(self):  # type: (...) -> None
        self.data_names = []
        self.data_types = []

    def to_simple_grammar(self):  # type: (...) -> SimpleGrammar
        return self
