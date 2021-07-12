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

from gemseo.core.grammars.abstract_grammar import AbstractGrammar
from gemseo.core.grammars.errors import InvalidDataException

LOGGER = logging.getLogger(__name__)


class SimpleGrammar(AbstractGrammar):
    """Store the data named and types."""

    def __init__(self, name):
        """Constructor.

        :param name : grammar name
        """
        super(SimpleGrammar, self).__init__(name)
        # Data names list
        self.data_names = []
        # Data types list in the same order as self.data_names
        self.data_types = []

    def load_data(self, data, raise_exception=True):
        """Loads the data dictionary in the grammar and checks it against self
        properties.

        :param data: the input data
        :param raise_exception: if False, no exception is raised
            when data is invalid (Default value = True)
        """
        self.check(data, raise_exception)
        return data

    def check(self, data, raise_exception=True):
        """Checks local data against self properties.

        :param raise_exception: if False, no exception is raised
            when data is invalid (Default value = True)
        """
        failed = False
        if not isinstance(data, dict):
            failed = True
            LOGGER.error("Grammar data is not a dict, in %s", self.name)
            if raise_exception:
                raise InvalidDataException("Invalid data in " + str(self.name))

        for data_type in self.data_types:
            if data_type is not None and not isinstance(data_type, type):
                msg = "Invalid data type in grammar {} , {} is not a type".format(
                    self.name, data_type
                )
                raise TypeError(msg)

        for data_type, data_name in zip(self.data_types, self.data_names):
            if data_name not in data:
                failed = True
                LOGGER.error("Missing input: %s in %s", data_name, self.name)
            elif not isinstance(data[data_name], data_type):
                failed = True
                LOGGER.error(
                    "Wrong input type for: %s in %s got %s instead of %s",
                    data_name,
                    self.name,
                    type(data[data_name]),
                    data_type,
                )
        if failed and raise_exception:
            raise InvalidDataException("Invalid data in " + str(self.name))

    def initialize_from_base_dict(self, typical_data_dict):
        """Initialize the grammar with types and names from a typical data entry.

        :param typical_data_dict: a data dictionary
        """
        self.data_names = []
        self.data_types = []
        for key, value in typical_data_dict.items():
            self.data_names.append(key)
            self.data_types.append(type(value))

    def get_data_names(self):
        """Returns the list of data names.

        :returns: the data names alphabetically sorted
        """
        return self.data_names

    def is_all_data_names_existing(self, data_names):
        """Checks if data_names are present in grammar.

        :param data_names: the data names list
        :returns: True if all data are in grammar
        """
        for data_name in data_names:
            if not self.is_data_name_existing(data_name):
                return False
        return True

    def _update_field(self, data_name, data_type):
        """Updates self properties with a new property of data_name,data_type Adds it if
        self has no property named data_name Updates self.data_types otherwise.

        :param data_name: the name of the property
        :param data_type: the type of the property
        """
        if data_name in self.data_names:
            indx = self.data_names.index(data_name)
            self.data_names[indx] = data_name
            self.data_types[indx] = data_type
        else:
            self.data_names.append(data_name)
            self.data_types.append(data_type)

    def get_type_of_data_named(self, data_name):
        """Gets the associated type to the data named data_name.

        :param data_name: the name of the property
        :returns: data type associated to data_name
        """
        if not self.is_data_name_existing(data_name):
            raise ValueError("Unknown data named :" + str(data_name))
        indx = self.data_names.index(data_name)
        return self.data_types[indx]

    def update_from(self, input_grammar):
        """Adds properties coming from another grammar.

        :param input_grammar: the grammar to take inputs from
        """
        if not isinstance(input_grammar, AbstractGrammar):
            msg = self._get_update_error_msg(self, input_grammar)
            raise TypeError(msg)

        input_grammar = input_grammar.to_simple_grammar()

        for g_name, g_type in zip(input_grammar.data_names, input_grammar.data_types):
            self._update_field(g_name, g_type)

    def update_from_if_not_in(self, input_grammar, exclude_grammar):
        """Adds properties coming from input_grammar if they are not in exclude_grammar.

        :param input_grammar: the grammar to take inputs from
        :param exclude_grammar: exclusion grammar
        """
        if not isinstance(input_grammar, AbstractGrammar) or not isinstance(
            exclude_grammar, AbstractGrammar
        ):
            msg = self._get_update_error_msg(self, input_grammar, exclude_grammar)
            raise TypeError(msg)

        input_grammar = input_grammar.to_simple_grammar()
        exclude_grammar = exclude_grammar.to_simple_grammar()

        for g_name, g_type in zip(input_grammar.data_names, input_grammar.data_types):
            if exclude_grammar.is_data_name_existing(g_name):
                ex_data_type = exclude_grammar.get_type_of_data_named(g_name)
                if g_type != ex_data_type:
                    raise ValueError(
                        "Inconsistent grammar update "
                        + str(g_type)
                        + " !="
                        + str(ex_data_type)
                    )
            else:
                self._update_field(g_name, g_type)

    def is_data_name_existing(self, data_name):
        """Checks if data_name is present in grammar.

        :param data_name: the data name
        :returns: True if data is in grammar
        """
        return data_name in self.data_names

    def clear(self):
        """Clears the data to produce an empty grammar."""
        self.data_names = []
        self.data_types = []

    def to_simple_grammar(self):
        """Converts to the base SimpleGrammar type.

        :returns: a SimpleGrammar instance equivalent to self
        """
        return self
