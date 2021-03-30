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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Data rules and checks for disciplines inputs/outputs validation
***************************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

standard_library.install_aliases()
from gemseo import LOGGER


class AbstractGrammar(object):
    """Abstract Grammar : defines the abstraction for a Grammar
    A grammar subclass instance stores the input or output data
    types and structure an MDODiscipline
    It is able to check the inputs and outputs against predefined types

    """

    INPUT_GRAMMAR = "input"
    OUTPUT_GRAMMAR = "output"

    def load_data(self, data_dict, raise_exception=True):
        """Loads the data dictionary in the grammar
        and checks it against self properties

        :param data_dict: the input data
        :param raise_exception: if False, no exception is raised
             when data is invalid (Default value = True)
        """
        raise NotImplementedError()

    def get_data_names(self):
        """Returns the list of data names

        :returns: the data names alphabetically sorted
        """
        raise NotImplementedError()

    def update_from(self, input_grammar):
        """Adds properties coming from another grammar

        :param input_grammar: the grammar to take inputs from
        """
        raise NotImplementedError()

    def update_from_if_not_in(self, input_grammar, exclude_grammar):
        """Adds properties coming from input_grammar if they are not in
        exclude_grammar

        :param input_grammar: the grammar to take inputs from
        :param exclude_grammar: exclusion grammar
        """
        raise NotImplementedError()

    def is_data_name_existing(self, data_name):
        """Checks if data_name is present in grammar

        :param data_name: the data name
        :returns: True if data is in grammar
        """
        raise NotImplementedError()

    def is_all_data_names_existing(self, data_names):
        """Checks if data_names are present in grammar

        :param data_names: the data names list
        :returns: True if all data are in grammar
        """
        raise NotImplementedError()

    def clear(self):
        """Clears the data to produce an empty grammar"""
        raise NotImplementedError()


class SimpleGrammar(object):
    """A grammar instance stores the input or output data types and structure a
    MDODiscipline
    It is able to check the inputs and outputs against predefined types


    """

    def __init__(self, name):
        """
        Constructor

        :param name : grammar name
        """
        super(SimpleGrammar, self).__init__()
        # Data names list
        self.data_names = []
        # Data types list in the same order as self.data_names
        self.data_types = []
        # Default data dict, keys must be in self.data_names
        self.defaults = {}
        self.name = name
        self.data = None

    def load_data(self, data_dict, raise_exception=True):
        """Loads the data dictionary in the grammar
        and checks it against self properties

        :param data_dict: the input data
        :param raise_exception: if False, no exception is raised
            when data is invalid (Default value = True)
        """
        self.data = data_dict
        self._load_defaults()
        self.check(raise_exception)
        return self.data

    def check(self, raise_exception=True):
        """Checks local data against self properties

        :param raise_exception: if False, no exception is raised
            when data is invalid (Default value = True)
        """
        failed = False
        if not isinstance(self.data, dict):
            failed = True
            LOGGER.error("Grammar data is not a dict, in %s", self.name)
            if raise_exception:
                raise InvalidDataException("Invalid data in " + str(self.name))

        for data_type in self.data_types:
            if not isinstance(data_type, type):
                raise TypeError(
                    "Invalid data_types in grammar :"
                    + str(self.name)
                    + ", "
                    + str(data_type)
                    + " is not a type"
                )

        for data_type, data_name in zip(self.data_types, self.data_names):
            if data_name not in self.data:
                failed = True
                LOGGER.error("Missing input: %s in %s", str(data_name), self.name)
            elif not isinstance(self.data[data_name], data_type):
                failed = True
                LOGGER.error(
                    "Wrong input type for: %s in %s got %s instead of %s",
                    str(data_name),
                    self.name,
                    str(type(self.data[data_name])),
                    str(data_type),
                )
        if failed and raise_exception:
            raise InvalidDataException("Invalid data in " + str(self.name))

    def initialize_from_base_dict(self, typical_data_dict):
        """Initialize the grammar with types and names from a
        typical data entry

        :param typical_data_dict: a data dictionary
        """
        self.data_names = []
        self.data_types = []
        for key, value in typical_data_dict.items():
            self.data_names.append(key)
            self.data_types.append(type(value))

    def _load_defaults(self):
        """Loads defaults values in self"""
        for key, value in self.defaults.items():
            if key not in self.data:
                self.data[key] = value

    def get_data_names(self):
        """Returns the list of data names

        :returns: the data names alphabetically sorted
        """
        return self.data_names

    def is_all_data_names_existing(self, data_names):
        """Checks if data_names are present in grammar

        :param data_names: the data names list
        :returns: True if all data are in grammar
        """
        for data_name in data_names:
            if not self.is_data_name_existing(data_name):
                return False
        return True

    def _update_field(self, data_name, data_type):
        """Updates self properties with a new property of data_name,data_type
        Adds it if self has no property named data_name
        Updates self.data_types otherwise

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
        """Gets the associated type to the data named data_name

        :param data_name: the name of the property
        :returns: data type associated to data_name
        """
        if not self.is_data_name_existing(data_name):
            raise ValueError("Unknown data named :" + str(data_name))
        indx = self.data_names.index(data_name)
        return self.data_types[indx]

    def update_from(self, input_grammar):
        """Adds properties coming from another grammar

        :param input_grammar: the grammar to take inputs from
        """
        if not isinstance(input_grammar, SimpleGrammar):
            LOGGER.warning(
                "Cannot update grammar %s of type %s with %s of type %s",
                self.name,
                type(self).__name__,
                input_grammar.name,
                type(input_grammar).__name__,
            )
            return

        for g_name, g_type in zip(input_grammar.data_names, input_grammar.data_types):
            self._update_field(g_name, g_type)

    def update_from_if_not_in(self, input_grammar, exclude_grammar):
        """Adds properties coming from input_grammar if they are not in
        exclude_grammar

        :param input_grammar: the grammar to take inputs from
        :param exclude_grammar: exclusion grammar
        """
        if not isinstance(input_grammar, SimpleGrammar) or not isinstance(
            exclude_grammar, SimpleGrammar
        ):
            raise TypeError(
                "Cannot update grammar %s of type %s with %s of type %s "
                + " and %s, of type %s",
                self.name,
                type(self).__name__,
                input_grammar.name,
                type(input_grammar).__name__,
                exclude_grammar.name,
                type(exclude_grammar).__name__,
            )

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
        """Checks if data_name is present in grammar

        :param data_name: the data name
        :returns: True if data is in grammar
        """
        return data_name in self.data_names

    def clear(self):
        """Clears the data to produce an empty grammar"""
        self.data_names = []
        self.data_types = []
        self.defaults.clear()


# AbstractGrammar.register(SimpleGrammar)


class InvalidDataException(Exception):
    """Exception raised when data is not valid against grammar rules."""
