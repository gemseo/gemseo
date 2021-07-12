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

"""Rules and checks for disciplines inputs/outputs validation."""

from __future__ import division, unicode_literals

import logging

from numpy import zeros

LOGGER = logging.getLogger(__name__)


class AbstractGrammar(object):
    """Abstract Grammar : defines the abstraction for a Grammar.

    A grammar subclass instance stores the input or output data
    types and structure an MDODiscipline.
    It is able to check the inputs and outputs against predefined types.
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "grammar name: {}".format(self.name)

    def load_data(self, data, raise_exception=True):
        """Loads the data dictionary in the grammar and checks it against self
        properties.

        :param data: the input data
        :param raise_exception: if False, no exception is raised
             when data is invalid (Default value = True)
        """
        raise NotImplementedError()

    def get_data_names(self):
        """Returns the list of data names.

        :returns: the data names alphabetically sorted
        """
        raise NotImplementedError()

    def update_from(self, input_grammar):
        """Adds properties coming from another grammar.

        :param input_grammar: the grammar to take inputs from
        """
        raise NotImplementedError()

    def update_from_if_not_in(self, input_grammar, exclude_grammar):
        """Adds properties coming from input_grammar if they are not in exclude_grammar.

        :param input_grammar: the grammar to take inputs from
        :param exclude_grammar: exclusion grammar
        """
        raise NotImplementedError()

    def is_data_name_existing(self, data_name):
        """Checks if data_name is present in grammar.

        :param data_name: the data name
        :returns: True if data is in grammar
        """
        raise NotImplementedError()

    def is_all_data_names_existing(self, data_names):
        """Checks if data_names are present in grammar.

        :param data_names: the data names list
        :returns: True if all data are in grammar
        """
        raise NotImplementedError()

    def clear(self):
        """Clears the data to produce an empty grammar."""
        raise NotImplementedError()

    def to_simple_grammar(self):
        """Converts to the base SimpleGrammar type.

        :returns: a SimpleGrammar instance equivalent to self
        """
        raise NotImplementedError()

    def initialize_from_data_names(self, data_names):
        """Initializes a Grammar from a list of data. All data of the grammar will be
        set as arrays.

        :param data_names: a data names list
        """
        data = zeros(1)
        typical_data_dict = {k: data for k in data_names}
        self.initialize_from_base_dict(typical_data_dict)

    def initialize_from_base_dict(self, typical_data_dict):
        """Initialize the grammar with types and names from a typical data entry.

        :param typical_data_dict: a data dictionary
            keys are used as data names
            values are used to detect the data types
        """
        raise NotImplementedError()

    @staticmethod
    def _get_update_error_msg(grammar1, grammar2, grammar3=None):
        """Create a message for grammar update error.

        Args:
            grammar1: A grammar.
            grammar2: A grammar.
            grammar3: A grammar, optional.

        Returns:
            str: The error message.
        """
        msg = "Cannot update grammar {} of type {} with {} of type {}".format(
            grammar1.name,
            grammar1.__class__.__name__,
            grammar2.name,
            grammar2.__class__.__name__,
        )
        if grammar3 is not None:
            msg += " and {} of type {}".format(
                grammar1.name,
                grammar1.__class__.__name__,
            )
        return msg
