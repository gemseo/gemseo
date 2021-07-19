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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Factory to create |g| disciplines
**************************************
"""
from __future__ import division, unicode_literals

from copy import deepcopy
from os.path import dirname, join

from gemseo.core import discipline
from gemseo.core.discipline import MDODiscipline
from gemseo.core.factory import Factory
from gemseo.core.json_grammar import JSONGrammar


class DisciplinesFactory(object):
    """The **DisciplinesFactory** is used to create :class:`.MDODiscipline` objects that
    are known to |g|

    Three types of directories are scanned
    to import the :class:`.MDODiscipline` classes:

    - the environment variable "GEMSEO_PATH" may contain
      the list of directories to scan,
    - the present directory (:doc:`gemseo.problems`) contains some
      benchmark test cases,
    """

    def __init__(self):
        """The constructor initializes the factory by scanning the directories to search
        for subclasses of :class:`.MDODiscipline` objects.

        Searches in "GEMSEO_PATH" and :doc:`gemseo.problems`.
        """
        # Defines the benchmark problems to be imported
        internal_modules_paths = ("gemseo.problems", "gemseo.core", "gemseo.wrappers")
        self.factory = Factory(MDODiscipline, internal_modules_paths)

        base_gram_path = join(
            dirname(discipline.__file__), "MDODiscipline_options.json"
        )
        self.__base_grammar = JSONGrammar(
            "MDODiscipline_options", schema_file=base_gram_path
        )
        self.__base_grammar_names = self.__base_grammar.get_data_names()

    def create(self, discipline_name, **options):
        """Create a :class:`.MDODiscipline` from its name.

        :param discipline_name: name of the discipline
        :type discipline_name: str
        :param options: options of the discipline,
            both the options to be passed to the constructor
            and the options that are generic to all the disciplines
        :type options: dict
        :returns: the discipline instance
        """
        com_opts_dict, spec_opts_dict = self.__filter_common_options(options)
        self.__base_grammar.load_data(com_opts_dict)
        disc = self.factory.create(discipline_name, **spec_opts_dict)
        if "linearization_mode" in com_opts_dict:
            disc.linearization_mode = com_opts_dict["linearization_mode"]

        cache_opts = self.__filter_opts_dict(com_opts_dict, "cache")
        if cache_opts:
            disc.set_cache_policy(**cache_opts)

        jac_opts = self.__filter_opts_dict(com_opts_dict, "jac")
        if jac_opts:
            disc.set_jacobian_approximation(**jac_opts)

        return disc

    @staticmethod
    def __filter_opts_dict(options, startstring):
        """Filters the options that start with a string.

        :param options: discipline options
        :type options: dict
        :param startstring: predicate for the option
        :type startstring: str
        """
        return {k: v for k, v in options.items() if k.startswith(startstring)}

    def __filter_common_options(self, options):
        """Separates options:

        - from the :class:`.MDODiscipline` options grammar
        - from the options that are specific to the discipline.

        :param options: options of the discipline
        :type options: dict
        """
        com_opts_names = self.__base_grammar_names
        com_opts_dict = {k: v for k, v in options.items() if k in com_opts_names}
        spec_opts_dict = {k: v for k, v in options.items() if k not in com_opts_dict}
        return com_opts_dict, spec_opts_dict

    def update(self):
        """Updates the paths, to be used if GEMSEO_PATH was changed."""
        self.factory.update()

    @property
    def disciplines(self):
        """Lists the available :class:`.MDODiscipline`, known to this factory.

        :returns: the list of available disciplines names
            (ie their class names)
        """
        return self.factory.classes

    def get_options_grammar(self, name, write_schema=False, schema_file=None):
        """Get the options default values for the given class name Only addresses kwargs
        Generates.

        :param name: name of the class
        :type name: str
        :param schema_file: the output json file path. If None: input.json or
            output.json depending on grammar type.
            (Default value = None)
        :type schema_file: str
        :param write_schema: if True, writes the schema files
            (Default value = False)
        :type write_schema: bool
        :returns: the json grammar for options
        """
        disc_gram = self.factory.get_options_grammar(name, write_schema, schema_file)
        base_grammar = deepcopy(self.__base_grammar)
        base_grammar.update_from(disc_gram)
        return base_grammar
