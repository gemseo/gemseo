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
Factory base class
******************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import importlib
import os
import pkgutil
import sys

from future import standard_library
from future.utils import with_metaclass

from gemseo.core.json_grammar import JSONGrammar
from gemseo.third_party.prettytable import PrettyTable
from gemseo.utils.singleton import SingleInstancePerAttributeEq
from gemseo.utils.source_parsing import SourceParsing

standard_library.install_aliases()

from gemseo import LOGGER


class Factory(with_metaclass(SingleInstancePerAttributeEq, object)):
    """
    Factory to create extensions that are known to |g|:
        can be a MDODiscipline, MDOFormulation... Depending on the subclass

    Three types of directories are scanned :

    - the environment variable "GEMSEO_PATH" may contain
      the list of directories to scan
    - internal_modules_paths (such as gemseo.problems...)
    - a directory list may be passed to the factory
    """

    # Name of the environment variable to search for classes
    GEMSEO_PATH = "GEMSEO_PATH"
    GEMS_PATH = "GEMS_PATH"

    # Allowed prefix for naming a plugin in sys.path
    PLUGIN_PREFIX = "gemseo_"

    def __init__(self, base_class=None, internal_modules_paths=None):
        """Initializes the factory.

        Scans the directories to search for subclasses of MDODiscipline.
        Searches in "GEMSEO_PATH", "GEMS_PATH",  and gemseo.problems

        :param base_class: class to search in the modules
            (MDOFormulation, MDODiscipline...) depending on the subclass
        :param internal_modules_paths: import paths (such as gemseo.problems)
            which are already imported
        :param name: name of the factory to print when configuration
                     is printed
        :param possible_plugin_names: tuple of plugins packages names to be
            scanned if they can be imported. The last plugin name has the
            priority. For instance, if the same class MDAJacobi exists in
            gemseo.mda, gemseo_plugins.mda and gemseo_private.mda, the used one will
            be gemseo_private.mda
        """
        if not isinstance(base_class, type):
            raise TypeError("Class to search must be a class!")

        self.base_class = base_class
        self.internal_modules_paths = internal_modules_paths or []

        self.failed_imports = {}
        self.__names_to_classes = {}

        self.update()

    def _update_path_from_env_variable(self, env_variable):
        """Update the classes that can be instanciated from a factory from an
        environment variable.

        param env_variable: name of the environment variable
        """
        g_path = os.environ.get(env_variable)
        if g_path is None:
            return

        if ":" in g_path:
            paths = g_path.split(":")
        else:
            paths = [g_path]

        # temporary make the gemseo paths visible to the import machinery
        for path in paths:
            sys.path.insert(0, path)
        for _, mod_name, _ in pkgutil.iter_modules(path=paths):
            self.__import_modules_from(mod_name)
        for path in paths:
            sys.path.pop(0)

    def update(self):
        """Update the classes that can be created by the factory.

        In order, scan in the internal modules, then in plugins, then in
        GEMSEO_PATH.
        """
        # Scan internal packages
        for mod_name in self.internal_modules_paths:
            self.__import_modules_from(mod_name)

        # Scan plugins packages
        for _, mod_name, _ in pkgutil.iter_modules():
            if mod_name.startswith(self.PLUGIN_PREFIX):
                self.__import_modules_from(mod_name)

        gems_path = os.environ.get(self.GEMS_PATH)
        if gems_path is not None:
            msg = """GEMS is now named GEMSEO. The GEMS_PATH environment
             variable is now deprecated and it is strongly recommended to use
             the GEMSEO_PATH environment variable instead to register your
             GEMSEO plugins."""
            LOGGER.warn(msg)

        # Scan environment variable paths
        env_variables = [self.GEMSEO_PATH, self.GEMS_PATH]
        for env_variable in env_variables:
            self._update_path_from_env_variable(env_variable)

        self.__names_to_classes = self.__get_sub_classes(self.base_class)

    def __log_import_failure(self, pkg_name):
        """Log import failures."""
        LOGGER.debug("Failed to import package %s", pkg_name)
        self.failed_imports[pkg_name] = ""

    def __import_modules_from(self, pkg_name):
        """Import modules from a package."""
        pkg = importlib.import_module(pkg_name)

        if not hasattr(pkg, "__path__"):
            # not a package so no more module to import
            return

        for _, mod_name, _ in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + ".", self.__log_import_failure
        ):
            try:
                importlib.import_module(mod_name)
            except Exception as err:  # pylint: disable=(broad-except
                LOGGER.debug("Failed to import module: %s, %s", mod_name, err)
                self.failed_imports[mod_name] = err

    def __get_sub_classes(self, cls):
        """Return all the sub classes of cls.

        The class names are unique, the last imported is kept when more than
        one class have the same name.
        """
        all_sub_classes = {}
        for sub_class in cls.__subclasses__():
            sub_classes = {sub_class.__name__: sub_class}
            sub_classes.update(self.__get_sub_classes(sub_class))
            for cls_name, _cls in sub_classes.items():
                all_sub_classes[cls_name] = _cls
        return all_sub_classes

    @property
    def classes(self):
        """Return the available classes.

        :returns : the list of classes names
        """
        return sorted(self.__names_to_classes.keys())

    # TODO: rename to has_class
    def is_available(self, name):
        """Return whether a class is available.

        :param name : name of the class
        :returns: True if the class is available
        """
        return name in self.__names_to_classes

    def get_class(self, name):
        """Return the class from its name.

        :param name : name of the class
        """
        try:
            return self.__names_to_classes[name]
        except KeyError:
            msg = "Class {} is not available!\nAvailable ones are: {}".format(
                name, ", ".join(sorted(self.__names_to_classes.keys()))
            )
            raise ImportError(msg)

    def create(self, class_name, **options):
        """Return an instance with given class name.

        :param class_name : name of the class
        :parma options: options to be passed to the constructor
        """
        cls = self.get_class(class_name)
        try:
            return cls(**options)
        except TypeError:
            # TODO: raise an error with message and let the callers handle
            # logging
            LOGGER.error(
                "Failed to create class %s with arguments %s", class_name, options
            )
            raise

    def get_options_doc(self, name):
        """Return the options documentation for the given class name.

        :param name: name of the class
        :returns: the dict of option name: option documentation
        """
        cls = self.get_class(name)
        return SourceParsing.get_options_doc(cls.__init__)

    def get_default_options_values(self, name):
        """Return the options default values for the given class name.

        Only addresses kwargs

        :param name : name of the class
        :returns: the dict option name: option default value
        """
        cls = self.get_class(name)
        return SourceParsing.get_default_options_values(cls)

    def get_options_grammar(self, name, write_schema=False, schema_file=None):
        """Return the options grammar for a class.

        Attempts to generate a JSONGrammar from the arguments of the __init__
        method of the class

        :param name: name of the class
        :param schema_file: the output json file path. If None: input.json or
            output.json depending on gramamr type.
            (Default value = None)
        :param write_schema: if True, writes the schema files
            (Default value = False)
        :returns: the json grammar for options
        """
        args_dict = self.get_default_options_values(name)
        opts_doc = self.get_options_doc(name)
        opts_doc = {k: v for k, v in opts_doc.items() if k in args_dict}
        gramm = JSONGrammar(name)

        gramm.initialize_from_base_dict(
            args_dict,
            schema_file=schema_file,
            write_schema=write_schema,
            description_dict=opts_doc,
        )
        # Remove None args from required
        sch_dict = gramm.schema.to_dict()
        required = sch_dict["required"]
        has_changed = False
        for opt, val in args_dict.items():
            if val is None and opt in required:
                required.remove(opt)
                has_changed = True
        if has_changed:
            gramm = JSONGrammar(name, schema=sch_dict)
        return gramm

    def get_sub_options_grammar(self, class_name, **options):
        """Return the JSONGrammar of the sub options of a class.

        :param class_name: name of the class
        :param options: options to be passed to the class required to deduce
            the sub options
        """
        cls = self.get_class(class_name)
        return cls.get_sub_options_grammar(**options)

    def get_default_sub_options_values(self, class_name, **options):
        """Return the default values of the sub options of a class.

        :param class_name: name of the class
        :param options: options to be passed to the class required to deduce
            the sub options
        """
        cls = self.get_class(class_name)
        return cls.get_default_sub_options_values(**options)

    def __str__(self):
        """Return the representation of a factory.

        Gives the configuration with the successfully loaded modules and
        failed imports with the reason.
        """
        table = PrettyTable(
            ["Module", "Is available ?", "Purpose or error message"],
            title=self.base_class.__name__,
            min_table_width=120,
            max_table_width=120,
        )

        row_dict = {}
        for cls in self.__names_to_classes.values():
            msg = ""
            try:
                msgs = cls.__doc__.split("\n")
                while msgs and msg == "":
                    msg = msgs[0]
                    del msgs[0]
            except Exception as err:  # pylint: disable=(broad-except
                pass

            key = cls.__name__
            row_dict[key] = [key, "Yes", msg]

        for key, err in self.failed_imports.items():
            row_dict[key] = [key, "No", str(err)]

        # Take them all and then sort them for pretty printing
        for key in sorted(row_dict.keys()):
            table.add_row(row_dict[key])

        return table.get_string()
