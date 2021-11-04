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

"""Factory base class."""

from __future__ import division, unicode_literals

import importlib
import logging
import os
import pkgutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from gemseo.core.json_grammar import JSONGrammar
from gemseo.third_party.prettytable import PrettyTable
from gemseo.utils.py23_compat import PY3, importlib_metadata, lru_cache
from gemseo.utils.singleton import Multiton, _Multiton
from gemseo.utils.source_parsing import get_default_options_values, get_options_doc

LOGGER = logging.getLogger(__name__)


class Factory(Multiton):
    """Factory of class objects with cache.

    This factory can create an object from a base class
    or any of its sub-classes
    that can be imported from the given modules sources.
    There are 3 sources of modules that can be searched:

    - fully qualified module names (such as gemseo.problems, ...),
    - the environment variable "GEMSEO_PATH" may contain the list of directories,
    - |g| plugins, i.e. packages which have declared a setuptools entry point.

    A setuptools entry point is declared in a plugin :file:`setup.cfg` file,
    with a section::

        [options.entry_points]
        gemseo_plugins =
            a-name = plugin_package_name

    Above ``a-name`` is not used
    and can be any name
    but we advise to use the plugin name.

    The plugin entry point searched by the factory could be changed
    with :class:`.Factory.PLUGIN_ENTRY_POINT`.

    If a class,
    despite being a sub-class of the base class,
    or even the base class itself,
    does not belong to the modules sources
    then it is not taken into account
    by the factory.

    The created objects are cached:
    more calls to the constructor with the same call signature will return
    the object in cache instead of instantiating a new one.
    """

    # Names of the environment variable to search for classes
    __GEMSEO_PATH = "GEMSEO_PATH"
    __GEMS_PATH = "GEMS_PATH"

    # Allowed prefix for naming a plugin importable from sys.path
    __PLUGIN_PREFIX = "gemseo_"

    # The name of the setuptools entry point for declaring plugins.
    PLUGIN_ENTRY_POINT = "gemseo_plugins"

    def __init__(
        self,
        base_class,  # type: Type[Any]
        module_names=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        # noqa: D205, D212, D405, D415
        """
        Args:
            base_class: The base class to be considered.
            module_names: The fully qualified modules names to be searched.
        """
        if not isinstance(base_class, type):
            raise TypeError("Class to search must be a class!")

        self.__base_class = base_class
        self.__module_names = module_names or []
        self.__names_to_classes = {}
        self.failed_imports = {}

        self.update()

    def update(self):  # type: (...) -> None
        """Search for the classes that can be instantiated.

        The search is done in the following order:
            1. The fully qualified module names
            2. The plugin packages
            3. The packages from the environment variables
        """
        module_names = list(self.__module_names)

        # Import the fully qualified modules names.
        for module_name in module_names:
            self.__import_modules_from(module_name)

        # Import the plugins packages.

        # Do not search the current working directory.
        # See https://docs.python.org/3.9/library/sys.html#sys.path
        sys_path = list(sys.path)
        sys_path.pop(0)

        for _, module_name, _ in pkgutil.iter_modules(path=sys_path):
            if module_name.startswith(self.__PLUGIN_PREFIX):
                self.__import_modules_from(module_name)
                module_names += [module_name]

        if PY3:
            # Import from the setuptools entry points.
            for entry_point in importlib_metadata.entry_points().get(
                self.PLUGIN_ENTRY_POINT, []
            ):
                module_name = entry_point.value
                self.__import_modules_from(module_name)
                module_names += [module_name]

        gems_path = os.environ.get(self.__GEMS_PATH)
        if gems_path is not None:
            msg = (
                "GEMS is now named GEMSEO. "
                "The GEMS_PATH environment variable is now deprecated "
                "and it is strongly recommended "
                "to use the GEMSEO_PATH environment variable "
                "instead to register your GEMSEO plugins."
            )
            LOGGER.warning(msg)

        # Import from the environment variable paths.
        for env_variable in [self.__GEMSEO_PATH, self.__GEMS_PATH]:
            module_names += self.__import_modules_from_env_var(env_variable)

        names_to_classes = self.__get_sub_classes(self.__base_class)
        for name, cls in names_to_classes.items():
            if self.__is_class_in_modules(module_names, cls):
                self.__names_to_classes[name] = cls

    def __log_import_failure(
        self, pkg_name  # type: str
    ):  # type: (...) -> None
        """Log import failures.

        Args:
            pkg_name: The name of a package that failed to be imported.
        """
        LOGGER.debug("Failed to import package %s", pkg_name)
        self.failed_imports[pkg_name] = ""

    def __import_modules_from_env_var(
        self, env_variable  # type: str
    ):  # type: (...) -> List[str]
        """Import the modules from the path given by an environment variable.

        Args:
            env_variable: The name of an environment variable.

        Returns:
            The imported fully qualified module names.
        """
        g_path = os.environ.get(env_variable)
        if g_path is None:
            return []

        if ":" in g_path:
            paths = g_path.split(":")
        else:
            paths = [g_path]

        # temporary make the paths visible to the import machinery
        for path in paths:
            sys.path.insert(0, path)

        mod_names = list()
        for _, mod_name, _ in pkgutil.iter_modules(path=paths):
            self.__import_modules_from(mod_name)
            mod_names += [mod_name]

        for _ in paths:
            sys.path.pop(0)

        return mod_names

    def __import_modules_from(
        self, pkg_name  # type: str
    ):  # type: (...) -> None
        """Import all the modules from a package.

        Args:
            pkg_name: The name of the package.
        """
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
                LOGGER.debug("Failed to import module: %s", mod_name, exc_info=True)
                self.failed_imports[mod_name] = err

    def __get_sub_classes(
        self, cls  # type: Type[Any]
    ):  # type: (...) -> Dict[str, Type[Any]]
        """Find all the sub classes of a class.

        The class names are unique,
        the last imported is kept when more than one class have the same name.

        Args:
            cls: A class.

        Returns:
            A mapping from the names to the unique sub-classes.
        """
        all_sub_classes = {}
        for sub_class in cls.__subclasses__():
            sub_classes = {sub_class.__name__: sub_class}
            sub_classes.update(self.__get_sub_classes(sub_class))
            for cls_name, _cls in sub_classes.items():
                all_sub_classes[cls_name] = _cls
        return all_sub_classes

    @staticmethod
    def __is_class_in_modules(
        module_names,  # type: str
        cls,  # type: Type[Any]
    ):  # type: (...) -> bool
        """Return whether a class belongs to given modules.

         Args:
            module_names: The names of the modules.
            cls: The class.

        Returns:
            Whether the class belongs to the modules.
        """
        for name in module_names:
            if cls.__module__.startswith(name):
                return True
        return False

    @property
    @lru_cache()
    def classes(self):  # type: (...) -> List[str]
        """Return the available classes.

        Returns:
            The sorted names of the available classes.
        """
        return sorted(self.__names_to_classes.keys())

    def is_available(
        self, name  # type: str
    ):  # type: (...) -> bool
        """Return whether a class can be instantiated.

        Args:
            name: The name of the class.

        Returns:
            Whether the class can be instantiated.
        """
        return name in self.__names_to_classes

    def get_class(
        self, name  # type: str
    ):  # type: (...) -> Type[Any]
        """Return a class from its name.

        Args:
            name: The name of the class.

        Returns:
            The class.

        Raises:
            ImportError: If the class is not available.
        """
        try:
            return self.__names_to_classes[name]
        except KeyError:
            msg = "Class {} is not available!\nAvailable ones are: {}".format(
                name, ", ".join(sorted(self.__names_to_classes.keys()))
            )
            raise ImportError(msg)

    def create(
        self,
        class_name,  # type: str
        **options  # type: Any
    ):  # type: (...) -> Any
        """Return an instance of a class.

        Args:
            class_name: The name of the class.
            **options: The arguments to be passed to the class constructor.

        Returns:
            The instance of the class.

        Raises:
            TypeError: If the class cannot be instantiated.
        """
        cls = self.get_class(class_name)
        try:
            return cls(**options)
        except TypeError:
            LOGGER.error(
                "Failed to create class %s with arguments %s", class_name, options
            )
            raise

    def get_options_doc(
        self, name  # type: str
    ):  # type: (...) -> Dict[str, str]
        """Return the constructor documentation of a class.

        Args:
            name: The name of the class.

        Returns:
            The mapping from the argument names to their documentation.
        """
        cls = self.get_class(name)
        return get_options_doc(cls.__init__)

    def get_default_options_values(
        self, name  # type: str
    ):  # type: (...) -> Dict[str, Union[str,int,float,bool]]
        """Return the constructor kwargs default values of a class.

        Args:
            name: The name of the class.

        Returns:
            The mapping from the argument names to their default values.
        """
        cls = self.get_class(name)
        return get_default_options_values(cls)

    def get_options_grammar(
        self,
        name,  # type: str
        write_schema=False,  # type: bool
        schema_file=None,  # type: Optional[str]
    ):  # type: (...) -> JSONGrammar
        """Return the options JSON grammar for a class.

        Attempt to generate a JSONGrammar
        from the arguments of the __init__ method of the class.

        Args:
            name: The name of the class.
            write_schema: If True, write the JSON schema to a file.
            schema_file: The path to the JSON schema file.
                If None, the file is saved in the current directory in a file named
                after the name of the class.

        Returns:
            The JSON grammar.
        """
        args_dict = self.get_default_options_values(name)
        opts_doc = self.get_options_doc(name)
        opts_doc = {k: v for k, v in opts_doc.items() if k in args_dict}
        grammar = JSONGrammar(name)

        grammar.initialize_from_base_dict(
            args_dict,
            description_dict=opts_doc,
        )

        if write_schema:
            grammar.write_schema(schema_file)

        # Remove None args from required
        sch_dict = grammar.schema.to_dict()
        required = sch_dict["required"]
        has_changed = False

        for opt, val in args_dict.items():
            if val is None and opt in required:
                required.remove(opt)
                has_changed = True

        if has_changed:
            grammar = JSONGrammar(name, schema=sch_dict)

        return grammar

    def get_sub_options_grammar(
        self,
        name,  # type: str
        **options  # type: str
    ):  # type: (...) -> JSONGrammar
        """Return the JSONGrammar of the sub options of a class.

        Args:
            name: The name of the class.
            **options: The options to be passed to the class required to deduce
                the sub options.

        Returns:
            The JSON grammar.
        """
        cls = self.get_class(name)
        return cls.get_sub_options_grammar(**options)

    def get_default_sub_options_values(
        self,
        name,  # type: str
        **options  # type: str
    ):  # type: (...) -> JSONGrammar
        """Return the default values of the sub options of a class.

        Args:
            name: The name of the class.
            **options: The options to be passed to the class required to deduce
                the sub options.

        Returns:
            The JSON grammar.
        """
        cls = self.get_class(name)
        return cls.get_default_sub_options_values(**options)

    @staticmethod
    def cache_clear():
        _Multiton.cache_clear()

    def __str__(self):  # type: (...) -> str
        return "Factory({})".format(self.__base_class.__name__)

    def __repr__(self):  # type: (...) -> str
        # Display the successfully loaded modules and the failed imports with the reason
        table = PrettyTable(
            ["Module", "Is available ?", "Purpose or error message"],
            title=self.__base_class.__name__,
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
            except Exception:  # pylint: disable=broad-except
                pass

            key = cls.__name__
            row_dict[key] = [key, "Yes", msg]

        for key, err in self.failed_imports.items():
            row_dict[key] = [key, "No", str(err)]

        # Take them all and then sort them for pretty printing
        for key in sorted(row_dict.keys()):
            table.add_row(row_dict[key])

        return table.get_string()
