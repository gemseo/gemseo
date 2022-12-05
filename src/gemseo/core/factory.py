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
from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys
from inspect import isabstract
from typing import Any
from typing import Iterable

from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.third_party.prettytable import PrettyTable
from gemseo.utils.python_compatibility import importlib_metadata
from gemseo.utils.singleton import _Multiton
from gemseo.utils.singleton import Multiton
from gemseo.utils.source_parsing import get_default_options_values
from gemseo.utils.source_parsing import get_options_doc

LOGGER = logging.getLogger(__name__)


class Factory(Multiton):
    """Factory of class objects with cache.

    This factory can create an object from a base class
    or any of its subclasses that can be imported from the given modules sources.
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
    and can be any name,
    but we advise to use the plugin name.

    The plugin entry point searched by the factory could be changed
    with :class:`.Factory.PLUGIN_ENTRY_POINT`.

    If a class,
    despite being a subclass of the base class,
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

    # The name of the setuptools entry point for declaring plugins.
    PLUGIN_ENTRY_POINT = "gemseo_plugins"

    def __init__(
        self,
        base_class: type[Any],
        module_names: Iterable[str] | None = None,
    ) -> None:
        """
        Args:
            base_class: The base class to be considered.
            module_names: The fully qualified modules names to be searched.
        """  # noqa: D205, D212, D415
        if not isinstance(base_class, type):
            raise TypeError("Class to search must be a class!")

        self.__base_class = base_class
        self.__module_names = module_names or []
        self.__names_to_classes = {}
        self.__names_to_library_names = {}
        self.failed_imports = {}
        self.update()

    def update(self) -> None:
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
            if self.__is_class_in_modules(module_names, cls) and not isabstract(cls):
                self.__names_to_classes[name] = cls
                self.__names_to_library_names[name] = cls.__module__.split(".")[0]

    def __log_import_failure(self, pkg_name: str) -> None:
        """Log import failures.

        Args:
            pkg_name: The name of a package that failed to be imported.
        """
        LOGGER.debug("Failed to import package %s", pkg_name)
        self.failed_imports[pkg_name] = ""

    def __import_modules_from_env_var(self, env_variable: str) -> list[str]:
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

    def __import_modules_from(self, pkg_name: str) -> None:
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

    def __get_sub_classes(self, cls: type[Any]) -> dict[str, type[Any]]:
        """Find all the subclasses of a class.

        The class names are unique,
        the last imported is kept when more than one class have the same name.

        Args:
            cls: A class.

        Returns:
            A mapping from the names to the unique subclasses.
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
        module_names: str,
        cls: type[Any],
    ) -> bool:
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

    # TODO: API: rename classes to class_names
    @property
    def classes(self) -> list[str]:
        """The sorted names of the available classes."""
        return sorted(self.__names_to_classes.keys())

    def is_available(self, name: str) -> bool:
        """Return whether a class can be instantiated.

        Args:
            name: The name of the class.

        Returns:
            Whether the class can be instantiated.
        """
        return name in self.__names_to_classes

    def get_library_name(self, name: str) -> str:
        """Return the name of the library related to the name of a class.

        Args:
            name: The name of the class.

        Returns:
            The name of the library.
        """
        return self.__names_to_library_names[name]

    def get_class(self, name: str) -> type[Any]:
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
            raise ImportError(
                "Class {} is not available; \navailable ones are: {}.".format(
                    name, ", ".join(sorted(self.__names_to_classes.keys()))
                )
            )

    def create(
        self,
        class_name: str,
        **options: Any,
    ) -> Any:
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

    def get_options_doc(self, name: str) -> dict[str, str]:
        """Return the constructor documentation of a class.

        Args:
            name: The name of the class.

        Returns:
            The mapping from the argument names to their documentation.
        """
        cls = self.get_class(name)
        return get_options_doc(cls.__init__)

    def get_default_options_values(
        self, name: str
    ) -> dict[str, str | int | float | bool]:
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
        name: str,
        write_schema: bool = False,
        schema_path: str | None = None,
    ) -> JSONGrammar:
        """Return the options JSON grammar for a class.

        Attempt to generate a JSONGrammar
        from the arguments of the __init__ method of the class.

        Args:
            name: The name of the class.
            write_schema: If True, write the JSON schema to a file.
            schema_path: The path to the JSON schema file.
                If None, the file is saved in the current directory in a file named
                after the name of the class.

        Returns:
            The JSON grammar.
        """
        default_option_values = self.get_default_options_values(name)
        option_descriptions = {
            # The parsed docstrings contain carriage returns
            # in the descriptions of the arguments for a better HTML rendering
            # but the JSON grammars do not contain this special character.
            option_name: option_description.replace("\n", " ")
            for option_name, option_description in self.get_options_doc(name).items()
            if option_name in default_option_values
        }
        grammar = JSONGrammar(name)
        grammar.update_from_data(default_option_values)
        grammar.set_descriptions(option_descriptions)

        # Remove args bound to None from the required properties
        # because they are optional.
        for opt, val in default_option_values.items():
            if val is None:
                grammar.required_names.remove(opt)

        if write_schema:
            grammar.write(schema_path)

        return grammar

    def get_sub_options_grammar(
        self,
        name: str,
        **options: str,
    ) -> JSONGrammar:
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
        name: str,
        **options: str,
    ) -> JSONGrammar:
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
    def cache_clear() -> None:
        """Clear the cache."""
        _Multiton.cache_clear()

    def __str__(self) -> str:
        return f"Factory({self.__base_class.__name__})"

    def __repr__(self) -> str:
        # Display the successfully loaded modules and the failed imports with the reason
        table = PrettyTable(
            ["Module", "Is available ?", "Purpose or error message"],
            title=self.__base_class.__name__,
            min_table_width=120,
            max_table_width=120,
        )

        names_to_import_statuses = {}
        for cls in self.__names_to_classes.values():
            msg = ""
            try:
                class_docstring_lines = cls.__doc__.split("\n")
                while class_docstring_lines and msg == "":
                    msg = class_docstring_lines[0]
                    del class_docstring_lines[0]
            except Exception:  # pylint: disable=broad-except
                pass

            class_name = cls.__name__
            names_to_import_statuses[class_name] = [class_name, "Yes", msg]

        for package_name, err in self.failed_imports.items():
            names_to_import_statuses[package_name] = [package_name, "No", str(err)]

        # Take them all and then sort them for pretty printing
        for name in sorted(names_to_import_statuses.keys()):
            table.add_row(names_to_import_statuses[name])

        return table.get_string()
