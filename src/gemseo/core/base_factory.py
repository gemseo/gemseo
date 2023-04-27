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
from abc import ABCMeta
from abc import abstractmethod
from importlib import metadata
from inspect import isabstract
from typing import Any
from typing import ClassVar
from typing import Iterable
from typing import NamedTuple

from docstring_inheritance import GoogleDocstringInheritanceMeta

from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.third_party.prettytable import PrettyTable
from gemseo.utils.source_parsing import get_default_option_values
from gemseo.utils.source_parsing import get_options_doc

LOGGER = logging.getLogger(__name__)


class _FactoryMultitonMeta(ABCMeta, GoogleDocstringInheritanceMeta):
    """A metaclass for implementing the Multiton design pattern.

    See `Multiton <https://en.wikipedia.org/wiki/Multiton_pattern>`.

    As opposed to the functools.lru_cache,
    the objects built from this metaclass can be pickled.

    A cache entry is bound to the tuple combining :attr:`.Factory._CLASS` and
    :attr:`.Factory._MODULE_NAMES`.
    When instantiating a factory, if an instance has already been created for this
    tuple then this instance is used, otherwise a new instance is created is stored
    into the cache.
    """

    __cache: ClassVar[dict[tuple, BaseFactory]] = {}
    """The cache that keeps the factory instances."""

    def __call__(cls) -> BaseFactory:  # noqa: D107
        key = (cls._CLASS,) + tuple(cls._MODULE_NAMES)
        # Either return an instance that match an already existing key
        # or create and return a new instance.
        obj = cls.__cache.get(key)
        if obj is not None:
            return obj
        return cls.__cache.setdefault(key, type.__call__(cls))

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cache."""
        cls.__cache.clear()


class _ClassInfo(NamedTuple):
    """Information about a class exposed via the factory."""

    class_: type
    """The class."""

    library_name: str
    """The name of the library (the module) that contains the class."""


class BaseFactory(metaclass=_FactoryMultitonMeta):
    """A base class for factory of objects.

    This factory can create objects from a base class
    or any of its subclasses that can be imported from the given module sources.
    The base class and the module sources shall be defined as class attributes of the
    factory class,
    for instance::

        class AFactory(BaseFactory):
            _CLASS = ABaseClass
            _MODULE_NAMES = ("first.module.fully.qualified.name",
                             "second.module.fully.qualified.name")

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

    _ENV_VAR_WITH_SEARCH_PATHS: ClassVar[str] = "GEMSEO_PATH"
    """The name of the environment variable that contains the paths to search for
    classes."""

    PLUGIN_ENTRY_POINT: ClassVar[str] = "gemseo_plugins"
    """The name of the setuptools entry point for declaring plugins."""

    _names_to_class_info: dict[str, _ClassInfo]
    """The class names bound to the class information."""

    failed_imports: dict[str, str]
    """The class names bound to the import errors."""

    def __init__(self) -> None:  # noqa: D107
        self._names_to_class_info = {}
        self.failed_imports = {}
        self.update()

    @property
    @abstractmethod
    def _CLASS(self) -> type:  # noqa: N802
        """The base class that the factory can build."""

    @property
    @abstractmethod
    def _MODULE_NAMES(self) -> list[str]:  # noqa: N802
        """The fully qualified names of the modules to search."""

    def update(self) -> None:
        """Search for the classes that can be instantiated.

        The search is done in the following order:
            1. The fully qualified module names
            2. The plugin packages
            3. The packages from the environment variables
        """
        module_names = list(self._MODULE_NAMES)

        # Import the fully qualified modules names.
        for module_name in module_names:
            self.__import_modules_from(module_name)

        # Import the plugins packages.

        # Do not search the current working directory.
        # See https://docs.python.org/3.9/library/sys.html#sys.path
        sys_path = list(sys.path)
        sys_path.pop(0)

        # Import from the setuptools entry points.
        for entry_point in metadata.entry_points().get(self.PLUGIN_ENTRY_POINT, []):
            module_name = entry_point.value
            self.__import_modules_from(module_name)
            module_names += [module_name]

        module_names += self.__import_modules_from_env_var()

        names_to_classes = self.__get_sub_classes(self._CLASS)

        if not isabstract(self._CLASS):
            names_to_classes[self._CLASS.__name__] = self._CLASS

        for name, cls in names_to_classes.items():
            if self.__is_class_in_modules(module_names, cls) and not isabstract(cls):
                self._names_to_class_info[name] = _ClassInfo(
                    cls, cls.__module__.split(".")[0]
                )

    def __log_import_failure(self, pkg_name: str) -> None:
        """Log import failures.

        Args:
            pkg_name: The name of a package that failed to be imported.
        """
        LOGGER.debug("Failed to import package %s", pkg_name)
        self.failed_imports[pkg_name] = ""

    def __import_modules_from_env_var(self) -> list[str]:
        """Import the modules from the path given by an environment variable.

        Returns:
            The imported fully qualified module names.
        """
        search_paths = os.environ.get(self._ENV_VAR_WITH_SEARCH_PATHS)
        if search_paths is None:
            return []

        if ":" in search_paths:
            paths = search_paths.split(":")
        else:
            paths = [search_paths]

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
                self.failed_imports[mod_name] = str(err)

    def __get_sub_classes(self, cls: type) -> dict[str, type]:
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
        module_names: Iterable[str],
        cls: type,
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

    @property
    def class_names(self) -> list[str]:
        """The sorted names of the available classes."""
        return sorted(self._names_to_class_info.keys())

    def is_available(self, name: str) -> bool:
        """Return whether a class can be instantiated.

        Args:
            name: The name of the class.

        Returns:
            Whether the class can be instantiated.
        """
        return name in self._names_to_class_info

    def get_library_name(self, name: str) -> str:
        """Return the name of the library related to the name of a class.

        Args:
            name: The name of the class.

        Returns:
            The name of the library.
        """
        return self._names_to_class_info[name].library_name

    def get_class(self, name: str) -> type:
        """Return a class from its name.

        Args:
            name: The name of the class.

        Returns:
            The class.

        Raises:
            ImportError: If the class is not available.
        """
        class_info = self._names_to_class_info.get(name)
        if class_info is None:
            names = ", ".join(self.class_names)
            raise ImportError(
                f"The class {name} is not available; the available ones are: {names}.",
            )
        return class_info.class_

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
        return get_options_doc(self.get_class(name).__init__)

    def get_default_option_values(
        self, name: str
    ) -> dict[str, str | int | float | bool]:
        """Return the constructor kwargs default values of a class.

        Args:
            name: The name of the class.

        Returns:
            The mapping from the argument names to their default values.
        """
        return get_default_option_values(self.get_class(name))

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
        default_option_values = self.get_default_option_values(name)
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
            grammar.to_file(schema_path)

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
        return self.get_class(name).get_sub_options_grammar(**options)

    def get_default_sub_option_values(
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
        return self.get_class(name).get_default_sub_option_values(**options)

    def __str__(self) -> str:
        return f"Factory of {self._CLASS.__name__} objects"

    def __repr__(self) -> str:
        # Display the successfully loaded modules and the failed imports with the reason
        table = PrettyTable(
            ["Module", "Is available?", "Purpose or error message"],
            title=self._CLASS.__name__,
            min_table_width=120,
            max_table_width=120,
        )

        names_to_import_statuses = {}
        for class_info in self._names_to_class_info.values():
            cls = class_info.class_
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
            names_to_import_statuses[package_name] = [package_name, "No", err]

        # Take them all and then sort them for pretty printing
        for name in sorted(names_to_import_statuses.keys()):
            table.add_row(names_to_import_statuses[name])

        return table.get_string()
