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
#                           documentation
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Abstract factory to create libraries of algorithms."""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.base_factory import BaseFactory
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.algos.base_algorithm_library import BaseAlgorithmLibrary
    from gemseo.algos.base_problem import BaseProblem
    from gemseo.algos.optimization_result import OptimizationResult


class _AlgoFactoryMeta(ABCMeta):
    """A metaclass to add an internal factory class derived from :class:`.Factory`."""

    _CLASS: ClassVar[type]
    """The base class that the factory can build."""

    _MODULE_NAMES: ClassVar[list[str]]
    """The fully qualified names of the modules to search."""

    _Factory: type[BaseFactory[BaseAlgorithmLibrary]]
    """The internal factory class."""

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> None:  # noqa: D107
        super().__init__(name, bases, namespace)
        # Do not create the internal factory for BaseAlgoFactory which is abstract.
        if name != "BaseAlgoFactory":
            cls._Factory = type(
                "_Factory",
                (BaseFactory,),
                {"_CLASS": cls._CLASS, "_MODULE_NAMES": cls._MODULE_NAMES},
            )
            # Set the correct fully qualified name for pickling.
            cls._Factory.__module__ = cls.__module__
            cls._Factory.__qualname__ = (
                f"{cls.__qualname__}.{cls._Factory.__qualname__}"
            )


class BaseAlgoFactory(metaclass=_AlgoFactoryMeta):
    """A base class for creating factories for objects of kind :class:`AlgoLib`.

    This factory can create objects from a base class
    or any of its subclasses that can be imported from the given module sources.
    The base class and the module sources shall be defined as class attributes of the
    factory class,
    for instance::

    class AFactory(BaseAlgoFactory):
        _CLASS = ABaseClass
        _MODULE_NAMES = ("first.module.fully.qualified.name",
                         "second.module.fully.qualified.name")

    A factory instance can use a cache for the objects it creates, this cache is only
    used by one factory instance and is not shared with another instance.
    The cache is activated by passing ``use_cache = True`` to the constructor.
    When the cache is activated, a factory will return an object already created when
    possible and will create a new object otherwise.
    """

    __lib_cache: dict[str, dict[str, BaseAlgorithmLibrary]]
    """The library cache."""

    __use_cache: bool
    """Whether to cache the created objects."""

    def __init__(self, use_cache: bool = False) -> None:
        """
        Args:
            use_cache: Whether to cache the created objects.
        """  # noqa:D205 D212 D415
        self.__lib_cache = {}
        self.__use_cache = use_cache
        self._factory = self._Factory()
        self.__algo_name_to_lib_name = {}
        for lib_name in self.libraries:
            cls = self._factory.get_class(lib_name)
            self.__lib_cache[lib_name] = {}
            for algo_name in cls.ALGORITHM_INFOS:
                self.__algo_name_to_lib_name[algo_name] = lib_name

    @property
    @abstractmethod
    def _CLASS(self) -> type:  # noqa:N802
        """The base class that the factory can build."""

    @property
    @abstractmethod
    def _MODULE_NAMES(self) -> list[str]:  # noqa:N802
        """The fully qualified names of the modules to search."""

    def is_available(self, name: str) -> bool:
        """Check the availability of a library name or algorithm name.

        Args:
            name: The name of the library name or algorithm name.

        Returns:
            Whether the library or algorithm is available.
        """
        return self._factory.is_available(self.__algo_name_to_lib_name.get(name, name))

    @property
    def algorithms(self) -> list[str]:
        """The available algorithms names."""
        return list(self.__algo_name_to_lib_name.keys())

    @property
    def algo_names_to_libraries(self) -> dict[str, str]:
        """The mapping from the algorithm names to the libraries."""
        return self.__algo_name_to_lib_name

    @property
    def libraries(self) -> list[str]:
        """List the available library names in the present configuration.

        Returns:
            The names of the available libraries.
        """
        return self._factory.class_names

    def create(self, algo_name: str) -> BaseAlgorithmLibrary:
        """Create an algorithm library from an algorithm name.

        Args:
            algo_name: The name of an algorithm.

        Returns:
             The algorithm library.

        Raises:
            ImportError: If the algorithm is not available.
        """
        if algo_name not in self.__algo_name_to_lib_name:
            msg = (
                f"No algorithm named {algo_name} is available; "
                f"available algorithms are {pretty_str(self.algorithms, use_and=True)}."
            )
            raise ValueError(msg)

        lib_name = self.__algo_name_to_lib_name.get(algo_name)
        lib_cache = self.__lib_cache[lib_name]
        if self.__use_cache:
            algo = lib_cache.get(algo_name)
            if algo is None:
                algo = self._factory.create(lib_name, algo_name=algo_name)
                lib_cache[algo_name] = algo

            return algo

        return self._factory.create(lib_name, algo_name=algo_name)

    def execute(
        self,
        problem: BaseProblem,
        algo_name: str,
        **options: Any,
    ) -> OptimizationResult:
        """Execute a problem with an algorithm.

        Args:
            problem: The problem to execute.
            algo_name: The name of the algorithm.
            **options: The options of the algorithm.

        Returns:
            The optimization result.
        """
        return self.create(algo_name).execute(problem, **options)

    def clear_lib_cache(self) -> None:
        """Clear the library cache."""
        for value in self.__lib_cache.values():
            value.clear()

    def get_library_name(self, name: str) -> str:
        """Return the name of the library related to the name of a class.

        Args:
            name: The name of the class.

        Returns:
            The name of the library.
        """
        return self._factory.get_library_name(name)

    def get_class(self, name: str) -> type:
        """Return a class from its name.

        Args:
            name: The name of the class.

        Returns:
            The class.

        Raises:
            ImportError: If the class is not available.
        """
        return self._factory.get_class(name)

    def _repr_html(self) -> str:
        return self._factory._repr_html_()

    def __repr__(self) -> str:
        return repr(self._factory)
