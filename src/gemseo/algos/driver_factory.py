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
"""Abstract factory to create drivers."""
from __future__ import annotations

from typing import Any

from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.factory import Factory


class DriverFactory:
    """Base class for definition of optimization and/or DOE factory.

    Automates the creation of library interfaces given a name of the algorithm.
    """

    def __init__(self, driver_lib_class: type[DriverLib], driver_package: str) -> None:
        """Initializes the factory: scans the directories to search for subclasses of
        DriverLib.

        Searches subclasses of driver_lib_class in "GEMSEO_PATH" and driver_package.
        """  # noqa: D205, D212, D415
        self.factory = Factory(driver_lib_class, (driver_package,))
        self.__algo_name_to_lib_name = {}
        for lib_name in self.libraries:
            lib = self.create(lib_name)
            for algo_name in lib.algorithms:
                self.__algo_name_to_lib_name[algo_name] = lib_name

    def is_available(self, name: str) -> bool:
        """Check the availability of a library name or algorithm name.

        Args:
            name: The name of the library name or algorithm name.

        Returns:
            Whether the library or algorithm is available.
        """
        return name in self.__algo_name_to_lib_name or self.factory.is_available(name)

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
        return self.factory.classes

    def create(self, name: str) -> DriverLib:
        """Create a driver library from an algorithm name or a library name.

        Args:
            name: The name of a library or algorithm.

        Returns:
             The driver library.
        """
        lib_name = self.__algo_name_to_lib_name.get(name)
        algo_name = None
        if lib_name is not None:
            algo_name = name
        elif self.factory.is_available(name):
            lib_name = name
        else:
            algorithms = ", ".join(sorted(self.algorithms))
            raise ImportError(
                f"No algorithm or library of algorithms named '{name}' is available; "
                f"available algorithms are {algorithms}."
            )
        lib_created = self.factory.create(lib_name)
        # Set the algo name if it was passed by the user as "name" arg
        lib_created.algo_name = algo_name
        return lib_created

    def execute(
        self,
        problem: OptimizationProblem,
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
        return self.create(algo_name).execute(problem, algo_name=algo_name, **options)
