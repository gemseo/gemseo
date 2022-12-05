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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory to instantiate linear solvers from their class names."""
from __future__ import annotations

from typing import Any

from numpy import ndarray

from gemseo.algos.driver_factory import DriverFactory
from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverLib


class LinearSolversFactory(DriverFactory):
    """MDA factory to create the MDA from a name or a class."""

    def __init__(self) -> None:  # noqa:D107
        super().__init__(LinearSolverLib, "gemseo.algos.linear_solvers")

    @property
    def linear_solvers(self) -> list[str]:
        """The names of the available classes."""
        return self.factory.classes

    def is_available(self, solver_name: str) -> bool:
        """Check the availability of a LinearSolver.

        Args:
            solver_name: The name of the LinearSolver.

        Returns:
            Whether the :class:`.LinearSolver` is available.
        """
        return super().is_available(solver_name)

    def execute(
        self,
        problem: LinearProblem,
        algo_name: str,
        **options: Any,
    ) -> ndarray:
        """Execute the driver.

        Find the appropriate library and execute the driver on the problem to solve
        the linear system LHS.x = RHS.

        Args:
            problem: The linear equations and right hand side
             (lhs, rhs) that defines the linear problem. XXX is a tuple expected?
            algo_name: The algorithm name.
            **options: The options for the algorithm,
                see associated JSON file.

        Returns:
            The solution.
        """
        lib = self.create(algo_name)
        return lib.execute(problem, algo_name=algo_name, **options)
