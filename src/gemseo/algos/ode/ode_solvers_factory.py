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
#        :author: Isabelle Santos
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory to instantiate ODE solvers from their class names."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.algos.base_algo_factory import BaseAlgoFactory
from gemseo.algos.ode.ode_solver_lib import ODESolverLib

if TYPE_CHECKING:
    from gemseo.algos.ode.ode_problem import ODEProblem
    from gemseo.algos.opt_result import OptimizationResult


class ODESolversFactory(BaseAlgoFactory):
    """This class instantiates and ODE solver from its class name."""

    _CLASS = ODESolverLib
    _MODULE_NAMES = ("gemseo.algos.ode",)

    def execute(
        self,
        problem: ODEProblem,
        algo_name: str = "RK45",
        **options: Any,
    ) -> OptimizationResult:
        """Execute the solver.

        Find the appropriate library and execute the solver on the problem to
        solve the ordinary differential equation ``s(t)' = f(t, s(t))``

        Args:
            problem: The ordinary differential equation that defines the problem
            algo_name: The algorithm name.
            **options: The options for the algorithm,
                see associated JSON file.

        Returns:
            The solution.
        """
        return super().execute(problem, algo_name, **options)
