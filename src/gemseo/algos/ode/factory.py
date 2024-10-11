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
"""A factory of ODE solver libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.algos.base_algo_factory import BaseAlgoFactory
from gemseo.algos.ode.base_ode_solver_library import BaseODESolverLibrary

if TYPE_CHECKING:
    from gemseo.algos.ode.ode_problem import ODEProblem
    from gemseo.algos.ode.ode_problem import ODEResult


class ODESolverLibraryFactory(BaseAlgoFactory):
    """A factory of ODE solver libraries."""

    _CLASS = BaseODESolverLibrary
    _MODULE_NAMES = ("gemseo.algos.ode",)

    def execute(
        self,
        problem: ODEProblem,
        algo_name: str = "RK45",
        **settings: Any,
    ) -> ODEResult:
        """Execute the solver.

        Find the appropriate library and execute the solver on the problem to
        solve the ordinary differential equation ``s(t)' = f(t, s(t))``
        """
        return super().execute(problem, algo_name, **settings)
