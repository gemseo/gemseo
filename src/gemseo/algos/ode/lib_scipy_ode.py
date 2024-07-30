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
#        :author: Isabelle Santos
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Wrappers for SciPy's ODE solvers.

ODE stands for ordinary differential equation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import inf
from scipy.integrate import solve_ivp

from gemseo.algos.ode.base_ode_solver_library import BaseODESolverLibrary
from gemseo.algos.ode.base_ode_solver_library import ODESolverDescription

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.algos.ode.ode_problem import ODEProblem
    from gemseo.algos.ode.ode_result import ODEResult

LOGGER = logging.getLogger(__name__)


class ScipyODEAlgos(BaseODESolverLibrary):
    """Wrapper for SciPy's ODE solvers.

    ODE stands for ordinary differential equation.
    """

    ALGORITHM_INFOS: ClassVar[dict[str, ODESolverDescription]] = {
        name: ODESolverDescription(
            algorithm_name=name,
            internal_algorithm_name=name,
            description="ODE solver implemented in the SciPy library.",
            library_name="SciPy",
            website=f"https://docs.scipy.org/doc/scipy/reference/generated/{name}.html",
        )
        for name in [
            "RK45",
            "RK23",
            "DOP853",
            "Radau",
            "BDF",
            "LSODA",
        ]
    }

    def _get_options(
        self,
        first_step: float | None = None,
        max_step: float = inf,
        rtol: float | NDArray[float] = 1e-3,
        atol: float | NDArray[float] = 1e-6,
        jac_sparsity: NDArray[float] | None = None,
        lband: int | None = None,
        uband: int | None = None,
        min_step: float = 0,
    ) -> dict[str, Any]:
        """Check the options and set the default values.

        For more information, see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        Args:
            first_step: Initial step size. If ``None``, let the algorithm choose.
            max_step: Maximum allowed step size.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            jac_sparsity: Sparsity structure of the Jacobian matrix.
            lband: Lower boundary of the bandwidth for the "LSODA" method.
            uband: Upper boundary of the bandwidth for the "LSODA" method.
            min_step: Minimum allowed step for the "LSODA" method.

        Returns:
            The options of the solver.

        Raises:
            ValueError: When the LHR and RHS shapes are inconsistent, or
                when the preconditioner options are inconsistent.
        """
        return self._process_options(
            first_step=first_step,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
            jac_sparsity=jac_sparsity,
            lband=lband,
            uband=uband,
            min_step=min_step,
        )

    def _run(
        self, problem: ODEProblem, **options: bool | float | NDArray[float] | None
    ) -> ODEResult:
        if problem.time_vector is not None:
            options["t_eval"] = problem.time_vector

        if problem.jac is not None:
            options["jac"] = problem.jac

        solution = solve_ivp(
            fun=problem.rhs_function,
            y0=problem.initial_state,
            method=self._algo_name,
            t_span=problem.integration_interval,
            **options,
        )

        problem.result.is_converged = solution.status == 0
        problem.result.solver_message = solution.message
        if not problem.result.is_converged:
            LOGGER.warning(solution.message)

        problem.result.state_vector = solution.y
        problem.result.time_vector = solution.t
        problem.result.n_func_evaluations = solution.nfev
        problem.result.n_jac_evaluations = solution.njev

        return problem.result
