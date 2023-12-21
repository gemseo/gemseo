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

from numpy import inf
from scipy.integrate import solve_ivp

from gemseo.algos.ode.ode_solver_lib import ODESolverDescription
from gemseo.algos.ode.ode_solver_lib import ODESolverLib

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.algos.ode.ode_result import ODEResult

LOGGER = logging.getLogger(__name__)


class ScipyODEAlgos(ODESolverLib):
    """Wrapper for SciPy's ODE solvers.

    ODE stands for ordinary differential equation.
    """

    __WEBSITE = "https://docs.scipy.org/doc/scipy/reference/generated/{}.html"
    __WEBPAGE = "scipy.integrate.solve_ivp"

    LIBRARY_NAME = "SciPy"

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        self.descriptions = {
            name: ODESolverDescription(
                algorithm_name=name,
                internal_algorithm_name=name,
                description="ODE solver implemented in the SciPy library.",
                library_name="SciPy",
                website=self.__WEBSITE.format(name),
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

    def _run(self, **options: bool | int | float | NDArray[float] | None) -> ODEResult:
        if self.problem.time_vector is not None:
            options["t_eval"] = self.problem.time_vector
        if self.problem.jac is not None:
            options["jac"] = self.problem.jac

        solution = solve_ivp(
            fun=self.problem.rhs_function,
            y0=self.problem.initial_state,
            method=self.algo_name,
            t_span=self.problem.integration_interval,
            **options,
        )

        self.problem.result.is_converged = solution.status == 0
        self.problem.result.solver_message = solution.message
        if not self.problem.result.is_converged:
            LOGGER.warning(solution.message)

        self.problem.result.state_vector = solution.y
        self.problem.result.time_vector = solution.t
        self.problem.result.n_func_evaluations = solution.nfev
        self.problem.result.n_jac_evaluations = solution.njev

        return self.problem.result
