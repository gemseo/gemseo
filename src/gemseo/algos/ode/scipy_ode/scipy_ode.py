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
from typing import Final

from numpy import newaxis
from scipy.integrate import solve_ivp

from gemseo.algos.ode.base_ode_solver_library import BaseODESolverLibrary
from gemseo.algos.ode.base_ode_solver_library import ODESolverDescription
from gemseo.algos.ode.base_ode_solver_library_settings import (
    BaseODESolverLibrarySettings,
)
from gemseo.algos.ode.scipy_ode.settings.bdf import BDFSettings
from gemseo.algos.ode.scipy_ode.settings.dop853 import DOP853Settings
from gemseo.algos.ode.scipy_ode.settings.lsoda import LSODASettings
from gemseo.algos.ode.scipy_ode.settings.radau import RadauSettings
from gemseo.algos.ode.scipy_ode.settings.rk23 import RK23Settings
from gemseo.algos.ode.scipy_ode.settings.rk45 import RK45Settings

if TYPE_CHECKING:
    from gemseo.algos.ode.ode_problem import ODEProblem
    from gemseo.algos.ode.ode_result import ODEResult

LOGGER = logging.getLogger(__name__)


class ScipyODESolverDescription(ODESolverDescription):
    """The description of SciPy ODE solvers."""

    library_name: str = "SciPy ODE"
    """The name of the wrapped library."""


class ScipyODEAlgos(BaseODESolverLibrary):
    """Wrapper for SciPy's ODE solvers.

    ODE stands for ordinary differential equation.
    """

    __DOC: Final[str] = "https://docs.scipy.org/doc/scipy/reference/generated/"

    ALGORITHM_INFOS: ClassVar[dict[str, ScipyODESolverDescription]] = {
        "RK45": ScipyODESolverDescription(
            algorithm_name="RK45",
            internal_algorithm_name="RK45",
            description="Explicit Runge-Kutta method of order 5(4)",
            website=f"{__DOC}scipy.integrate.RK45.html",
            Settings=RK45Settings,
        ),
        "RK23": ScipyODESolverDescription(
            algorithm_name="RK23",
            internal_algorithm_name="RK23",
            description="Explicit Runge-Kutta method of order 3(2)",
            website=f"{__DOC}scipy.integrate.RK23.html",
            Settings=RK23Settings,
        ),
        "DOP853": ScipyODESolverDescription(
            algorithm_name="DOP853",
            internal_algorithm_name="DOP853",
            description="Explicit Runge-Kutta method of order 8",
            website=f"{__DOC}scipy.integrate.DOP853.html",
            Settings=DOP853Settings,
        ),
        "Radau": ScipyODESolverDescription(
            algorithm_name="Radau",
            internal_algorithm_name="Radau",
            description="Implicit Runge-Kutta method of the Radau IIA type of order 5",
            website=f"{__DOC}scipy.integrate.Radau.html",
            Settings=RadauSettings,
        ),
        "BDF": ScipyODESolverDescription(
            algorithm_name="BDF",
            internal_algorithm_name="BDF",
            description=(
                "Implicit multi-step variable-order (1 to 5) method based on a backward"
                " differentiation formula for the derivative approximation"
            ),
            website=f"{__DOC}scipy.integrate.BDF.html",
            Settings=BDFSettings,
        ),
        "LSODA": ScipyODESolverDescription(
            algorithm_name="LSODA",
            internal_algorithm_name="LSODA",
            description="Adams/BDF method with automatic stiffness detection/switching",
            website=f"{__DOC}scipy.integrate.LSODA.html",
            Settings=LSODASettings,
        ),
    }

    def _run(self, problem: ODEProblem, **settings: Any) -> ODEResult:
        settings_ = self._filter_settings(
            settings, model_to_exclude=BaseODESolverLibrarySettings
        )
        settings_["jac"] = problem.jac.state

        if problem.solve_at_algorithm_times:
            settings_["t_eval"] = problem.times

        solution = solve_ivp(
            fun=problem.rhs_function,
            y0=problem.initial_state,
            method=self._algo_name,
            t_span=problem.time_interval,
            events=problem.event_functions,
            **settings_,
        )

        problem.result.algorithm_name = self._algo_name
        problem.result.algorithm_settings = settings_
        problem.result.algorithm_has_converged = solution.status >= 0
        problem.result.algorithm_termination_message = solution.message
        problem.result.state_trajectories = solution.y
        problem.result.times = solution.t
        problem.result.n_func_evaluations = solution.nfev
        problem.result.n_jac_evaluations = solution.njev

        if not problem.result.algorithm_has_converged:
            LOGGER.warning(solution.message)

        index = None
        if solution.t_events:
            for i, t_event in enumerate(solution.t_events):
                if len(t_event):
                    index = i
                break

        problem.result.terminal_event_index = index
        if problem.event_functions and index is not None:
            problem.result.terminal_event_time = solution.t_events[index]
            problem.result.terminal_event_state = solution.y_events[index][0][
                :, newaxis
            ]
        else:
            problem.result.terminal_event_time = solution.t[-1:]
            problem.result.terminal_event_state = solution.y[:, -1][:, newaxis]

    def _get_result(self, problem: ODEProblem) -> ODEResult:
        return problem.result
