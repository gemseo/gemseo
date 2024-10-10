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

from scipy.integrate import solve_ivp

from gemseo.algos.ode._base_ode_solver_library_settings import (
    BaseODESolverLibrarySettings,
)
from gemseo.algos.ode.base_ode_solver_library import BaseODESolverLibrary
from gemseo.algos.ode.base_ode_solver_library import ODESolverDescription
from gemseo.algos.ode.scipy_ode._settings.bdf import BDFSettings
from gemseo.algos.ode.scipy_ode._settings.dop853 import DOP853Settings
from gemseo.algos.ode.scipy_ode._settings.lsoda import LSODASettings
from gemseo.algos.ode.scipy_ode._settings.radau import RadauSettings
from gemseo.algos.ode.scipy_ode._settings.rk23 import RK23Settings
from gemseo.algos.ode.scipy_ode._settings.rk45 import RK45Settings

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
            settings=RK45Settings,
        ),
        "RK23": ScipyODESolverDescription(
            algorithm_name="RK23",
            internal_algorithm_name="RK23",
            description="Explicit Runge-Kutta method of order 3(2)",
            website=f"{__DOC}scipy.integrate.RK23.html",
            settings=RK23Settings,
        ),
        "DOP853": ScipyODESolverDescription(
            algorithm_name="DOP853",
            internal_algorithm_name="DOP853",
            description="Explicit Runge-Kutta method of order 8",
            website=f"{__DOC}scipy.integrate.DOP853.html",
            settings=DOP853Settings,
        ),
        "Radau": ScipyODESolverDescription(
            algorithm_name="Radau",
            internal_algorithm_name="Radau",
            description="Implicit Runge-Kutta method of the Radau IIA type of order 5",
            website=f"{__DOC}scipy.integrate.Radau.html",
            settings=RadauSettings,
        ),
        "BDF": ScipyODESolverDescription(
            algorithm_name="BDF",
            internal_algorithm_name="BDF",
            description=(
                "Implicit multi-step variable-order (1 to 5) method based on a backward"
                " differentiation formula for the derivative approximation"
            ),
            website=f"{__DOC}scipy.integrate.BDF.html",
            settings=BDFSettings,
        ),
        "LSODA": ScipyODESolverDescription(
            algorithm_name="LSODA",
            internal_algorithm_name="LSODA",
            description="Adams/BDF method with automatic stiffness detection/switching",
            website=f"{__DOC}scipy.integrate.LSODA.html",
            settings=LSODASettings,
        ),
    }

    def _run(self, problem: ODEProblem, **settings: Any) -> ODEResult:
        settings_ = self._filter_settings(
            settings, model_to_exclude=BaseODESolverLibrarySettings
        )

        if problem.time_vector is not None:
            settings_["t_eval"] = problem.time_vector

        if problem.jac is not None:
            settings_["jac"] = problem.jac

        solution = solve_ivp(
            fun=problem.rhs_function,
            y0=problem.initial_state,
            method=self._algo_name,
            t_span=problem.integration_interval,
            **settings_,
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
