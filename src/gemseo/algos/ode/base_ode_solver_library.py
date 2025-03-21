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
"""Base class for libraries of ODE solvers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from gemseo.algos.base_algorithm_library import AlgorithmDescription
from gemseo.algos.base_algorithm_library import BaseAlgorithmLibrary
from gemseo.algos.ode.base_ode_solver_settings import BaseODESolverSettings

if TYPE_CHECKING:
    from gemseo.algos.ode.ode_problem import ODEProblem
    from gemseo.algos.ode.ode_result import ODEResult

LOGGER = logging.getLogger(__name__)


class ODESolverDescription(AlgorithmDescription):
    """Description for the ODE solver."""

    Settings: type[BaseODESolverSettings] = BaseODESolverSettings
    """The settings validation model."""


class BaseODESolverLibrary(BaseAlgorithmLibrary):
    """Base class for libraries of ODE solvers."""

    def _pre_run(
        self,
        problem: ODEProblem,
        **settings: Any,
    ) -> None:
        problem.result.solver_options = settings
        problem.result.solver_name = self._algo_name

    def _post_run(
        self,
        problem: ODEProblem,
        result: ODEResult,
        **settings: Any,
    ) -> None:  # noqa: D107
        if not problem.result.algorithm_has_converged:
            LOGGER.warning(
                "The ODE solver %s did not converge.", problem.result.solver_name
            )
