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
"""Base wrapper for all ODE solvers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from gemseo.algos.algorithm_library import AlgorithmDescription
from gemseo.algos.algorithm_library import AlgorithmLibrary

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.algos.ode.ode_problem import ODEProblem

LOGGER = logging.getLogger(__name__)


class ODESolverDescription(AlgorithmDescription):
    """Description for the ODE solver."""


class ODESolverLib(AlgorithmLibrary):
    """Abstract class for libraries of ODE solvers."""

    problem: ODEProblem = None

    def _pre_run(
        self,
        problem: ODEProblem,
        algo_name: str,
        **options: Any,
    ) -> None:
        problem.result.solver_options = options
        problem.result.solver_name = algo_name

    def _post_run(
        self,
        problem: ODEProblem,
        algo_name: str,
        result: NDArray[float],
        **options: Any,
    ) -> None:  # noqa: D107
        if not self.problem.result.is_converged:
            LOGGER.warning(
                "The ODE solver %s did not converge.", self.problem.result.solver_name
            )
