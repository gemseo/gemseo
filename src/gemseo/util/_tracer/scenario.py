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
"""Tracer for a scenario."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.optimization.problem import OptimizationProblem
from gemseo.util._tracer.base import BaseTracer

if TYPE_CHECKING:
    from gemseo.scenario.evaluation import EvaluationScenario
    from gemseo.util.typing import MutableStrKeyMapping
    from gemseo.util.typing import StrKeyMapping


class ScenarioTracer(BaseTracer):
    """Tracer for recording scenario execution data.

    Records the objective and the optimum, when available, at the end of the
    scenario execution.
    """

    _observed_object: EvaluationScenario

    def _get_end_trace(
        self, call_arguments: StrKeyMapping, returned_data: Any
    ) -> MutableStrKeyMapping:  # noqa: D102
        trace = super()._get_end_trace(call_arguments, returned_data)
        problem = self._observed_object.formulation.problem
        if not isinstance(problem, OptimizationProblem) or problem.objective is None:
            # A plain `EvaluationScenario` (e.g. as built by
            # `gemseo.sample_disciplines`) has an `EvaluationProblem`, which
            # has no notion of objective nor optimum; only an
            # `OptimizationProblem` (used by `MDOScenario`) defines them, and
            # only once an objective has been set, since both
            # `OptimizationProblem.objective_name` and
            # `OptimizationProblem.optimum` read the objective function.
            return trace
        trace["objective"] = problem.objective_name
        try:
            optimum = problem.optimum
            iteration = problem.database.get_iteration(optimum.design)
        except ValueError:
            # The execution failed before any complete evaluation, e.g. the
            # database is empty.
            return trace
        except KeyError:
            # `OptimizationHistory.optimum` returns an empty design vector when
            # no feasible point carries the value of the objective, and
            # `Database.get_iteration` does not hold such a vector.
            return trace
        trace["optimum"] = {"objective": optimum.objective, "iteration": iteration}
        return trace
