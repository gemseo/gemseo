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
"""BiLevel scenario result."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.algos.opt_result import OptimizationResult
from gemseo.scenarios.scenario_results.scenario_result import ScenarioResult

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.core.scenario import Scenario


class BiLevelScenarioResult(ScenarioResult):
    """The result of a :class:`.Scenario` using a :class:`.BiLevel` formulation."""

    __SUB_LABEL_FORMATTER: Final[str] = "sub_{}"
    """The formatter to get the name of the key of a sub-problem from its index.

    To be used as ``__SUB_LABEL_FORMATTER.format(i)`` where ``i`` is an integer.
    """

    __n_sub_problems: int
    """The number of sub-optimization problems."""

    def __init__(self, scenario: Scenario | str | Path) -> None:  # noqa: D107
        super().__init__(scenario)
        formulation = scenario.formulation
        main_problem = formulation.opt_problem
        x_shared_opt = main_problem.solution.x_opt
        i_opt = main_problem.database.get_iteration(x_shared_opt) - 1
        scenario_adapters = formulation.scenario_adapters
        self.__n_sub_problems = len(scenario_adapters)
        for index, scenario_adapter in enumerate(scenario_adapters):
            sub_problem = scenario_adapter.scenario.formulation.opt_problem
            database = sub_problem.database
            sub_problem.database = scenario_adapter.databases[i_opt]
            result = OptimizationResult.from_optimization_problem(sub_problem)
            x_local_opt = result.x_opt
            label = self.__SUB_LABEL_FORMATTER.format(index)
            self.optimization_problems_to_results[label] = result
            self.design_variable_names_to_values.update(
                sub_problem.design_space.array_to_dict(x_local_opt)
            )
            sub_problem.database = database

    def get_top_optimization_result(self) -> OptimizationResult:
        """Return the optimization result of the top-level optimization problem."""
        return self.optimization_result

    def get_sub_optimization_result(self, index: int) -> OptimizationResult | None:
        """Return the optimization result of a sub-optimization problem if any.

        Args:
            index: The index of the sub-optimization problem,
                between 0 and N-1 where N is the number of sub-optimization problems.

        Returns:
            The optimization result of a sub-optimization problem.

        Raises:
            ValueError: If the index is greater than N-1.
        """
        max_index = self.__n_sub_problems - 1
        if index > max_index:
            raise ValueError(
                f"The index ({index}) of a sub-optimization result "
                f"must be between 0 and {max_index}."
            )
        return self.optimization_problems_to_results[
            self.__SUB_LABEL_FORMATTER.format(index)
        ]
