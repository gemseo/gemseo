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
"""Scenario result."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post.factory import POST_FACTORY

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.optimization_result import OptimizationResult
    from gemseo.post.base_post import BasePost
    from gemseo.post.base_post_settings import BasePostSettings
    from gemseo.post.factory import PostFactory
    from gemseo.scenarios.mdo import MDOScenario


class ScenarioResult:
    """The result of an [MDOScenario][gemseo.scenarios.mdo.MDOScenario]."""

    _MAIN_PROBLEM_LABEL: Final[str] = "main"
    """The default label for the main problem."""

    optimization_problem_to_result: dict[str, OptimizationResult]
    """The optimization results associated with the different optimization problems."""

    design_variable_name_to_value: dict[str, ndarray]
    """The design variable names bound to the optimal values."""

    __obj_to_be_post_processed: MDOScenario | OptimizationProblem
    """The object to be post-processed."""

    POST_FACTORY: ClassVar[PostFactory] = POST_FACTORY
    """The factory of [BasePost][gemseo.post.base_post.BasePost], if created."""

    def __init__(self, scenario: MDOScenario | str | Path) -> None:
        """
        Args:
            scenario: The scenario to post-process or the path to its HDF5 file.

        Raises:
            ValueError: When the scenario has not yet been executed.
        """  # noqa: D205 D212 D415
        if isinstance(scenario, (str, Path)):
            self.__obj_to_be_post_processed = OptimizationProblem.from_hdf(scenario)
            optimization_result = self.__obj_to_be_post_processed.solution
        else:
            self.__obj_to_be_post_processed = scenario.formulation.problem
            optimization_result = scenario.optimization_result

        if optimization_result is None:
            msg = "A ScenarioResult requires a scenario that has been executed."
            raise ValueError(msg)

        self.design_variable_name_to_value = optimization_result.x_opt_as_dict
        self.optimization_problem_to_result = {
            self._MAIN_PROBLEM_LABEL: optimization_result
        }

    @property
    def optimization_result(self) -> OptimizationResult:
        """The optimization result of the main optimization problem.

        For some scenarios, such as those based on multi-level formulations, there are
        several optimization problems including a main one. The current optimization
        result corresponds to this main optimization problem.

        For scenarios with a single optimization problem, the current optimization
        result corresponds to this unique optimization problem.
        """
        return self.optimization_problem_to_result[self._MAIN_PROBLEM_LABEL]

    def plot(self, settings: BasePostSettings) -> BasePost:
        """Visualize the result.

        Args:
            settings: The post-processor settings.

        Returns:
            The post-processing of the result.
        """
        return self.POST_FACTORY.execute(self.__obj_to_be_post_processed, settings)
