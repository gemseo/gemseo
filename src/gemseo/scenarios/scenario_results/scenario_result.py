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
from typing import Any
from typing import ClassVar
from typing import Final

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.opt_result import OptimizationResult
    from gemseo.core.scenario import Scenario
    from gemseo.post.opt_post_processor import OptPostProcessor


class ScenarioResult:
    """The result of a :class:`.Scenario`."""

    _MAIN_PROBLEM_LABEL: Final[str] = "main"
    """The default label for the main problem."""

    optimization_problems_to_results: dict[str, OptimizationResult]
    """The optimization results associated with the different optimization problems."""

    design_variable_names_to_values: dict[str, ndarray]
    """The design variable names bound to the optimal values."""

    __obj_to_be_post_processed: Scenario | OptimizationProblem
    """The object to be post-processed."""

    _POST_FACTORY: ClassVar[PostFactory | None] = None
    """The factory of :class:`.OptPostProcessor`, if created."""

    def __init__(self, scenario: Scenario | str | Path) -> None:
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
            self.__obj_to_be_post_processed = scenario.formulation.opt_problem
            optimization_result = scenario.optimization_result

        if optimization_result is None:
            raise ValueError(
                "A ScenarioResult requires a scenario that has been executed."
            )

        self.design_variable_names_to_values = optimization_result.x_opt_as_dict
        self.optimization_problems_to_results = {
            self._MAIN_PROBLEM_LABEL: optimization_result
        }

    @classmethod
    @property
    def POST_FACTORY(cls) -> PostFactory:  # noqa: N802
        """The factory of post-processors."""
        if cls._POST_FACTORY is None:
            cls._POST_FACTORY = PostFactory()
        return cls._POST_FACTORY

    @property
    def optimization_result(self) -> OptimizationResult:
        """The optimization result of the main optimization problem.

        For some scenarios, such as those based on multi-level formulations, there are
        several optimization problems including a main one. The current optimization
        result corresponds to this main optimization problem.

        For scenarios with a single optimization problem, the current optimization
        result corresponds to this unique optimization problem.
        """
        return self.optimization_problems_to_results[self._MAIN_PROBLEM_LABEL]

    def plot(self, name: str, **options: Any) -> OptPostProcessor:
        """Visualize the result.

        Args:
            name: The name of the post-processing.
            **options: The options of the post-processing.

        Returns:
            The post-processing of the result.
        """
        return self.POST_FACTORY.execute(
            self.__obj_to_be_post_processed, name, **options
        )
