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

import logging
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import arange
from numpy import column_stack
from numpy import concatenate
from numpy import full
from numpy import int64
from numpy import ndarray
from numpy import tile

from gemseo.core.problem.database import Database
from gemseo.dataset.dataset import Dataset
from gemseo.optimization.result import OptimizationResult
from gemseo.scenario.scenario_result.scenario_result import ScenarioResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo.core.discipline import Discipline
    from gemseo.optimization.problem import OptimizationProblem
    from gemseo.scenario.adapter.mdo import MDOScenarioAdapter
    from gemseo.scenario.mdo import MDOScenario
    from gemseo.util.typing import StrPath

logger = logging.getLogger(__name__)


class BiLevelScenarioResult(ScenarioResult):
    """The result of an [MDOScenario][gemseo.scenario.mdo.MDOScenario] using a [BiLevel][gemseo.formulation.bilevel.BiLevel] formulation."""  # noqa: E501

    __sub_label_formatter: Final[str] = "sub_{}"
    """The formatter to get the name of the key of a sub-problem from its index.

    To be used as `__sub_label_formatter.format(i)` where `i` is an integer.
    """

    __n_sub_problems: int
    """The number of sub-optimization problems."""

    __scenario_adapters: list[MDOScenarioAdapter | Discipline]
    """The adapters, and possibly disciplines, of the sub-scenarios."""

    __n_scenario_adapters: int
    """The number of adapters of sub-scenarios.

    They come first in the list of adapters,
    followed by the disciplines treated as sub-scenarios, if any.
    """

    executions_group: ClassVar[str] = "executions"
    """The group storing the sub-scenario execution and iteration numbers."""

    upper_level_designs_group: ClassVar[str] = "upper_level_designs"
    """The name of the group storing the upper-level values passed to the sub-scenario."""  # noqa: E501

    __execution_variable: Final[str] = "execution"
    """The name of the variable storing the sub-scenario execution number."""

    __sub_iteration_variable: Final[str] = "sub_iteration"
    """The name of the variable storing the sub-scenario iteration number."""

    def __init__(self, scenario: MDOScenario | StrPath) -> None:
        """
        Args:
            scenario: The scenario to post-process or the path to its HDF5 file.

        Raises:
            ValueError: If the scenario has not yet been executed,
                or if the HDF5 file of a sub-scenario execution
                that is needed to build the sub-optimization results
                was overwritten after this execution.
        """  # noqa: D205 D212 D415
        super().__init__(scenario)
        formulation = scenario.formulation
        scenario_adapters = formulation.scenario_adapters
        main_problem = formulation.problem
        self.__scenario_adapters = scenario_adapters
        y_opt = main_problem.database[main_problem.solution.x_opt]
        # The adapters of the sub-scenarios come first,
        # followed by the disciplines treated as sub-scenarios, which have no scenario.
        self.__n_scenario_adapters = len(formulation.sub_scenario_execution_indices)
        optimal_local_design_values = {
            variable_name: y_opt[variable_name]
            for scenario_adapter in scenario_adapters[: self.__n_scenario_adapters]
            for variable_name in scenario_adapter.scenario.design_space.variables
        }
        self.design_variable_name_to_value.update(optimal_local_design_values)
        self.__n_sub_problems = len(scenario_adapters)
        x_opt = Database.get_hashable_ndarray(main_problem.solution.x_opt)
        for index, execution_indices in enumerate(
            formulation.sub_scenario_execution_indices
        ):
            execution_index = execution_indices.get(x_opt)
            if execution_index is None:
                continue

            scenario_adapter = scenario_adapters[index]
            sub_problem = scenario_adapter.scenario.formulation.problem
            database = self.__get_database(
                scenario_adapter, execution_index, sub_problem.database.name
            )
            if database is None:
                continue

            with self.__use_database(sub_problem, database):
                result = OptimizationResult.from_optimization_problem(sub_problem)

            # The number of objective calls of the result is read from the counter
            # of the objective, which is the one of the last execution.
            result = replace(
                result,
                n_obj_call=self.__count_objective_evaluations(sub_problem, database),
            )
            label = self.__sub_label_formatter.format(index)
            self.optimization_problem_to_result[label] = result

    @staticmethod
    def __count_objective_evaluations(
        sub_problem: OptimizationProblem, database: Database
    ) -> int:
        """Count the evaluations of the objective stored in a database.

        Args:
            sub_problem: The sub-optimization problem.
            database: The database of an execution of the sub-problem.

        Returns:
            The number of entries of the database
            that hold a value of the objective.
        """
        name = sub_problem.objective.name
        return sum(outputs.get(name) is not None for outputs in database.values())

    @staticmethod
    @contextmanager
    def __use_database(
        sub_problem: OptimizationProblem, database: Database
    ) -> Iterator[None]:
        """Temporarily replace the database of a sub-optimization problem.

        Args:
            sub_problem: The sub-optimization problem.
            database: The database to substitute for the current one.

        Yields:
            Nothing;
            `sub_problem.database` is `database` for the duration of the `with` block,
            and is restored to its original value afterward,
            even if an exception is raised.
        """
        original_database = sub_problem.database
        sub_problem.database = database
        try:
            yield
        finally:
            sub_problem.database = original_database

    @staticmethod
    def __get_database(
        scenario_adapter: MDOScenarioAdapter,
        execution_index: int,
        name: str,
    ) -> Database | None:
        """Return the database of an execution of a sub-scenario adapter.

        Args:
            scenario_adapter: The adapter of the sub-scenario.
            execution_index: The index of the execution.
            name: The name to give the database loaded from an HDF5 file.

        Returns:
            The in-memory database of the execution
            when the optimization histories were kept in memory,
            otherwise the database loaded from the HDF5 file exported after it,
            or `None` when this file no longer exists.

        Raises:
            ValueError: If the HDF5 file was overwritten after the execution,
                e.g. by another sub-scenario exporting to the same path.
                The detection relies on the modification time of the file,
                so an overwrite made within the timestamp resolution
                of the filesystem is not detected.
        """
        if scenario_adapter.databases:
            return scenario_adapter.databases[execution_index]

        path = scenario_adapter.database_file_paths[execution_index]
        if not path.exists():
            logger.warning(
                "The HDF5 file %s does not exist anymore; "
                "its optimization history is no longer available.",
                path,
            )
            return None

        if (
            path.stat().st_mtime_ns
            != scenario_adapter.database_file_mtimes[execution_index]
        ):
            msg = (
                f"The HDF5 file {path} was overwritten after the execution; "
                "its optimization history is no longer available."
            )
            raise ValueError(msg)

        return Database.from_hdf(path, name=name, log=False)

    def __check_index(self, index: int) -> None:
        """Check the index of a sub-optimization problem.

        Args:
            index: The index of the sub-optimization problem.

        Raises:
            ValueError: If the index is negative or greater than N-1,
                where N is the number of sub-optimization problems.
        """
        max_index = self.__n_sub_problems - 1
        if not 0 <= index <= max_index:
            msg = (
                f"The index ({index}) of a sub-scenario "
                f"must be between 0 and {max_index}."
            )
            raise ValueError(msg)

    def get_top_optimization_result(self) -> OptimizationResult:
        """Return the optimization result of the top-level optimization problem."""
        return self.optimization_result

    def get_sub_optimization_result(self, index: int) -> OptimizationResult | None:
        """Return the optimization result of a sub-optimization problem if any.

        This result is the one of the last execution of the sub-scenario
        before the system-level optimum was stored in the system-level database,
        i.e. the execution that produced the system-level outputs at the optimum.
        It is unavailable, and so `None` is returned,
        when the optimization history of the sub-scenario
        was neither kept in memory nor saved to disk,
        or when the HDF5 file saved for it no longer exists,
        or when the sub-scenario was executed in separate processes,
        or when the sub-scenario is a discipline,
        or when the adapter of the sub-scenario uses a cache
        that can hold several entries.

        Args:
            index: The index of the sub-optimization problem,
                between 0 and N-1 where N is the number of sub-optimization problems.

        Returns:
            The optimization result of a sub-optimization problem, if any.

        Raises:
            ValueError: If the index is negative or greater than N-1.
        """
        self.__check_index(index)
        return self.optimization_problem_to_result.get(
            self.__sub_label_formatter.format(index)
        )

    def get_sub_scenario_history_dataset(self, index: int) -> Dataset:
        """Return the combined optimization history of a sub-scenario.

        The returned dataset stacks, for every execution of the sub-scenario
        whose optimization history was retained,
        one row per sub-scenario iteration,
        tagged with the execution number,
        the sub-scenario iteration number
        and the upper-level values passed to the sub-scenario for that execution.

        As this dataset stacks several independent optimization histories,
        it is a plain [Dataset][gemseo.dataset.dataset.Dataset]
        carrying no optimization metadata,
        and so is not meant to be passed to
        [execute_post][gemseo.execute_post].

        Args:
            index: The index of the sub-optimization problem,
                between 0 and N-1 where N is the number of sub-optimization problems.

        Returns:
            The combined optimization history of the sub-scenario.

        Note:
            When the histories are read from HDF5 files,
            an overwrite of a file is detected through its modification time.
            An overwrite made within the timestamp resolution of the filesystem
            is not detected
            and the history of another run may then be returned without error.
            Give each run its own export directory to avoid this.

        Raises:
            ValueError: If the index is negative or greater than N-1,
                or if the sub-scenario is a discipline,
                or if the sub-scenario has no retained optimization history,
                or if the HDF5 file of an execution no longer exists
                or was overwritten after this execution,
                or if an upper-level value passed to the sub-scenario
                is not a NumPy array.
        """
        self.__check_index(index)
        if index >= self.__n_scenario_adapters:
            msg = (
                f"The sub-scenario at index {index} is a discipline "
                "and has no optimization history."
            )
            raise ValueError(msg)

        scenario_adapter = self.__scenario_adapters[index]
        sub_problem = scenario_adapter.scenario.formulation.problem
        input_data_history = scenario_adapter.input_data_history
        if not input_data_history:
            if scenario_adapter.keep_databases or scenario_adapter.save_databases:
                msg = (
                    "No sub-scenario history is available; "
                    "the sub-scenario histories are not collected "
                    "when the sub-scenarios are executed in separate processes."
                )
                raise ValueError(msg)

            msg = (
                "No sub-scenario history is available; "
                "enable keep_opt_history or save_opt_history "
                "on the BiLevel formulation."
            )
            raise ValueError(msg)

        iteration_datasets = []
        for execution_index in range(len(input_data_history)):
            database = self.__get_database(
                scenario_adapter, execution_index, sub_problem.database.name
            )
            if database is None:
                msg = (
                    f"The HDF5 file of the execution {execution_index + 1} "
                    "of the sub-scenario no longer exists; "
                    "its optimization history is no longer available."
                )
                raise ValueError(msg)

            with self.__use_database(sub_problem, database):
                iteration_datasets.append(sub_problem.to_dataset(opt_naming=True))

        # The order of the adapter input names is not deterministic,
        # hence the sorting to get a deterministic layout of the group.
        variable_names = sorted(input_data_history[0])
        # The upper-level values become columns of the dataset,
        # and so must be numeric.
        for name in variable_names:
            value = input_data_history[0][name]
            if not isinstance(value, ndarray):
                msg = (
                    f"The upper-level value of the variable {name!r} "
                    f"is of type {type(value).__name__} instead of a NumPy array; "
                    "a non-numeric upper-level value "
                    "cannot be used to label the sub-scenario history."
                )
                # ValueError, and not TypeError,
                # for consistency with the other errors raised by this method.
                raise ValueError(msg)  # noqa: TRY004

        variable_sizes = {
            name: input_data_history[0][name].size for name in variable_names
        }
        n_sub_iterations = [len(dataset) for dataset in iteration_datasets]
        # The dataset stacks several independent optimization histories
        # and so is not an optimization history.
        dataset = Dataset.concatenate(iteration_datasets)
        dataset.add_group(
            self.executions_group,
            column_stack([
                concatenate([
                    full(size, i + 1, dtype=int64)
                    for i, size in enumerate(n_sub_iterations)
                ]),
                concatenate([
                    arange(1, size + 1, dtype=int64) for size in n_sub_iterations
                ]),
            ]),
            variable_names=[
                self.__execution_variable,
                self.__sub_iteration_variable,
            ],
        )
        if not variable_names:
            return dataset

        dataset.add_group(
            self.upper_level_designs_group,
            concatenate(
                [
                    tile(
                        concatenate([input_data[name].real for name in variable_names]),
                        (size, 1),
                    )
                    for input_data, size in zip(
                        input_data_history, n_sub_iterations, strict=True
                    )
                ],
                axis=0,
            ),
            variable_names=variable_names,
            variable_name_to_n_components=variable_sizes,
        )
        return dataset
