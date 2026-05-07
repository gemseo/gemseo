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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Pierre-Jean Barjhoux, Benoit Pauwels - MDOScenarioAdapter
#                                                        Jacobian computation
"""A scenario to solve an MDO problem, using an optimizer or a DOE algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.algos.driver_library import DriverLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.core.functions.concatenate import Concatenate
from gemseo.post import OptHistoryView_Settings
from gemseo.post.factory import POST_FACTORY
from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.scenarios.scenario_results.factory import ScenarioResultFactory
from gemseo.utils.string_tools import convert_strings_to_iterable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from pathlib import Path

    from numpy import ndarray

    from gemseo.algos.base_driver_settings import BaseDriverSettings
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_result import OptimizationResult
    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.datasets.dataset import Dataset
    from gemseo.formulations.base_settings import BaseFormulationSettings
    from gemseo.post.base_post import BasePost
    from gemseo.post.base_post_settings import BasePostSettings
    from gemseo.post.factory import PostFactory
    from gemseo.scenarios.scenario_results.scenario_result import ScenarioResult


class MDOScenario(EvaluationScenario):
    """A scenario to solve an MDO problem, using an optimizer or a DOE algorithm.

    The outputs of interest can be declared as objectives, constraints or observables
    using the method
    [add_objective()][gemseo.scenarios.mdo.MDOScenario.add_objective],
    [add_constraint()][gemseo.scenarios.mdo.MDOScenario.add_constraint]
    or [add_observable()][gemseo.scenarios.mdo.MDOScenario.add_observable] respectively.
    These objective, constraints and observables are attached to
    an [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem],
    built over the [DesignSpace][gemseo.algos.design_space.DesignSpace],
    that is passed at instantiation.

    Then,
    the [MDOScenario.execute()][gemseo.scenarios.mdo.MDOScenario.execute] method
    takes a driver
    (see
    [BaseDriverLibrary][gemseo.algos.base_driver_library.BaseDriverLibrary]
    )
    with options as input data
    and uses it to solve the optimization problem.
    This driver is in charge of executing the multidisciplinary process.

    To view the results,
    use the
    [MDOScenario.post_process()][gemseo.scenarios.mdo.MDOScenario.post_process]
    method after execution
    with one of the available post-processors
    that can be listed by
    [MDOScenario.posts][gemseo.scenarios.mdo.MDOScenario.posts].
    """

    _ALGO_FACTORY_CLASS: ClassVar[type[DriverLibraryFactory]] = DriverLibraryFactory

    _evaluation_problem_class: type[OptimizationProblem] = OptimizationProblem

    __objectives_to_minimize: dict[ArrayFunction, bool]
    """The objectives and if it must be minimized."""

    post_factory: ClassVar[PostFactory] = POST_FACTORY
    """The factory of post-processors."""

    posts: ClassVar[list[str]] = post_factory.class_names
    """The names of the post-processors."""

    ConstraintType = ArrayFunction.ConstraintType

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[BaseDiscipline],
        design_space: DesignSpace,
        name: str = "",
        formulation_settings: BaseFormulationSettings | None = None,
    ) -> None:
        super().__init__(
            disciplines,
            design_space,
            name=name,
            formulation_settings=formulation_settings,
        )
        self.__objectives_to_minimize = {}

    def _add_extra_constraint(self, constraint: ArrayFunction) -> None:
        self.formulation.problem.add_constraint(constraint)

    def add_objective(
        self,
        output_names: str | Iterable[str],
        minimize: bool = True,
        objective_name: str = "",
    ) -> None:
        """Set the objective.

        Args:
            output_names: The names of the outputs used as objective.
                If multiple names are passed,
                the objective will be a vector
                and a single top-level discipline must provide all outputs.
            minimize: Whether to minimize the objective.
            objective_name: The name of the objective to be stored.
                If empty, the name is generated from `output_names` and `minimize`.
        """
        problem = self.formulation.problem
        output_names = convert_strings_to_iterable(output_names)
        objective = self.formulation.create_objective(
            output_names, objective_name=objective_name
        )

        if problem.objective is None:
            # There is a single objective.
            problem.objective = objective
            # When minimize is True,
            # the following does "problem.objective = -problem.objective"
            # because in GEMSEO, the optimizers expect an objective to minimize.
            problem.minimize_objective = minimize
            self.__objectives_to_minimize[problem.objective] = minimize
        else:
            # There are multiple objectives.
            if set(self.__objectives_to_minimize.values()) | {minimize} == {
                False,
                True,
            }:
                # There are objectives to minimize AND objectives to maximize.
                # Arbitrarily,
                # we decide to write the optimization problem as a minimization problem.
                problem.minimize_objective = True

            if not minimize:
                # We transform the current objective to maximize
                # into an objective to minimize.
                objective = -objective

            self.__objectives_to_minimize[objective] = minimize

            problem.objective = Concatenate(self.__objectives_to_minimize)

    @property
    def use_standardized_objective(self) -> bool:
        """Whether to use the standardized objective for logging and post-processing.

        The standardized objective corresponds to
        the original objective expressed as an objective to minimize.
        In other words,
        $f$ (resp. $-f$) is the standardized objective
        associated with the objective $f$ to minimize (resp. maximize).
        Drivers and databases handle standardized objectives
        but for convenience,
        it may be more relevant
        to log the expression and the values of the original objective.
        """
        return self.formulation.problem.use_standardized_objective

    @use_standardized_objective.setter
    def use_standardized_objective(self, value: bool) -> None:
        self.formulation.problem.use_standardized_objective = value

    def add_constraint(
        self,
        output_names: str | Iterable[str],
        constraint_type: ConstraintType = ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Add an equality or inequality constraint to the optimization problem.

        An equality constraint is written as $c(x)=a$,
        a positive inequality constraint is written as $c(x)\geq a$
        and a negative inequality constraint is written as $c(x)\leq a$.

        This constraint is in addition to those created by the formulation,
        e.g. consistency constraints in IDF.

        The strategy of repartition of the constraints is defined by the formulation.

        Args:
            output_names: The name(s) of the outputs computed by $c(x)$.
                If multiple names are passed,
                the constraint will be a vector
                and a single top-level discipline must provide all outputs.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name of the constraint is generated
                from `output_name`, `constraint_type`, `value` and `positive`.
            value: The value $a$ parameterizing the constraint.
            positive: Whether the inequality constraint is positive.
            **kwargs: Additional arguments specific to the MDO formulation.
        """
        output_names = convert_strings_to_iterable(output_names)
        constraint = self.formulation.create_constraint(
            output_names,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **kwargs,
        )
        if constraint is None:
            # The constraint is not added to this scenario but managed internally.
            # For example, the Bilevel formulation can add it to sub-scenarios.
            return

        self.formulation.problem._check_function_name(constraint)
        self.formulation.problem.add_constraint(constraint)

    # TODO: API: remove and use scenario.design_space.variable_names instead, or rename.
    def get_optim_variable_names(self) -> list[str]:
        """A convenience function to access the optimization variables.

        Returns:
            The optimization variables of the scenario.
        """
        return self.formulation.problem.design_space.variable_names

    def set_backup_settings(
        self,
        file_path: str | Path,
        at_each_iteration: bool = False,
        at_each_function_call: bool = True,
        erase: bool = False,
        load: bool = False,
        plot: bool = False,
    ) -> None:
        """
        Args:
            plot: Whether to plot the optimization history view at each iteration.
                The plots will be generated only after the first two iterations.
        """  # noqa: D205, D212
        super().set_backup_settings(
            file_path,
            at_each_iteration=at_each_iteration,
            at_each_function_call=at_each_function_call,
            erase=erase,
            load=load,
        )
        if plot:
            self.formulation.problem.add_listener(
                self._execute_plot_callback,
                at_each_iteration=True,
                at_each_function_call=False,
            )

    def _execute_plot_callback(self, x_vect: ndarray) -> None:
        """A callback function to plot the OptHistoryView of the current database.

        Args:
            x_vect: The input value.
        """
        if len(self.formulation.problem.database) > 2:
            self.post_process(
                OptHistoryView_Settings(
                    save=True, show=False, file_path=self._backup_file_path.stem
                )
            )

    def post_process(self, settings: BasePostSettings) -> BasePost:
        """Post-process the optimization history.

        Args:
            settings: The post-processor settings.

        Returns:
            The post-processor.
        """
        return self.post_factory.execute(self.formulation.problem, settings=settings)

    @property
    def optimization_result(self) -> OptimizationResult | None:
        """The optimization result of the last execution."""
        return self._execution_result

    @optimization_result.setter
    def optimization_result(self, value: OptimizationResult) -> None:
        self._execution_result = value

    def to_dataset(
        self,
        name: str = "",
        categorize: bool = True,
        opt_naming: bool = True,
        export_gradients: bool = False,
    ) -> Dataset:
        """
        Args:
            opt_naming: Whether to use
                [DESIGN_GROUP][gemseo.datasets.optimization_dataset.OptimizationDataset.DESIGN_GROUP]
                and
                [FUNCTION_GROUP][gemseo.datasets.optimization_dataset.OptimizationDataset.FUNCTION_GROUP]
                as groups.
                Otherwise,
                [INPUT_GROUP][gemseo.datasets.io_dataset.IODataset.INPUT_GROUP]
                and
                [OUTPUT_GROUP][gemseo.datasets.io_dataset.IODataset.OUTPUT_GROUP].
        """  # noqa: D205, D212
        return self.formulation.problem.to_dataset(
            name=name,
            categorize=categorize,
            opt_naming=opt_naming,
            export_gradients=export_gradients,
            input_values=self._get_input_values(),
        )

    def get_result(self, name: str = "", **options: Any) -> ScenarioResult | None:
        """Return the scenario result.

        This result differs from the optimization result
        returned by [execute()][gemseo.scenarios.mdo.MDOScenario.execute].
        It is a post-processed version of the latter,
        which can be retrieved using its `optimization_result` attribute.
        For the `BiLevel` formulation,
        its `get_sub_optimization_result` method returns the optimization result
        associated with a sub-scenario.

        Args:
            name: The class name of the
                [ScenarioResult][gemseo.scenarios.scenario_results.scenario_result.ScenarioResult].
                If empty, use the default class associated with the formulation.
            **options: The options of the
                [ScenarioResult][gemseo.scenarios.scenario_results.scenario_result.ScenarioResult].

        Returns:
            The scenario result.
        """
        if self._execution_result is None:
            return None

        return ScenarioResultFactory().create(
            name or self.formulation.DEFAULT_SCENARIO_RESULT_CLASS_NAME,
            scenario=self,
            **options,
        )

    def execute(
        self,
        algorithm_settings: BaseDriverSettings | None = None,
    ) -> OptimizationResult:
        """
        Returns:
            The optimization result.
            This result is part of the scenario result,
            accessible via [get_result()][gemseo.scenarios.mdo.MDOScenario.get_result].
        """  # noqa: D205, D212
        return super().execute(algorithm_settings=algorithm_settings)
