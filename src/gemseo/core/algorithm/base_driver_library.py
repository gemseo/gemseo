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
#       :author: Damien Guenot - 26 avr. 2016
#       :author: Francois Gallard, refactoring
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for libraries of drivers.

A driver is an algorithm evaluating the functions
of an [EvaluationProblem][gemseo.core.problem.evaluation.EvaluationProblem]
at different points of the input space,
using the
[execute()][gemseo.core.algorithm.base_driver_library.BaseDriverLibrary.execute]
method.
In the case
of an [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem],
this method also returns
an [OptimizationResult][gemseo.optimization.result.OptimizationResult].

There are two main families of drivers:
the optimizers with the base class
[BaseOptimizationLibrary][gemseo.optimization.core.base_optimization_library.BaseOptimizationLibrary]
and the design of experiments (DOE) with the base class
[BaseDOELibrary][gemseo.doe.core.base_doe_library.BaseDOELibrary].
"""

from __future__ import annotations

import dataclasses
import logging
from abc import abstractmethod
from collections.abc import Generator
from collections.abc import Iterable
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from numpy import array_equal
from numpy import ndarray

from gemseo.core.algorithm._progress_bar.custom import logger as tqdm_logger
from gemseo.core.algorithm._progress_bar.standard import ProgressBar
from gemseo.core.algorithm._progress_bar.unsuffixed import UnsuffixedProgressBar
from gemseo.core.algorithm._unsuitability_reason import _UnsuitabilityReason
from gemseo.core.algorithm.base_algorithm_library import AlgorithmDescription
from gemseo.core.algorithm.base_algorithm_library import BaseAlgorithmLibrary
from gemseo.core.algorithm.base_driver_settings import BaseDriverSettings
from gemseo.core.algorithm.progress_bar_data.data import ProgressBarData
from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.core.problem.termination_criterion import MaxIterReachedException
from gemseo.core.problem.termination_criterion import MaxTimeReached
from gemseo.core.problem.termination_criterion import TerminationCriterion
from gemseo.enum._variable_type import VariableType
from gemseo.space.transformation._working import get_default_transformation_factories
from gemseo.space.transformation._working import project_onto_declared_domain
from gemseo.space.transformation.composition import SpaceComposition
from gemseo.util._workflow_observer.injector import WorkflowObserverMeta
from gemseo.util.constant import _enable_progress_bar
from gemseo.util.derivative.approximation_mode import ApproximationMode
from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.logging import OneLineLogging
from gemseo.util.pydantic import create_model
from gemseo.util.string import MultiLineString
from gemseo.util.string import _convert_camel_case_to_lower_case_words
from gemseo.util.typing import StrKeyMapping

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Set as AbstractSet

    from gemseo.core.algorithm._progress_bar.base import BaseProgressBar
    from gemseo.core.algorithm.progress_bar_data.factory import ProgressBarDataName
    from gemseo.optimization.problem import OptimizationProblem
    from gemseo.optimization.result import OptimizationResult
    from gemseo.space.base import BaseVariableSpace
    from gemseo.space.design import DesignSpace
    from gemseo.space.transformation.base import BaseSpaceTransformation

DriverSettingType = (
    str
    | float
    | int
    | bool
    | list[str]
    | ndarray
    | Iterable[CallbackType]
    | StrKeyMapping
)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseDriverSettings)
_SpaceT = TypeVar("_SpaceT", bound="BaseVariableSpace")


@dataclass
class DriverDescription(AlgorithmDescription):
    """The description of a driver."""

    handle_catalog_variables: bool = False
    """Whether the driver handles catalog variables."""
    handle_categorical_variables: bool = False
    """Whether the driver handles categorical variables."""

    handle_discrete_variables: bool = False
    """Whether the driver handles discrete variables."""

    handle_integer_variables: bool = False
    """Whether the driver handles integer variables."""

    settings_class: type[BaseDriverSettings] = BaseDriverSettings
    """The Pydantic model for the driver library settings."""


class BaseDriverLibrary(
    BaseAlgorithmLibrary[T], Generic[T, _SpaceT], metaclass=WorkflowObserverMeta
):
    """Base class for libraries of drivers.

    The type of the input space of the problems handled by the library
    is the second type parameter of this class:
    a family of drivers requiring a specific kind of space,
    e.g. an optimization library requiring a design space,
    passes it to its base.
    """

    ApproximationMode = ApproximationMode

    DifferentiationMethod = EvaluationProblem.DifferentiationMethod

    ALGORITHM_INFOS: ClassVar[dict[str, DriverDescription]] = {}
    """The description of the algorithms contained in the library."""

    _result_class: ClassVar[type[OptimizationResult] | None] = None
    """The class used to present the result of the optimization.

    Set by the driver families whose drivers can solve an optimization problem.
    """

    _support_sparse_jacobian: ClassVar[bool] = False
    """Whether the library support sparse Jacobians."""

    enable_progress_bar: bool = _enable_progress_bar
    """Whether to enable the progress bar in the evaluation log.

    Driven by the `enable_progress_bar` option of the global configuration.
    """

    _problem: EvaluationProblem[_SpaceT] | None
    """The optimization problem the driver library is bonded to."""

    _progress_bar: BaseProgressBar | None
    """The progress bar used during the execution, if any."""

    __start_time: float
    """The time at which the execution begins."""

    __transformed_kinds: frozenset[VariableType] | None
    """The kinds of variables the working transformation of the run takes care of.

    These are the kinds of the variables it relaxes or encodes into other ones.
    It is `None` when there is no working transformation to read them from,
    and the relaxation settings are read instead.
    """

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self._progress_bar = None
        self.__start_time = 0.0
        self.__transformed_kinds = None

    @classmethod
    def _get_unsuitability_reason(
        cls, algorithm_description: DriverDescription, problem: EvaluationProblem
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if reason:
            return reason

        if not problem.input_space:
            return _UnsuitabilityReason.EMPTY_VARIABLE_SPACE

        variables = problem.input_space.variables
        # A relaxation setting is opt-in and a description knows no settings,
        # so an algorithm is suited to a kind of variable
        # only when it handles that kind natively.
        if (
            variables.has_variables_of_type(VariableType.INTEGER)
            and not algorithm_description.handle_integer_variables
        ):
            return _UnsuitabilityReason.INTEGER_VARIABLES

        if (
            variables.has_variables_of_type(VariableType.DISCRETE)
            and not algorithm_description.handle_discrete_variables
        ):
            return _UnsuitabilityReason.DISCRETE_VARIABLES

        if (
            variables.has_variables_of_type(VariableType.CATALOG)
            and not algorithm_description.handle_catalog_variables
        ):
            return _UnsuitabilityReason.CATALOG_VARIABLES

        if (
            variables.has_variables_of_type(VariableType.CATEGORICAL)
            and not algorithm_description.handle_categorical_variables
        ):
            return _UnsuitabilityReason.CATEGORICAL_VARIABLES

        return _UnsuitabilityReason.NO_REASON

    def _get_algorithm_description_to_check(self) -> DriverDescription:
        description = super()._get_algorithm_description_to_check()
        if self._settings is None:
            return description

        # A kind the working transformation relaxes or encodes is handled.
        # Without a transformation, e.g. for a meta-driver,
        # the relaxation settings decide instead,
        # and the catalog and categorical kinds are not handled.
        transformed_kinds = self.__transformed_kinds
        if transformed_kinds is None:
            relax_integer_variables = self._settings.relax_integer_variables
            relax_discrete_variables = self._settings.relax_discrete_variables
            encode_categorical = False
            encode_catalog = False
        else:
            relax_integer_variables = VariableType.INTEGER in transformed_kinds
            relax_discrete_variables = VariableType.DISCRETE in transformed_kinds
            encode_categorical = VariableType.CATEGORICAL in transformed_kinds
            encode_catalog = VariableType.CATALOG in transformed_kinds

        return dataclasses.replace(
            description,
            handle_integer_variables=(
                description.handle_integer_variables or relax_integer_variables
            ),
            handle_discrete_variables=(
                description.handle_discrete_variables or relax_discrete_variables
            ),
            handle_categorical_variables=(
                description.handle_categorical_variables or encode_categorical
            ),
            handle_catalog_variables=(
                description.handle_catalog_variables or encode_catalog
            ),
        )

    def _init_iter_observer(
        self,
        problem: EvaluationProblem[_SpaceT],
        max_iter: int,
        message: str = "",
        progress_bar_data_name: ProgressBarDataName = ProgressBarData.__name__,
    ) -> None:
        """Initialize the iteration observer.

        It will handle the termination criteria and the update of the progress bar.

        Args:
            problem: The evaluation problem.
            max_iter: The maximum number of iterations.
            message: The message to display at the beginning of the progress bar status.
            progress_bar_data_name: The name of a
                [BaseProgressBarData][
                gemseo.core.algorithm.progress_bar_data.base.BaseProgressBarData]
                class
                to define the data of an optimization problem
                to be displayed in the progress bar.
        """
        from gemseo.util.global_configuration import _configuration

        problem.evaluation_counter.maximum = max_iter
        if self._settings.reset_iteration_counters:
            problem.evaluation_counter.current = 0

        if self.enable_progress_bar and _configuration.logging.enable:
            cls = ProgressBar if self._settings.log_problem else UnsuffixedProgressBar
            self._progress_bar = cls(max_iter, problem, message, progress_bar_data_name)
        else:
            self._progress_bar = None

        self.__start_time = time()

    def _finalize_previous_iteration_using_database(self) -> None:
        """Finalize the previous iteration using the database."""
        # This is the start of the current iteration.
        counter = self._problem.evaluation_counter
        if not counter.enabled:
            counter.enabled = True
            self._check_stopping_criteria()
            return

        self._finalize_previous_iteration()
        self._check_stopping_criteria()

    def _check_stopping_criteria(self) -> None:
        """Check the termination criteria at the current iteration.

        Raises:
            MaxTimeReached: If the elapsed time is greater
                than the maximum execution time.
            MaxTimeReached If the maximum number of evaluations is reached.
        """
        t = time()
        if 0 < self._settings.max_time < t - self.__start_time:
            msg = f"Maximum time reached: {self._settings.max_time} seconds. "
            raise MaxTimeReached(msg)

        if self._problem.evaluation_counter.maximum_is_reached:
            raise MaxIterReachedException

    def _post_run(
        self,
        problem: OptimizationProblem,
        result: OptimizationResult,
        max_input_space_dimension_to_log: int,
    ) -> None:
        """
        Args:
            max_input_space_dimension_to_log: The maximum dimension of an input space
                to be logged.
                If this number is higher than the dimension of the input space
                then the input space will not be logged.
        """  # noqa: D205, D212
        result.objective_name = problem.objective.name
        result.design_space = problem.input_space
        problem.solution = result
        # An empty optimum is no optimum,
        # as when a run stops before evaluating one,
        # and there is nothing to write back into the space
        # and nothing to project for the fields set below.
        # The optimum is expressed in the coordinates the user declared
        # but not necessarily in the domain they declared,
        # e.g. a relaxed integer variable,
        # and the space accepts that domain only,
        # so the value written back into it is projected onto it,
        # once, here, and never at an evaluation.
        if result.x_opt is not None and result.x_opt.size:
            if self._iterates_on_working_problem:
                projected = self._transformation.project(result.x_opt)
            else:
                # This driver applies no coordinate transformation itself,
                # so it knows nothing of what a sub-driver,
                # or the algorithm itself, relaxed;
                # projecting onto the declared domain relaxes every kind,
                # and is the identity on a point already in it.
                projected = project_onto_declared_domain(
                    problem.input_space, result.x_opt
                )

            problem.input_space.set_current_value(projected)
            self.__set_projected_optimum(problem, result, projected)

        if self._settings.log_problem:
            self._log_result(problem, max_input_space_dimension_to_log)

    def __set_projected_optimum(
        self,
        problem: OptimizationProblem,
        result: OptimizationResult,
        projected: ndarray,
    ) -> None:
        """Set the fields of the optimum projected onto the declared domain.

        `x_opt` and `f_opt` keep the relaxed optimum, at the point where the
        algorithm stopped, so that they agree with each other;
        this sets the counterparts of the design variables, the objective and
        the feasibility at the point the design space actually receives,
        `x_opt` itself when the projection changed nothing.

        Args:
            problem: The problem the user built.
            result: The result to set the fields of.
            projected: The optimum, projected onto the domain the user declared.
        """
        if array_equal(projected, result.x_opt):
            result.x_opt_projected = result.x_opt
            result.x_opt_projected_as_dict = result.x_opt_as_dict
            result.f_opt_projected = result.f_opt
            result.is_feasible_projected = result.is_feasible
            return

        result.x_opt_projected = projected
        result.x_opt_projected_as_dict = problem.input_space.convert_array_to_dict(
            projected
        )
        # Evaluated on the bound functions of the problem the user built,
        # with their database disabled for the occasion: the result is
        # already built at this point, and recording this evaluation would
        # misalign `problem.database` with `result.x_opt`, e.g. after a
        # round trip through HDF, or as one iteration too many in a history
        # plot. The database is only switched off, not bypassed altogether,
        # so this evaluation still goes through the same NaN check as any
        # other, and still counts towards the number of calls.
        # Only the objective and the constraints are asked for:
        # an observable failing at the projected point, e.g. one undefined
        # off the domain the algorithm explored, must not prevent
        # `f_opt_projected` and `is_feasible_projected` from being set.
        output_functions, _ = problem.get_functions(observable_names=None)
        original_databases = [function._database for function in output_functions]
        for function in output_functions:
            function._database = None
        try:
            outputs, _ = problem.evaluate_functions(
                input_value=projected,
                input_value_is_normalized=False,
                output_functions=output_functions,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Could not evaluate the functions of %s at the optimum projected "
                "onto the declared domain, %s: %s.",
                problem.__class__.__name__,
                projected,
                error,
            )
            result.f_opt_projected = None
            result.is_feasible_projected = None
            return
        finally:
            for function, database in zip(
                output_functions, original_databases, strict=True
            ):
                function._database = database

        f_opt_projected = outputs[problem.objective.name]
        if not problem.minimize_objective and not problem.use_standardized_objective:
            # `f_opt` itself is un-negated the same way in
            # `OptimizationResult.from_optimization_problem`.
            f_opt_projected = -f_opt_projected

        result.f_opt_projected = f_opt_projected
        result.is_feasible_projected = problem.constraints.is_point_feasible(outputs)

    def _log_result(
        self, problem: OptimizationProblem, max_input_space_dimension_to_log: int
    ) -> None:
        """Log the optimization result.

        Args:
            problem: The problem to be solved.
            max_input_space_dimension_to_log: The maximum dimension of an input space
                to be logged.
                If this number is higher than the dimension of the input space
                then the input space will not be logged.
        """
        result = problem.solution
        opt_result_str = result._strings
        logger.info("%s", opt_result_str[0])
        if result.constraint_values:
            if result.is_feasible:
                logger.info("%s", opt_result_str[1])
            else:
                logger.warning("%s", opt_result_str[1])
        logger.info("%s", opt_result_str[2])
        if result.x_opt_projected is not None and not array_equal(
            result.x_opt, result.x_opt_projected
        ):
            # Logged only when the projection changed something,
            # e.g. a relaxed integer or discrete variable,
            # so a run relaxing nothing logs nothing more than before.
            logger.info(
                "Optimum projected onto the declared domain: %s, "
                "giving an objective of %s (feasible: %s).",
                result.x_opt_projected,
                result.f_opt_projected,
                result.is_feasible_projected,
            )
        input_space = problem.input_space
        space_name = _convert_camel_case_to_lower_case_words(type(input_space).__name__)
        self._log_input_space(
            input_space,
            max_input_space_dimension_to_log,
            f"{space_name.capitalize()}:",
            2,
        )

    @staticmethod
    def _log_input_space(
        input_space: BaseVariableSpace,
        max_dimension: int,
        heading: str,
        indentation: int,
    ) -> None:
        """Log an input space under a heading, unless it is too wide to read.

        Args:
            input_space: The input space to log.
            max_dimension: The dimension above which the input space is not logged.
            heading: The line introducing the input space.
            indentation: The number of indentation levels to log the heading at.
        """
        if input_space.dimension > max_dimension:
            return

        log = MultiLineString()
        for _ in range(indentation):
            log.indent()

        log.add("{}", heading)
        log.indent()
        for line in str(input_space).split("\n")[1:]:
            log.add(line)

        log.dedent()
        logger.info("%s", log)

    def __check_categorical_kinds(self, space: BaseVariableSpace) -> None:
        """Check that the algo handles the catalog and categorical variables of a space.

        A catalog variable is a categorical one
        and no setting relaxes these variables.

        Args:
            space: The space to check,
                e.g. the input space of the problem
                or the working space of the algorithm.

        Raises:
            ValueError: If the algo does not handle catalog variables
                and the space includes at least one catalog variable,
                or if the algo does not handle categorical variables
                and the space includes at least one categorical variable.
        """
        self._check_catalog_variables(space)
        if (
            space.variables.has_variables_of_type(VariableType.CATEGORICAL)
            and not self.ALGORITHM_INFOS[self._algo_name].handle_categorical_variables
        ):
            # A categorical variable has no relaxation.
            msg = self.__create_unhandled_kind_message(
                VariableType.CATEGORICAL,
                ".\nUse an algorithm handling them, e.g. a DOE algorithm.",
            )
            raise ValueError(msg)

    def _check_catalog_variables(self, input_space: BaseVariableSpace) -> None:
        """Check that the algo handles the catalog variables of a space.

        No setting relaxes a catalog variable.

        Args:
            input_space: The space to check.

        Raises:
            ValueError: If the algo does not handle catalog variables
                and the space includes at least one catalog variable.
        """
        if (
            input_space.variables.has_variables_of_type(VariableType.CATALOG)
            and not self.ALGORITHM_INFOS[self._algo_name].handle_catalog_variables
        ):
            # A catalog variable has no relaxation.
            msg = self.__create_unhandled_kind_message(
                VariableType.CATALOG,
                ".\nUse an algorithm handling them; no GEMSEO driver does yet.",
            )
            raise ValueError(msg)

    def _check_variable_handling(
        self,
        input_space: BaseVariableSpace,
    ) -> None:
        """Check if the algo handles the variables of the input space.

        This method is called only for a driver
        iterating on the problem the user built,
        e.g. one handing sub-problems to sub-drivers,
        which relax the variables according to their own settings,
        before the run.
        A driver iterating on a working problem does not call it:
        the variables of its working space are checked instead,
        once its working transformation is built,
        so overriding this method has no effect on such a driver.
        The user may relax the variables the algorithm does not handle,
        in this case a warning is logged.

        Args:
            input_space: The input space of the problem.

        Raises:
            ValueError: If the algo does not handle catalog (resp. categorical)
                variables and the input space includes at least one catalog
                (resp. categorical) variable,
                or if the corresponding relaxation setting is set to `False`
                and the algo does not handle integer (resp. discrete) variables
                and the input space includes at least one integer (resp. discrete)
                variable.
        """
        self.__check_categorical_kinds(input_space)
        self.__check_relaxation_settings(input_space)
        variables = input_space.variables
        for kind in self.__get_unhandled_kinds():
            if not variables.has_variables_of_type(kind):
                continue

            # The setting of this kind is `True`, see the check above.
            # An algorithm iterating on no working problem
            # applies no relaxation itself:
            # it passes the variables on as declared,
            # and one handing sub-problems to sub-algorithms
            # leaves the relaxation to their own settings.
            logger.warning(
                "%s does not relax the problem it is handed; "
                "the %s variables are passed on as declared.",
                self._algo_name,
                kind,
            )

    def __check_relaxation_settings(self, input_space: BaseVariableSpace) -> None:
        """Check that the settings relax the kinds of variables the algorithm needs.

        The settings alone decide here, whatever the transformations of the run.
        It is the check of a driver without a working problem,
        and of a run whose working transformation cannot be built,
        so that the relaxation hint still comes first.

        Args:
            input_space: The input space of the problem.

        Raises:
            ValueError: If the input space includes at least one variable
                of a kind the algorithm does not handle
                and the corresponding relaxation setting is set to `False`.
        """
        variables = input_space.variables
        for kind in self.__get_unhandled_kinds():
            if variables.has_variables_of_type(
                kind
            ) and not self.__get_relaxation_setting(kind):
                raise ValueError(self.__create_unhandled_kind_message(kind))

    def __get_unhandled_kinds(self) -> tuple[VariableType, ...]:
        """Return the kinds of variables the algorithm does not handle natively.

        Returns:
            The kinds of variables the algorithm does not handle natively.
        """
        description = self.ALGORITHM_INFOS[self._algo_name]
        return tuple(
            kind
            for kind, handles_kind in (
                (VariableType.INTEGER, description.handle_integer_variables),
                (VariableType.DISCRETE, description.handle_discrete_variables),
            )
            if not handles_kind
        )

    def __get_relaxation_setting(self, kind: VariableType) -> bool:
        """Return the setting relaxing a kind of variables.

        Args:
            kind: The kind of variables.

        Returns:
            Whether the variables of this kind are relaxed.
        """
        return getattr(self._settings, f"relax_{kind}_variables")

    def __create_unhandled_kind_message(
        self, kind: VariableType, end: str | None = None
    ) -> str:
        """Create the message of an algorithm not handling a kind of variables.

        Args:
            kind: The kind of variables.
            end: The end of the message,
                following "it does not handle <kind> variables".
                If `None`,
                the relaxation setting is the reason why the variables are not relaxed,
                so the message tells to change it.

        Returns:
            The message.
        """
        if end is None:
            end = (
                ".\n"
                f"Set 'relax_{kind}_variables' to 'True' to relax them to float ones, "
                "or use an algorithm handling them."
            )

        return (
            f"Algorithm {self._algo_name} is not adapted to the problem, "
            f"it does not handle {kind} variables{end}"
        )

    @property
    def _is_solving_optimization_problem(self) -> bool:
        """Whether is solving an optimization problem."""
        return self._problem._is_optimization

    @property
    def _evaluates_in_parallel(self) -> bool:
        """Whether the algorithm evaluates the functions in parallel."""
        # TODO: Have a better class hierarchy to avoid getattr,
        # or have this field in all settings but forced to be 1 as needed.
        return getattr(self._settings, "n_processes", 1) > 1

    _original_problem: EvaluationProblem[_SpaceT] | None
    """The problem the user built, which the algorithm does not iterate on.

    Its functions take a point in the coordinates the user declared,
    which is what a database listener receives.
    It is `None` outside a run.
    """

    _transformation: BaseSpaceTransformation | None
    """The map from the coordinates the user declared to the working ones.

    It is `None` outside a run,
    since a transformation is built for a space
    and there is none to build it for.
    During a run it is set whatever the driver does,
    an empty composition when the algorithm iterates on the problem the user built,
    so that a caller needing a point in the user's coordinates
    can apply the backward map without asking whether there is one.
    """

    _iterates_on_working_problem: ClassVar[bool] = True
    """Whether the algorithm iterates on a working problem built from the user's one.

    A meta-algorithm that only builds sub-problems
    and hands them to sub-drivers does not.
    It builds each sub-problem from a copy of its input space *and* its functions,
    so the two have to come from the same place,
    and each sub-driver adapts its own sub-problem.
    """

    def _reset(self) -> None:
        super()._reset()
        # These two are the state of a run,
        # like the problem and the settings the base class clears.
        # Left as they are,
        # they would keep the problem the user built, its input space and its functions
        # alive for as long as the library,
        # and a driver library is serializable,
        # so a pickle taken after the run would carry that whole graph.
        self._original_problem = None
        self._transformation = None
        self.__transformed_kinds = None

    def _attach_criteria(self, problem: EvaluationProblem[_SpaceT]) -> None:
        """Attach the termination criteria to the problem the user built.

        The user supplies the tolerances in the scales of their own functions
        and variables,
        so the criteria are attached to,
        and evaluated against,
        the original problem,
        even when the algorithm iterates on a working one.

        Args:
            problem: The problem the user built.
        """

    def execute(
        self,
        problem: EvaluationProblem[_SpaceT],
        settings: BaseDriverSettings | None = None,
    ) -> OptimizationResult:
        """
        Raises:
            ValueError: If there is no function in the problem.
        """  # noqa: D205, D212
        # The state of a run is cleared
        # whatever the outcome of that run:
        # an algorithm raising would otherwise leave the problems and the settings
        # on the library,
        # where the next run would not need them
        # and a pickle taken in between would drag them along.
        try:
            return self.__execute(problem, settings)
        finally:
            # `__build_working_problem` tells the evaluation halves of the
            # original problem the transformation, whatever the algorithm does
            # with the working one, so it is released here, whether the run
            # succeeded or raised, before it is forgotten below.
            if self._original_problem is not None:
                self._original_problem._release_transformation()
            self._reset()

    def __execute(
        self,
        problem: EvaluationProblem[_SpaceT],
        settings: BaseDriverSettings | None,
    ) -> OptimizationResult:
        """Run the algorithm on a problem.

        Args:
            problem: The problem to be solved.
            settings: The settings of the algorithm,
                or `None` to use the default ones.

        Returns:
            The result of the run,
            for a problem the driver family can build a result for.

        Raises:
            ValueError: If there is no function in the problem.
        """
        solve_optimization_problem, transformation, relaxed_names = self.__prepare_run(
            problem, settings
        )
        problem = self.__build_working_problem(problem, transformation, relaxed_names)
        self.__log_problem(solve_optimization_problem)
        with self.__record_iterations(problem):
            result = self.__run_algorithm(problem, solve_optimization_problem)

        if solve_optimization_problem:
            # The result is built on the problem the user built,
            # whose database holds the history in their own coordinates.
            self._post_run(
                self._original_problem,
                result,
                self._settings.max_input_space_dimension_to_log,
            )

        return result

    def __prepare_run(
        self,
        problem: EvaluationProblem[_SpaceT],
        settings: BaseDriverSettings | None,
    ) -> tuple[bool, SpaceComposition, set[str]]:
        """Validate the problem, build the settings of the run and apply them.

        Args:
            problem: The problem the user built.
            settings: The settings of the algorithm,
                or `None` to use the default ones.

        Returns:
            Whether the algorithm solves an optimization problem,
            the map from the coordinates the user declared to the working ones,
            and the names of the variables it relaxes.

        Raises:
            ValueError: If there is no function in the problem.
        """
        self._problem = problem
        if not problem.functions:
            msg = "A driver requires a problem with at least one function."
            raise ValueError(msg)

        self._settings = create_model(
            self.ALGORITHM_INFOS[self.algo_name].settings_class, settings_model=settings
        )
        # The variable-handling messages, with their
        # "Set 'relax_..._variables' to 'True'" hint,
        # must be raised before the generic unsuitability message,
        # and before the settings are applied to the problem.
        # The problem is checked before a transformation is built on its space,
        # so that an inconsistent space gives its own message,
        # and the working space is built here,
        # the composition building the one of its last transformation on demand.
        self.__transformed_kinds = None
        try:
            problem.check()
            transformation = self.__create_working_transformation(problem)
            transformation.working_space  # noqa: B018
        except Exception as error:  # noqa: BLE001
            check_error = error
        else:
            check_error = None

        if check_error is not None:
            # A problem that is not ready, e.g. an empty input space,
            # or a transformation that cannot be built,
            # e.g. a normalization of a space that has no bounds to normalize
            # against, must not hide that the algorithm is not suited anyway,
            # nor the relaxation hint, which comes first.
            # What the transformations would have relaxed or encoded is unknown,
            # so the settings decide, whatever the driver,
            # and the error of the build is chained to the raised one,
            # since a hint may be misleading
            # for a chain relaxing variables whatever the settings.
            try:
                if self._iterates_on_working_problem:
                    self.__check_categorical_kinds(problem.input_space)
                    self.__check_relaxation_settings(problem.input_space)
                else:
                    self._check_variable_handling(problem.input_space)

                self._check_algorithm(problem)
            except ValueError as error:
                raise error from check_error

            raise check_error

        relaxed_names = transformation.get_relaxed_names()
        if self._iterates_on_working_problem:
            self.__transformed_kinds = self.__check_working_space_handling(
                problem.input_space, transformation, relaxed_names
            )
        else:
            self._check_variable_handling(problem.input_space)

        self._check_algorithm(problem)
        self.__log_relaxed_kinds(problem.input_space, relaxed_names)
        self.__warn_if_normalize_design_space_is_ignored()

        solve_optimization_problem = self._is_solving_optimization_problem
        if solve_optimization_problem:
            problem: OptimizationProblem
            problem.tolerances.equality = self._settings.eq_tolerance
            problem.tolerances.inequality = self._settings.ineq_tolerance

        enable_progress_bar = self._settings.enable_progress_bar
        if enable_progress_bar is not None:
            self.enable_progress_bar = enable_progress_bar

        return solve_optimization_problem, transformation, relaxed_names

    def __warn_if_normalize_design_space_is_ignored(self) -> None:
        """Warn when `normalize_design_space` is explicitly set but has no effect.

        A library that does not iterate on a working problem does not
        transform the problem it is handed, for one of two reasons: a
        meta-algorithm, such as the augmented Lagrangian or `MultiStart`,
        delegates to sub-algorithms, which normalize according to their own
        settings; a linear solver, such as `ScipyLinprog` or `ScipyMILP`,
        reads the coefficients of the functions and the bounds of the
        problem directly, both in the coordinates the user declared, and has
        no sub-algorithm at all. Either way, the setting has no effect at the
        top level. The check reads `model_fields_set` rather than the value
        alone, so that a default left untouched, e.g. the augmented
        Lagrangian's own default of `True`, stays quiet and only an explicit
        request warns.
        """
        if (
            not self._iterates_on_working_problem
            and "normalize_design_space" in self._settings.model_fields_set
            and self._settings.normalize_design_space
        ):
            logger.warning(
                "The setting normalize_design_space is ignored by %s, which "
                "does not transform the problem it is handed.",
                self._algo_name,
            )

    def _get_default_transformation_factories(
        self, problem: EvaluationProblem[_SpaceT]
    ) -> list[Callable[[DesignSpace], BaseSpaceTransformation[DesignSpace]]]:
        """Return the factories of the map from the user's coordinates to the working ones.

        The working space is built by applying the transformations in order,
        each one built on the working space of the transformation before it,
        see [SpaceComposition][gemseo.space.transformation.composition.SpaceComposition].
        By default,
        the variables of a kind the algorithm does not handle are relaxed,
        according to the settings of the run,
        then the space is normalized,
        according to the settings of the run too.

        A driver changing the chain overrides this method,
        either calling it to append, insert or replace a factory,
        or returning its own chain,
        and may read the settings of the run and the name of the algorithm
        to make the chain depend on them.
        It is called only when the algorithm iterates on a working problem.

        Args:
            problem: The problem the user built.

        Returns:
            The factories of the transformations,
            from the original space to the working one,
            e.g. a transformation class
            or a `functools.partial` of it setting its other arguments.
        """  # noqa: E501
        # A kind of variable the algorithm handles is left as declared:
        # the algorithm, and the functions of the problem,
        # such as a discipline whose grammar declares an integer,
        # expect a value of the domain the user declared, not a relaxed one.
        unhandled_kinds = self.__get_unhandled_kinds()
        return get_default_transformation_factories(
            problem.input_space,
            normalize=self._settings.normalize_design_space,
            relax_integer=(
                VariableType.INTEGER in unhandled_kinds
                and self.__get_relaxation_setting(VariableType.INTEGER)
            ),
            relax_discrete=(
                VariableType.DISCRETE in unhandled_kinds
                and self.__get_relaxation_setting(VariableType.DISCRETE)
            ),
        )

    def __create_working_transformation(
        self, problem: EvaluationProblem[_SpaceT]
    ) -> SpaceComposition:
        """Create the map from the coordinates the user declared to the working ones.

        Args:
            problem: The problem the user built.

        Returns:
            The map from the coordinates the user declared to the working ones.
        """
        if self._iterates_on_working_problem:
            return SpaceComposition(
                problem.input_space,
                *self._get_default_transformation_factories(problem),
            )

        # A driver iterating on the problem the user built
        # applies no transformation and gets an empty composition,
        # which is the identity;
        # the composition is built whatever the case,
        # so that a caller applying the backward map
        # need not ask whether there is one.
        return SpaceComposition(problem.input_space)

    def __check_working_space_handling(
        self,
        input_space: BaseVariableSpace,
        transformation: SpaceComposition,
        relaxed_names: AbstractSet[str],
    ) -> frozenset[VariableType]:
        """Check that the algorithm handles the variables of the working space.

        The transformations are chosen by
        [_get_default_transformation_factories][gemseo.core.algorithm.base_driver_library.BaseDriverLibrary._get_default_transformation_factories],
        which a driver may override,
        so the variables are checked on the working space they lead to,
        whatever the relaxation settings,
        the transformations, their order,
        and the variables they replace or rename.

        Args:
            input_space: The input space of the problem.
            transformation: The working transformation.
            relaxed_names: The names of the variables it relaxes.

        Returns:
            The kinds of the declared variables
            the transformation relaxes or encodes into other ones.

        Raises:
            ValueError: When a variable of the working space
                has a type the algorithm does not handle.
        """  # noqa: E501
        # A catalog or categorical variable has no relaxation,
        # so one remaining in the working space is refused,
        # whereas one encoded away is accepted, as below.
        self.__check_categorical_kinds(transformation.working_space)
        # A declared variable of a kind the algorithm does not handle
        # is accepted when the transformation relaxes it,
        # or when the working space has no variable of its name:
        # the transformation then encodes it into other variables
        # and its backward map decodes a value into the domain of the variable,
        # see `BaseSpaceTransformation.get_relaxed_names`.
        # Otherwise,
        # its working variable may have another type,
        # e.g. a normalized integer variable is a float one,
        # but the algorithm still iterates on an image of an integer variable.
        # A working variable of such a kind is refused whatever its name,
        # since it is what the algorithm iterates on.
        original_variables = input_space.variables
        working_variables = transformation.working_space.variables
        unhandled_kinds = self.__get_unhandled_kinds()
        for kind in unhandled_kinds:
            unrelaxed_names = [
                name
                for name, variable in original_variables.items()
                if variable.type == kind
                and name not in relaxed_names
                and name in working_variables
            ]
            working_names = [
                name
                for name, variable in working_variables.items()
                if variable.type == kind
            ]
            if not (unrelaxed_names or working_names):
                continue

            if unrelaxed_names and not self.__get_relaxation_setting(kind):
                # The usual case: the user did not ask for a relaxation,
                # so the message tells to.
                msg = self.__create_unhandled_kind_message(kind)
            elif unrelaxed_names:
                # The settings ask for a relaxation the transformation
                # does not apply.
                msg = self.__create_unhandled_kind_message(
                    kind,
                    " and its working transformation leaves "
                    f"{', '.join(unrelaxed_names)} unrelaxed.",
                )
            else:
                msg = self.__create_unhandled_kind_message(
                    kind,
                    f" and its working space has the {kind} variables "
                    f"{', '.join(working_names)}.",
                )

            raise ValueError(msg)

        return frozenset(
            variable.type
            for name, variable in original_variables.items()
            if name in relaxed_names or name not in working_variables
        )

    def __log_relaxed_kinds(
        self, input_space: BaseVariableSpace, relaxed_names: AbstractSet[str]
    ) -> None:
        """Warn for each kind of variables the algorithm does not handle and relaxes.

        It is called once every check has accepted the run,
        so that the log reports only a relaxation that does run.

        Args:
            input_space: The input space of the problem.
            relaxed_names: The names of the variables the transformation relaxes.
        """
        original_variables = input_space.variables
        for kind in self.__get_unhandled_kinds():
            if any(original_variables[name].type == kind for name in relaxed_names):
                logger.warning(
                    "Running an algorithm that does not handle "
                    "%s variables; they are relaxed to float ones.",
                    kind,
                )

    def __build_working_problem(
        self,
        problem: EvaluationProblem[_SpaceT],
        transformation: SpaceComposition,
        relaxed_names: AbstractSet[str],
    ) -> EvaluationProblem[_SpaceT]:
        """Bind the functions of the problem and build the working one to iterate on.

        Args:
            problem: The problem the user built.
            transformation: The map from the coordinates the user declared
                to the working ones.
            relaxed_names: The names of the variables it relaxes.

        Returns:
            The working problem the algorithm iterates on,
            which is the one the user built when the driver builds none.
        """
        # Base drivers have no 'vectorize' option,
        # unlike certain specialized drivers, such as DOEs.
        vectorize = getattr(self._settings, "vectorize", False)
        # The transformation may relax some of the variables to float ones;
        # the database is told which, so
        # `to_dataset()` can export them as float columns from this flag
        # rather than from the values it records, which may happen to be
        # integral even for a relaxed run.
        # Set, not merged with what a previous run may have left there:
        # the database may outlive this run, e.g. a second, non-relaxed,
        # one on the same problem, which relaxes nothing and must not keep
        # exporting the columns of the first run as float.
        # Set only once the run is accepted:
        # a refused run evaluates nothing,
        # so the names keep describing the evaluations of the previous runs.
        problem.database.relaxed_variable_names = set(relaxed_names)

        # The problem binds its own functions to its own database and counters,
        # in the coordinates the user declared.
        problem.bind_functions(
            use_database=self._settings.use_database,
            store_jacobian=self._settings.store_jacobian,
            support_sparse_jacobian=self._support_sparse_jacobian,
            evaluate_observable_jacobian=self._settings.evaluate_observable_jacobian,
            vectorize=vectorize,
        )
        # The algorithm iterates on the working problem;
        # the one the user built is left untouched
        # and is what the results are returned on.
        # `create_working_problem` tells the evaluation halves the map,
        # so an approximated Jacobian perturbs in the working coordinates.
        # A working problem is built even when the composition is empty,
        # e.g. a default DOE or an optimizer with `normalize_design_space`
        # set to `False` and nothing to relax: an algorithm iterating on the
        # bound problem itself, such as `BaseOptimizationLibrary._pre_run`,
        # replaces its objective and constraints with scaled versions,
        # and doing so on the problem the user built would leave it mutated
        # once the run is over. An empty composition is the identity,
        # so its working space is the original one, the very same object,
        # and building the working problem costs no copy of it.
        self._original_problem = problem
        self._transformation = transformation
        if self._iterates_on_working_problem:
            problem = problem.create_working_problem(transformation)

        self._problem = problem
        return problem

    def __log_problem(self, solve_optimization_problem: bool) -> None:
        """Log the problem the user built and announce the run.

        Args:
            solve_optimization_problem: Whether the algorithm solves
                an optimization problem.
        """
        original_problem = self._original_problem
        if self._settings.log_problem:
            # What is logged is the problem the user built,
            # in their own coordinates:
            # the working one exists for the algorithm only.
            logger.info("%s", original_problem)
            input_space = original_problem.input_space
            space_name = _convert_camel_case_to_lower_case_words(
                type(input_space).__name__
            )
            self._log_input_space(
                input_space,
                self._settings.max_input_space_dimension_to_log,
                f"over the {space_name}:",
                1,
            )

        if self._settings.log_problem and solve_optimization_problem:
            progress_bar_title = "Solving optimization problem with algorithm %s:"
        else:
            progress_bar_title = "Running the algorithm %s:"

        if self.enable_progress_bar:
            logger.info(progress_bar_title, self._algo_name)

    @contextmanager
    def __record_iterations(
        self, problem: EvaluationProblem[_SpaceT]
    ) -> Generator[None, None, None]:
        """Hook the driver onto the evaluation layer of the problem.

        The hooks are released whatever the run does,
        an algorithm that raises included:
        left in place,
        the one finalizing an iteration would keep the library alive
        through the functions of the user
        and fire on the next run.

        Args:
            problem: The problem the algorithm iterates on.

        Yields:
            Nothing; the hooks are set for the duration of the context.
        """
        # The hook belongs to the evaluation layer,
        # which the original problem owns:
        # it fires on a point absent from the database,
        # in the user's coordinates.
        # The evaluation layer is rebuilt at each run,
        # so the hook is this driver's own
        # and the run it belongs to is the one releasing it.
        functions = self._original_problem.functions
        if not self._evaluates_in_parallel:
            for function in functions:
                function.pre_compute_at_new_point = (
                    self._finalize_previous_iteration_using_database
                )

        if problem.new_iter_observables:
            problem.database.add_new_iter_listener(
                problem.new_iter_observables.evaluate
            )

        try:
            yield
        finally:
            problem.database.clear_listeners(
                new_iter_listeners=(
                    (o.evaluate,) if (o := problem.new_iter_observables) else None
                ),
                store_listeners=None,
            )
            for function in functions:
                function.pre_compute_at_new_point = None

    def __run_algorithm(
        self,
        problem: EvaluationProblem[_SpaceT],
        solve_optimization_problem: bool,
    ) -> OptimizationResult | None:
        """Run the algorithm on the problem it iterates on.

        Args:
            problem: The problem the algorithm iterates on.
            solve_optimization_problem: Whether the algorithm solves
                an optimization problem.

        Returns:
            The result of the run,
            or `None` when the driver family builds no result.
        """
        original_problem = self._original_problem
        parallelize = self._evaluates_in_parallel
        result = None
        try:
            with (
                OneLineLogging(tqdm_logger)
                if self._settings.use_one_line_progress_bar
                else nullcontext()
            ):
                get_result = self._get_result
                try:
                    # The criteria are attached to the problem the user built,
                    # since the tolerances are expressed in their scales.
                    self._attach_criteria(original_problem)
                    # pre_run can trigger termination criteria,
                    # e.g., max_iter or max_time.
                    self._pre_run(problem)
                    args = self._run(problem) or (None, None)
                except TerminationCriterion as termination_criterion:
                    args = (termination_criterion,)
                    get_result = self._get_early_stopping_result
                    # Disable the counter
                    # because the iteration has been finalized
                    # just before raising the TerminationCriterion
                    # (see the _finalize_previous_iteration_using_database method).
                    problem.evaluation_counter.enabled = False

                if solve_optimization_problem:
                    # The result is built on the problem the user built,
                    # whose database holds the history in their own coordinates.
                    result = get_result(original_problem, *args)

            if self._problem.evaluation_counter.enabled and not parallelize:
                self._finalize_previous_iteration()

            problem.evaluation_counter.enabled = False
        finally:
            # Close the progress bar even when the algorithm raises,
            # otherwise it would log its last status when garbage collected.
            if self._progress_bar is not None:
                self._progress_bar.close()

        return result

    def _finalize_previous_iteration(self):
        """Finalize the previous iteration."""
        problem = self._problem
        problem.evaluation_counter.current += 1
        if self._progress_bar is not None:
            input_value = HashableNdarray(problem.database.get_last_n_x_vect(1)[0])
            self._progress_bar.update(input_value)

    @abstractmethod
    def _run(self, problem: EvaluationProblem[_SpaceT]) -> tuple[Any, Any]:
        """
        Returns:
            The message and status of the algorithm if any.
        """  # noqa: D205 D212

    def _get_early_stopping_result(
        self,
        problem: EvaluationProblem[_SpaceT],
        termination_criterion: TerminationCriterion,
    ) -> OptimizationResult:
        """Retrieve the best known result when a termination criterion is met.

        Args:
            problem: The problem to be solved.
            termination_criterion: A termination criterion.

        Returns:
            The best known optimization result when the termination criterion is met.
        """
        message = termination_criterion.message + "GEMSEO stopped the driver."
        return self._get_result(problem, message, None)

    def _get_result(
        self,
        problem: OptimizationProblem,
        message: Any,
        status: Any,
        *args: Any,
    ) -> OptimizationResult:
        """Return the result of the resolution of the problem.

        Args:
            problem: The problem that was solved.
            message: The message associated with the termination criterion if any.
            status: The status associated with the termination criterion if any.
            *args: Specific arguments.

        Returns:
            The result of the resolution of the problem.

        Raises:
            NotImplementedError: When the driver family did not set `_result_class`.
        """
        if self._result_class is None:
            msg = (
                f"The driver library {type(self).__name__} cannot build a result; "
                "set its class attribute _result_class."
            )
            raise NotImplementedError(msg)
        return self._result_class.from_optimization_problem(
            problem, message=message, status=status, optimizer_name=self._algo_name
        )
