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

A driver is an algorithm evaluating the functions of an :class:`.EvaluationProblem`
at different points of the design space,
using the :meth:`~.BaseDriverLibrary.execute` method.
In the case of an :class:`.OptimizationProblem`,
this method also returns an :class:`.OptimizationResult`.

There are two main families of drivers:
the optimizers with the base class :class:`.BaseOptimizationLibrary`
and the design of experiments (DOE) with the base class :class:`.BaseDOELibrary`.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import TypeVar

from numpy import ndarray

from gemseo.algos._progress_bar.custom import LOGGER as TQDM_LOGGER
from gemseo.algos._progress_bar.standard import ProgressBar
from gemseo.algos._progress_bar.unsuffixed import UnsuffixedProgressBar
from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.base_algorithm_library import AlgorithmDescription
from gemseo.algos.base_algorithm_library import BaseAlgorithmLibrary
from gemseo.algos.base_driver_settings import BaseDriverSettings
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.algos.hashable_ndarray import HashableNdarray
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.algos.progress_bar_data.data import ProgressBarData
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FtolReached
from gemseo.algos.stop_criteria import FunctionIsNan
from gemseo.algos.stop_criteria import KKTReached
from gemseo.algos.stop_criteria import MaxIterReachedException
from gemseo.algos.stop_criteria import MaxTimeReached
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.algos.stop_criteria import XtolReached
from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType
from gemseo.typing import StrKeyMapping
from gemseo.utils.constants import _ENABLE_PROGRESS_BAR
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.logging import OneLineLogging
from gemseo.utils.pydantic import create_model
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from gemseo.algos._progress_bar.base import BaseProgressBar
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.progress_bar_data.factory import ProgressBarDataName

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
LOGGER = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseDriverSettings)


@dataclass
class DriverDescription(AlgorithmDescription):
    """The description of a driver."""

    handle_integer_variables: bool = False
    """Whether the driver handles integer variables."""

    Settings: type[BaseDriverSettings] = BaseDriverSettings
    """The Pydantic model for the driver library settings."""


class BaseDriverLibrary(BaseAlgorithmLibrary[T]):
    """Base class for libraries of drivers."""

    ApproximationMode = ApproximationMode

    DifferentiationMethod = EvaluationProblem.DifferentiationMethod

    ALGORITHM_INFOS: ClassVar[dict[str, DriverDescription]] = {}
    """The description of the algorithms contained in the library."""

    _RESULT_CLASS: ClassVar[type[OptimizationResult]] = OptimizationResult
    """The class used to present the result of the optimization."""

    _SUPPORT_SPARSE_JACOBIAN: ClassVar[bool] = False
    """Whether the library support sparse Jacobians."""

    enable_progress_bar: bool = _ENABLE_PROGRESS_BAR
    """Whether to enable the progress bar in the evaluation log."""

    _problem: EvaluationProblem | None
    """The optimization problem the driver library is bonded to."""

    _progress_bar: BaseProgressBar | None
    """The progress bar used during the execution, if any."""

    __start_time: float
    """The time at which the execution begins."""

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self._progress_bar = None
        self.__start_time = 0.0

    @classmethod
    def _get_unsuitability_reason(
        cls, algorithm_description: DriverDescription, problem: OptimizationProblem
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if reason or problem.design_space:
            return reason

        return _UnsuitabilityReason.EMPTY_DESIGN_SPACE

    def _init_iter_observer(
        self,
        problem: EvaluationProblem,
        max_iter: int,
        message: str = "",
        progress_bar_data_name: ProgressBarDataName = ProgressBarData.__name__,
    ) -> None:
        """Initialize the iteration observer.

        It will handle the stopping criteria and the update of the progress bar.

        Args:
            problem: The evaluation problem.
            max_iter: The maximum number of iterations.
            message: The message to display at the beginning of the progress bar status.
            progress_bar_data_name: The name
                of a :class:`.BaseProgressBarData` class
                to define the data of an optimization problem
                to be displayed in the progress bar.
        """
        from gemseo.utils.global_configuration import _configuration

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
        """Check the stopping criteria at the current iteration.

        Raises:
            MaxTimeReached: If the elapsed time is greater
                than the maximum execution time.
            MaxTimeReached If the maximum number of evaluations is reached.
        """
        t = time()
        if 0 < self._settings.max_time < t - self.__start_time:
            raise MaxTimeReached

        if self._problem.evaluation_counter.maximum_is_reached:
            raise MaxIterReachedException

    def _post_run(
        self,
        problem: OptimizationProblem,
        result: OptimizationResult,
        max_design_space_dimension_to_log: int,
    ) -> None:
        """
        Args:
            max_design_space_dimension_to_log: The maximum dimension of a design space
                to be logged.
                If this number is higher than the dimension of the design space
                then the design space will not be logged.
        """  # noqa: D205, D212
        result.objective_name = problem.objective.name
        result.design_space = problem.design_space
        problem.solution = result
        if result.x_opt is not None:
            problem.design_space.set_current_value(result)

        if self._settings.log_problem:
            self._log_result(problem, max_design_space_dimension_to_log)

    def _log_result(
        self, problem: OptimizationProblem, max_design_space_dimension_to_log: int
    ) -> None:
        """Log the optimization result.

        Args:
            problem: The problem to be solved.
            max_design_space_dimension_to_log: The maximum dimension of a design space
                to be logged.
                If this number is higher than the dimension of the design space
                then the design space will not be logged.
        """
        result = problem.solution
        opt_result_str = result._strings
        LOGGER.info("%s", opt_result_str[0])
        if result.constraint_values:
            if result.is_feasible:
                LOGGER.info("%s", opt_result_str[1])
            else:
                LOGGER.warning("%s", opt_result_str[1])
        LOGGER.info("%s", opt_result_str[2])
        if problem.design_space.dimension <= max_design_space_dimension_to_log:
            log = MultiLineString()
            log.indent()
            log.indent()
            log.add("Design space:")
            log.indent()
            for line in str(problem.design_space).split("\n")[1:]:
                log.add(line)
            log.dedent()
            LOGGER.info("%s", log)

    def _check_integer_handling(
        self,
        design_space: DesignSpace,
        force_execution: bool,
    ) -> None:
        """Check if the algo handles integer variables.

        The user may force the execution if needed, in this case a warning is logged.

        Args:
            design_space: The design space of the problem.
            force_execution: Whether to force the execution of the algorithm when
                the problem includes integer variables and the algo does not handle
                them.

        Raises:
            ValueError: If `force_execution` is set to `False` and
                the algo does not handle integer variables and the
                design space includes at least one integer variable.
        """
        if (
            design_space.has_integer_variables
            and not self.ALGORITHM_INFOS[self._algo_name].handle_integer_variables
        ):
            if not force_execution:
                msg = (
                    f"Algorithm {self._algo_name} is not adapted to the problem, "
                    "it does not handle integer variables.\n"
                    "Execution may be forced setting the 'skip_int_check' "
                    "argument to 'True'."
                )
                raise ValueError(msg)

            LOGGER.warning(
                "Forcing the execution of an algorithm that does not handle "
                "integer variables."
            )

    # TODO: API: state this in the class hierarchy instead of at runtime.
    @property
    def _is_solving_optimization_problem(self) -> bool:
        """Whether is solving an optimization problem."""
        return isinstance(self._problem, OptimizationProblem)

    # TODO: API: move the following arguments into the settings_model
    # - eval_obs_jac
    # - skip_int_check
    # - max_design_space_dimension_to_log
    def execute(
        self,
        problem: EvaluationProblem,
        eval_obs_jac: bool = False,
        skip_int_check: bool = False,
        max_design_space_dimension_to_log: int = 40,
        settings_model: BaseDriverSettings | None = None,
        **settings: Any,
    ) -> OptimizationResult:
        """
        Args:
            eval_obs_jac: Whether to evaluate the Jacobian of the observables.
            skip_int_check: Whether to skip the integer variable handling check
                of the selected algorithm.
            max_design_space_dimension_to_log: The maximum dimension of a design space
                to be logged.
                If this number is higher than the dimension of the design space
                then the design space will not be logged.
        """  # noqa: D205, D212
        self._problem = problem
        self._check_algorithm(problem)
        self._check_integer_handling(problem.design_space, skip_int_check)

        self._settings = create_model(
            self.ALGORITHM_INFOS[self.algo_name].Settings,
            settings_model=settings_model,
            **settings,
        )

        solve_optimization_problem = self._is_solving_optimization_problem
        if solve_optimization_problem:
            problem: OptimizationProblem
            problem.tolerances.equality = self._settings.eq_tolerance
            problem.tolerances.inequality = self._settings.ineq_tolerance

        enable_progress_bar = self._settings.enable_progress_bar
        if enable_progress_bar is not None:
            self.enable_progress_bar = enable_progress_bar

        problem.check()
        problem.preprocess_functions(
            is_function_input_normalized=self._settings.normalize_design_space,
            use_database=self._settings.use_database,
            round_ints=self._settings.round_ints,
            eval_obs_jac=eval_obs_jac,
            support_sparse_jacobian=self._SUPPORT_SPARSE_JACOBIAN,
            store_jacobian=self._settings.store_jacobian,
            # Base drivers have no 'vectorize' option,
            # unlike certain specialized drivers, such as DOEs.
            vectorize=getattr(self._settings, "vectorize", False),
        )
        # TODO: Have a better class hierarchy to avoid getattr,
        # or have this field in all settings but forced to be 1 as needed.
        parallelize = getattr(self._settings, "n_processes", 1) > 1

        functions = problem.functions

        set_pre_compute_at_new_point = functions[0].pre_compute_at_new_point is None
        if set_pre_compute_at_new_point and not parallelize:
            for function in functions:
                function.pre_compute_at_new_point = (
                    self._finalize_previous_iteration_using_database
                )

        if problem.new_iter_observables:
            problem.database.add_new_iter_listener(
                problem.new_iter_observables.evaluate
            )
        if self._settings.log_problem:
            LOGGER.info("%s", problem)
            if problem.design_space.dimension <= max_design_space_dimension_to_log:
                log = MultiLineString()
                log.indent()
                log.add("over the design space:")
                log.indent()
                for line in str(problem.design_space).split("\n")[1:]:
                    log.add(line)
                log.dedent()
                LOGGER.info("%s", log)

        if self._settings.log_problem and solve_optimization_problem:
            progress_bar_title = "Solving optimization problem with algorithm %s:"
        else:
            progress_bar_title = "Running the algorithm %s:"

        if self.enable_progress_bar:
            LOGGER.info(progress_bar_title, self._algo_name)

        result = None
        with (
            OneLineLogging(TQDM_LOGGER)
            if self._settings.use_one_line_progress_bar
            else nullcontext()
        ):
            get_result = self._get_result
            try:
                # pre_run can trigger stopping criteria, e.g., max_iter or max_time.
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
                problem: OptimizationProblem
                result = get_result(problem, *args)

        if self._problem.evaluation_counter.enabled and not parallelize:
            self._finalize_previous_iteration()

        problem.evaluation_counter.enabled = False
        if self._progress_bar is not None:
            self._progress_bar.close()

        problem.database.clear_listeners(
            new_iter_listeners=(
                (o.evaluate,) if (o := problem.new_iter_observables) else None
            ),
            store_listeners=None,
        )
        if set_pre_compute_at_new_point:
            for function in problem.functions:
                function.pre_compute_at_new_point = None

        if solve_optimization_problem:
            self._post_run(
                problem,
                result,
                max_design_space_dimension_to_log,
            )

        self._reset()
        return result

    def _finalize_previous_iteration(self):
        """Finalize the previous iteration."""
        problem = self._problem
        problem.evaluation_counter.current += 1
        if self._progress_bar is not None:
            input_value = HashableNdarray(problem.database.get_last_n_x_vect(1)[0])
            self._progress_bar.update(input_value)

    @abstractmethod
    def _run(self, problem: EvaluationProblem) -> tuple[Any, Any]:
        """
        Returns:
            The message and status of the algorithm if any.
        """  # noqa: D205 D212

    def _get_early_stopping_result(
        self, problem: OptimizationProblem, termination_criterion: TerminationCriterion
    ) -> OptimizationResult:
        """Retrieve the best known result when a termination criterion is met.

        Args:
            problem: The problem to be solved.
            termination_criterion: A termination criterion.

        Returns:
            The best known optimization result when the termination criterion is met.
        """
        if isinstance(termination_criterion, MaxIterReachedException):
            message = "Maximum number of iterations reached. "
        elif isinstance(termination_criterion, FunctionIsNan):
            message = (
                "Function value or gradient or constraint is NaN, "
                "and problem.stop_if_nan is set to True. "
            )
        elif isinstance(termination_criterion, DesvarIsNan):
            message = "Design variables are NaN. "
        elif isinstance(termination_criterion, XtolReached):
            message = (
                "Successive iterates of the design variables "
                "are closer than xtol_rel or xtol_abs. "
            )
        elif isinstance(termination_criterion, FtolReached):
            message = (
                "Successive iterates of the objective function "
                "are closer than ftol_rel or ftol_abs. "
            )
        elif isinstance(termination_criterion, MaxTimeReached):
            message = f"Maximum time reached: {self._settings.max_time} seconds. "
        elif isinstance(termination_criterion, KKTReached):
            message = (
                "The KKT residual norm is smaller than the tolerance "
                "kkt_tol_abs or kkt_tol_rel. "
            )
        else:
            message = ""

        message += "GEMSEO stopped the driver."
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
            message: The message associated with the termination criterion if any.
            status: The status associated with the termination criterion if any.
            *args: Specific arguments.
        """
        return self._RESULT_CLASS.from_optimization_problem(
            problem, message=message, status=status, optimizer_name=self._algo_name
        )
