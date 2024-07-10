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
"""Base class for libraries of algorithms handling :class:`.OptimizationProblem`.

A driver library aims to solve an :class:`.OptimizationProblem`
using a particular algorithm from a particular family of numerical methods.
This algorithm will be in charge of evaluating the objective and constraints
functions at different points of the design space, using the
:meth:`.BaseDriverLibrary.execute` method.
The most famous kinds of numerical methods to solve an optimization problem
are optimization algorithms and design of experiments (DOE). A DOE driver
browses the design space agnostically, i.e. without taking into
account the function evaluations. On the contrary, an optimization algorithm
uses this information to make the journey through design space
as relevant as possible in order to reach as soon as possible the optimum.
These families are implemented in :class:`.BaseDOELibrary`
and :class:`.BaseOptimizationLibrary`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Union

from numpy import ndarray
from strenum import StrEnum

from gemseo.algos._progress_bars.custom_tqdm_progress_bar import LOGGER as TQDM_LOGGER
from gemseo.algos._progress_bars.dummy_progress_bar import DummyProgressBar
from gemseo.algos._progress_bars.progress_bar import ProgressBar
from gemseo.algos._progress_bars.unsuffixed_progress_bar import UnsuffixedProgressBar
from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.base_algorithm_library import AlgorithmDescription
from gemseo.algos.base_algorithm_library import BaseAlgorithmLibrary
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FtolReached
from gemseo.algos.stop_criteria import FunctionIsNan
from gemseo.algos.stop_criteria import KKTReached
from gemseo.algos.stop_criteria import MaxIterReachedException
from gemseo.algos.stop_criteria import MaxTimeReached
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.algos.stop_criteria import XtolReached
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.enumeration import merge_enums
from gemseo.utils.logging_tools import OneLineLogging
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from gemseo.algos._progress_bars.base_progress_bar import BaseProgressBar
    from gemseo.algos.database import ListenerType
    from gemseo.algos.design_space import DesignSpace

DriverLibraryOptionType = Union[
    str, float, int, bool, list[str], ndarray, Iterable[CallbackType], Mapping[str, Any]
]
LOGGER = logging.getLogger(__name__)


@dataclass
class DriverDescription(AlgorithmDescription):
    """The description of a driver."""

    handle_integer_variables: bool = False
    """Whether the optimization algorithm handles integer variables."""

    require_gradient: bool = False
    """Whether the optimization algorithm requires the gradient."""


class BaseDriverLibrary(BaseAlgorithmLibrary):
    """Base class for libraries of algorithms handling :class:`.OptimizationProblem`.

    Lists available methods in the library for the proposed problem to be solved.

    To integrate an optimization package, inherit from this class and put your file in
    gemseo.algos.doe or gemseo.algo.opt packages.
    """

    ApproximationMode = ApproximationMode

    class _DifferentiationMethod(StrEnum):
        """The additional differentiation methods."""

        USER_GRAD = OptimizationProblem.DifferentiationMethod.USER_GRAD

    DifferentiationMethod = merge_enums(
        "DifferentiationMethod",
        StrEnum,
        ApproximationMode,
        _DifferentiationMethod,
        doc="The differentiation methods.",
    )

    # Option names
    _ACTIVATE_PROGRESS_BAR_OPTION_NAME = "activate_progress_bar"
    _EQ_TOLERANCE: Final[str] = "eq_tolerance"
    _EVAL_OBS_JAC_OPTION: Final[str] = "eval_obs_jac"
    _INEQ_TOLERANCE: Final[str] = "ineq_tolerance"
    _MAX_TIME: Final[str] = "max_time"
    _NORMALIZE_DESIGN_SPACE_OPTION: Final[str] = "normalize_design_space"
    _ROUND_INTS_OPTION: Final[str] = "round_ints"
    _USE_DATABASE_OPTION: Final[str] = "use_database"

    _NORMALIZE_DS: ClassVar[bool] = True

    _RESULT_CLASS: ClassVar[type[OptimizationResult]] = OptimizationResult
    """The class used to present the result of the optimization."""

    _SUPPORT_SPARSE_JACOBIAN: ClassVar[bool] = False
    """Whether the library support sparse Jacobians."""

    _COMMON_OPTIONS_GRAMMAR: ClassVar[JSONGrammar] = JSONGrammar(
        "DriverLibOptions",
        file_path=Path(__file__).parent / "driver_lib_options.json",
    )

    activate_progress_bar: bool = True
    """Whether to activate the progress bar in the optimization log."""

    _max_time: float
    """The maximum duration of the execution."""

    _start_time: float
    """The time at which the execution begins."""

    __log_problem: bool
    """Whether to log the definition and result of the problem."""

    __one_line_progress_bar: bool
    """Whether to log the progress bar on a single line."""

    __progress_bar: BaseProgressBar
    """The progress bar used during the execution."""

    __reset_iteration_counters: bool
    """Whether to reset the iteration counters of the OptimizationProblem before each
    execution."""

    _problem: OptimizationProblem
    """The optimization problem the driver library is bonded to."""

    __new_iter_listeners: set[ListenerType]
    """The functions to be called when a new iteration is stored to the database."""

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self._deactivate_progress_bar()
        self._start_time = 0.0
        self._max_time = 0.0
        self.__reset_iteration_counters = True
        self.__log_problem = True
        self.__one_line_progress_bar = False
        self.__new_iter_listeners = set()

    @classmethod
    def _get_unsuitability_reason(
        cls, algorithm_description: DriverDescription, problem: OptimizationProblem
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if reason or problem.design_space:
            return reason

        return _UnsuitabilityReason.EMPTY_DESIGN_SPACE

    def _deactivate_progress_bar(self) -> None:
        """Deactivate the progress bar."""
        self.__progress_bar = DummyProgressBar()

    def _init_iter_observer(
        self,
        problem: OptimizationProblem,
        max_iter: int,
        message: str = "",
    ) -> None:
        """Initialize the iteration observer.

        It will handle the stopping criterion and the logging of the progress bar.

        Args:
            max_iter: The maximum number of iterations.
            message: The message to display at the beginning of the progress bar status.

        Raises:
            ValueError: If ``max_iter`` is lower than one.
        """
        if max_iter < 1:
            msg = f"max_iter must be >=1, got {max_iter}"
            raise ValueError(msg)
        problem.evaluation_counter.maximum = max_iter
        problem.evaluation_counter.current = (
            0 if self.__reset_iteration_counters else problem.evaluation_counter.current
        )
        if self.activate_progress_bar:
            cls = ProgressBar if self.__log_problem else UnsuffixedProgressBar
            self.__progress_bar = cls(
                max_iter,
                problem,
                message,
            )
        else:
            self._deactivate_progress_bar()

        self._start_time = time()

    def _new_iteration_callback(self, x_vect: ndarray) -> None:
        """Iterate the progress bar, implement the stop criteria.

        Args:
            x_vect: The design variables values.

        Raises:
            MaxTimeReached: If the elapsed time is greater than the maximum
                execution time.
        """
        self.__progress_bar.set_objective_value(None, True)
        self.problem.evaluation_counter.current += 1
        if 0 < self._max_time < time() - self._start_time:
            raise MaxTimeReached

        self.__progress_bar.set_objective_value(x_vect)

    def _pre_run(
        self,
        problem: OptimizationProblem,
        **options: DriverLibraryOptionType,
    ) -> None:
        self._max_time = options.get(self._MAX_TIME, 0.0)

    def _post_run(
        self,
        problem: OptimizationProblem,
        result: OptimizationResult,
        max_design_space_dimension_to_log: int,
        **options: Any,
    ) -> None:
        """
        Args:
            max_design_space_dimension_to_log: The maximum dimension of a design space
                to be logged.
                If this number is higher than the dimension of the design space
                then the design space will not be logged.
        """  # noqa: D205, D212
        problem.solution = result
        if result.x_opt is not None:
            problem.design_space.set_current_value(result)

        if self.__log_problem:
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
            design_space.has_integer_variables()
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

    def execute(
        self,
        problem: OptimizationProblem,
        eval_obs_jac: bool = False,
        skip_int_check: bool = False,
        max_design_space_dimension_to_log: int = 40,
        **settings: DriverLibraryOptionType,
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
        self.problem = problem
        self._check_algorithm(problem)
        self._check_integer_handling(problem.design_space, skip_int_check)
        activate_progress_bar = settings.pop(
            self._ACTIVATE_PROGRESS_BAR_OPTION_NAME, None
        )
        if activate_progress_bar is not None:
            self.activate_progress_bar = activate_progress_bar

        use_one_line_progress_bar = settings.pop("use_one_line_progress_bar", False)

        self.__reset_iteration_counters = settings.pop("reset_iteration_counters", True)
        self.__log_problem = settings.pop("log_problem", True)

        options = self._update_algorithm_options(**settings)
        problem.check()
        problem.preprocess_functions(
            is_function_input_normalized=options.get(
                self._NORMALIZE_DESIGN_SPACE_OPTION, self._NORMALIZE_DS
            ),
            use_database=options.get(self._USE_DATABASE_OPTION, True),
            round_ints=options.get(self._ROUND_INTS_OPTION, True),
            eval_obs_jac=eval_obs_jac,
            support_sparse_jacobian=self._SUPPORT_SPARSE_JACOBIAN,
        )
        # A database contains both shared listeners
        # and listeners specific to a BaseDriverLibrary instance.
        # At execution,
        # a BaseDriverLibrary instance must be able
        # to list the listeners it has added to the database
        # in order to remove them at the end of the execution.
        listeners = []
        if problem.new_iter_observables:
            listeners.append(problem.new_iter_observables.evaluate)
        listeners.append(self._new_iteration_callback)
        for listener in listeners:
            if problem.database.add_new_iter_listener(listener):
                # The listener was not in the database.
                self.__new_iter_listeners.add(listener)

        if self.__log_problem:
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

            progress_bar_title = "Solving optimization problem with algorithm %s:"
        else:
            progress_bar_title = "Running the algorithm %s:"

        if self.activate_progress_bar:
            LOGGER.info(progress_bar_title, self._algo_name)

        with (
            OneLineLogging(TQDM_LOGGER) if use_one_line_progress_bar else nullcontext()
        ):
            # Term criteria such as max iter or max_time can be triggered in pre_run
            try:
                self._pre_run(problem, **options)
                result = self._run(problem, **options)
            except TerminationCriterion as termination_criterion:
                result = self._get_early_stopping_result(problem, termination_criterion)

        result.objective_name = problem.objective.name
        result.design_space = problem.design_space
        self.__progress_bar.finalize_iter_observer()
        self._clear_listeners(problem)
        self._post_run(
            problem,
            result,
            max_design_space_dimension_to_log,
            **options,
        )
        # Clear the state of _problem; the cache of the AlgoFactory can be used.
        self._problem = None
        return result

    def _clear_listeners(self, problem: OptimizationProblem) -> None:
        """Remove the listeners from the :attr:`.database`.

        Args:
            problem: The problem to be solved.
        """
        problem.database.clear_listeners(
            new_iter_listeners=self.__new_iter_listeners or None, store_listeners=None
        )
        self.__new_iter_listeners.clear()

    def _process_specific_option(self, options, option_key: str) -> None:
        if option_key == self._INEQ_TOLERANCE:
            self.problem.tolerances.inequality = options[option_key]
            del options[option_key]
        elif option_key == self._EQ_TOLERANCE:
            self.problem.tolerances.equality = options[option_key]
            del options[option_key]

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
            message = f"Maximum time reached: {self._max_time} seconds. "
        elif isinstance(termination_criterion, KKTReached):
            message = (
                "The KKT residual norm is smaller than the tolerance "
                "kkt_tol_abs or kkt_tol_rel. "
            )
        else:
            message = ""

        message += "GEMSEO stopped the driver."
        return self._get_optimum_from_database(problem, message)

    def _get_optimum_from_database(
        self, problem: OptimizationProblem, message=None, status=None
    ) -> OptimizationResult:
        """Return the optimization result from the database.

        Args:
            problem: The problem to be solved.
            message: The message associated with the termination criterion.
            status: The status associated with the termination criterion.
        """
        return self._RESULT_CLASS.from_optimization_problem(
            problem, message=message, status=status, optimizer_name=self._algo_name
        )
