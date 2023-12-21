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
"""Driver library.

A driver library aims to solve an :class:`.OptimizationProblem`
using a particular algorithm from a particular family of numerical methods.
This algorithm will be in charge of evaluating the objective and constraints
functions at different points of the design space, using the
:meth:`.DriverLibrary.execute` method.
The most famous kinds of numerical methods to solve an optimization problem
are optimization algorithms and design of experiments (DOE). A DOE driver
browses the design space agnostically, i.e. without taking into
account the function evaluations. On the contrary, an optimization algorithm
uses this information to make the journey through design space
as relevant as possible in order to reach as soon as possible the optimum.
These families are implemented in :class:`.DOELibrary`
and :class:`.OptimizationLibrary`.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import Union
from typing import overload

from numpy import ndarray
from strenum import StrEnum

from gemseo.algos._progress_bars.custom_tqdm_progress_bar import LOGGER as TQDM_LOGGER
from gemseo.algos._progress_bars.dummy_progress_bar import DummyProgressBar
from gemseo.algos._progress_bars.progress_bar import ProgressBar
from gemseo.algos._progress_bars.unsuffixed_progress_bar import UnsuffixedProgressBar
from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.algorithm_library import AlgorithmDescription
from gemseo.algos.algorithm_library import AlgorithmLibrary
from gemseo.algos.first_order_stop_criteria import KKTReached
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FtolReached
from gemseo.algos.stop_criteria import FunctionIsNan
from gemseo.algos.stop_criteria import MaxIterReachedException
from gemseo.algos.stop_criteria import MaxTimeReached
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.algos.stop_criteria import XtolReached
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.enumeration import merge_enums
from gemseo.utils.logging_tools import OneLineLogging
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from gemseo.algos._progress_bars.base_progress_bar import BaseProgressBar
    from gemseo.algos.design_space import DesignSpace

DriverLibOptionType = Union[str, float, int, bool, list[str], ndarray]
LOGGER = logging.getLogger(__name__)


@dataclass
class DriverDescription(AlgorithmDescription):
    """The description of a driver."""

    handle_integer_variables: bool = False
    """Whether the optimization algorithm handles integer variables."""

    require_gradient: bool = False
    """Whether the optimization algorithm requires the gradient."""


class DriverLibrary(AlgorithmLibrary):
    """Abstract class for driver library interfaces.

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

    INEQ_TOLERANCE = "ineq_tolerance"
    EQ_TOLERANCE = "eq_tolerance"
    MAX_TIME = "max_time"
    USE_DATABASE_OPTION = "use_database"
    NORMALIZE_DESIGN_SPACE_OPTION = "normalize_design_space"
    _NORMALIZE_DS = True
    ROUND_INTS_OPTION = "round_ints"
    EVAL_OBS_JAC_OPTION = "eval_obs_jac"
    MAX_DS_SIZE_PRINT = 40

    _SUPPORT_SPARSE_JACOBIAN: ClassVar[bool] = False
    """Whether the library support sparse Jacobians."""

    __USE_ONE_LINE_PROGRESS_BAR: Final[str] = "use_one_line_progress_bar"
    """The name of the option to use a one line progress bar."""

    USE_ONE_LINE_PROGRESS_BAR: ClassVar[bool] = False
    """Whether to use a one line progress bar."""

    _ACTIVATE_PROGRESS_BAR_OPTION_NAME = "activate_progress_bar"
    """The name of the option to activate the progress bar in the optimization log."""

    _COMMON_OPTIONS_GRAMMAR: ClassVar[JSONGrammar] = JSONGrammar(
        "DriverLibOptions",
        file_path=Path(__file__).parent / "driver_lib_options.json",
    )

    __LOG_PROBLEM: Final[str] = "log_problem"
    """The name of the option to log the definition and result of the problem."""

    __RESET_ITERATION_COUNTERS_OPTION: Final[str] = "reset_iteration_counters"
    """The name of the option to reset the iteration counters of the OptimizationProblem
    before each execution."""

    activate_progress_bar: ClassVar[bool] = True
    """Whether to activate the progress bar in the optimization log."""

    # TODO: API: use 0.0 instead of None
    _max_time: float | None
    """The maximum duration of the execution."""

    # TODO: API: use 0.0 instead of None
    _start_time: float | None
    """The time at which the execution begins."""

    __iter: int
    """The current iteration."""

    __log_problem: bool
    """Whether to log the definition and result of the problem."""

    __max_iter: int
    """The maximum number of iterations."""

    __one_line_progress_bar: bool
    """Whether to log the progress bar on a single line."""

    __progress_bar: BaseProgressBar
    """The progress bar used during the execution."""

    __reset_iteration_counters: bool
    """Whether to reset the iteration counters of the OptimizationProblem before each
    execution."""

    problem: OptimizationProblem
    """The optimization problem the driver library is bonded to."""

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        self.deactivate_progress_bar()
        self.__activate_progress_bar = self.activate_progress_bar
        self.__max_iter = 0
        self.__iter = 0
        self._start_time = None
        self._max_time = None
        self.__reset_iteration_counters = True
        self.__log_problem = True
        self.__one_line_progress_bar = False

    @classmethod
    def _get_unsuitability_reason(
        cls, algorithm_description: DriverDescription, problem: OptimizationProblem
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if reason or problem.design_space:
            return reason

        return _UnsuitabilityReason.EMPTY_DESIGN_SPACE

    def deactivate_progress_bar(self) -> None:
        """Deactivate the progress bar."""
        self.__progress_bar = DummyProgressBar()

    def init_iter_observer(
        self,
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
            raise ValueError(f"max_iter must be >=1, got {max_iter}")
        self.problem.max_iter = self.__max_iter = max_iter
        self.problem.current_iter = self.__iter = (
            0 if self.__reset_iteration_counters else self.problem.current_iter
        )
        if self.__activate_progress_bar:
            cls = ProgressBar if self.__log_problem else UnsuffixedProgressBar
            self.__progress_bar = cls(
                max_iter,
                self.__iter,
                self.problem,
                message,
            )
        else:
            self.deactivate_progress_bar()

        self._start_time = time()

    def new_iteration_callback(self, x_vect: ndarray | None = None) -> None:
        """Iterate the progress bar, implement the stop criteria.

        Args:
            x_vect: The design variables values. If ``None``, use the values of the
                last iteration.

        Raises:
            MaxTimeReached: If the elapsed time is greater than the maximum
                execution time.
        """
        self.__progress_bar.set_objective_value(None, True)
        self.__iter += 1
        self.problem.current_iter = self.__iter
        if 0 < self._max_time < time() - self._start_time:
            raise MaxTimeReached

        self.__progress_bar.set_objective_value(x_vect)

    def finalize_iter_observer(self) -> None:
        """Finalize the iteration observer."""
        self.__progress_bar.finalize_iter_observer()

    def _pre_run(
        self,
        problem: OptimizationProblem,
        algo_name: str,
        **options: DriverLibOptionType,
    ) -> None:
        """To be overridden by subclasses.

        Specific method to be executed just before _run method call.

        Args:
            problem: The optimization problem.
            algo_name: The name of the algorithm.
            **options: The options of the algorithm,
                see the associated JSON file.
        """
        self._max_time = options.get(self.MAX_TIME, 0.0)

    def _post_run(
        self,
        problem: OptimizationProblem,
        algo_name: str,
        result: OptimizationResult,
        **options: Any,
    ) -> None:
        """To be overridden by subclasses.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            result: The result of the run.
            **options: The options of the algorithm.
        """
        problem.solution = result
        if result.x_opt is not None:
            problem.design_space.set_current_value(result)

        if self.__log_problem:
            opt_result_str = result._strings
            LOGGER.info("%s", opt_result_str[0])
            if result.constraint_values:
                if result.is_feasible:
                    LOGGER.info("%s", opt_result_str[1])
                else:
                    LOGGER.warning("%s", opt_result_str[1])

            LOGGER.info("%s", opt_result_str[2])
            if problem.design_space.dimension <= self.MAX_DS_SIZE_PRINT:
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
            and not self.descriptions[self.algo_name].handle_integer_variables
        ):
            if not force_execution:
                raise ValueError(
                    "Algorithm {} is not adapted to the problem, it does not handle "
                    "integer variables.\n"
                    "Execution may be forced setting the 'skip_int_check' "
                    "argument to 'True'.".format(self.algo_name)
                )

            LOGGER.warning(
                "Forcing the execution of an algorithm that does not handle "
                "integer variables."
            )

    def execute(
        self,
        problem: OptimizationProblem,
        algo_name: str | None = None,
        eval_obs_jac: bool = False,
        skip_int_check: bool = False,
        **options: DriverLibOptionType,
    ) -> OptimizationResult:
        """Execute the driver.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
                If ``None``, use the algo_name attribute
                which may have been set by the factory.
            eval_obs_jac: Whether to evaluate the Jacobian of the observables.
            skip_int_check: Whether to skip the integer variable handling check
                of the selected algorithm.
            **options: The options for the algorithm.

        Returns:
            The optimization result.

        Raises:
            ValueError: If `algo_name` was not either set by the factory or given
                as an argument.
        """
        self.problem = problem
        if algo_name is not None:
            self.algo_name = algo_name

        if self.algo_name is None:
            raise ValueError(
                "Algorithm name must be either passed as "
                "argument or set by the attribute 'algo_name'."
            )

        self._check_algorithm(self.algo_name, problem)
        self._check_integer_handling(problem.design_space, skip_int_check)
        activate_progress_bar = options.pop(
            self._ACTIVATE_PROGRESS_BAR_OPTION_NAME, None
        )
        if activate_progress_bar is not None:
            self.__activate_progress_bar = activate_progress_bar

        use_one_line_progress_bar = options.pop(
            self.__USE_ONE_LINE_PROGRESS_BAR, self.USE_ONE_LINE_PROGRESS_BAR
        )

        self.__reset_iteration_counters = options.pop(
            self.__RESET_ITERATION_COUNTERS_OPTION, True
        )
        self.__log_problem = options.pop(self.__LOG_PROBLEM, True)

        options = self._update_algorithm_options(**options)
        self.internal_algo_name = self.descriptions[
            self.algo_name
        ].internal_algorithm_name

        problem.check()
        problem.preprocess_functions(
            is_function_input_normalized=options.get(
                self.NORMALIZE_DESIGN_SPACE_OPTION, self._NORMALIZE_DS
            ),
            use_database=options.get(self.USE_DATABASE_OPTION, True),
            round_ints=options.get(self.ROUND_INTS_OPTION, True),
            eval_obs_jac=eval_obs_jac,
            support_sparse_jacobian=self._SUPPORT_SPARSE_JACOBIAN,
        )
        problem.database.add_new_iter_listener(problem.execute_observables_callback)
        problem.database.add_new_iter_listener(self.new_iteration_callback)
        if self.__log_problem:
            LOGGER.info("%s", problem)
            if problem.design_space.dimension <= self.MAX_DS_SIZE_PRINT:
                log = MultiLineString()
                log.indent()
                log.add("over the design space:")
                log.indent()
                for line in str(problem.design_space).split("\n")[1:]:
                    log.add(line)
                log.dedent()
                LOGGER.info("%s", log)
                LOGGER.info(
                    "Solving optimization problem with algorithm %s:",
                    self.algo_name,
                )
        else:
            LOGGER.info("Running the algorithm %s:", self.algo_name)

        with OneLineLogging(
            TQDM_LOGGER
        ) if use_one_line_progress_bar else nullcontext():
            # Term criteria such as max iter or max_time can be triggered in pre_run
            try:
                self._pre_run(problem, self.algo_name, **options)
                result = self._run(**options)
            except TerminationCriterion as error:
                result = self._termination_criterion_raised(error)

        result.objective_name = problem.objective.name
        result.design_space = problem.design_space
        self.finalize_iter_observer()
        problem.database.clear_listeners()
        self._post_run(problem, self.algo_name, result, **options)
        return result

    def _process_specific_option(self, options, option_key: str) -> None:
        """Process one option as a special treatment.

        Args:
            options: The options as preprocessed by _process_options.
            option_key: The current option key to process.
        """
        if option_key == self.INEQ_TOLERANCE:
            self.problem.ineq_tolerance = options[option_key]
            del options[option_key]
        elif option_key == self.EQ_TOLERANCE:
            self.problem.eq_tolerance = options[option_key]
            del options[option_key]

    def _termination_criterion_raised(
        self, error: TerminationCriterion
    ) -> OptimizationResult:  # pylint: disable=W0613
        """Retrieve the best known iterate when max iter has been reached.

        Args:
            error: The obtained error from the algorithm.
        """
        if isinstance(error, TerminationCriterion):
            message = ""
            if isinstance(error, MaxIterReachedException):
                message = "Maximum number of iterations reached."
            elif isinstance(error, FunctionIsNan):
                message = "Function value or gradient or constraint is NaN, "
                message += "and problem.stop_if_nan is set to True."
            elif isinstance(error, DesvarIsNan):
                message = "Design variables are NaN."
            elif isinstance(error, XtolReached):
                message = "Successive iterates of the design variables "
                message += "are closer than xtol_rel or xtol_abs."
            elif isinstance(error, FtolReached):
                message = "Successive iterates of the objective function "
                message += "are closer than ftol_rel or ftol_abs."
            elif isinstance(error, MaxTimeReached):
                message = f"Maximum time reached: {self._max_time} seconds."
            elif isinstance(error, KKTReached):
                message = (
                    "The KKT residual norm is smaller than the tolerance "
                    "kkt_tol_abs or kkt_tol_rel."
                )
            message += " GEMSEO Stopped the driver"
        else:
            message = error.args[0]

        return self.get_optimum_from_database(message)

    def get_optimum_from_database(
        self, message=None, status=None
    ) -> OptimizationResult:
        """Return the optimization result from the database."""
        return OptimizationResult.from_optimization_problem(
            self.problem, message=message, status=status, optimizer_name=self.algo_name
        )

    def requires_gradient(self, driver_name: str) -> bool:
        """Check if a driver requires the gradient.

        Args:
            driver_name: The name of the driver.

        Returns:
            Whether the driver requires the gradient.
        """
        if driver_name not in self.descriptions:
            raise ValueError(f"Algorithm {driver_name} is not available.")

        return self.descriptions[driver_name].require_gradient

    @overload
    def get_x0_and_bounds_vects(
        self, normalize_ds: bool, as_dict: Literal[False] = False
    ) -> tuple[ndarray, ndarray, ndarray]: ...

    @overload
    def get_x0_and_bounds_vects(
        self, normalize_ds: bool, as_dict: Literal[True] = False
    ) -> tuple[dict[str, ndarray], dict[str, ndarray], dict[str, ndarray]]: ...

    # TODO: API: remove "_vects" from the following method name
    # TODO: return the design space to be used by the solver instead of a tuple
    def get_x0_and_bounds_vects(
        self, normalize_ds: bool, as_dict: bool = False
    ) -> (
        tuple[ndarray, ndarray, ndarray]
        | tuple[dict[str, ndarray], dict[str, ndarray], dict[str, ndarray]]
    ):
        """Return the initial design variable values and their lower and upper bounds.

        Args:
            normalize_ds: Whether to normalize the design variables.
            as_dict: Whether to return dictionaries instead of NumPy arrays.

        Returns:
            The initial values of the design variables,
            their lower bounds,
            and their upper bounds.
        """
        space = self.problem.design_space
        if not normalize_ds:
            return (
                space.get_current_value(None, True, as_dict),
                space.get_lower_bounds(None, as_dict),
                space.get_upper_bounds(None, as_dict),
            )

        current_value = self.problem.get_x0_normalized(True, as_dict)
        lower_bounds = space.normalize_vect(space.get_lower_bounds())
        upper_bounds = space.normalize_vect(space.get_upper_bounds())
        if not as_dict:
            return current_value, lower_bounds, upper_bounds

        return (
            current_value,
            space.array_to_dict(lower_bounds),
            space.array_to_dict(upper_bounds),
        )

    def ensure_bounds(self, orig_func, normalize: bool = True):
        """Project the design vector onto the design space before execution.

        Args:
            orig_func: The original function.
            normalize: Whether to use the normalized design space.

        Returns:
            A function calling the original function
            with the input data projected onto the design space.
        """

        def wrapped_func(x_vect):
            x_proj = self.problem.design_space.project_into_bounds(x_vect, normalize)
            return orig_func(x_proj)

        return wrapped_func
