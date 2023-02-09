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
:meth:`.DriverLib.execute` method.
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

import io
import logging
import string
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Callable
from typing import ClassVar
from typing import List
from typing import Union

import tqdm
from numpy import ndarray
from numpy import ones
from numpy import where
from numpy import zeros
from tqdm.utils import _unicode
from tqdm.utils import disp_len

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.algo_lib import AlgoLib
from gemseo.algos.algo_lib import AlgorithmDescription
from gemseo.algos.design_space import DesignSpace
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
from gemseo.utils.python_compatibility import Final
from gemseo.utils.string_tools import MultiLineString

DriverLibOptionType = Union[str, float, int, bool, List[str], ndarray]
LOGGER = logging.getLogger(__name__)


class TqdmToLogger(io.StringIO):
    """Redirect tqdm output to the gemseo logger."""

    def write(self, buf):
        """Write buffer."""
        buf = buf.strip(string.whitespace)
        if buf:
            LOGGER.info(buf)


@dataclass
class DriverDescription(AlgorithmDescription):
    """The description of a driver."""

    handle_integer_variables: bool = False
    """Whether the optimization algorithm handles integer variables."""

    require_gradient: bool = False
    """Whether the optimization algorithm requires the gradient."""


class ProgressBar(tqdm.tqdm):
    """Extend tqdm progress bar with better time units.

    Use hour, day or week for slower processes.
    """

    @classmethod
    def format_meter(cls, n, total, elapsed, **kwargs):  # noqa: D102
        meter = tqdm.tqdm.format_meter(n, total, elapsed, **kwargs)
        if elapsed != 0.0:
            rate, unit = cls.__convert_rate(n, elapsed)
            lstr = meter.split(",")
            lstr[1] = f" {rate:5.2f}{unit}"
            meter = ",".join(lstr)
        # remove the unit suffix that is hard coded in tqdm
        return meter.replace("/s,", ",").replace("/s]", "]")

    @staticmethod
    def __convert_rate(total, elapsed):
        rps = float(total) / elapsed
        if rps >= 0:
            rate = rps
            unit = "sec"

        rpm = rps * 60.0
        if rpm < 60.0:
            rate = rpm
            unit = "min"

        rph = rpm * 60.0
        if rph < 60.0:
            rate = rph
            unit = "hour"

        rpd = rph * 24.0
        if rpd < 24.0:
            rate = rpd
            unit = "day"

        return rate, f" it/{unit}"

    def status_printer(
        self, file: io.TextIOWrapper | io.StringIO
    ) -> Callable[[str], None]:
        """Overload the status_printer method to avoid the use of closures.

        Args:
            file: Specifies where to output the progress messages.

        Returns:
            The function to print the status in the progress bar.
        """
        self._last_len = [0]
        return self._print_status

    def _print_status(self, s):
        len_s = disp_len(s)
        self.fp.write(
            _unicode("\r{}{}".format(s, (" " * max(self._last_len[0] - len_s, 0))))
        )
        self.fp.flush()
        self._last_len[0] = len_s

    def __getstate__(self):
        state = self.__dict__.copy()
        # A file-like stream cannot be pickled.
        del state["fp"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Set back the file-like stream to its state as done in tqdm.__init__.
        self.fp = tqdm.utils.DisableOnWriteError(TqdmToLogger(), tqdm_instance=self)


class DriverLib(AlgoLib):
    """Abstract class for DOE & optimization libraries interfaces.

    Lists available methods in the library for the proposed
    problem to be solved.

    To integrate an optimization package, inherit from this class
    and put your file in gemseo.algos.doe or gemseo.algo.opt packages.
    """

    USER_DEFINED_GRADIENT = OptimizationProblem.USER_GRAD
    COMPLEX_STEP_METHOD = OptimizationProblem.COMPLEX_STEP
    FINITE_DIFF_METHOD = OptimizationProblem.FINITE_DIFFERENCES

    DIFFERENTIATION_METHODS = [
        USER_DEFINED_GRADIENT,
        COMPLEX_STEP_METHOD,
        FINITE_DIFF_METHOD,
    ]

    INEQ_TOLERANCE = "ineq_tolerance"
    EQ_TOLERANCE = "eq_tolerance"
    MAX_TIME = "max_time"
    USE_DATABASE_OPTION = "use_database"
    NORMALIZE_DESIGN_SPACE_OPTION = "normalize_design_space"
    _NORMALIZE_DS = True
    ROUND_INTS_OPTION = "round_ints"
    EVAL_OBS_JAC_OPTION = "eval_obs_jac"
    MAX_DS_SIZE_PRINT = 40

    _ACTIVATE_PROGRESS_BAR_OPTION_NAME = "activate_progress_bar"
    """The name of the option to activate the progress bar in the optimization log."""

    activate_progress_bar: ClassVar[bool] = True
    """Whether to activate the progress bar in the optimization log."""

    _COMMON_OPTIONS_GRAMMAR: ClassVar[JSONGrammar] = JSONGrammar(
        "DriverLibOptions",
        schema_path=Path(__file__).parent / "driver_lib_options.json",
    )

    __RESET_ITERATION_COUNTERS_OPTION: Final[str] = "reset_iteration_counters"
    """The name of the option to reset the iteration counters of the OptimizationProblem
    before each execution."""

    __reset_iteration_counters: bool
    """Whether to reset the iteration counters of the OptimizationProblem before each
    execution."""

    def __init__(self):  # noqa:D107
        # Library settings and check
        super().__init__()
        self.__progress_bar = None
        self.__activate_progress_bar = self.activate_progress_bar
        self.__max_iter = 0
        self.__iter = 0
        self._start_time = None
        self._max_time = None
        self.__message = None
        self.__is_current_iteration_logged = True
        self.__reset_iteration_counters = True

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
        self.__progress_bar = None

    def init_iter_observer(
        self,
        max_iter: int,
        message: str = "...",
    ) -> None:
        """Initialize the iteration observer.

        It will handle the stopping criterion and the logging of the progress bar.

        Args:
            max_iter: The maximum number of iterations.
            message: The message to display at the beginning.

        Raises:
            ValueError: If ``max_iter`` is lower than one.
        """
        if max_iter < 1:
            raise ValueError(f"max_iter must be >=1, got {max_iter}")
        self.problem.max_iter = self.__max_iter = max_iter
        self.problem.current_iter = self.__iter = (
            0 if self.__reset_iteration_counters else self.problem.current_iter
        )
        self.__message = message
        if self.__activate_progress_bar:
            self.__progress_bar = ProgressBar(
                total=max_iter,
                desc=message,
                ascii=False,
                bar_format="{desc} {percentage:3.0f}%|{bar}{r_bar}",
                file=TqdmToLogger(),
            )
            self.__progress_bar.n = self.__iter
        else:
            self.deactivate_progress_bar()

        self._start_time = time()

    def __set_progress_bar_objective_value(self, x_vect: ndarray | None) -> None:
        """Set the objective value in the progress bar.

        Args:
            x_vect: The design variables values.
                If None, consider the objective at the last iteration.
        """
        if x_vect is None:
            value = self.problem.objective.last_eval
        else:
            value = self.problem.database.get_f_of_x(
                self.problem.objective.name, x_vect
            )

        if value is not None:
            self.__is_current_iteration_logged = True
            # if maximization problem: take the opposite
            if (
                not self.problem.minimize_objective
                and not self.problem.use_standardized_objective
            ):
                value = -value
            self.__progress_bar.n += 1
            if isinstance(value, ndarray):
                if len(value) == 1:
                    value = value[0]
            self.__progress_bar.set_postfix(refresh=True, obj=value)
            #
        else:
            if self.__is_current_iteration_logged:
                self.__is_current_iteration_logged = False
            else:
                self.__is_current_iteration_logged = True
                self.__progress_bar.n += 1
                self.__progress_bar.set_postfix(refresh=True, obj="Not evaluated")

    def new_iteration_callback(self, x_vect: ndarray | None = None) -> None:
        """Iterate the progress bar, implement the stop criteria.

        Args:
            x_vect: The design variables values. If None, use the values of the
                last iteration.

        Raises:
            MaxTimeReached: If the elapsed time is greater than the maximum
                execution time.
        """
        # First check if the max_iter is reached and update the progress bar
        if self.__progress_bar is not None and not self.__is_current_iteration_logged:
            self.__set_progress_bar_objective_value(
                self.problem.database.get_x_by_iter(self.problem.current_iter - 1)
            )
        self.__iter += 1
        self.problem.current_iter = self.__iter
        if self._max_time > 0:
            delta_t = time() - self._start_time
            if delta_t > self._max_time:
                raise MaxTimeReached()

        if self.__progress_bar is not None:
            self.__set_progress_bar_objective_value(x_vect)

    def finalize_iter_observer(self) -> None:
        """Finalize the iteration observer."""
        if self.__progress_bar is not None:
            if not self.__is_current_iteration_logged:
                self.__set_progress_bar_objective_value(
                    self.problem.database.get_x_by_iter(self.problem.current_iter - 1)
                )
            self.__progress_bar.leave = False
            self.__progress_bar.close()

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
        LOGGER.info("%s", problem)
        if problem.design_space.dimension <= self.MAX_DS_SIZE_PRINT:
            log = MultiLineString()
            log.indent()
            log.add("over the design space:")
            for line in str(problem.design_space).split("\n")[1:]:
                log.add(line)
            LOGGER.info("%s", log)
        LOGGER.info("Solving optimization problem with algorithm %s:", algo_name)

    def _post_run(self, problem, algo_name, result, **options):
        """To be overridden by subclasses.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            result: The result of the run, e.g. an :class:`.OptimizationResult`.
            **options: The options of the algorithm.
        """
        opt_result_str = result._strings
        LOGGER.info("%s", opt_result_str[0])
        if result.constraints_values:
            if result.is_feasible:
                LOGGER.info("%s", opt_result_str[1])
            else:
                LOGGER.warning("%s", opt_result_str[1])
        LOGGER.info("%s", opt_result_str[2])
        problem.solution = result
        if result.x_opt is not None:
            problem.design_space.set_current_value(result)
        if problem.design_space.dimension <= self.MAX_DS_SIZE_PRINT:
            log = MultiLineString()
            log.indent()
            log.indent()
            for line in str(problem.design_space).split("\n"):
                log.add(line)
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
            else:
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
                If None, use the algo_name attribute
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

        self.__reset_iteration_counters = options.pop(
            self.__RESET_ITERATION_COUNTERS_OPTION, True
        )

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
        )
        problem.database.add_new_iter_listener(problem.execute_observables_callback)
        problem.database.add_new_iter_listener(self.new_iteration_callback)
        try:  # Term criteria such as max iter or max_time can be triggered in pre_run
            self._pre_run(problem, self.algo_name, **options)
            result = self._run(**options)
        except TerminationCriterion as error:
            result = self._termination_criterion_raised(error)
        self.finalize_iter_observer()
        problem.database.clear_listeners()
        self._post_run(problem, algo_name, result, **options)
        return result

    def _process_specific_option(self, options, option_key):
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

    def _termination_criterion_raised(self, error):  # pylint: disable=W0613
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

        result = self.get_optimum_from_database(message)
        return result

    def get_optimum_from_database(self, message=None, status=None):
        """Retrieve the optimum from the database and build an optimization."""
        problem = self.problem
        if len(problem.database) == 0:
            return OptimizationResult(
                optimizer_name=self.algo_name,
                message=message,
                status=status,
                n_obj_call=0,
            )
        x_0 = problem.database.get_x_by_iter(0)
        # compute the best feasible or infeasible point
        f_opt, x_opt, is_feas, c_opt, c_opt_grad = problem.get_optimum()
        if (
            f_opt is not None
            and not problem.minimize_objective
            and not problem.use_standardized_objective
        ):
            f_opt = -f_opt

        if x_opt is None:
            optimum_index = None
        else:
            optimum_index = problem.database.get_index_of(x_opt)

        return OptimizationResult(
            x_0=x_0,
            x_opt=x_opt,
            f_opt=f_opt,
            optimizer_name=self.algo_name,
            message=message,
            status=status,
            n_obj_call=problem.objective.n_calls,
            is_feasible=is_feas,
            constraints_values=c_opt,
            constraints_grad=c_opt_grad,
            optimum_index=optimum_index,
        )

    # TODO: API: rename to requires_gradient?
    def is_algo_requires_grad(self, algo_name):
        """Returns True if the algorithm requires a gradient evaluation.

        Args:
            algo_name: The name of the algorithm.
        """
        if algo_name not in self.descriptions:
            raise ValueError(f"Algorithm {algo_name} is not available.")

        return self.descriptions[algo_name].require_gradient

    def get_x0_and_bounds_vects(self, normalize_ds):
        """Return x0 and bounds.

        Args:
            normalize_ds: Whether to normalize the input variables
                that are not integers,
                according to the normalization policy of the design space.

        Returns:
            The current value, the lower bounds and the upper bounds.
        """
        design_space = self.problem.design_space
        l_b = design_space.get_lower_bounds()
        u_b = design_space.get_upper_bounds()

        # remove normalization from options for algo
        if normalize_ds:
            norm_array = design_space.dict_to_array(design_space.normalize)
            l_b = where(norm_array, zeros(norm_array.shape), l_b)
            u_b = where(norm_array, ones(norm_array.shape), u_b)
            current_x = self.problem.get_x0_normalized(cast_to_real=True)
        else:
            current_x = self.problem.design_space.get_current_value(
                complex_to_real=True
            )

        return current_x, l_b, u_b

    def ensure_bounds(self, orig_func, normalize=True):
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
