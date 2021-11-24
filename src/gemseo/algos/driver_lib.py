# -*- coding: utf-8 -*-
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

"""
Driver library
==============

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

from __future__ import division, unicode_literals

import io
import logging
import string
from time import time
from typing import Callable, List, Optional, Union

import tqdm
from numpy import ndarray, ones_like, where, zeros_like
from tqdm.utils import _unicode, disp_len

from gemseo.algos.algo_lib import AlgoLib
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.stop_criteria import (
    DesvarIsNan,
    FtolReached,
    FunctionIsNan,
    MaxIterReachedException,
    MaxTimeReached,
    TerminationCriterion,
    XtolReached,
)

DriverLibOptionType = Union[str, float, int, bool, List[str], ndarray]
LOGGER = logging.getLogger(__name__)


class TqdmToLogger(io.StringIO):
    """Redirect tqdm output to the gemseo logger."""

    def write(self, buf):
        """Write buffer."""
        buf = buf.strip(string.whitespace)
        if buf:
            LOGGER.info(buf)


class ProgressBar(tqdm.tqdm):
    """Extend tqdm progress bar with better time units.

    Use hour, day or week for slower processes.
    """

    @classmethod
    def format_meter(cls, n, total, elapsed, **kwargs):
        if elapsed != 0.0:
            rate, unit = cls.__convert_rate(total, elapsed)
            kwargs["rate"] = rate
            kwargs["unit"] = unit
        meter = tqdm.tqdm.format_meter(n, total, elapsed, **kwargs)
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

        return rate, " it/{}".format(unit)

    def status_printer(
        self, file  # type: Union[io.TextIOWrapper, io.StringIO]
    ):  # type: (...) -> Callable[[str], None]
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

    REQUIRE_GRAD = "require_grad"
    HANDLE_EQ_CONS = "handle_equality_constraints"
    HANDLE_INEQ_CONS = "handle_inequality_constraints"
    POSITIVE_CONSTRAINTS = "positive_constraints"
    INEQ_TOLERANCE = "ineq_tolerance"
    EQ_TOLERANCE = "eq_tolerance"
    MAX_TIME = "max_time"
    USE_DATABASE_OPTION = "use_database"
    NORMALIZE_DESIGN_SPACE_OPTION = "normalize_design_space"
    ROUND_INTS_OPTION = "round_ints"
    WEBSITE = "website"
    DESCRIPTION = "description"
    MAX_DS_SIZE_PRINT = 40

    def __init__(self):
        """Constructor."""
        # Library settings and check
        super(DriverLib, self).__init__()
        self.__progress_bar = None
        self.__max_iter = 0
        self.__iter = 0
        self._start_time = None
        self._max_time = None
        self.__message = None

    def deactivate_progress_bar(self):  # type: (...) -> None
        """Deactivate the progress bar."""
        self.__progress_bar = None

    def init_iter_observer(
        self,
        max_iter,  # type: int
        message,  # type: str
    ):  # type: (...) -> None
        """Initialize the iteration observer.

        It will handle the stopping criterion and the logging of the progress bar.

        Args:
            max_iter: The maximum number of iterations.
            message: The message to display at the beginning.

        Raises:
            ValueError: If the `max_iter` is not greater than or equal to one.
        """
        if max_iter < 1:
            raise ValueError("max_iter must be >=1, got {}".format(max_iter))
        self.__max_iter = max_iter
        self.__iter = 0
        self.__message = message
        self.__progress_bar = ProgressBar(
            total=self.__max_iter,
            desc=self.__message,
            ascii=False,
            file=TqdmToLogger(),
        )
        self._start_time = time()
        self.problem.max_iter = max_iter

    def __set_progress_bar_objective_value(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> None
        """Set the objective value in the progress bar.

        Args:
            x_vect: The design variables values.
        """
        value = self.problem.database.get_f_of_x(self.problem.objective.name, x_vect)

        if value is not None:
            # if maximization problem: take the opposite
            if not self.problem.minimize_objective:
                value = -value

            self.__progress_bar.set_postfix(refresh=False, obj=value)
            self.__progress_bar.update()
        else:
            self.__progress_bar.update()

    def new_iteration_callback(
        self, x_vect=None  # type: Optional[ndarray]
    ):  # type: (...) -> None
        """Callback called at each new iteration, i.e. every time a design vector that
        is not already in the database is proposed by the optimizer.

        Iterate the progress bar, implement the stop criteria.

        Args:
            x_vect: The design variables values. If None, use the values of the
                last iteration.

        Raises:
            MaxTimeReached: If the elapsed time is greater than the maximum
                execution time.
        """
        # First check if the max_iter is reached and update the progress bar
        self.__iter += 1
        self.problem.current_iter = self.__iter

        if self._max_time > 0:
            delta_t = time() - self._start_time
            if delta_t > self._max_time:
                raise MaxTimeReached()

        if self.__progress_bar is not None:
            if x_vect is None:
                x_vect = self.problem.database.get_x_by_iter(-1)
            self.__set_progress_bar_objective_value(x_vect)

    def finalize_iter_observer(self):  # type: (...) -> None
        """Finalize the iteration observer."""
        if self.__progress_bar is not None:
            self.__progress_bar.close()

    def _pre_run(
        self,
        problem,  # type: OptimizationProblem
        algo_name,  # type: str
        **options  # type: DriverLibOptionType
    ):  # type: (...) -> None
        """To be overridden by subclasses. Specific method to be executed just before
        _run method call.

        Args:
            problem: The optimization problem.
            algo_name: The name of the algorithm.
            **options: The options of the algorithm,
                see the associated JSON file.
        """
        self._max_time = options.get(self.MAX_TIME, 0.0)

    def _post_run(self, problem, algo_name, result, **options):  # pylint: disable=W0613
        """To be overridden by subclasses Specific method to be executed just after _run
        method call.

        :param problem: the problem to be solved
        :param algo_name: name of the algorithm
        :param result: result of the run such as an OptimizationResult
        :param options: the options dict for the algorithm, see associated JSON file
        """
        LOGGER.info("%s", result)
        problem.solution = result
        if result.x_opt is not None:
            problem.design_space.set_current_x(result)
        if problem.design_space.dimension <= self.MAX_DS_SIZE_PRINT:
            LOGGER.info("%s", problem.design_space)

    def execute(self, problem, algo_name=None, **options):
        """Executes the driver.

        :param problem: the problem to be solved
        :param algo_name: name of the algorithm
            if None, use self.algo_name
            which may have been set by the factory (Default value = None)
        :param options: the options dict for the algorithm
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
        options = self._update_algorithm_options(**options)
        self.internal_algo_name = self.lib_dict[self.algo_name][self.INTERNAL_NAME]

        problem.check()
        problem.preprocess_functions(
            normalize=options.get(self.NORMALIZE_DESIGN_SPACE_OPTION, True),
            use_database=options.get(self.USE_DATABASE_OPTION, True),
            round_ints=options.get(self.ROUND_INTS_OPTION, True),
        )

        try:  # Term criteria such as max iter or max_time can be triggered in pre_run
            self._pre_run(problem, self.algo_name, **options)
            result = self._run(**options)
        except TerminationCriterion as error:
            result = self._termination_criterion_raised(error)
        self.finalize_iter_observer()
        self.problem.clear_listeners()
        self._post_run(problem, algo_name, result, **options)

        return result

    def _process_specific_option(self, options, option_key):
        """Process one option as a special treatment, at the begining of the general
        treatment and checks of _process_options.

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

        :param error: the obtained error from the algorithm
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
                message = "Maximum time reached : {} seconds.".format(self._max_time)
            message += " GEMSEO Stopped the driver"
        else:
            message = error.args[0]

        result = self.get_optimum_from_database(message)
        return result

    def get_optimum_from_database(self, message=None, status=None):
        """Retrieves the optimum from the database and builds an optimization result
        object from it.

        :param message: Default value = None)
        :param status: Default value = None)
        """
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
        if f_opt is not None and not problem.minimize_objective:
            f_opt = -f_opt
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
        )

    def _get_options(self, **options):
        """Retrieves the options of the library To be overloaded by subclasses Used to
        define default values for options using keyword arguments.

        :param options: options of the driver
        """
        raise NotImplementedError()

    def _run(self, **options):
        """Runs the algorithm, to be overloaded by subclasses.

        :param options: the options dict for the algorithm
        """
        raise NotImplementedError()

    def is_algo_requires_grad(self, algo_name):
        """Returns True if the algorithm requires a gradient evaluation.

        :param algo_name: name of the algorithm
        """
        lib_alg = self.lib_dict.get(algo_name, None)
        if lib_alg is None:
            raise ValueError("Algorithm {} is not available.".format(algo_name))

        return lib_alg.get(self.REQUIRE_GRAD, False)

    def get_x0_and_bounds_vects(self, normalize_ds):
        """Gets x0, bounds, normalized or not depending on algo options, all as numpy
        arrays.

        :param normalize_ds: if True, normalizes all input vars
               that are not integers, according to design space
               normalization policy
        :returns: x, lower bounds, upper bounds
        """
        dspace = self.problem.design_space
        l_b = dspace.get_lower_bounds()
        u_b = dspace.get_upper_bounds()

        # remove normalization from options for algo
        if normalize_ds:
            norm_dict = dspace.normalize
            norm_array = dspace.dict_to_array(norm_dict)
            xvec = self.problem.get_x0_normalized()
            l_b = where(norm_array, zeros_like(xvec), l_b)
            u_b = where(norm_array, ones_like(xvec), u_b)
        else:
            xvec = self.problem.design_space.get_current_x()

        return xvec, l_b, u_b

    def ensure_bounds(self, orig_func, normalize=True):
        """Project the design vector onto the design space before execution.

        :param orig_func: the original function
        :param normalize: if True, use the normalized design space
        :returns: the wrapped function
        """

        def wrapped_func(x_vect):
            x_proj = self.problem.design_space.project_into_bounds(x_vect, normalize)
            return orig_func(x_proj)

        return wrapped_func
