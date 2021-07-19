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

A driver library aims to solve an :class:`.DriverLib`
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

import inspect
import io
import logging
import string
from os.path import dirname, exists, join
from time import time

import tqdm
from numpy import ones_like, where, zeros_like

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
from gemseo.core.grammar import InvalidDataException
from gemseo.core.json_grammar import JSONGrammar
from gemseo.utils.source_parsing import get_options_doc

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


class DriverLib(object):
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

    LIB = "lib"
    INTERNAL_NAME = "internal_algo_name"
    OPTIONS_DIR = "options"
    OPTIONS_MAP = {}
    PROBLEM_TYPE = "problem_type"
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
        self.lib_dict = {}
        self.algo_name = None
        self.internal_algo_name = None
        self.problem = None
        self.opt_grammar = None
        self.__progress_bar = None
        self.__max_iter = 0
        self.__iter = 0
        self._start_time = None
        self._max_time = None

    def init_options_grammar(self, algo_name):
        """Initializes the options grammar.

        :param algo_name: name of the algorithm
        """
        # Try algo name convention which has the prioryty over
        # library options
        basename = algo_name.upper() + "_options.json"
        lib_dir = inspect.getfile(self.__class__)
        opt_dir = join(dirname(lib_dir), self.OPTIONS_DIR)
        algo_schema_file = join(opt_dir, basename)
        if exists(algo_schema_file):
            schema_file = algo_schema_file
        else:
            # Try to load library options convention by default
            basename = self.__class__.__name__.upper() + "_options.json"
            lib_schema_file = join(opt_dir, basename)

            if exists(lib_schema_file):
                schema_file = lib_schema_file
            else:
                msg = "Options grammar file " + algo_schema_file
                msg += " for algorithm: " + algo_name
                msg += " not found. And library options grammar file "
                msg += lib_schema_file + " not found either."
                raise ValueError(msg)

        descr_dict = get_options_doc(self.__class__._get_options)
        self.opt_grammar = JSONGrammar(
            algo_name, schema_file=schema_file, descriptions=descr_dict
        )

        return self.opt_grammar

    @property
    def algorithms(self):
        """Return the available algorithms."""
        return list(self.lib_dict.keys())

    def init_iter_observer(self, max_iter, message):
        """Initialize the iteration observer.

        It will handle the stopping criterion and the logging of the progress bar.

        :param max_iter: maximum number of calls
        :param message: message to display at the beginning
        """
        if max_iter < 1:
            raise ValueError("max_iter must be >=1, got {}".format(max_iter))
        self.__max_iter = max_iter
        self.__iter = len(self.problem.database)
        self.__progress_bar = ProgressBar(
            total=self.__max_iter,
            desc=message,
            ascii=False,
            file=TqdmToLogger(),
        )

    def __set_progress_bar_objective_value(self):
        value = self.problem.objective.last_eval
        if value is not None:
            # if maximization problem: take the opposite
            if not self.problem.minimize_objective:
                value = -value
            self.__progress_bar.set_postfix(refresh=False, obj=value)

    def new_iteration_callback(self):
        """Callback called at each new iteration, ie every time a design vector that is
        not already in the database is proposed by the optimizer.

        Iterates the progress bar, implements the stop criteria.
        """
        # First check if the max_iter is reached and update the progress bar
        self.__iter += 1
        if self.__iter > self.__max_iter:
            raise MaxIterReachedException()
        if self._max_time > 0:
            delta_t = time() - self._start_time
            if delta_t > self._max_time:
                raise MaxTimeReached()

        self.__set_progress_bar_objective_value()
        self.__progress_bar.update()

    def finalize_iter_observer(self):
        """Finalize the iteration observer."""
        self.__set_progress_bar_objective_value()
        self.__progress_bar.close()

    def _pre_run(self, problem, algo_name, **options):
        """To be overridden by subclasses Specific method to be executed just before
        _run method call.

        :param problem: the problem to be solved
        :param algo_name: name of the algorithm
        :param options: the options dict for the algorithm, see associated JSON file
        """
        self._start_time = time()
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

    def driver_has_option(self, option_key):
        """Checks if the option key exists.

        :param option_key: the name of the option
        :return: True if the option is in the grammar
        """
        return self.opt_grammar.is_data_name_existing(option_key)

    def _process_options(self, **options):
        """After _get_options is called, the options are converted to algorithm specific
        options, and checked.

        :param options: driver options
        """
        for option_key in list(options.keys()):  # Copy keys on purpose
            # Remove extra options added in the _get_option method of the
            # driver
            if not self.driver_has_option(option_key):
                del options[option_key]
            elif option_key == self.INEQ_TOLERANCE:
                self.problem.ineq_tolerance = options[option_key]
                del options[option_key]
            elif option_key == self.EQ_TOLERANCE:
                self.problem.eq_tolerance = options[option_key]
                del options[option_key]
            elif options[option_key] is None:
                del options[option_key]

        try:
            self.opt_grammar.load_data(options)
        except InvalidDataException:
            raise ValueError("Invalid options for algorithm " + self.opt_grammar.name)
        for option_key in list(options.keys()):  # Copy keys on purpose
            lib_option_key = self.OPTIONS_MAP.get(option_key)
            # Overload with specific keys
            if lib_option_key is not None:
                options[lib_option_key] = options[option_key]
                if lib_option_key != option_key:
                    del options[option_key]

        return options

    def _check_ignored_options(self, options):
        """Check that the user did not passed options that do not exist for this driver.

        Raises a warning if it is the case
        :param options: options dict
        """
        for option_key in options:
            if not self.driver_has_option(option_key):
                msg = "Driver " + self.algo_name + " has no option " + option_key
                msg += ", option is ignored !"
                LOGGER.warning(msg)

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
                + "argument or set by the attribute self.algo_name"
            )
        self.__check_algorithm(self.algo_name, problem)
        self.init_options_grammar(self.algo_name)
        use_database = options.get(self.USE_DATABASE_OPTION, True)
        normalize = options.get(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        round_ints = options.get(self.ROUND_INTS_OPTION, True)

        self._check_ignored_options(options)
        options = self._get_options(**options)
        self.internal_algo_name = self.lib_dict[self.algo_name][self.INTERNAL_NAME]
        problem.check()
        problem.preprocess_functions(
            normalize=normalize, use_database=use_database, round_ints=round_ints
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

    def __check_algorithm(self, algo_name, problem):
        """Checks that algorithm required by user is available and adapted to the
        problem. Set optimization library Set algorithm name according to optimization
        library requirements.

        :param algo_name : name of algorithm
        :type algo_name: str
        :param problem: optimization problem
        :type problem: OptimizationProblem
        """
        # Check that the algorithm is available
        if algo_name not in self.lib_dict:
            raise KeyError(
                "Requested optimization algorithm"
                + " %s is not in list of available algorithms %s"
                % (algo_name, list(self.lib_dict.keys()))
            )

        # Check that the algorithm is suited to the problem
        algo_dict = self.lib_dict[self.algo_name]
        if not self.is_algorithm_suited(algo_dict, problem):
            raise ValueError(
                "Algorithm {} is not adapted to the problem.".format(algo_name)
            )

    @staticmethod
    def _display_result(result):
        """Displays the optimization result.

        :param result: the result to display
        """
        LOGGER.info("Algorithm execution finished, result is:")
        LOGGER.info(result)

    @staticmethod
    def is_algorithm_suited(algo_dict, problem):
        """Checks if the algorithm is suited to the problem according to its algo dict.

        :param algo_dict: the algorithm characteristics
        :param problem: the opt_problem to be solved
        """
        raise NotImplementedError()

    def filter_adapted_algorithms(self, problem):
        """Filters the algorithms capable of solving the problem.

        :param problem: the opt_problem to be solved
        :returns: the list of adapted algorithms names
        """
        available = []
        for algo_name, algo_dict in self.lib_dict.items():
            if self.is_algorithm_suited(algo_dict, problem):
                available.append(algo_name)
        return available

    def is_algo_requires_grad(self, algo_name):
        """Returns True if the algorithm requires a gradient evaluation.

        :param algo_name: name of the algorithm
        """
        lib_alg = self.lib_dict.get(algo_name, None)
        if lib_alg is None:
            raise ValueError("Algorithm " + str(algo_name) + " is not available !")

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
