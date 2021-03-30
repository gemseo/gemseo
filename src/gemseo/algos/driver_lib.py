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
from __future__ import absolute_import, division, unicode_literals

import inspect
from builtins import str
from os.path import dirname, exists, join

from future import standard_library
from numpy import ones_like, where, zeros_like

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.algos.stop_criteria import (
    DesvarIsNan,
    FunctionIsNan,
    MaxIterReachedException,
    TerminationCriterion,
)
from gemseo.core.grammar import InvalidDataException
from gemseo.core.json_grammar import JSONGrammar
from gemseo.third_party.tqdm import Tqdm
from gemseo.utils.source_parsing import SourceParsing

standard_library.install_aliases()


from gemseo import LOGGER


class DriverLib(object):
    """Abstract class for DOE & optimization libraries interfaces

    Lists available methods in the library for the proposed
    problem to be solved

    To integrate an optimization package, inherit from this class
    and put your file in gemseo.algos.doe or gemseo.algo.opt packages


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
    REQUIRE_GRAD = "require_grad"
    HANDLE_EQ_CONS = "handle_equality_constraints"
    HANDLE_INEQ_CONS = "handle_inequality_constraints"
    POSITIVE_CONSTRAINTS = "positive_constraints"
    INEQ_TOLERANCE = "ineq_tolerance"
    EQ_TOLERANCE = "eq_tolerance"
    USE_DATABASE_OPTION = "use_database"
    NORMALIZE_DESIGN_SPACE_OPTION = "normalize_design_space"
    WEBSITE = "website"
    DESCRIPTION = "description"
    MAX_DS_SIZE_PRINT = 40

    def __init__(self):
        """
        Constructor
        """
        # Library settings and check
        self.lib_dict = {}
        self.algo_name = None
        self.internal_algo_name = None
        self.problem = None
        self.opt_grammar = None
        self._progress_bar = None

    def init_options_grammar(self, algo_name):
        """Initializes the options grammar

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

        self.opt_grammar = JSONGrammar(
            algo_name, schema_file=schema_file, grammar_type="input"
        )

        descr_dict = SourceParsing.get_options_doc(self.__class__._get_options)
        self.opt_grammar.add_description(descr_dict)
        return self.opt_grammar

    def init_progress_bar(self, max_iter, message):
        """Initializes the progress bar

        :param max_iter: maximum number of calls
        :param message: message to display at the beginning
        """
        if max_iter <= 0:
            raise ValueError("max_iter must be >=1, got " + str(max_iter))
        self._progress_bar = Tqdm(max_iter, desc=message, mininterval=0.1)

    def __get_signed_obj_val(self):
        """
        Gets the objective value with the right sign
        """
        value = self.problem.objective.last_eval
        # if maximization problem, take the opposite
        if value is not None and not self.problem.minimize_objective:
            value = -value
        return value

    @property
    def algorithms(self):
        """Return the available algorithms."""
        return list(self.lib_dict.keys())

    def close_progress_bar(self):
        """Closes the progress bar"""
        if self._progress_bar is not None:
            value = self.__get_signed_obj_val()
            self._progress_bar.close(value)

    def iterate_progress_bar(self):
        """Iterates the progress bar"""
        try:
            value = self.__get_signed_obj_val()
            self._progress_bar.next(value)
        except StopIteration:
            raise MaxIterReachedException

    def _pre_run(self, problem, algo_name, **options):
        """To be overriden by subclasses
        Specific method to be executed just before _run method call

        :param problem: the problem to be solved
        :param algo_name: name of the algorithm
        :param options: the options dict for the algorithm,
            see associated JSON file
        """

    def _post_run(self, problem, algo_name, result, **options):  # pylint: disable=W0613
        """To be overriden by subclasses
        Specific method to be executed just after _run method call

        :param problem: the problem to be solved
        :param algo_name: name of the algorithm
        :param result: result of the run such as an OptimizationResult
        :param options: the options dict for the algorithm,
            see associated JSON file
        """
        result.log_me()
        problem.solution = result
        if result.x_opt is not None:
            problem.design_space.set_current_x(result)
        if problem.design_space.dimension <= self.MAX_DS_SIZE_PRINT:
            problem.design_space.log_me()

    def _process_options(self, **options):
        """After _get_options is called, the options are
        converted to algorithm specific options, and checked

        :param options: driver options
        """

        for option_key in list(options.keys()):  # Copy keys on purpose
            # Remove extra options
            if not self.opt_grammar.is_data_name_existing(option_key):
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

    def execute(self, problem, algo_name=None, **options):
        """Executes the driver

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
        self.__check_algorithm(self.algo_name)
        self.init_options_grammar(self.algo_name)
        use_database = options.get(self.USE_DATABASE_OPTION, True)
        normalize = options.get(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        options = self._get_options(**options)
        self.internal_algo_name = self.lib_dict[self.algo_name][self.INTERNAL_NAME]
        problem.check()
        problem.preprocess_functions(
            normalize=normalize, use_database=use_database, round_ints=True
        )
        try:
            self._pre_run(problem, self.algo_name, **options)
            result = self._run(**options)
        except TerminationCriterion as error:
            result = self._termination_criterion_raised(error)
        finally:
            self.close_progress_bar()
            self.problem.clear_listeners()

        self._post_run(problem, algo_name, result, **options)

        return result

    def _termination_criterion_raised(self, error):  # pylint: disable=W0613
        """Retrieve the best known iterate when max iter has been reached

        :param error: the obtained error from the algorithm
        """
        if len(error.args) >= 1:
            message = error.args[0]
        else:
            message = ""
        if isinstance(error, MaxIterReachedException):
            message = "Maximum number of iterations reached, "
            message += "the solver has been stopped."
        elif isinstance(error, FunctionIsNan):
            message = "Function value or gradient or constraint is NaN, "
            message += "and problem.stop_if_nan is set to True. "
            message += "The solver has been stopped."
        elif isinstance(error, DesvarIsNan):
            message = "Design variables are NaN, "
            message += "The solver has been stopped."

        result = self.get_optimum_from_database(message)
        return result

    def get_optimum_from_database(self, message=None, status=None):
        """Retrieves the optimum from the database and
        builds an optimization result object from it

        :param message: Default value = None)
        :param status: Default value = None)

        """
        problem = self.problem
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
        """Retrieves the options of the library
        To be overloaded by subclasses
        Used to define default values for options using keyword arguments

        :param options: options of the driver
        """
        raise NotImplementedError()

    def _run(self, **options):
        """Runs the algorithm, to be overloaded by subclasses

        :param options: the options dict for the algorithm
        """
        raise NotImplementedError()

    def __check_algorithm(self, algo_name):
        """Check that algorithm required by user is available.
        Set optimization library
        Set algorithm name according to optimization library requirements
        :param algo_name : name of algorithm
        """
        if algo_name not in self.lib_dict:
            raise KeyError(
                "Requested optimization algorithm"
                + " %s is not in list of available algorithms %s"
                % (algo_name, list(self.lib_dict.keys()))
            )

    @staticmethod
    def _display_result(result):
        """Displays the optimization result

        :param result: the result to display
        """
        LOGGER.info("Algorithm execution finished, result is:")
        LOGGER.info(result)

    @staticmethod
    def is_algorithm_suited(algo_dict, problem):
        """Checks if the algorithm is suited to the problem
        according to its algo dict

        :param algo_dict: the algorithm characteristics
        :param problem: the opt_problem to be solved
        """
        raise NotImplementedError()

    def filter_adapted_algorithms(self, problem):
        """Filters the algorithms capable of solving the problem

        :param problem: the opt_problem to be solved
        :returns: the list of adapted algorithms names
        """
        available = []
        for algo_name, algo_dict in self.lib_dict.items():
            if self.is_algorithm_suited(algo_dict, problem):
                available.append(algo_name)
        return available

    def is_algo_requires_grad(self, algo_name):
        """
        Returns True if the algorithm requries a gradient evaluation

        :param algo_name: name of the algorithm
        """
        lib_alg = self.lib_dict.get(algo_name, None)
        if lib_alg is None:
            raise ValueError("Algorithm " + str(algo_name) + " is not available !")

        return lib_alg.get(self.REQUIRE_GRAD, False)

    def get_x0_and_bounds_vects(self, normalize_ds):
        """
        Gets x0, bounds, normalized or not depending
        on algo options, all as numpy arrays

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
