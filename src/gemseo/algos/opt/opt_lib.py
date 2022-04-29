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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Damien Guenot
#        :author: Francois Gallard, refactoring
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Optimization library wrappers base class."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from numpy import ndarray

from gemseo.algos.driver_lib import DriverDescription
from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.stop_criteria import FtolReached
from gemseo.algos.stop_criteria import is_f_tol_reached
from gemseo.algos.stop_criteria import is_x_tol_reached
from gemseo.algos.stop_criteria import XtolReached

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationAlgorithmDescription(DriverDescription):
    """The description of an optimization algorithm."""

    handle_equality_constraints: bool = False
    """Whether the optimization algorithm handles equality constraints."""

    handle_inequality_constraints: bool = False
    """Whether the optimization algorithm handles inequality constraints."""

    handle_multiobjective: bool = False
    """Whether the optimization algorithm handles multiple objectives."""

    positive_constraints: bool = False
    """Whether the optimization algorithm requires positive constraints."""

    problem_type: str = OptimizationProblem.NON_LINEAR_PB
    """The type of problem (see :attr:`.OptimizationProblem.AVAILABLE_PB_TYPES`)."""


class OptimizationLibrary(DriverLib):
    """Optimization library base class See DriverLib."""

    MAX_ITER = "max_iter"
    F_TOL_REL = "ftol_rel"
    F_TOL_ABS = "ftol_abs"
    X_TOL_REL = "xtol_rel"
    X_TOL_ABS = "xtol_abs"
    STOP_CRIT_NX = "stop_crit_n_x"
    # Maximum step for the line search
    LS_STEP_SIZE_MAX = "max_ls_step_size"
    # Maximum number of line search steps (per iteration).
    LS_STEP_NB_MAX = "max_ls_step_nb"
    MAX_FUN_EVAL = "max_fun_eval"
    MAX_TIME = "max_time"
    PG_TOL = "pg_tol"
    VERBOSE = "verbose"

    def __init__(self):
        super(OptimizationLibrary, self).__init__()
        self._ftol_rel = 0.0
        self._ftol_abs = 0.0
        self._xtol_rel = 0.0
        self._xtol_abs = 0.0
        self._stop_crit_n_x = 3

    def __algorithm_handles(self, algo_name: str, eq_constraint: bool):
        """Check if the algorithm handles equality or inequality constraints.

        Args:
            algo_name: The name of the algorithm.
            eq_constraint: Whether the constraints are equality ones.

        Returns:
            Whether the algorithm handles the passed type of constraints.
        """
        if algo_name not in self.lib_dict:
            raise KeyError(
                f"Algorithm {algo_name} not in library {self.__class__.__name__}."
            )
        if eq_constraint:
            return self.lib_dict[algo_name].handle_equality_constraints
        else:
            return self.lib_dict[algo_name].handle_inequality_constraints

    def algorithm_handles_eqcstr(self, algo_name):
        """Returns True if the algorithms handles equality constraints.

        :param algo_name: the name of the algorithm
        :returns: True or False
        """
        return self.__algorithm_handles(algo_name, True)

    def algorithm_handles_ineqcstr(self, algo_name):
        """Returns True if the algorithms handles inequality constraints.

        :param algo_name: the name of the algorithm
        :returns: True or False
        """
        return self.__algorithm_handles(algo_name, False)

    def is_algo_requires_positive_cstr(self, algo_name):
        """Returns True if the algorithm requires positive constraints False otherwise.

        :param algo_name: the name of the algorithm
        :returns: True if constraints must be positive
        :rtype: bool
        """
        return self.lib_dict[algo_name].positive_constraints

    def _check_constraints_handling(self, algo_name, problem):
        """Check if problem and algorithm are consistent for constraints handling."""
        if problem.has_eq_constraints() and not self.algorithm_handles_eqcstr(
            algo_name
        ):
            raise ValueError(
                "Requested optimization algorithm "
                "%s can not handle equality constraints." % algo_name
            )
        if problem.has_ineq_constraints() and not self.algorithm_handles_ineqcstr(
            algo_name
        ):
            raise ValueError(
                "Requested optimization algorithm "
                "%s can not handle inequality constraints." % algo_name
            )

    def get_right_sign_constraints(self):
        """Transforms the problem constraints into their opposite sign counterpart if
        the algorithm requires positive constraints."""
        if self.problem.has_ineq_constraints() and self.is_algo_requires_positive_cstr(
            self.algo_name
        ):
            return [-cstr for cstr in self.problem.constraints]
        return self.problem.constraints

    def _run(self, **options):
        """Runs the algorithm, to be overloaded by subclasses.

        :param options: the options dict for the algorithm,
            see associated JSON file
        """
        raise NotImplementedError()

    def _pre_run(self, problem, algo_name, **options):
        """To be overriden by subclasses Specific method to be executed just before _run
        method call.

        :param problem: the problem to be solved
        :param algo_name: name of the algorithm
        :param options: the options dict for the algorithm,
            see associated JSON file
        """
        super(OptimizationLibrary, self)._pre_run(problem, algo_name, **options)
        self._check_constraints_handling(algo_name, problem)

        if self.MAX_ITER in options:
            max_iter = options[self.MAX_ITER]
        elif (
            self.MAX_ITER in self.OPTIONS_MAP
            and self.OPTIONS_MAP[self.MAX_ITER] in options
        ):
            max_iter = options[self.OPTIONS_MAP[self.MAX_ITER]]
        else:
            raise ValueError("Could not determine the maximum number of iterations.")

        self._ftol_rel = options.get(self.F_TOL_REL, 0.0)
        self._ftol_abs = options.get(self.F_TOL_ABS, 0.0)
        self._xtol_rel = options.get(self.X_TOL_REL, 0.0)
        self._xtol_abs = options.get(self.X_TOL_ABS, 0.0)
        self._stop_crit_n_x = options.get(self.STOP_CRIT_NX, 3)
        self.init_iter_observer(max_iter, " ")
        problem.add_callback(self.new_iteration_callback)
        eval_jac = self.is_algo_requires_grad(algo_name)
        normalize = options.get(self.NORMALIZE_DESIGN_SPACE_OPTION, self._NORMALIZE_DS)
        # First, evaluate all functions at x_0. Some algorithms don't do this
        self.problem.evaluate_functions(
            eval_jac=eval_jac, eval_obj=True, normalize=normalize
        )

    @staticmethod
    def is_algorithm_suited(algo_dict, problem):
        """Checks if the algorithm is suited to the problem according to its algo dict.

        :param algo_dict: the algorithm characteristics
        :param problem: the opt_problem to be solved
        """
        if problem.has_eq_constraints() and not algo_dict.handle_equality_constraints:
            return False

        if (
            problem.has_ineq_constraints()
            and not algo_dict.handle_inequality_constraints
        ):
            return False

        if (
            problem.pb_type == problem.NON_LINEAR_PB
            and algo_dict.problem_type == problem.LINEAR_PB
        ):
            return False

        return True

    def new_iteration_callback(
        self, x_vect=None  # type: Optional[ndarray]
    ):  # type: (...) -> None
        """
        Raises:
            FtolReached: If the defined relative or absolute function
                tolerance is reached.
            XtolReached: If the defined relative or absolute x tolerance
                is reached.
        """
        # First check if the max_iter is reached and update the progress bar
        super(OptimizationLibrary, self).new_iteration_callback(x_vect)
        if is_f_tol_reached(
            self.problem, self._ftol_rel, self._ftol_abs, self._stop_crit_n_x
        ):
            raise FtolReached()
        if is_x_tol_reached(
            self.problem, self._xtol_rel, self._xtol_abs, self._stop_crit_n_x
        ):
            raise XtolReached()
