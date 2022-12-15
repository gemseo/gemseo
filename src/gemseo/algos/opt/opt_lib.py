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

from dataclasses import dataclass

from numpy import ndarray

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.driver_lib import DriverDescription
from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.first_order_stop_criteria import is_kkt_residual_norm_reached
from gemseo.algos.first_order_stop_criteria import kkt_residual_computation
from gemseo.algos.first_order_stop_criteria import KKTReached
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.stop_criteria import FtolReached
from gemseo.algos.stop_criteria import is_f_tol_reached
from gemseo.algos.stop_criteria import is_x_tol_reached
from gemseo.algos.stop_criteria import XtolReached


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
    """Base optimization library defining a collection of optimization algorithms.

    Typically used as:

    #. Instantiate an :class:`.OptimizationLibrary`.
    #. Select the algorithm with :attr:`.algo_name`.
    #. Solve an :class:`.OptimizationProblem` with :meth:`.execute`.

    Note:
        The missing current values
        of the :class:`.DesignSpace` attached to the :class:`.OptimizationProblem`
        are automatically initialized
        with the method :meth:`.DesignSpace.initialize_missing_current_values`.
    """

    MAX_ITER = "max_iter"
    F_TOL_REL = "ftol_rel"
    F_TOL_ABS = "ftol_abs"
    X_TOL_REL = "xtol_rel"
    X_TOL_ABS = "xtol_abs"
    _KKT_TOL_ABS = "kkt_tol_abs"
    _KKT_TOL_REL = "kkt_tol_rel"
    STOP_CRIT_NX = "stop_crit_n_x"
    # Maximum step for the line search
    LS_STEP_SIZE_MAX = "max_ls_step_size"
    # Maximum number of line search steps (per iteration).
    LS_STEP_NB_MAX = "max_ls_step_nb"
    MAX_FUN_EVAL = "max_fun_eval"
    MAX_TIME = "max_time"
    PG_TOL = "pg_tol"
    VERBOSE = "verbose"

    def __init__(self):  # noqa:D107
        super().__init__()
        self._ftol_rel = 0.0
        self._ftol_abs = 0.0
        self._xtol_rel = 0.0
        self._xtol_abs = 0.0
        self.__kkt_abs_tol = 0.0
        self.__kkt_rel_tol = 0.0
        self.__ref_kkt_norm = None
        self._stop_crit_n_x = 3

    def __algorithm_handles(self, algo_name: str, eq_constraint: bool):
        """Check if the algorithm handles equality or inequality constraints.

        Args:
            algo_name: The name of the algorithm.
            eq_constraint: Whether the constraints are equality ones.

        Returns:
            Whether the algorithm handles the passed type of constraints.
        """
        if algo_name not in self.descriptions:
            raise KeyError(
                f"Algorithm {algo_name} not in library {self.__class__.__name__}."
            )
        if eq_constraint:
            return self.descriptions[algo_name].handle_equality_constraints
        else:
            return self.descriptions[algo_name].handle_inequality_constraints

    def algorithm_handles_eqcstr(self, algo_name: str) -> bool:
        """Check if an algorithm handles equality constraints.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            Whether the algorithm handles equality constraints.
        """
        return self.__algorithm_handles(algo_name, True)

    def algorithm_handles_ineqcstr(self, algo_name: str) -> bool:
        """Check if an algorithm handles inequality constraints.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            Whether the algorithm handles inequality constraints.
        """
        return self.__algorithm_handles(algo_name, False)

    def is_algo_requires_positive_cstr(self, algo_name: str) -> bool:
        """Check if an algorithm requires positive constraints.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            Whether the algorithm requires positive constraints.
        """
        return self.descriptions[algo_name].positive_constraints

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
        """Transform the problem constraints into their opposite sign counterpart.

        This is done if the algorithm requires positive constraints.
        """
        if self.problem.has_ineq_constraints() and self.is_algo_requires_positive_cstr(
            self.algo_name
        ):
            return [-cstr for cstr in self.problem.constraints]
        return self.problem.constraints

    def _pre_run(self, problem, algo_name, **options):
        """To be overridden by subclasses.

        Specific method to be executed just before _run method call.

        The missing current values of the :class:`.DesignSpace` are initialized
        with the method :meth:`.DesignSpace.initialize_missing_current_values`.

        Args:
            problem: The optimization problem.
            algo_name: The name of the algorithm.
            **options: The options of the algorithm,
                see the associated JSON file.
        """
        super()._pre_run(problem, algo_name, **options)
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
        self.__ineq_tolerance = options.get(self.INEQ_TOLERANCE, problem.ineq_tolerance)
        self._stop_crit_n_x = options.get(self.STOP_CRIT_NX, 3)
        self.__kkt_abs_tol = options.get(self._KKT_TOL_ABS, None)
        self.__kkt_rel_tol = options.get(self._KKT_TOL_REL, None)
        self.init_iter_observer(max_iter)
        if (
            self.__kkt_abs_tol is not None or self.__kkt_rel_tol is not None
        ) and self.descriptions[self.algo_name].require_gradient:
            problem.add_callback(
                self._check_kkt_from_database, each_new_iter=False, each_store=True
            )
        # First, evaluate all functions at x_0. Some algorithms don't do this
        self.problem.design_space.initialize_missing_current_values()
        self.problem.evaluate_functions(
            eval_jac=self.is_algo_requires_grad(algo_name),
            eval_obj=True,
            normalize=options.get(
                self.NORMALIZE_DESIGN_SPACE_OPTION, self._NORMALIZE_DS
            ),
        )

    @classmethod
    def _get_unsuitability_reason(
        cls,
        algorithm_description: OptimizationAlgorithmDescription,
        problem: OptimizationProblem,
    ) -> _UnsuitabilityReason:
        reason = super()._get_unsuitability_reason(algorithm_description, problem)
        if reason:
            return reason

        if (
            problem.has_eq_constraints()
            and not algorithm_description.handle_equality_constraints
        ):
            return _UnsuitabilityReason.EQUALITY_CONSTRAINTS

        if (
            problem.has_ineq_constraints()
            and not algorithm_description.handle_inequality_constraints
        ):
            return _UnsuitabilityReason.INEQUALITY_CONSTRAINTS

        if (
            problem.pb_type == problem.NON_LINEAR_PB
            and algorithm_description.problem_type == problem.LINEAR_PB
        ):
            return _UnsuitabilityReason.NON_LINEAR_PROBLEM

        return reason

    def new_iteration_callback(self, x_vect: ndarray | None = None) -> None:
        """Verify the design variable and objective value stopping criteria.

        Raises:
            FtolReached: If the defined relative or absolute function
                tolerance is reached.
            XtolReached: If the defined relative or absolute x tolerance
                is reached.
        """
        # First check if the max_iter is reached and update the progress bar
        super().new_iteration_callback(x_vect)
        if is_f_tol_reached(
            self.problem, self._ftol_rel, self._ftol_abs, self._stop_crit_n_x
        ):
            raise FtolReached()

        if is_x_tol_reached(
            self.problem, self._xtol_rel, self._xtol_abs, self._stop_crit_n_x
        ):
            raise XtolReached()

    def _check_kkt_from_database(self, x_vect: ndarray | None = None) -> None:
        """Verify, if required, KKT norm stopping criterion at each database storage.

        Raises:
            KKTReached: If the absolute tolerance on the KKT residual is reached.
        """
        check_kkt = True
        function_names = [
            self.problem.get_objective_name()
        ] + self.problem.get_constraints_names()
        database = self.problem.database
        for function_name in function_names:
            if (
                database.get_f_of_x(database.get_gradient_name(function_name), x_vect)
                is None
            ) or (database.get_f_of_x(function_name, x_vect) is None):
                check_kkt = False
                break
        if check_kkt and (self.__ref_kkt_norm is None):
            self.__ref_kkt_norm = kkt_residual_computation(
                self.problem, x_vect, self.__ineq_tolerance
            )

        if check_kkt and is_kkt_residual_norm_reached(
            self.problem,
            x_vect,
            kkt_abs_tol=self.__kkt_abs_tol,
            kkt_rel_tol=self.__kkt_rel_tol,
            ineq_tolerance=self.__ineq_tolerance,
            reference_residual=self.__ref_kkt_norm,
        ):
            raise KKTReached()
