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
"""Base class for libraries of optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

import numpy

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.base_driver_library import DriverDescription
from gemseo.algos.stop_criteria import DesignToleranceTester
from gemseo.algos.stop_criteria import KKTConditionsTester
from gemseo.algos.stop_criteria import ObjectiveToleranceTester
from gemseo.algos.stop_criteria import kkt_residual_computation

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdofunctions.mdo_function import MDOFunction


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

    for_linear_problems: bool = False
    """Whether the optimization algorithm is dedicated to linear problems."""


class BaseOptimizationLibrary(BaseDriverLibrary):
    """Base class for libraries of optimizers.

    Typically used as:

    #. Instantiate an :class:`.BaseOptimizationLibrary`.
    #. Select the algorithm with :attr:`._algo_name`.
    #. Solve an :class:`.OptimizationProblem` with :meth:`.execute`.

    Notes:
        The missing current values
        of the :class:`.DesignSpace` attached to the :class:`.OptimizationProblem`
        are automatically initialized
        with the method :meth:`.DesignSpace.initialize_missing_current_values`.
    """

    _MAX_ITER: Final[str] = "max_iter"
    _F_TOL_REL: Final[str] = "ftol_rel"
    _F_TOL_ABS: Final[str] = "ftol_abs"
    _X_TOL_REL: Final[str] = "xtol_rel"
    _X_TOL_ABS: Final[str] = "xtol_abs"
    _KKT_TOL_ABS: Final[str] = "kkt_tol_abs"
    _KKT_TOL_REL: Final[str] = "kkt_tol_rel"
    _STOP_CRIT_NX: Final[str] = "stop_crit_n_x"
    _LS_STEP_SIZE_MAX: Final[str] = "max_ls_step_size"
    _LS_STEP_NB_MAX: Final[str] = "max_ls_step_nb"
    _MAX_FUN_EVAL: Final[str] = "max_fun_eval"
    _PG_TOL: Final[str] = "pg_tol"
    _SCALING_THRESHOLD: Final[str] = "scaling_threshold"
    _VERBOSE: Final[str] = "verbose"

    _f_tol_tester: ObjectiveToleranceTester
    """A tester for the termination criterion associated the objective."""

    _x_tol_tester: DesignToleranceTester
    """A tester for the termination criterion associated the design variables."""

    __kkt_tester: KKTConditionsTester
    """A tester for the termination criterion associated the KKT conditions."""

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self._f_tol_tester = ObjectiveToleranceTester()
        self._x_tol_tester = DesignToleranceTester()
        self.__kkt_tester = KKTConditionsTester()

    def _check_constraints_handling(self, problem: OptimizationProblem) -> None:
        """Check if problem and algorithm are consistent for constraints handling."""
        algo_name = self._algo_name
        if (
            tuple(problem.constraints.get_equality_constraints())
            and not self.ALGORITHM_INFOS[algo_name].handle_equality_constraints
        ):
            msg = (
                "Requested optimization algorithm "
                f"{algo_name} can not handle equality constraints."
            )
            raise ValueError(msg)
        if (
            tuple(problem.constraints.get_inequality_constraints())
            and not self.ALGORITHM_INFOS[algo_name].handle_inequality_constraints
        ):
            msg = (
                "Requested optimization algorithm "
                f"{algo_name} can not handle inequality constraints."
            )
            raise ValueError(msg)

    def _get_right_sign_constraints(self, problem: OptimizationProblem):
        """Transform the problem constraints into their opposite sign counterpart.

        This is done if the algorithm requires positive constraints.

        Args:
            problem: The problem to be solved.

        Returns:
            The constraints with the right sign.
        """
        if (
            tuple(problem.constraints.get_inequality_constraints())
            and self.ALGORITHM_INFOS[self._algo_name].positive_constraints
        ):
            return [-constraint for constraint in problem.constraints]
        return problem.constraints

    def _pre_run(self, problem: OptimizationProblem, **options: Any) -> None:
        super()._pre_run(problem, **options)
        self._check_constraints_handling(problem)

        if self._MAX_ITER in options:
            max_iter = options[self._MAX_ITER]
        elif (
            self._MAX_ITER in self._OPTIONS_MAP
            and self._OPTIONS_MAP[self._MAX_ITER] in options
        ):
            max_iter = options[self._OPTIONS_MAP[self._MAX_ITER]]
        else:
            msg = "Could not determine the maximum number of iterations."
            raise ValueError(msg)

        n_points = options.get(self._STOP_CRIT_NX, 3)
        self._f_tol_tester = ObjectiveToleranceTester(
            absolute=options.get(self._F_TOL_ABS, 0.0),
            relative=options.get(self._F_TOL_REL, 0.0),
            n_last_iterations=n_points,
        )
        self._x_tol_tester = DesignToleranceTester(
            absolute=options.get(self._X_TOL_ABS, 0.0),
            relative=options.get(self._X_TOL_REL, 0.0),
            n_last_iterations=n_points,
        )
        kkt_abs_tol = options.get(self._KKT_TOL_ABS)
        kkt_rel_tol = options.get(self._KKT_TOL_REL)
        self._init_iter_observer(problem, max_iter)
        require_gradient = self.ALGORITHM_INFOS[self._algo_name].require_gradient
        if require_gradient and (kkt_abs_tol is not None or kkt_rel_tol is not None):
            self.__kkt_tester = KKTConditionsTester(
                absolute=kkt_abs_tol or 0.0,
                relative=kkt_rel_tol or 0.0,
                ineq_tolerance=options.get(
                    self._INEQ_TOLERANCE, problem.tolerances.inequality
                ),
            )
            problem.add_listener(
                self._check_kkt_from_database,
                at_each_iteration=False,
                at_each_function_call=True,
            )
        problem.design_space.initialize_missing_current_values()
        if problem.differentiation_method == self.DifferentiationMethod.COMPLEX_STEP:
            problem.design_space.to_complex()
        # First, evaluate all functions at x_0. Some algorithms don't do this
        output_functions, jacobian_functions = problem.get_functions(
            jacobian_names=() if require_gradient else None,
            evaluate_objective=True,
            observable_names=None,
        )

        function_values, _ = problem.evaluate_functions(
            design_vector_is_normalized=options.get(
                self._NORMALIZE_DESIGN_SPACE_OPTION, self._NORMALIZE_DS
            ),
            output_functions=output_functions,
            jacobian_functions=jacobian_functions,
        )

        scaling_threshold = options.get(self._SCALING_THRESHOLD)
        if scaling_threshold is not None:
            self.problem.objective = self.__scale(
                self.problem.objective,
                function_values[self.problem.objective.name],
                scaling_threshold,
            )
            self.problem.constraints = [
                self.__scale(
                    constraint, function_values[constraint.name], scaling_threshold
                )
                for constraint in self.problem.constraints
            ]

            observables = tuple(self.problem.observables)
            self.problem.observables.clear()
            for observable in observables:
                self.problem.add_observable(
                    self.__scale(
                        observable, function_values[observable.name], scaling_threshold
                    )
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
            tuple(problem.constraints.get_equality_constraints())
            and not algorithm_description.handle_equality_constraints
        ):
            return _UnsuitabilityReason.EQUALITY_CONSTRAINTS

        if (
            tuple(problem.constraints.get_inequality_constraints())
            and not algorithm_description.handle_inequality_constraints
        ):
            return _UnsuitabilityReason.INEQUALITY_CONSTRAINTS

        if not problem.is_linear and algorithm_description.for_linear_problems:
            return _UnsuitabilityReason.NON_LINEAR_PROBLEM

        return reason

    def _new_iteration_callback(self, x_vect: ndarray) -> None:
        super()._new_iteration_callback(x_vect)
        self._f_tol_tester.check(self.problem, raise_exception=True)
        self._x_tol_tester.check(self.problem, raise_exception=True)

    def _check_kkt_from_database(self, x_vect: ndarray) -> None:
        """Verify, if required, KKT norm stopping criterion at each database storage.

        Raises:
            KKTReached: If the absolute tolerance on the KKT residual is reached.
        """
        check_kkt = True
        function_names = [
            self.problem.standardized_objective_name,
            *self.problem.constraints.get_names(),
        ]
        database = self.problem.database
        for function_name in function_names:
            if (
                database.get_function_value(
                    database.get_gradient_name(function_name), x_vect
                )
                is None
            ) or (database.get_function_value(function_name, x_vect) is None):
                check_kkt = False
                break

        if check_kkt:
            if not self.__kkt_tester.kkt_norm:
                self.__kkt_tester.kkt_norm = kkt_residual_computation(
                    self.problem, x_vect, self.__kkt_tester.ineq_tolerance
                )

            self.__kkt_tester.check(
                self.problem, raise_exception=True, input_vector=x_vect
            )

    @staticmethod
    def __scale(
        function: MDOFunction,
        function_value: ndarray,
        scaling_threshold: float,
    ) -> MDOFunction:
        """Scale a function based on its value on the current design values.

        Args:
            function: The function.
            function_value: The function value of reference for scaling.
            scaling_threshold: The threshold on the reference function value
                that triggers scaling.

        Returns:
            The scaled function.
        """
        reference_values = numpy.absolute(function_value)
        threshold_reached = reference_values > scaling_threshold
        if not threshold_reached.any():
            return function

        scaled_function = function / numpy.where(
            threshold_reached, reference_values, 1.0
        )
        # Use same function name for consistency with name used in database
        scaled_function.name = function.name
        return scaled_function
