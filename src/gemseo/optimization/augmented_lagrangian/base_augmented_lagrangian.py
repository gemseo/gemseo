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
"""An implementation of the augmented lagrangian algorithm."""

from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import TypeVar

from numpy import atleast_1d
from numpy import concatenate
from numpy import inf
from numpy import zeros_like
from numpy.linalg import norm
from numpy.ma import allequal

from gemseo.optimization.aggregation.aggregation_func import (
    aggregate_positive_sum_square,
)
from gemseo.optimization.aggregation.aggregation_func import aggregate_sum_square
from gemseo.optimization.augmented_lagrangian.settings.base import (
    BaseAugmentedLagrangianSettings,
)
from gemseo.optimization.core.base_optimization_library import BaseOptimizationLibrary
from gemseo.optimization.factory import OptimizationLibraryFactory
from gemseo.optimization.problem import OptimizationProblem
from gemseo.space.transformation._working import project_onto_declared_domain

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.optimization.core.base_optimizer_settings import BaseOptimizerSettings
    from gemseo.optimization.result import OptimizationResult
    from gemseo.util.typing import NumberArray

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseAugmentedLagrangianSettings)


class BaseAugmentedLagrangian(BaseOptimizationLibrary[T]):
    """This is an abstract base class for augmented lagrangian optimization algorithms.

    The abstract methods `_update_penalty()` and
    `_update_lagrange_multipliers()` need to be implemented by derived classes.
    """

    _iterates_on_working_problem: ClassVar[bool] = False
    """The augmented Lagrangian builds a sub-problem and hands it to a sub-driver."""

    __n_obj_func_calls: int
    """The total number of objective function calls."""

    _rho: float
    """The penalty value."""

    _function_outputs: dict[str, float | NumberArray]
    """The current iteration function outputs."""

    _sub_problems: list[OptimizationProblem]
    """The sub problems appended in the sequence of optimization problem."""

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self.__n_obj_func_calls = 0
        self._function_outputs = {}
        self._sub_problems = []

    @property
    def n_obj_func_calls(self) -> int:
        """The total number of objective function calls."""
        return self.__n_obj_func_calls

    def _run(self, problem: OptimizationProblem) -> tuple[str, Any]:
        self._rho = self._settings.initial_rho
        self._update_options_callback = self._settings.update_options_callback

        problem_ineq_constraints = [
            constr
            for constr in problem.constraints.get_inequality_constraints()
            if constr.name not in self._settings.sub_problem_constraints
        ]
        problem_eq_constraints = [
            constr
            for constr in problem.constraints.get_equality_constraints()
            if constr.name not in self._settings.sub_problem_constraints
        ]

        # This library derives no problem,
        # so its functions take a point in the coordinates the user declared
        # whatever `normalize_design_space` says.
        current_value = self._problem.input_space.get_current_value()
        eq_multipliers = {
            h.name: zeros_like(h.evaluate(current_value))
            for h in problem_eq_constraints
        }
        ineq_multipliers = {
            g.name: zeros_like(g.evaluate(current_value))
            for g in problem_ineq_constraints
        }

        active_constraint_residual = inf
        x = self._problem.input_space.get_current_value()
        message = None
        for iteration in range(self._settings.max_iter):
            logger.debug("iteration: %s", iteration)
            logger.debug(
                "inequality Lagrange multiplier approximations:  %s", ineq_multipliers
            )
            logger.debug(
                "equality Lagrange multiplier approximations:  %s", eq_multipliers
            )
            logger.debug("Active constraint residual:  %s", active_constraint_residual)
            logger.debug("penalty:  %s", self._rho)

            # Get the next design candidate solving the sub-problem.
            f_calls_sub_prob, x_new = self.__solve_sub_problem(
                eq_multipliers,
                ineq_multipliers,
                x,
            )

            self.__n_obj_func_calls += f_calls_sub_prob

            (_, hv, vk) = (
                self.__compute_objective_function_and_active_constraint_residual(
                    ineq_multipliers,
                    problem_eq_constraints,
                    problem_ineq_constraints,
                    x_new,
                )
            )

            self._rho = self._update_penalty(
                constraint_violation_current_iteration=max(norm(vk), norm(hv)),
                objective_function_current_iteration=self._function_outputs[
                    self._problem.objective.name
                ],
                constraint_violation_previous_iteration=active_constraint_residual,
                current_penalty=self._rho,
                iteration=iteration,
            )
            # Update the active constraint residual.
            active_constraint_residual = max(norm(vk), norm(hv))

            self._update_lagrange_multipliers(eq_multipliers, ineq_multipliers, x_new)

            has_converged, message = self._check_termination_criteria(
                x_new, x, eq_multipliers, ineq_multipliers
            )
            if has_converged:
                break

            x = x_new

        return message, None

    def _post_run(
        self,
        problem: OptimizationProblem,
        result: OptimizationResult,
        max_input_space_dimension_to_log: int,
    ) -> None:
        result.n_obj_call = self.__n_obj_func_calls
        super()._post_run(problem, result, max_input_space_dimension_to_log)

    @staticmethod
    def _check_termination_criteria(
        x_new: NumberArray,
        x: NumberArray,
        eq_lag: dict[str, NumberArray],
        ineq_lag: dict[str, NumberArray],
    ) -> tuple[bool, str]:
        """Check if the termination criteria are satisfied.

        Args:
            x_new: The new design vector.
            x: The old design vector.
            eq_lag: The equality constraint lagrangian multipliers.
            ineq_lag: The inequality constraint lagrangian multipliers.

        Returns:
            Whether the termination criteria are satisfied and the convergence message.
        """
        if len(eq_lag) + len(ineq_lag) == 0:
            return True, "The sub solver dealt with the constraints."
        if allequal(x_new, x):
            return True, "The solver stopped proposing new designs."
        return False, "Maximum number of iterations reached."

    def __compute_objective_function_and_active_constraint_residual(
        self,
        mu0: dict[str, NumberArray],
        problem_eq_constraints: Iterable[ArrayFunction],
        problem_ineq_constraints: Iterable[ArrayFunction],
        x_opt: NumberArray,
    ) -> tuple[float | NumberArray, NumberArray | Iterable, NumberArray | Iterable]:
        """Compute the objective function and active constraint residuals.

        Args:
            mu0: The lagrangian multipliers for inequality constraints.
            problem_eq_constraints: The optimization problem equality constraints dealt
                with Augmented Lagrangian.
            problem_ineq_constraints: The optimization problem inequality constraints
                dealt with Augmented Lagrangian.
            x_opt: The current design variable vector.

        Returns:
            The objective function value,
            the equality constraint violation value,
            the active inequality constraint residuals.
        """
        require_gradient = self.ALGORITHM_INFOS[self.algo_name].require_gradient
        output_functions, jacobian_functions = self._problem.get_functions(
            jacobian_names=() if require_gradient else None,
        )
        # Evaluated at the point rather than set as the current value,
        # and without the domain membership check that preprocessing would run:
        # a sub-algorithm relaxing some variables returns a point
        # outside the declared domain (e.g. a non-integral value
        # for an integer or discrete variable), which the space would refuse.
        # `self._problem` is the problem the user built, only bound to its own
        # database (`_iterates_on_working_problem` is `False`, so no working
        # problem is built on top of it), and `bind_functions()` refuses a
        # function expecting a normalized input, so `x_opt` is already in the
        # coordinates these functions take.
        self._function_outputs, _ = self._problem.evaluate_functions(
            input_value=x_opt,
            input_value_is_normalized=False,
            preprocess_input_value=False,
            output_functions=output_functions or None,
            jacobian_functions=jacobian_functions or None,
        )
        f_opt = self._function_outputs[self._problem.objective.name]
        gv = [
            atleast_1d(self._function_outputs[constr.name])
            for constr in problem_ineq_constraints
        ]
        hv = [
            atleast_1d(self._function_outputs[constr.name])
            for constr in problem_eq_constraints
        ]
        mu_vector = [
            atleast_1d(mu0[constr.name]) for constr in problem_ineq_constraints
        ]
        vk = [
            -g_i * (-g_i <= mu / self._rho) + mu / self._rho * (-g_i > mu / self._rho)
            for g_i, mu in zip(gv, mu_vector, strict=False)
        ]
        vk = concatenate(vk) if vk else vk
        hv = concatenate(hv) if hv else hv
        return f_opt, hv, vk

    @staticmethod
    def _check_for_preconditioner(
        sub_algorithm_settings: BaseOptimizerSettings,
    ) -> None:
        """Check if 'precond' is in sub_algorithm_settings and log if detected."""
        if "precond" in sub_algorithm_settings.model_fields_set:
            logger.info("Preconditioner Detected")

    def __solve_sub_problem(
        self,
        lambda0: dict[str, NumberArray],
        mu0: dict[str, NumberArray],
        x_init: NumberArray,
    ) -> tuple[int, NumberArray]:
        """Solve the sub-problem.

        Args:
            lambda0: The lagrangian multipliers for equality constraints.
            mu0: The lagrangian multipliers for inequality constraints.
            x_init: The design variable vector at the current iteration.

        Returns:
            The updated number of function call and the new design variable vector.
        """
        # Get the sub problem.
        lagrangian = self.__get_lagrangian_function(lambda0, mu0, self._rho)
        dspace = deepcopy(self._problem.input_space)
        # The previous sub-algorithm may have relaxed some variables,
        # and the space accepts a starting point of the declared domain only.
        dspace.set_current_value(project_onto_declared_domain(dspace, x_init))
        sub_problem = OptimizationProblem(dspace)
        sub_problem.objective = lagrangian
        for constraint in self._problem.constraints.get_originals():
            if constraint.name in self._settings.sub_problem_constraints:
                sub_problem.constraints.append(constraint)

        if self._update_options_callback is not None:
            self._update_options_callback(
                self._sub_problems, self._settings.sub_algorithm_settings
            )

        self._check_for_preconditioner(self._settings.sub_algorithm_settings)

        # Solve the sub-problem.
        lib = OptimizationLibraryFactory().create(
            self._settings.sub_algorithm_settings.target_class_name
        )
        opt = lib.execute(sub_problem, settings=self._settings.sub_algorithm_settings)

        self._sub_problems.append(sub_problem)
        # The Lagrangian and the sub-problem constraints are built from
        # `.original` / `get_originals()`, so the sub-algorithm's own
        # iterations are recorded in the sub-problem's database only,
        # not in the top-level one; only its final `x_opt`, evaluated on
        # the top-level functions afterward, reaches it. There is
        # therefore nothing in the sub-problem's function history worth
        # copying here, only the names of the variables the sub-run may
        # have relaxed, which this problem's `to_dataset()` needs too.
        self._problem.database.merge_function_histories(sub_problem.database, ())

        return sub_problem.objective.n_calls, opt.x_opt

    @abstractmethod
    def _update_penalty(
        self,
        constraint_violation_current_iteration: NumberArray | float,
        objective_function_current_iteration: NumberArray | float,
        constraint_violation_previous_iteration: NumberArray | float,
        current_penalty: NumberArray | float,
        iteration: int,
    ) -> float:
        """Update the penalty.

        This method must be implemented in a derived class
        in order to compute the penalty coefficient
        at each iteration of the Augmented Lagrangian algorithm.

        Args:
            objective_function_current_iteration: The objective function value at the
                current iteration.
            constraint_violation_current_iteration: The maximum constraint violation at
                the current iteration.
            constraint_violation_previous_iteration: The maximum constraint violation at
                the previous iteration.
            current_penalty: The penalty value at the current iteration.
            iteration: The iteration number.

        Returns:
            The updated penalty value.
        """

    @abstractmethod
    def _update_lagrange_multipliers(
        self,
        eq_lag: dict[str, NumberArray],
        ineq_lag: dict[str, NumberArray],
        x_opt: NumberArray,
    ) -> None:
        """Update the lagrange multipliers.

        This method must be implemented in a derived class
        in order to compute the lagrange multipliers
        at each iteration of the Augmented Lagrangian algorithm.

        Args:
            eq_lag: The lagrange multipliers for equality constraints.
            ineq_lag: The lagrange multipliers for inequality constraints.
            x_opt: The current design variables vector.
        """

    def __get_lagrangian_function(
        self,
        eq_lag: dict[str, NumberArray],
        ineq_lag: dict[str, NumberArray],
        rho: float,
    ) -> ArrayFunction:
        """Return the lagrangian function.

        Args:
            eq_lag: The lagrangian multipliers for equality constraints.
            ineq_lag: The lagrangian multipliers for inequality constraints.
            rho: The penalty.

        Returns:
            The lagrangian function.
        """
        lagrangian = self._problem.objective.original
        for constr in self._problem.constraints.get_originals():
            if constr.name in ineq_lag:
                lagrangian += aggregate_positive_sum_square(
                    constr + ineq_lag[constr.name] / rho, scale=rho / 2
                )
            if constr.name in eq_lag:
                lagrangian += aggregate_sum_square(
                    constr + eq_lag[constr.name] / rho, scale=rho / 2
                )
        return lagrangian
