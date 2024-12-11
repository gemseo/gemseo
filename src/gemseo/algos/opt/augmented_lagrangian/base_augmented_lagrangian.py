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
from typing import Final

from numpy import atleast_1d
from numpy import concatenate
from numpy import inf
from numpy import zeros_like
from numpy.linalg import norm
from numpy.ma import allequal

from gemseo.algos.aggregation.aggregation_func import aggregate_positive_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.optimization_result import OptimizationResult
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import NumberArray
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


class BaseAugmentedLagrangian(BaseOptimizationLibrary):
    """This is an abstract base class for augmented lagrangian optimization algorithms.

    The abstract methods :func:`_update_penalty` and
    :func:`_update_lagrange_multipliers` need to be implemented by derived classes.
    """

    __n_obj_func_calls: int
    """The total number of objective function calls."""

    __INITIAL_RHO: Final[str] = "initial_rho"
    """The name of the option for `initial_rho` parameter."""

    __SUB_PROBLEM_CONSTRAINTS: Final[str] = "sub_problem_constraints"
    """The name of the option that corresponds to sub problem constraints."""

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

    def _run(self, problem: OptimizationProblem, **settings: Any) -> tuple[str, Any]:
        self._rho = settings[self.__INITIAL_RHO]
        self._update_options_callback = settings["update_options_callback"]

        problem_ineq_constraints = [
            constr
            for constr in problem.constraints.get_inequality_constraints()
            if constr.name not in settings[self.__SUB_PROBLEM_CONSTRAINTS]
        ]
        problem_eq_constraints = [
            constr
            for constr in problem.constraints.get_equality_constraints()
            if constr.name not in settings[self.__SUB_PROBLEM_CONSTRAINTS]
        ]

        current_value = self._problem.design_space.get_current_value(
            normalize=self._normalize_ds
        )
        eq_multipliers = {
            h.name: zeros_like(h.evaluate(current_value))
            for h in problem_eq_constraints
        }
        ineq_multipliers = {
            g.name: zeros_like(g.evaluate(current_value))
            for g in problem_ineq_constraints
        }

        active_constraint_residual = inf
        x = self._problem.design_space.get_current_value()
        message = None
        for iteration in range(settings[self._MAX_ITER]):
            LOGGER.debug("iteration: %s", iteration)
            LOGGER.debug(
                "inequality Lagrange multiplier approximations:  %s", ineq_multipliers
            )
            LOGGER.debug(
                "equality Lagrange multiplier approximations:  %s", eq_multipliers
            )
            LOGGER.debug("Active constraint residual:  %s", active_constraint_residual)
            LOGGER.debug("penalty:  %s", self._rho)

            # Get the next design candidate solving the sub-problem.
            f_calls_sub_prob, x_new = self.__solve_sub_problem(
                eq_multipliers,
                ineq_multipliers,
                self._normalize_ds,
                settings[self.__SUB_PROBLEM_CONSTRAINTS],
                settings["sub_algorithm_name"],
                settings["sub_algorithm_settings"],
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
                **settings,
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
        max_design_space_dimension_to_log: int,
        **settings: Any,
    ) -> None:
        result.n_obj_call = self.__n_obj_func_calls
        super()._post_run(
            problem, result, max_design_space_dimension_to_log, **settings
        )

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
        return False, "Maximun number of iterations reached."

    def __compute_objective_function_and_active_constraint_residual(
        self,
        mu0: dict[str, NumberArray],
        problem_eq_constraints: Iterable[MDOFunction],
        problem_ineq_constraints: Iterable[MDOFunction],
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
        self._problem.design_space.set_current_value(x_opt)
        require_gradient = self.ALGORITHM_INFOS[self.algo_name].require_gradient
        output_functions, jacobian_functions = self._problem.get_functions(
            jacobian_names=() if require_gradient else None,
        )
        self._function_outputs, _ = self._problem.evaluate_functions(
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
            for g_i, mu in zip(gv, mu_vector)
        ]
        vk = concatenate(vk) if vk else vk
        hv = concatenate(hv) if hv else hv
        return f_opt, hv, vk

    @staticmethod
    def _check_for_preconditioner(sub_algorithm_settings: StrKeyMapping) -> None:
        """Check if 'precond' is in sub_algorithm_settings and log if detected."""
        if sub_algorithm_settings and "precond" in sub_algorithm_settings:
            LOGGER.info("Preconditioner Detected")

    def __solve_sub_problem(
        self,
        lambda0: dict[str, NumberArray],
        mu0: dict[str, NumberArray],
        normalize: bool,
        sub_problem_constraints: Iterable[str],
        sub_algorithm_name: str,
        sub_algorithm_settings: StrKeyMapping,
        x_init: NumberArray,
    ) -> tuple[int, NumberArray]:
        """Solve the sub-problem.

        Args:
            lambda0: The lagrangian multipliers for equality constraints.
            mu0: The lagrangian multipliers for inequality constraints.
            normalize: Whether to normalize the design space.
            sub_problem_constraints: The constraints to keep in the sub-problem.
                If ``empty`` all constraints are dealt by the Augmented Lagrange,
                which means that the sub-problem is unconstrained.
            sub_algorithm_name: The name of the optimization algorithm used to solve
                each sub-poblem.
            sub_algorithm_settings: The settings of the sub-problem optimization solver.
            x_init: The design variable vector at the current iteration.

        Returns:
            The updated number of function call and the new design variable vector.
        """
        # Get the sub problem.
        lagrangian = self.__get_lagrangian_function(lambda0, mu0, self._rho)
        dspace = deepcopy(self._problem.design_space)
        dspace.set_current_value(x_init)
        sub_problem = OptimizationProblem(dspace)
        sub_problem.objective = lagrangian
        for constraint in self._problem.constraints.get_originals():
            if constraint.name in sub_problem_constraints:
                sub_problem.constraints.append(constraint)
        sub_problem.preprocess_functions(is_function_input_normalized=normalize)

        if self._update_options_callback is not None:
            self._update_options_callback(self._sub_problems, sub_algorithm_settings)

        self._check_for_preconditioner(sub_algorithm_settings)

        # Solve the sub-problem.
        opt = OptimizationLibraryFactory().execute(
            sub_problem, algo_name=sub_algorithm_name, **sub_algorithm_settings
        )

        self._sub_problems.append(sub_problem)

        return sub_problem.objective.n_calls, opt.x_opt

    @abstractmethod
    def _update_penalty(
        self,
        constraint_violation_current_iteration: NumberArray | float,
        objective_function_current_iteration: NumberArray | float,
        constraint_violation_previous_iteration: NumberArray | float,
        current_penalty: NumberArray | float,
        iteration: int,
        **options: Any,
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
            **options: The other options of the update penalty method.

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
    ) -> MDOFunction:
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
