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

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from numpy import atleast_1d
from numpy import concatenate
from numpy import inf
from numpy import ndarray
from numpy import zeros_like
from numpy.linalg import norm
from numpy.ma import allequal

from gemseo import LOGGER
from gemseo.algos.aggregation.aggregation_func import aggregate_positive_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt.optimization_library import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.algos.opt_result import OptimizationResult
    from gemseo.core.mdofunctions.mdo_function import MDOFunction


class BaseAugmentedLagrangian(
    OptimizationLibrary, metaclass=ABCGoogleDocstringInheritanceMeta
):
    """This is an abstract base class for augmented lagrangian optimization algorithms.

    The abstract methods :func:`_update_penalty` and
    :func:`_update_lagrange_multipliers` need to be implemented by derived classes.
    """

    __n_obj_func_calls: int
    """The total number of objective function calls."""

    LIBRARY_NAME = "GEMSEO"

    __SUB_PROBLEM_CONSTRAINTS: Final[str] = "sub_problem_constraints"
    """The name of the option that corresponds to sub problem constraints."""

    __INITIAL_RHO: Final[str] = "initial_rho"
    """The name of the option for `initial_rho` parameter."""

    __SUB_SOLVER_ALGORITHM: Final[str] = "sub_solver_algorithm"
    """The name of the option for the sub solver algorithm."""

    __SUB_PROBLEM_OPTIONS: Final[str] = "sub_problem_options"
    """The name of the option for the sub problem options."""

    _rho: float
    """The penalty value."""
    _function_outputs: dict[str, float | ndarray]
    """The current iteration function outputs."""

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        self.__n_obj_func_calls = 0
        self._function_outputs = {}

    def _get_options(
        self,
        sub_solver_algorithm: str,
        normalize_design_space: bool = True,
        max_iter: int = 999,
        stop_crit_n_x: int = 3,
        ftol_rel: float = 1e-9,
        ftol_abs: float = 1e-9,
        xtol_rel: float = 1e-9,
        xtol_abs: float = 1e-9,
        max_fun_eval: int = 999,
        eq_tolerance: float = 1e-2,
        ineq_tolerance: float = 1e-4,
        kkt_tol_abs: float | None = None,
        kkt_tol_rel: float | None = None,
        sub_problem_options: Mapping[str, Any] | None = None,
        sub_problem_constraints: Iterable[str] = (),
        initial_rho: float = 10.0,
        **options: Any,
    ) -> dict[str, Any]:
        """

        Args:
            sub_solver_algorithm: The name of the optimization algorithm used to solve
                each sub-poblem.
            sub_problem_options: The options passed to the sub-problem optimization
                solver.
            stop_crit_n_x: The minimum number of design vectors to take into account in
                the stopping criteria.
            max_iter: The maximum number of iterations, i.e. unique calls to f(x).
            ftol_rel: A stop criteria, the relative tolerance on the
               objective function.
               If abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop.
            ftol_abs: A stop criteria, the absolute tolerance on the objective
               function. If abs(f(xk)-f(xk+1))<= ftol_rel: stop.
            xtol_rel: A stop criteria, the relative tolerance on the
               design variables. If norm(xk-xk+1)/norm(xk)<= xtol_rel: stop.
            xtol_abs: A stop criteria, absolute tolerance on the
               design variables.
               If norm(xk-xk+1)<= xtol_abs: stop.
            max_fun_eval: The internal stop criteria on the
               number of algorithm outer iterations.
            kkt_tol_abs: The absolute tolerance on the KKT residual norm.
                If ``None`` this criterion is not activated.
            kkt_tol_rel: The relative tolerance on the KKT residual norm.
                If ``None`` this criterion is not activated.
            eq_tolerance: The tolerance on the equality constraints.
            ineq_tolerance: The tolerance on the inequality constraints.
            normalize_design_space: Whether to scale the variables into ``[0, 1]``.
            sub_problem_constraints: The constraints to keep in the sub-problem.
                If ``empty`` all constraints are dealt by the Augmented Lagrange,
                which means that the sub-problem is unconstrained.
            initial_rho: The initial value of the penalty.
        """  # noqa: D205, D212, D415
        if sub_problem_options is None:
            sub_problem_options = {}
        return self._process_options(
            sub_solver_algorithm=sub_solver_algorithm,
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            stop_crit_n_x=stop_crit_n_x,
            normalize_design_space=normalize_design_space,
            ineq_tolerance=ineq_tolerance,
            eq_tolerance=eq_tolerance,
            kkt_tol_abs=kkt_tol_abs,
            kkt_tol_rel=kkt_tol_rel,
            sub_problem_options=sub_problem_options,
            sub_problem_constraints=sub_problem_constraints,
            initial_rho=initial_rho,
            **options,
        )

    def _run(self, **options: Any) -> OptimizationResult:
        # Initialize the penalty and the multipliers.
        self._rho = options[self.__INITIAL_RHO]
        active_constraint_residual = inf
        x = self.problem.design_space.get_current_value()
        normalize = options[self.NORMALIZE_DESIGN_SPACE_OPTION]
        problem_ineq_constraints = [
            constr
            for constr in self.problem.get_ineq_constraints()
            if constr.name not in options[self.__SUB_PROBLEM_CONSTRAINTS]
        ]
        problem_eq_constraints = [
            constr
            for constr in self.problem.get_eq_constraints()
            if constr.name not in options[self.__SUB_PROBLEM_CONSTRAINTS]
        ]
        eq_multipliers = {
            h.name: zeros_like(
                h(self.problem.design_space.get_current_value(normalize=normalize))
            )
            for h in problem_eq_constraints
        }
        ineq_multipliers = {
            g.name: zeros_like(
                g(self.problem.design_space.get_current_value(normalize=normalize))
            )
            for g in problem_ineq_constraints
        }

        for iteration in range(options[self.MAX_ITER]):
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
                normalize,
                options[self.__SUB_PROBLEM_CONSTRAINTS],
                options[self.__SUB_SOLVER_ALGORITHM],
                options[self.__SUB_PROBLEM_OPTIONS],
                x,
            )

            self.__n_obj_func_calls += f_calls_sub_prob

            (
                _f_opt,
                hv,
                vk,
            ) = self.__compute_objective_function_and_active_constraint_residual(
                ineq_multipliers,
                problem_eq_constraints,
                problem_ineq_constraints,
                x_new,
            )

            self._rho = self._update_penalty(
                constraint_violation_current_iteration=max(norm(vk), norm(hv)),
                objective_function_current_iteration=self._function_outputs[
                    self.problem.objective.name
                ],
                constraint_violation_previous_iteration=active_constraint_residual,
                current_penalty=self._rho,
                iteration=iteration,
                **options,
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
        return self.get_optimum_from_database(message)

    def _post_run(
        self,
        problem: OptimizationProblem,
        algo_name: str,
        result: OptimizationResult,
        **options: Any,
    ) -> None:
        result.n_obj_call = self.__n_obj_func_calls
        super()._post_run(problem, algo_name, result, **options)

    @staticmethod
    def _check_termination_criteria(
        x_new: ndarray,
        x: ndarray,
        eq_lag: dict[str, ndarray],
        ineq_lag: dict[str, ndarray],
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
        mu0: dict[str, ndarray],
        problem_eq_constraints: Iterable[MDOFunction],
        problem_ineq_constraints: Iterable[MDOFunction],
        x_opt: ndarray,
    ) -> tuple[float | ndarray, ndarray | Iterable, ndarray | Iterable]:
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
        self.problem.design_space.set_current_value(x_opt)
        self._function_outputs, _ = self.problem.evaluate_functions(
            eval_jac=self.descriptions[self.algo_name].require_gradient,
            eval_obj=True,
        )
        f_opt = self._function_outputs[self.problem.objective.name]
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
        if vk:
            vk = concatenate(vk)
        if hv:
            hv = concatenate(hv)
        return f_opt, hv, vk

    def __solve_sub_problem(
        self,
        lambda0: dict[str, ndarray],
        mu0: dict[str, ndarray],
        normalize: bool,
        sub_problem_constraints: Iterable[str],
        sub_solver_algorithm: str,
        sub_problem_options: Mapping[str, Any],
        x_init: ndarray,
    ) -> tuple[int, ndarray]:
        """Solve the sub-problem.

        Args:
            lambda0: The lagrangian multipliers for equality constraints.
            mu0: The lagrangian multipliers for inequality constraints.
            normalize: Whether to normalize the design space.
            sub_problem_constraints: The constraints to keep in the sub-problem.
                If ``empty`` all constraints are dealt by the Augmented Lagrange,
                which means that the sub-problem is unconstrained.
            sub_solver_algorithm: The name of the optimization algorithm used to solve
                each sub-poblem.
            sub_problem_options: The options passed to the sub-problem optimization
                solver.
            x_init: The design variable vector at the current iteration.

        Returns:
            The updated number of function call and the new design variable vector.
        """
        # Get the sub problem.
        lagrangian = self.__get_lagrangian_function(lambda0, mu0, self._rho)
        dspace = deepcopy(self.problem.design_space)
        dspace.set_current_value(x_init)
        sub_problem = OptimizationProblem(dspace)
        sub_problem.objective = lagrangian
        for constraint in self.problem.nonproc_constraints:
            if constraint.name in sub_problem_constraints:
                sub_problem.constraints.append(constraint)
        sub_problem.preprocess_functions(is_function_input_normalized=normalize)

        # Solve the sub-problem.
        opt = OptimizersFactory().execute(
            sub_problem,
            sub_solver_algorithm,
            **sub_problem_options,
        )
        return sub_problem.objective.n_calls, opt.x_opt

    @abstractmethod
    def _update_penalty(
        self,
        constraint_violation_current_iteration: ndarray | float,
        objective_function_current_iteration: ndarray | float,
        constraint_violation_previous_iteration: ndarray | float,
        current_penalty: ndarray | float,
        iteration: int,
        **options: Any,
    ) -> float | ndarray:
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
        self, eq_lag: dict[str, ndarray], ineq_lag: dict[str, ndarray], x_opt: ndarray
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
        self, eq_lag: dict[str, ndarray], ineq_lag: dict[str, ndarray], rho: float
    ) -> MDOFunction:
        """Return the lagrangian function.

        Args:
            eq_lag: The lagrangian multipliers for equality constraints.
            ineq_lag: The lagrangian multipliers for inequality constraints.
            rho: The penalty.

        Returns:
            The lagrangian function.
        """
        lagrangian = self.problem.nonproc_objective
        for constr in self.problem.nonproc_constraints:
            if constr.name in ineq_lag:
                lagrangian += aggregate_positive_sum_square(
                    constr + ineq_lag[constr.name] / rho, scale=rho / 2
                )
            if constr.name in eq_lag:
                lagrangian += aggregate_sum_square(
                    constr + eq_lag[constr.name] / rho, scale=rho / 2
                )
        return lagrangian
