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

from copy import deepcopy
from typing import Any
from typing import ClassVar
from typing import Mapping

from numpy import atleast_1d
from numpy import concatenate
from numpy import heaviside
from numpy import Infinity
from numpy import ndarray
from numpy import zeros_like
from numpy.linalg import norm

from gemseo import LOGGER
from gemseo.algos.aggregation.aggregation_func import aggregate_positive_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt.optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.optimization_library import OptimizationLibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_function import MDOFunction


class AugmentedLagrangian(OptimizationLibrary):
    """This class implements the Augmented Lagrangian optimization algorithm.

    See :cite:`birgin2014practical`
    """

    LIB_COMPUTE_GRAD = True

    LIBRARY_NAME = "GEMSEO"
    TAU: ClassVar[float] = 0.9
    """The threshold for the penalty term increase."""

    GAMMA: ClassVar[float] = 1.5
    """The increase of the penalty term."""

    RHO_0: ClassVar[float] = 10
    """The initial penalty."""

    RHO_MAX: ClassVar[float] = 10000
    """The max penalty value."""

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        self.descriptions = {
            "Augmented_Lagrangian": OptimizationAlgorithmDescription(
                algorithm_name="Augmented_Lagrangian",
                description=("Augmented Lagrangian algorithm."),
                internal_algorithm_name="Augmented_Lagrangian",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                require_gradient=True,
            ),
        }

    def _get_options(
        self,
        sub_solver_algorithm: str,
        normalize_design_space: bool = True,
        max_iter: int = 999,
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
        **options: Any,
    ) -> dict[str, Any]:
        """

        Args:
            sub_solver_algorithm: The name of the optimization algorithm used to solve
                each sub-poblem.
            sub_problem_options: The options passed to the sub-problem optimization
                solver.
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
            normalize_design_space=normalize_design_space,
            ineq_tolerance=ineq_tolerance,
            eq_tolerance=eq_tolerance,
            kkt_tol_abs=kkt_tol_abs,
            kkt_tol_rel=kkt_tol_rel,
            sub_problem_options=sub_problem_options,
            **options,
        )

    def _run(self, **options: Any) -> OptimizationResult:
        # Initialize the penalty and the multipliers.
        constraint_violation_k = Infinity
        x0 = self.problem.design_space.get_current_value()
        rho0 = self.RHO_0
        lambda0 = {
            h.name: zeros_like(h(self.problem.design_space.get_current_value()))
            for h in self.problem.get_eq_constraints()
        }
        mu0 = {
            g.name: zeros_like(g(self.problem.design_space.get_current_value()))
            for g in self.problem.get_ineq_constraints()
        }
        x_old = x0
        message = "Maximun number of iterations reached."
        for iteration in range(options["max_iter"]):
            LOGGER.debug("iteration: %s", iteration)
            LOGGER.debug("mu:  %s", mu0)
            LOGGER.debug("lambda:  %s", lambda0)
            LOGGER.debug("constraint violation:  %s", constraint_violation_k)
            LOGGER.debug("penalty:  %s", rho0)
            # Update the Lagrangian.
            lagrangian = self.__get_lagrangian_function(lambda0, mu0, rho0)
            # Get the sub problem.
            dspace = deepcopy(self.problem.design_space)
            dspace.set_current_value(x_old)
            sub_problem = OptimizationProblem(dspace)
            sub_problem.objective = lagrangian
            sub_problem.preprocess_functions()
            # Solve the sub-problem.
            opt = OptimizersFactory().execute(
                sub_problem,
                options["sub_solver_algorithm"],
                **options["sub_problem_options"],
            )
            x_opt = opt.x_opt
            self.problem.design_space.set_current_value(x_opt)
            val_opt, jac_opt = self.problem.evaluate_functions(
                eval_jac=True,
                eval_obj=True,
            )
            gv = [
                atleast_1d(val_opt[constr.name])
                for constr in self.problem.get_ineq_constraints()
            ]
            hv = [
                atleast_1d(val_opt[constr.name])
                for constr in self.problem.get_eq_constraints()
            ]
            mu_vector = [
                atleast_1d(mu0[constr.name])
                for constr in self.problem.get_ineq_constraints()
            ]
            vk = [
                -g_i if -g_i <= mu / rho0 else mu / rho0
                for g_i, mu in zip(gv, mu_vector)
            ]
            if vk:
                vk = concatenate(vk)
            if hv:
                hv = concatenate(hv)
            if iteration == 0 and max(norm(vk), norm(hv)) > 1e-9:
                rho0 *= val_opt[self.problem.objective.name] / max(norm(vk), norm(hv))
            if max(norm(vk), norm(hv)) > self.TAU * constraint_violation_k:
                rho0 *= self.GAMMA
                rho0 = min(rho0, self.RHO_MAX)
            constraint_violation_k = max(norm(vk), norm(hv))
            # update the multipliers.
            for constraint in self.problem.constraints:
                if constraint.name in mu0.keys():
                    mu_1 = mu0[constraint.name] + rho0 * val_opt[constraint.name]
                    mu0[constraint.name] = (mu_1) * heaviside(mu_1, 0.0)
                elif constraint.name in lambda0.keys():
                    lambda0[constraint.name] = (
                        lambda0[constraint.name] + rho0 * val_opt[constraint.name]
                    )
            if norm(x_opt - x_old) <= options["xtol_abs"]:
                message = (
                    "The algorithm converged based on variation of design variables."
                )
                break
            x_old = x_opt

        return self.get_optimum_from_database(message, "DONE.")

    def __get_lagrangian_function(
        self, lambda0: dict[str, ndarray], mu0: dict[str, ndarray], rho0: float
    ) -> MDOFunction:
        """Return the lagrangian function.

        Args:
            lambda0: The lagrangian multipliers for equality constraints.
            mu0: The lagrangian multipliers for inequality constraints.
            rho0: The penalty.

        Returns:
            The lagrangian function.
        """
        lagrangian = deepcopy(self.problem.nonproc_objective)
        for constr in self.problem.nonproc_constraints:
            if constr.f_type == MDOFunction.ConstraintType.INEQ:
                lagrangian += aggregate_positive_sum_square(
                    constr + mu0[constr.name] / rho0, scale=rho0 / 2
                )
            else:
                lagrangian += aggregate_sum_square(
                    constr + lambda0[constr.name] / rho0, scale=rho0 / 2
                )
        return lagrangian
