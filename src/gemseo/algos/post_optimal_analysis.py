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
#       :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Post-optimal analysis."""
from __future__ import annotations

import logging
from typing import Iterable
from typing import Mapping

from numpy import atleast_1d
from numpy import dot
from numpy import hstack
from numpy import ndarray
from numpy import vstack
from numpy import zeros_like
from numpy.linalg.linalg import norm

from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.opt_problem import OptimizationProblem

LOGGER = logging.getLogger(__name__)


class PostOptimalAnalysis:
    r"""Post-optimal analysis of a parameterized optimization problem.

    Consider the parameterized optimization problem below, whose objective and
    constraint functions depend on both the optimization variable :math:`x` and
    a parameter :math:`p`.

    .. math::

        \begin{aligned}
            & \text{Minimize}    & & f(x,p) \\
            & \text{relative to} & & x \\
            & \text{subject to}  & & \left\{\begin{aligned}
                                                & g(x,p)\le0, \\
                                                & h(x,p)=0, \\
                                                & \ell\le x\le u.
                                     \end{aligned}\right.
        \end{aligned}

    Denote :math:`x^\ast(p)` a solution of the problem, which depends on
    :math:`p`.
    The post-optimal analysis consists in computing the following total
    derivative:

    .. math::

        \newcommand{\total}{\mathrm{d}}
        \frac{\total f(x^\ast(p),p)}{\total p}(p)
        =\frac{\partial f}{\partial p}(x^\ast(p),p)
         +\lambda_g^\top\frac{\partial g}{\partial p}(x^\ast(p),p)
         +\lambda_h^\top\frac{\partial h}{\partial p}(x^\ast(p),p),

    where :math:`\lambda_g` and :math:`\lambda_h` are the Lagrange multipliers
    of :math:`x^\ast(p)`.
    N.B. the equality above relies on the assumption that

    .. math::

        \newcommand{\total}{\mathrm{d}}
        \lambda_g^\top\frac{\total g(x^\ast(p),p)}{\total p}(p)=0
        \text{ and }
        \lambda_h^\top\frac{\total h(x^\ast(p),p)}{\total p}(p)=0.
    """
    # Dictionary key for term "Lagrange multipliers dot constraints Jacobian"
    MULT_DOT_CONSTR_JAC = "mult_dot_constr_jac"

    def __init__(self, opt_problem: OptimizationProblem, ineq_tol: bool = None) -> None:
        """
        Args:
            opt_problem: The solved optimization problem to be analyzed.
            ineq_tol: The tolerance to determine active inequality constraints.
                If ``None``, its value is fetched in the optimization problem.

        Raises:
            ValueError: If the optimization problem is not solved.
        """  # noqa: D205, D212, D415
        if opt_problem.solution is None:
            raise ValueError(
                "The post-optimal analysis can only be conducted after the "
                "optimization problem is solved."
            )
        self.lagrange_computer = LagrangeMultipliers(opt_problem)
        # N.B. at creation LagrangeMultipliers checks the optimization problem
        self.opt_problem = opt_problem
        # Get the optimal solution
        self.x_opt = self.opt_problem.design_space.get_current_value()
        # Get the objective name
        outvars = self.opt_problem.objective.outvars
        if len(outvars) != 1:
            raise ValueError("The objective must be single-valued.")
        self.outvars = outvars
        # Set the tolerance on inequality constraints
        if ineq_tol is None:
            self.ineq_tol = self.opt_problem.ineq_tolerance
        else:
            self.ineq_tol = ineq_tol

    def check_validity(
        self,
        total_jac: dict[str, dict[str, ndarray]],
        partial_jac: dict[str, dict[str, ndarray]],
        parameters: list[str],
        threshold: float,
    ):
        """Check whether the assumption for post-optimal validity holds.

        Args:
            total_jac: The total derivatives of the post-optimal constraints.
            partial_jac: The partial derivatives of the constraints.
            parameters: The names of the optimization problem parameters.
            threshold: The tolerance on the validity assumption.
        """
        # Check the Jacobians
        func_names = self.opt_problem.get_constraints_names()
        self._check_jacobians(total_jac, func_names, parameters)
        self._check_jacobians(partial_jac, func_names, parameters)

        # Compute the Lagrange multipliers
        multipliers = self.lagrange_computer.compute(self.x_opt, self.ineq_tol)
        _, mul_ineq = multipliers.get(LagrangeMultipliers.INEQUALITY, ([], []))
        _, mul_eq = multipliers.get(LagrangeMultipliers.EQUALITY, ([], []))

        # Get the array to validate the inequality constraints
        total_ineq_jac = self._get_act_ineq_jac(total_jac, parameters)
        partial_ineq_jac = self._get_act_ineq_jac(partial_jac, parameters)
        ineq_tot, ineq_part, ineq_corr = self._compute_validity(
            total_ineq_jac, partial_ineq_jac, mul_ineq, parameters
        )

        # Get the array to validate the equality constraints
        total_eq_jac = self._get_eq_jac(total_jac, parameters)
        partial_eq_jac = self._get_eq_jac(partial_jac, parameters)
        eq_tot, eq_part, eq_corr = self._compute_validity(
            total_eq_jac, partial_eq_jac, mul_eq, parameters
        )

        # Compute the error
        errors = [arr for arr in [ineq_tot, eq_tot] if arr is not None]
        part_norms = [arr for arr in [ineq_part, eq_part] if arr is not None]
        if errors and part_norms:
            error = norm(vstack(errors))
            part_norm = norm(vstack(part_norms))
            if part_norm > threshold:
                error /= part_norm
        else:
            error = 0.0

        # Assess the validity
        valid = error < threshold
        if valid:
            LOGGER.info("Post-optimality is valid.")
        else:
            LOGGER.info("Post-optimality assumption is wrong by %s%%.", error * 100.0)

        return valid, ineq_corr, eq_corr

    def _compute_validity(
        self,
        total_jac: dict[str, dict[str, ndarray]],
        partial_jac: dict[str, dict[str, ndarray]],
        multipliers: ndarray,
        parameters: list[str],
    ) -> tuple[list[ndarray], list[ndarray], dict[str, ndarray]]:
        """Compute the arrays necessary to the validity check.

        Args:
            total_jac: The total derivatives of the post-optimal constraints.
            partial_jac: The partial derivatives of the constraints.
            multipliers: The Lagrange multiplier.
            parameters: The names of the optimization problem parameters.
        """
        corrections = dict.fromkeys(parameters, 0.0)  # corrections terms
        total_prod_blocks = []
        partial_prod_blocks = []
        for input_name in parameters:
            total_jac_block = total_jac.get(input_name)
            partial_jac_block = partial_jac.get(input_name)
            if total_jac_block is not None and partial_jac_block is not None:
                total_prod_blocks.append(dot(multipliers, total_jac_block))
                partial_prod_blocks.append(dot(multipliers, partial_jac_block))
                corrections[input_name] = -total_prod_blocks[-1]
                if not self.opt_problem.minimize_objective:
                    corrections[input_name] *= -1.0

        total_prod = hstack(total_prod_blocks) if total_prod_blocks else None
        partial_prod = hstack(partial_prod_blocks) if partial_prod_blocks else None
        return total_prod, partial_prod, corrections

    def execute(
        self,
        outputs: Iterable[str],
        inputs: Iterable[str],
        functions_jac: dict[str, dict[str, ndarray]],
    ) -> dict[str, dict[str, ndarray]]:
        """Perform the post-optimal analysis.

        Args:
            outputs: The names of the outputs to differentiate.
            inputs: The names of the inputs w.r.t. which to differentiate.
            functions_jac: The Jacobians of the optimization functions
                w.r.t. the differentiation inputs.

        Returns:
            The Jacobian of the Lagrangian.
        """
        # Check the outputs
        nondifferentiable_outputs = set(outputs) - set(self.outvars)
        if nondifferentiable_outputs:
            nondifferentiable_outputs = ", ".join(nondifferentiable_outputs)
            raise ValueError(
                f"Only the post-optimal Jacobian of {self.outvars[0]} can be computed, "
                f"not the one(s) of {nondifferentiable_outputs}."
            )

        # Check the inputs and Jacobians consistency
        func_names = self.outvars + self.opt_problem.get_constraints_names()
        PostOptimalAnalysis._check_jacobians(functions_jac, func_names, inputs)

        # Compute the Lagrange multipliers
        self._compute_lagrange_multipliers()

        # Compute the Jacobian of the Lagrangian
        return self.compute_lagrangian_jac(functions_jac, inputs)

    @staticmethod
    def _check_jacobians(
        functions_jac: dict[str, dict[str, ndarray]],
        func_names: Iterable[str],
        inputs: Iterable[str],
    ):
        """Check the consistency of the Jacobians with the required inputs.

        Args:
            functions_jac: The Jacobians of the optimization function
                w.r.t. the differentiation inputs.
            func_names: The naemes of the function to differentiate.
            inputs: The names of the inputs w.r.t. which to differentiate.

        Raises:
            ValueError: When the Jacobian is totally or partially missing or malformed.
        """
        # Check the consistency of the Jacobians
        for output_name in func_names:
            jac_out = functions_jac.get(output_name)
            if jac_out is None:
                raise ValueError(f"Jacobian of {output_name} is missing.")
            for input_name in inputs:
                jac_block = jac_out.get(input_name)
                if jac_block is None:
                    raise ValueError(
                        f"Jacobian of {output_name} "
                        f"with respect to {input_name} is missing."
                    )
                if not isinstance(jac_block, ndarray):
                    raise ValueError(
                        f"Jacobian of {output_name} "
                        f"with respect to {input_name} must be of type ndarray."
                    )
                if len(jac_block.shape) != 2:
                    raise ValueError(
                        f"Jacobian of {output_name} "
                        f"with respect to {input_name} must be a 2-dimensional ndarray."
                    )

    def _compute_lagrange_multipliers(self) -> None:
        """Compute the Lagrange multipliers at the optimum."""
        self.lagrange_computer.compute(self.x_opt, self.ineq_tol)

    def compute_lagrangian_jac(
        self, functions_jac: dict[str, dict[str, ndarray]], inputs: Iterable[str]
    ) -> dict[str, dict[str, ndarray]]:
        """Compute the Jacobian of the Lagrangian.

        Args:
            functions_jac: The Jacobians of the optimization function
                w.r.t. the differentiation inputs.
            inputs: The names of the inputs w.r.t. which to differentiate.

        Returns:
            The Jacobian of the Lagrangian.
        """
        # Get the Lagrange multipliers
        multipliers = self.lagrange_computer.lagrange_multipliers
        if multipliers is None:
            self._compute_lagrange_multipliers()
            multipliers = self.lagrange_computer.lagrange_multipliers
        _, mul_ineq = multipliers.get(LagrangeMultipliers.INEQUALITY, ([], []))
        _, mul_eq = multipliers.get(LagrangeMultipliers.EQUALITY, ([], []))

        # Build the Jacobians of the active constraints
        act_ineq_jac = self._get_act_ineq_jac(functions_jac, inputs)
        eq_jac = self._get_eq_jac(functions_jac, inputs)

        jac = {self.outvars[0]: dict(), self.MULT_DOT_CONSTR_JAC: dict()}
        for input_name in inputs:
            # Contribution of the objective
            jac_obj_arr = functions_jac[self.outvars[0]][input_name]
            jac_cstr_arr = zeros_like(jac_obj_arr)

            # Contributions of the inequality constraints
            jac_ineq_arr = act_ineq_jac.get(input_name)
            if jac_ineq_arr is not None:
                jac_cstr_arr += dot(mul_ineq, jac_ineq_arr)

            # Contributions of the equality constraints
            jac_eq_arr = eq_jac.get(input_name)
            if jac_eq_arr is not None:
                jac_cstr_arr += dot(mul_eq, jac_eq_arr)

            # Assemble the Jacobian of the Lagrangian
            if not self.opt_problem.minimize_objective:
                jac_cstr_arr *= -1.0
            jac[self.MULT_DOT_CONSTR_JAC][input_name] = jac_cstr_arr
            jac[self.outvars[0]][input_name] = jac_obj_arr + jac_cstr_arr

        return jac

    def _get_act_ineq_jac(
        self, jacobian: Mapping[str, Mapping[str, ndarray]], input_names: Iterable[str]
    ) -> dict[str, ndarray]:
        """Build the Jacobian of the active inequality constraints for each input name.

        Args:
            jacobian: The Jacobian of the inequality constraints.
            input_names: The names of the differentiation inputs.

        Returns:
            The Jacobian of the active inequality constraints for each input name.
        """
        active_ineq_constraints = self.opt_problem.get_active_ineq_constraints(
            self.x_opt, self.ineq_tol
        )
        input_names_to_jacobians = dict()
        for input_name in input_names:
            jacobians = [
                jacobian[constraint.name][input_name][atleast_1d(components_are_active)]
                for constraint, components_are_active in active_ineq_constraints.items()
                if True in components_are_active
            ]
            if jacobians:
                input_names_to_jacobians[input_name] = vstack(jacobians)

        return input_names_to_jacobians

    def _get_eq_jac(
        self, jacobians: dict[str, dict[str, ndarray]], inputs: Iterable[str]
    ) -> dict[str, ndarray]:
        """Build the Jacobian of the equality constraints.

        Args:
            jacobians: The Jacobians of the equality constraints
                w.r.t. the differentiation inputs.
            inputs: The names of the differentiation inputs.

        Returns:
            The jacobian of the equality constraints.
        """
        eq_constraints = self.opt_problem.get_eq_constraints()
        jacobian = dict()
        if eq_constraints:
            for input_name in inputs:
                jacobian[input_name] = vstack(
                    [jacobians[func.name][input_name] for func in eq_constraints]
                )

        return jacobian
