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
#       :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Post-optimal analysis
*********************
"""
from __future__ import division, unicode_literals

import logging

from numpy import atleast_1d, dot, hstack, ndarray, vstack, zeros_like
from numpy.linalg.linalg import norm

from gemseo.algos.lagrange_multipliers import LagrangeMultipliers

LOGGER = logging.getLogger(__name__)


class PostOptimalAnalysis(object):
    r"""
    Post-optimal analysis of a parameterized optimization problem.

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

    def __init__(self, opt_problem, ineq_tol=None):
        """Constructor.

        :param opt_problem: solved optimization problem to be analyzed
        :type opt_problem: OptimizationProblem
        :param ineq_tol: tolerance to determine active inequality constraints.
            If None, its value is fetched in the optimization problem.
        """
        self.lagrange_computer = LagrangeMultipliers(opt_problem)
        # N.B. at creation LagrangeMultipliers checks the optimization problem
        self.opt_problem = opt_problem
        # Get the optimal solution
        self.x_opt = self.opt_problem.design_space.get_current_x()
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

    def check_validity(self, total_jac, partial_jac, parameters, threshold):
        """Checks whether the assumption for post-optimal validity holds.

        :param total_jac: total derivatives of the post-optimal constraints
        :type total_jac: dict(dict(ndarray))
        :param partial_jac: partial derivatives of the constraints
        :type total_jac: dict(dict(ndarray))
        :param parameters: names list of the optimization problem parameters
        :type parameters: list(str)
        :param threshold: tolerance on the validity assumption
        :type threshold: number
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
        error_list = [arr for arr in [ineq_tot, eq_tot] if arr is not None]
        part_norm_list = [arr for arr in [ineq_part, eq_part] if arr is not None]
        if error_list and part_norm_list:
            error = norm(vstack(error_list))
            part_norm = norm(vstack(part_norm_list))
            if part_norm > threshold:
                error /= part_norm
        else:
            error = 0.0

        # Assess the validity
        valid = error < threshold
        if valid:
            LOGGER.info("Post-optimality is valid.")
        else:
            msg = "Post-optimality assumption is wrong by "
            msg += str(error * 100.0) + "%."
            LOGGER.info(msg)

        return valid, ineq_corr, eq_corr

    def _compute_validity(self, total_jac, partial_jac, multipliers, parameters):
        """Computes the arrays necessary to the validity check.

        :param total_jac: total derivatives of the post-optimal constraints
        :type total_jac: dict(dict(ndarray))
        :param partial_jac: partial derivatives of the constraints
        :type total_jac: dict(dict(ndarray))
        :param multipliers: Lagrange multipliers
        :type multipliers: ndarray
        :param parameters: names list of the optimization problem parameters
        :type parameters: list(str)
        """
        corrections = dict.fromkeys(parameters, 0.0)  # corrections terms
        total_prod_list = []
        partial_prod_list = []
        for input_name in parameters:
            total_jac_block = total_jac.get(input_name)
            partial_jac_block = partial_jac.get(input_name)
            if total_jac_block is not None and partial_jac_block is not None:
                total_prod_block = dot(multipliers, total_jac_block)
                partial_prod_block = dot(multipliers, partial_jac_block)
                total_prod_list.append(total_prod_block)
                partial_prod_list.append(partial_prod_block)
                # Store the correction term
                corrections[input_name] = -total_prod_block
                if not self.opt_problem.minimize_objective:
                    corrections[input_name] *= -1.0
        total_prod = hstack(total_prod_list) if total_prod_list else None
        partial_prod = hstack(partial_prod_list) if partial_prod_list else None

        return total_prod, partial_prod, corrections

    def execute(self, outputs, inputs, functions_jac):
        """Performs the post-optimal analysis.

        :param outputs: names list of the outputs to differentiate
        :type outputs: list(str)
        :param inputs: names list of the inputs w.r.t. which to differentiate
        :type inputs: list(str)
        :param functions_jac: Jacobians of the optimization functions w.r.t.
            the differentiation inputs
        :type functions_jac: dict(dict(ndarray))
        """
        # Check the outputs
        nondifferentiable_outputs = set(outputs) - set(self.outvars)
        if nondifferentiable_outputs:
            raise ValueError(
                "Only the post-optimal Jacobian of "
                + self.outvars[0]
                + " can be computed"
                + ", not the one(s) of "
                + ", ".join(nondifferentiable_outputs)
                + "."
            )

        # Check the inputs and Jacobians consistency
        func_names = self.outvars + self.opt_problem.get_constraints_names()
        PostOptimalAnalysis._check_jacobians(functions_jac, func_names, inputs)

        # Compute the Lagrange multipliers
        self._compute_lagrange_multipliers()

        # Compute the Jacobian of the Lagrangian
        jac = self.compute_lagrangian_jac(functions_jac, inputs)

        return jac

    @staticmethod
    def _check_jacobians(functions_jac, func_names, inputs):
        """Checks the consistency of the Jacobians with the required inputs.

        :param functions_jac: Jacobians of the optimization function w.r.t. the
            differentiation inputs
        :type functions_jac: dict(dict(ndarray))
        :param func_names: names list of the functions differentiated
        :type func_names: list(str)
        :param inputs: names list of the inputs w.r.t. which to differentiate
        :type inputs: list(str)
        """
        # Check the consistency of the Jacobians
        for output_name in func_names:
            jac_out = functions_jac.get(output_name)
            if jac_out is None:
                raise ValueError("Jacobian of " + output_name + " is missing.")
            for input_name in inputs:
                jac_block = jac_out.get(input_name)
                if jac_block is None:
                    raise ValueError(
                        "Jacobian of "
                        + output_name
                        + " with respect to "
                        + input_name
                        + " is missing."
                    )
                if not isinstance(jac_block, ndarray):
                    raise ValueError(
                        "Jacobian of "
                        + output_name
                        + " with respect to "
                        + input_name
                        + " must be of type ndarray."
                    )
                if len(jac_block.shape) != 2:
                    raise ValueError(
                        "Jacobian of "
                        + output_name
                        + " with respect to "
                        + input_name
                        + " must be a 2-dimensional ndarray."
                    )

    def _compute_lagrange_multipliers(self):
        """Computes the Lagrange multipliers at the solution."""
        self.lagrange_computer.compute(self.x_opt, self.ineq_tol)

    def compute_lagrangian_jac(self, functions_jac, inputs):
        """Computes the Jacobian of the Lagrangian.

        :param functions_jac: Jacobians of the optimization function w.r.t. the
            differentiation inputs
        :type functions_jac: dict(dict(ndarray))
        :param inputs: names list of the inputs w.r.t. which to differentiate
        :type inputs: list(str)
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

    def _get_act_ineq_jac(self, jacobians, inputs):
        """Builds the Jacobian of the active inequality constraints.

        :param jacobians: Jacobians of the inequality constraints w.r.t. the
            differentiation inputs
        :type jacobians: dict(dict(ndarray))
        :param inputs: names list of the differentiation inputs
        :type inputs: list(str)
        """
        # Get the active constraints
        ineq_cstr = self.opt_problem.get_active_ineq_constraints(
            self.x_opt, self.ineq_tol
        )

        # Build the Jacobian
        jac_dict = dict()
        for input_name in inputs:
            jac_input_list = []
            for func, act_set in ineq_cstr.items():
                if True in act_set:
                    jac_block = jacobians[func.name][input_name]
                    jac_block = jac_block[atleast_1d(act_set), :]
                    jac_input_list.append(jac_block)
            if jac_input_list:
                jac_input_arr = vstack(jac_input_list)
                jac_dict[input_name] = jac_input_arr

        return jac_dict

    def _get_eq_jac(self, jacobians, inputs):
        """Builds the Jacobian of the equality constraints.

        :param jacobians: Jacobians of the equality constraints w.r.t. the
            differentiation inputs
        :type jacobians: dict(dict(ndarray))
        :param inputs: names list of the differentiation inputs
        :type inputs: list(str)
        """
        eq_cstr = self.opt_problem.get_eq_constraints()
        jac_dict = dict()
        for input_name in inputs:
            jac_input_list = [jacobians[func.name][input_name] for func in eq_cstr]
            if jac_input_list:
                jac_input_arr = vstack(jac_input_list)
                jac_dict[input_name] = jac_input_arr

        return jac_dict
