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
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import unittest

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.post_optimal_analysis import PostOptimalAnalysis
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import allclose
from numpy import array
from numpy import dot
from numpy.linalg import norm


class TestPostOptimalAnalysis(unittest.TestCase):
    """Tests for the post-optimal analysis. The following parameterized optimization
    problem is considered. Minimize    x^2 + y^2 + p^2 relative to x and y.

    subject to  p - x - y <= 0
                p*x - y = 0
                0 <= x <= 1
                0 <= y <= 1
    If p<0 then the unique minimizer is [0, 0].
    If 0<=p<=1 then the unique optimizer is [1, p]*p/(p+1).
    If p>1 then there is no solution.
    """

    def get_solution(self, p=0.5, minimize=True):
        """Returns the solution of the parameterized optimization problem.

        :param p: parameter of the optimization problem
        :param minimize: if True, returns the solution of the minimization
            problem
        """
        # Set the parameterized optimal sol
        sol = array([1, p]) * p / (p + 1)
        sol_der = array([1, p * (p + 2)]) / (p + 1) ** 2

        factor = 1.0 if minimize else -1.0
        jac_at_sol = {
            "f": {"p": factor * array([[2.0 * p]])},
            "g": {"p": array([[1.0]])},
            "h": {"p": array([[sol[0]]])},
        }

        return sol, sol_der, jac_at_sol

    def get_problem(self, p=0.5, minimize=True, solve=True):
        """Returns the parameterized optimization problem.

        :param p: parameter of the optimization problem
        :param minimize: if True, the problem is formulated as a minimization
            problem problem
        :param solve: if True, the problem is solved
        """
        # Create the design space
        design_space = DesignSpace()
        sol = self.get_solution(p)[0]
        design_space.add_variable("x", l_b=0.0, u_b=1.0, value=sol[0])
        design_space.add_variable("y", l_b=0.0, u_b=1.0, value=sol[1])

        # Create the optimization problem
        opt_problem = OptimizationProblem(design_space)
        if minimize:
            obj_func = MDOFunction(
                lambda xy: norm(xy) ** 2 + p**2,
                "f",
                jac=lambda xy: 2.0 * xy,
                expr="x^2+y^2+p^2",
                args=["x", "y"],
                dim=1,
                outvars=["f"],
            )
        else:
            obj_func = MDOFunction(
                lambda xy: -(norm(xy) ** 2 + p**2),
                "f",
                jac=lambda xy: -2.0 * xy,
                expr="-x^2-y^2-p^2",
                args=["x", "y"],
                dim=1,
                outvars=["f"],
            )
        opt_problem.objective = obj_func
        ineq_func = MDOFunction(
            lambda x: array([p - x[0] - x[1]]),
            "g",
            jac=lambda _: array([-1.0, -1.0]),
            expr="p-x-y",
            args=["x", "y"],
            dim=1,
        )
        opt_problem.add_ineq_constraint(ineq_func)
        eq_func = MDOFunction(
            lambda x: array([x[1] - p * x[0]]),
            "h",
            jac=lambda _: array([p, -1.0]),
            expr="p*x-y",
            args=["x", "y"],
            dim=1,
        )
        opt_problem.add_eq_constraint(eq_func)

        # Solve the problem
        if solve:
            if not minimize:
                opt_problem.change_objective_sign()
            OptimizersFactory().execute(opt_problem, algo_name="SLSQP")

        return opt_problem

    def test_invalid_problem(self):
        """Tests for an exception raise when the passed problem is unsolved or has a
        multi-named objective."""
        # Pass an unsolved problem
        opt_problem = self.get_problem(solve=False)
        self.assertRaises(ValueError, PostOptimalAnalysis, opt_problem)

        # Pass a multi-named objective
        opt_problem = self.get_problem()
        opt_problem.objective.outvars = ["f", "f"]
        self.assertRaises(ValueError, PostOptimalAnalysis, opt_problem)

    def test_invalid_output(self):
        """Tests for an exception raise when the passed outputs are invalid."""
        opt_problem = self.get_problem()
        post_optimal_analyser = PostOptimalAnalysis(opt_problem)
        _, jac_opt, _ = self.get_solution()

        # Pass a non-objective output
        self.assertRaises(
            ValueError, post_optimal_analyser.execute, ["g"], ["p"], jac_opt
        )

    def test_invalid_jacobians(self):
        """Tests for an exception raise when the passed Jacobians are invalid."""
        opt_problem = self.get_problem()
        post_optimal_analyser = PostOptimalAnalysis(opt_problem)

        # Pass Jacobians with a missing output
        jac_at_sol = self.get_solution()[2]
        del jac_at_sol["f"]
        self.assertRaises(
            ValueError, post_optimal_analyser.execute, ["f"], ["p"], jac_at_sol
        )

        # Pass Jacobians with a missing block
        jac_at_sol = self.get_solution()[2]
        del jac_at_sol["f"]["p"]
        self.assertRaises(
            ValueError, post_optimal_analyser.execute, ["f"], ["p"], jac_at_sol
        )

        # Pass Jacobians with a non-ndarray block:
        jac_at_sol = self.get_solution()[2]
        jac_at_sol["f"]["p"] = 1.0
        self.assertRaises(
            ValueError, post_optimal_analyser.execute, ["f"], ["p"], jac_at_sol
        )

        # Pass Jacobians with an ill-shaped block:
        jac_at_sol = self.get_solution()[2]
        jac_at_sol["f"]["p"] = array([1.0])
        self.assertRaises(
            ValueError, post_optimal_analyser.execute, ["f"], ["p"], jac_at_sol
        )
        jac_at_sol["f"]["p"] = array([[[1.0]]])
        self.assertRaises(
            ValueError, post_optimal_analyser.execute, ["f"], ["p"], jac_at_sol
        )

    def test_validity(self):
        """Tests the validity check."""
        # Set up a post-optimal analyzer
        p = 0.5
        opt_problem = self.get_problem(p)
        analyzer = PostOptimalAnalysis(opt_problem)
        sol, sol_der, jac_at_sol = self.get_solution(p)
        total_jac = {
            "g": {"p": array([[-sol_der[0] - sol_der[1] + 1.0]])},
            "h": {"p": array([[p * sol_der[0] - sol_der[1] + sol[0]]])},
        }

        # Check validity for a minimization problem
        valid, ineq_corr, eq_corr = analyzer.check_validity(
            total_jac, jac_at_sol, ["p"], threshold=1e-10
        )
        assert valid
        assert allclose(ineq_corr["p"], 0.0)
        assert allclose(eq_corr["p"], 0.0)

        # Check validity for a maximization problem
        opt_problem = self.get_problem(p, minimize=False)
        valid, ineq_corr, eq_corr = analyzer.check_validity(
            total_jac, jac_at_sol, ["p"], threshold=1e-10
        )
        assert valid
        assert allclose(ineq_corr["p"], 0.0)
        assert allclose(eq_corr["p"], 0.0)

    def test_execute(self):
        """Tests the validity of the post-optimal analysis."""
        p = 1.0

        # Pass a minimization problem
        min_problem = self.get_problem(p)
        sol, sol_der, jac_at_sol = self.get_solution(p)
        jac_target = 2.0 * dot(sol, sol_der) + 2.0 * p
        post_optimal_analyzer = PostOptimalAnalysis(min_problem)
        jac_computed = post_optimal_analyzer.execute(["f"], ["p"], jac_at_sol)
        assert allclose(jac_computed["f"]["p"], jac_target)

        # Pass a maximization problem
        max_problem = self.get_problem(p, minimize=False)
        _, _, jac_at_sol = self.get_solution(p, minimize=False)
        post_optimal_analyzer = PostOptimalAnalysis(max_problem)
        jac_computed = post_optimal_analyzer.execute(["f"], ["p"], jac_at_sol)
        assert allclose(jac_computed["f"]["p"], -jac_target)
