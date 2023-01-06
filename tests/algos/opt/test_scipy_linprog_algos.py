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
#        :author: Benoit Pauwels
from __future__ import annotations

from unittest.case import TestCase

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.lib_scipy_linprog import ScipyLinprog
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import allclose
from numpy import array


class TestScipyLinprog(TestCase):
    """"""

    OPT_LIB_NAME = "ScipyLinprog"

    @staticmethod
    def get_problem():
        # Design space
        design_space = DesignSpace()
        design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
        design_space.add_variable("y", l_b=0.0, u_b=1.0, value=0.5)

        # Optimization functions
        args = ["x", "y"]
        problem = OptimizationProblem(design_space, OptimizationProblem.LINEAR_PB)
        problem.objective = MDOLinearFunction(
            array([1.0, 1.0]), "f", MDOFunction.TYPE_OBJ, args, -1.0
        )
        ineq_constraint = MDOLinearFunction(array([1.0, 1.0]), "g", args=args)
        problem.add_constraint(ineq_constraint, 1.0, MDOFunction.TYPE_INEQ)
        eq_constraint = MDOLinearFunction(array([-2.0, 1.0]), "h", args=args)
        problem.add_constraint(eq_constraint, 0.0, MDOFunction.TYPE_EQ)

        return problem

    def test_init(self):
        factory = OptimizersFactory()
        if factory.is_available(self.OPT_LIB_NAME):
            factory.create(self.OPT_LIB_NAME)

    def test_nonlinear_pb(self):
        problem = Rosenbrock()
        library = OptimizersFactory().create(self.OPT_LIB_NAME)
        adapted_algorithms = library.filter_adapted_algorithms(problem)
        assert not adapted_algorithms

    def test_linprog_algorithms(self):
        library = OptimizersFactory().create(self.OPT_LIB_NAME)
        for algo_name in library.descriptions.keys():
            self.check_algorithm(algo_name)

    def check_algorithm(self, algo_name):
        # Check that the problem must be linear
        problem = Rosenbrock()
        self.assertRaises(ValueError, OptimizersFactory().execute, problem, algo_name)

        # Test on a linear minimization problem
        problem = self.get_problem()
        optim_result = OptimizersFactory().execute(problem, algo_name)
        assert allclose(optim_result.x_opt, array([0.0, 0.0]))
        self.assertAlmostEqual(optim_result.f_opt, -1.0)

        # Test on a linear maximization problem
        problem = self.get_problem()
        problem.change_objective_sign()
        optim_result = OptimizersFactory().execute(problem, algo_name)
        assert allclose(optim_result.x_opt, array([1.0, 2.0]) / 3.0)
        self.assertAlmostEqual(optim_result.f_opt, 0.0)


def test_library_name():
    """Check the library name."""
    assert ScipyLinprog.LIBRARY_NAME == "SciPy"
