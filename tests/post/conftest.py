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
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.lib_scalable import DiagonalDOE
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.rosenbrock import Rosenbrock


@pytest.fixture
def common_problem():
    """A dummy optimization problem to check post-processors."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=2, l_b=0, u_b=1, value=0.5)
    problem = OptimizationProblem(design_space)
    func = MDOFunction(lambda x: sum(x), "obj")
    func.has_default_name = True
    problem.objective = func
    problem.change_objective_sign()
    func = MDOFunction(lambda x: x * 0.5, "eq")
    func.has_default_name = True
    problem.add_constraint(func, cstr_type="eq")
    func = MDOFunction(lambda x: x * 1.5, "pos")
    func.has_default_name = True
    problem.add_constraint(func, cstr_type="ineq", positive=True)
    func = MDOFunction(lambda x: x * 1.5, "pos")
    func.has_default_name = True
    problem.add_constraint(
        func,
        cstr_type="ineq",
        value=0.5,
        positive=True,
    )
    func = MDOFunction(lambda x: x * 2.5, "neg")
    func.has_default_name = True
    problem.add_constraint(func, cstr_type="ineq")
    func = MDOFunction(lambda x: x * 2.5, "neg")
    func.has_default_name = True
    problem.add_constraint(func, cstr_type="ineq", value=0.5)
    problem.differentiation_method = problem.FINITE_DIFFERENCES
    lib = DiagonalDOE()
    lib.algo_name = "DiagonalDOE"
    lib.execute(problem, n_samples=2, eval_jac=True)
    return problem


@pytest.fixture
def common_problem_():
    """A dummy optimization problem to check post-processors."""
    problem = Rosenbrock()
    problem.change_objective_sign()
    lib = DiagonalDOE()
    lib.algo_name = "DiagonalDOE"
    lib.execute(problem, n_samples=10, eval_jac=True)
    return problem
