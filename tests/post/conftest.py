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
from gemseo.algos.doe.diagonal_doe.diagonal_doe import DiagonalDOE
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.optimization.rosenbrock import Rosenbrock


@pytest.fixture
def __common_problem():  # noqa: PT005
    """A dummy optimization problem to check post-processors."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=2, lower_bound=0, upper_bound=1, value=0.5)
    problem = OptimizationProblem(design_space)
    func = MDOFunction(sum, "obj")
    func.has_default_name = True
    problem.objective = func
    problem.minimize_objective = False
    func = MDOFunction(lambda x: x * 0.5, "eq")
    func.has_default_name = True
    problem.add_constraint(func, constraint_type="eq")
    func = MDOFunction(lambda x: x * 1.5, "pos")
    func.has_default_name = True
    problem.add_constraint(func, constraint_type="ineq", positive=True)
    func = MDOFunction(lambda x: x * 1.5, "pos")
    func.has_default_name = True
    problem.add_constraint(
        func,
        constraint_type="ineq",
        value=0.5,
        positive=True,
    )
    func = MDOFunction(lambda x: x * 2.5, "neg")
    func.has_default_name = True
    problem.add_constraint(func, constraint_type="ineq")
    func = MDOFunction(lambda x: x * 2.5, "neg")
    func.has_default_name = True
    problem.add_constraint(func, constraint_type="ineq", value=0.5)
    problem.differentiation_method = problem.ApproximationMode.FINITE_DIFFERENCES
    return problem


@pytest.fixture
def three_length_common_problem(__common_problem):  # noqa: PYI063, RUF052
    """The __common_problem sampled three times on a diagonal of its input space."""
    lib = DiagonalDOE()
    lib._algo_name = "DiagonalDOE"
    lib.execute(__common_problem, n_samples=3, eval_jac=True)
    return __common_problem


@pytest.fixture
def common_problem(__common_problem):  # noqa: PYI063, RUF052
    """The __common_problem sampled twice on a diagonal of its input space."""
    lib = DiagonalDOE()
    lib._algo_name = "DiagonalDOE"
    lib.execute(__common_problem, n_samples=2, eval_jac=True)
    return __common_problem


@pytest.fixture
def large_common_problem(__common_problem):  # noqa: PYI063, RUF052
    """The __common_problem sampled 20 times on a diagonal of its input space."""
    lib = DiagonalDOE()
    lib._algo_name = "DiagonalDOE"
    lib.execute(__common_problem, n_samples=20, eval_jac=True)
    return __common_problem


@pytest.fixture
def common_problem_():
    """A dummy optimization problem to check post-processors."""
    problem = Rosenbrock()
    problem.minimize_objective = False
    lib = DiagonalDOE()
    lib._algo_name = "DiagonalDOE"
    lib.execute(problem, n_samples=10, eval_jac=True)
    return problem
