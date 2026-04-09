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
from gemseo.algos.doe.diagonal_doe.settings.diagonal_doe_settings import (
    DiagonalDOE_Settings,
)
from gemseo.algos.doe.factory import DOE_LIBRARY_FACTORY
from gemseo.algos.doe.openturns.openturns import OT_LHS_Settings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.problems.optimization.rosenbrock import Rosenbrock


@pytest.fixture
def __common_problem():  # noqa: PT005
    """A dummy optimization problem to check post-processors."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=2, lower_bound=0, upper_bound=1, value=0.5)
    problem = OptimizationProblem(design_space)
    func = ArrayFunction(sum, name="obj")
    func.has_default_name = True
    problem.objective = func
    problem.minimize_objective = False
    func = ArrayFunction(lambda x: x * 0.5, name="eq")
    func.has_default_name = True
    problem.add_constraint(func, constraint_type=ArrayFunction.ConstraintType.EQ)
    func = ArrayFunction(lambda x: x * 1.5, name="pos")
    func.has_default_name = True
    problem.add_constraint(
        func, constraint_type=ArrayFunction.ConstraintType.INEQ, positive=True
    )
    func = ArrayFunction(lambda x: x * 1.5, name="pos")
    func.has_default_name = True
    problem.add_constraint(
        func,
        constraint_type=ArrayFunction.ConstraintType.INEQ,
        value=0.5,
        positive=True,
    )
    func = ArrayFunction(lambda x: x * 2.5, name="neg")
    func.has_default_name = True
    problem.add_constraint(func, constraint_type=ArrayFunction.ConstraintType.INEQ)
    func = ArrayFunction(lambda x: x * 2.5, name="neg")
    func.has_default_name = True
    problem.add_constraint(
        func, constraint_type=ArrayFunction.ConstraintType.INEQ, value=0.5
    )
    problem.differentiation_method = problem.ApproximationMode.FINITE_DIFFERENCES
    return problem


@pytest.fixture
def three_length_common_problem(__common_problem):  # noqa: PYI063, RUF052
    """The __common_problem sampled three times on a diagonal of its input space."""
    lib = DiagonalDOE()
    lib._algo_name = "DiagonalDOE"
    lib.execute(
        __common_problem, settings=DiagonalDOE_Settings(n_samples=3, eval_jac=True)
    )
    return __common_problem


@pytest.fixture
def common_problem(__common_problem):  # noqa: PYI063, RUF052
    """The __common_problem sampled twice on a diagonal of its input space."""
    lib = DiagonalDOE()
    lib._algo_name = "DiagonalDOE"
    lib.execute(
        __common_problem, settings=DiagonalDOE_Settings(n_samples=2, eval_jac=True)
    )
    return __common_problem


@pytest.fixture
def large_common_problem(__common_problem):  # noqa: PYI063, RUF052
    """The __common_problem sampled 20 times on a diagonal of its input space."""
    lib = DiagonalDOE()
    lib._algo_name = "DiagonalDOE"
    lib.execute(
        __common_problem, settings=DiagonalDOE_Settings(n_samples=20, eval_jac=True)
    )
    return __common_problem


@pytest.fixture
def common_problem_():
    """A dummy optimization problem to check post-processors."""
    problem = Rosenbrock()
    problem.minimize_objective = False
    lib = DiagonalDOE()
    lib._algo_name = "DiagonalDOE"
    lib.execute(problem, settings=DiagonalDOE_Settings(n_samples=10, eval_jac=True))
    return problem


@pytest.fixture
def common_problem_lhs_():
    """A dummy optimization problem to check post-processors."""
    problem = Rosenbrock()
    DOE_LIBRARY_FACTORY.execute(
        problem,
        settings=OT_LHS_Settings(n_samples=20, eval_jac=True),
    )
    return problem
