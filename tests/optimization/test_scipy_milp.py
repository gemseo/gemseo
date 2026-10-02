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
from numpy import array
from scipy.sparse import csr_array

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.linear_function import LinearFunction
from gemseo.optimization.factory import optimization_library_factory
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_milp import MILP_Settings
from gemseo.optimization.scipy_milp.scipy_milp import ScipyMILP
from gemseo.space.design import DesignSpace


@pytest.fixture(params=[True, False])
def problem_is_feasible(request) -> bool:
    """Whether to construct a feasible optimization problem."""
    return request.param


@pytest.fixture(params=[True, False])
def jacobians_are_sparse(request) -> bool:
    """Whether the Jacobians of array functions are sparse."""
    return request.param


@pytest.fixture
def milp_problem(
    problem_is_feasible: bool, jacobians_are_sparse: bool
) -> OptimizationProblem:
    """A MILP problem.

    Args:
        feasible_problem: Whether the optimization problem is feasible.
        sparse_jacobian: Whether the objective and constraints Jacobians are sparse.
    """
    array_ = csr_array if jacobians_are_sparse else array

    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=1.0)
    design_space.add_integer_variable("y", lower_bound=0, upper_bound=5, value=5)
    design_space.add_integer_variable("z", lower_bound=0, upper_bound=5, value=0)

    args = ["x", "y", "z"]
    problem = OptimizationProblem(design_space)

    problem.objective = LinearFunction(
        array_([[1.0, 1.0, -1]]), "f", ArrayFunction.FunctionType.OBJ, args, -1.0
    )
    ineq_constraint = LinearFunction(
        array_([[0, 0.5, -0.25]]),
        "g",
        input_names=args,
    )
    problem.add_constraint(
        ineq_constraint,
        value=0.333,
        positive=True,
        constraint_type=LinearFunction.ConstraintType.INEQ,
    )
    if not problem_is_feasible:
        problem.add_constraint(
            ineq_constraint,
            value=0.0,
            positive=False,
            constraint_type=LinearFunction.ConstraintType.INEQ,
        )

    problem.add_constraint(
        LinearFunction(
            array_([[-2.0, 1.0, 1.0]]),
            "h",
            input_names=args,
            f_type=LinearFunction.ConstraintType.EQ,
        )
    )
    return problem


def test_init() -> None:
    """Test solver is correctly initialized."""
    factory = optimization_library_factory
    assert factory.is_available("MILP")
    assert isinstance(factory.create("MILP"), ScipyMILP)


@pytest.mark.parametrize(
    "algo_options",
    [
        {"node_limit": 1},
        {"presolve": False, "node_limit": 1},
        {"max_time": 0, "node_limit": 1},
        {"mip_rel_gap": 100, "node_limit": 1},
        {"disp": True, "node_limit": 1},
        {"disp": True},
        {"eq_tolerance": 1e-6},
    ],
)
def test_solve_milp(milp_problem, problem_is_feasible, algo_options) -> None:
    """Test Scipy MILP solver."""
    optim_result = optimization_library_factory.execute(
        milp_problem, settings=MILP_Settings(**algo_options)
    )
    time_limit = algo_options.get("time_limit", 1)
    tolerance = algo_options.get("eq_tolerance", 1e-2)
    if problem_is_feasible and time_limit >= 1:
        assert pytest.approx(array([0.5, 1, 0.0]), abs=tolerance) == optim_result.x_opt
        assert pytest.approx(optim_result.f_opt, abs=tolerance) == 0.5
    else:
        assert pytest.approx(array([1.0, 5, 0]), abs=tolerance) == optim_result.x_opt


def test_design_space_that_is_not_the_unit_box() -> None:
    """Check a mixed-integer program over a box that is not the unit one.

    This library reads the coefficients of the functions
    and hands the solver the bounds of the space,
    so the two have to describe the same space.
    A design space that is already the unit box hides a mismatch between them.
    """
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=-2.0, upper_bound=6.0, value=0.0)
    design_space.add_integer_variable("i", lower_bound=-3, upper_bound=7, value=0)

    problem = OptimizationProblem(design_space)
    problem.objective = LinearFunction(array([[1.0, 2.0]]), "obj", value_at_zero=0.0)
    problem.add_constraint(
        LinearFunction(array([[1.0, 1.0]]), "cstr", value_at_zero=3.0),
        constraint_type=LinearFunction.ConstraintType.INEQ,
    )

    optimization_result = optimization_library_factory.execute(
        problem, settings=MILP_Settings()
    )

    assert pytest.approx(array([-2.0, -3.0])) == optimization_result.x_opt
    assert pytest.approx(-8.0) == optimization_result.f_opt
