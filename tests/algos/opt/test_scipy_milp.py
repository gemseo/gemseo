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

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.scipy_milp.scipy_milp import ScipyMILP
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction


@pytest.fixture(params=[True, False])
def problem_is_feasible(request) -> bool:
    """Whether to construct a feasible optimization problem."""
    return request.param


@pytest.fixture(params=[True, False])
def jacobians_are_sparse(request) -> bool:
    """Whether the Jacobians of MDO Functions are sparse."""
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
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=1.0)
    design_space.add_variable(
        "y",
        lower_bound=0.0,
        upper_bound=5.0,
        value=5,
        type_=design_space.DesignVariableType.INTEGER,
    )
    design_space.add_variable(
        "z",
        lower_bound=0.0,
        upper_bound=5.0,
        value=0,
        type_=design_space.DesignVariableType.INTEGER,
    )

    args = ["x", "y", "z"]
    problem = OptimizationProblem(design_space)

    problem.objective = MDOLinearFunction(
        array_([[1.0, 1.0, -1]]), "f", MDOFunction.FunctionType.OBJ, args, -1.0
    )
    ineq_constraint = MDOLinearFunction(
        array_([[0, 0.5, -0.25]]),
        "g",
        input_names=args,
        f_type=MDOLinearFunction.ConstraintType.INEQ,
    )
    problem.add_constraint(ineq_constraint, value=0.333, positive=True)
    if not problem_is_feasible:
        problem.add_constraint(ineq_constraint, value=0.0, positive=False)

    problem.add_constraint(
        MDOLinearFunction(
            array_([[-2.0, 1.0, 1.0]]),
            "h",
            input_names=args,
            f_type=MDOLinearFunction.ConstraintType.EQ,
        )
    )
    return problem


def test_init() -> None:
    """Test solver is correctly initialized."""
    factory = OptimizationLibraryFactory()
    assert factory.is_available("ScipyMILP")
    assert isinstance(factory.create("Scipy_MILP"), ScipyMILP)


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
    optim_result = OptimizationLibraryFactory().execute(
        milp_problem, algo_name="Scipy_MILP", **algo_options
    )
    time_limit = algo_options.get("time_limit", 1)
    tolerance = algo_options.get("eq_tolerance", 1e-2)
    if problem_is_feasible and time_limit >= 1:
        assert pytest.approx(array([0.5, 1, 0.0]), abs=tolerance) == optim_result.x_opt
        assert pytest.approx(optim_result.f_opt, abs=tolerance) == 0.5
    else:
        assert pytest.approx(array([1.0, 5, 0]), abs=tolerance) == optim_result.x_opt
