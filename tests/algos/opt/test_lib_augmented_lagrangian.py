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

import logging

import pytest
from numpy import array
from numpy import zeros

from gemseo import execute_algo
from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.stop_criteria import KKT_RESIDUAL_NORM
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.optimization.power_2 import Power2
from gemseo.problems.optimization.rosenbrock import Rosenbrock


@pytest.mark.parametrize("problem", [Power2(), Rosenbrock(l_b=0, u_b=1.0)])
def test_kkt_norm_correctly_stored(problem) -> None:
    """Test that kkt norm is stored at each iteration requiring gradient."""
    problem.preprocess_functions()
    options = {
        "normalize_design_space": True,
        "kkt_tol_abs": 1e-5,
        "kkt_tol_rel": 1e-5,
        "max_iter": 100,
        "sub_algorithm_name": "L-BFGS-B",
    }
    problem.reset()
    OptimizationLibraryFactory().execute(
        problem, algo_name="Augmented_Lagrangian_order_1", **options
    )
    kkt_hist = problem.database.get_function_history(KKT_RESIDUAL_NORM)
    obj_grad_hist = problem.database.get_gradient_history(problem.objective.name)
    obj_hist = problem.database.get_function_history(problem.objective.name)
    assert len(kkt_hist) == obj_grad_hist.shape[0]
    assert len(obj_hist) >= len(kkt_hist)
    assert pytest.approx(problem.get_solution()[0], abs=1e-2) == problem.solution.x_opt
    assert pytest.approx(problem.get_solution()[1], abs=1e-2) == problem.solution.f_opt


parametrized_options = pytest.mark.parametrize(
    "options",
    [
        {
            "max_iter": 50,
            "kkt_tol_abs": 1e-4,
            "sub_algorithm_name": "SLSQP",
            "sub_algorithm_settings": {"max_iter": 50},
        },
        {
            "max_iter": 50,
            "sub_algorithm_name": "SLSQP",
            "sub_algorithm_settings": {"max_iter": 50},
        },
    ],
)
parametrized_reformulate = pytest.mark.parametrize(
    "reformulate_constraints_with_slack_var", [True, False]
)
parametrized_algo = pytest.mark.parametrize(
    "algo",
    [
        "Augmented_Lagrangian_order_0",
        "Augmented_Lagrangian_order_1",
    ],
)


@parametrized_options
@parametrized_algo
@parametrized_reformulate
def test_2d_ineq(
    analytical_test_2d_ineq, options, algo, reformulate_constraints_with_slack_var
) -> None:
    """Test for lagrange multiplier inequality almost optimum."""
    if algo == "Augmented_Lagrangian_order_0" and "kkt_tol_abs" in options:
        options.pop("kkt_tol_abs")

    problem = analytical_test_2d_ineq.formulation.optimization_problem
    if reformulate_constraints_with_slack_var:
        problem = problem.get_reformulated_problem_with_slack_variables()
    execute_algo(problem, algo_name=algo, algo_type="opt", **options.copy())
    lagrange = LagrangeMultipliers(problem)
    epsilon = 1e-3
    if reformulate_constraints_with_slack_var:
        coef1 = array([0.0, 1.0, 0.0])
        coef2 = 10
        lag_kind = "equality"
    else:
        coef1 = array([0.0, 1.0])
        coef2 = 1.1
        lag_kind = "inequality"

    lag = lagrange.compute(
        problem.solution.x_opt - epsilon * coef1,
        ineq_tolerance=2.5 * epsilon,
    )

    assert pytest.approx(lag[lag_kind][1], coef2 * epsilon) == array([1.0])


@parametrized_options
@parametrized_algo
def test_2d_eq(analytical_test_2d_eq, options, algo) -> None:
    """Test for lagrange multiplier inequality almost optimum."""
    analytical_test_2d_eq.execute(algo_name=algo, **options.copy())
    problem = analytical_test_2d_eq.formulation.optimization_problem
    lagrange = LagrangeMultipliers(problem)
    epsilon = 1e-3
    lag = lagrange.compute(
        problem.solution.x_opt - epsilon * array([0.0, 1.0]),
        ineq_tolerance=2.5 * epsilon,
    )
    assert pytest.approx(lag["equality"][1], 10 * epsilon) == array([-1.0])


@parametrized_options
@parametrized_algo
def test_2d_multiple_eq(analytical_test_2d__multiple_eq, options, algo) -> None:
    """Test for lagrange multiplier inequality almost optimum."""
    analytical_test_2d__multiple_eq.execute(algo_name=algo, **options.copy())
    problem = analytical_test_2d__multiple_eq.formulation.optimization_problem
    lagrange = LagrangeMultipliers(problem)
    epsilon = 1e-3
    lag = lagrange.compute(
        problem.solution.x_opt - epsilon * array([0.0, 1.0, 0.0]),
        ineq_tolerance=2.5 * epsilon,
    )
    assert pytest.approx(lag["equality"][1][1], 10 * epsilon) == array([-8.0])


@pytest.mark.parametrize(
    "subsolver_constraints",
    [
        (),
        ["g"],
        ["h"],
        ["g", "h"],
    ],
)
@parametrized_options
@parametrized_algo
@parametrized_reformulate
def test_2d_mixed(
    analytical_test_2d_mixed_rank_deficient,
    options,
    algo,
    reformulate_constraints_with_slack_var,
    subsolver_constraints,
) -> None:
    """Test for lagrange multiplier inequality almost optimum."""
    opt = options.copy()
    opt["sub_problem_constraints"] = subsolver_constraints
    opt["ftol_rel"] = 1e-3
    problem = analytical_test_2d_mixed_rank_deficient.formulation.optimization_problem
    if reformulate_constraints_with_slack_var:
        problem = problem.get_reformulated_problem_with_slack_variables()
    execute_algo(problem, algo_name=algo, algo_type="opt", **opt)
    lagrange = LagrangeMultipliers(problem)
    epsilon = 1e-3
    if reformulate_constraints_with_slack_var:
        lag_approx = lagrange.compute(
            problem.solution.x_opt - epsilon * array([0.0, 1.0, 0.0, 0.0]),
            ineq_tolerance=2.5 * epsilon,
        )
    else:
        lag_approx = lagrange.compute(
            problem.solution.x_opt + epsilon * array([0.0, 1.0, 0.0]),
            ineq_tolerance=2.5 * epsilon,
        )
    if not reformulate_constraints_with_slack_var:
        assert lag_approx["inequality"][1] > 0
    else:
        assert lag_approx["equality"][1][0] > 0


def test_n_obj_func_calls():
    """Test that n_obj_func_calls property returns correct number of function calls."""
    problem = Power2()
    problem.preprocess_functions()

    options = {
        "normalize_design_space": True,
        "kkt_tol_abs": 1e-5,
        "kkt_tol_rel": 1e-5,
        "max_iter": 100,
        "sub_algorithm_name": "L-BFGS-B",
    }

    optimizer = OptimizationLibraryFactory().create("Augmented_Lagrangian_order_1")
    problem.reset()
    optimizer.execute(problem, **options)
    n_calls = optimizer.n_obj_func_calls  # Accessing as property, not as a method
    assert n_calls > 0


@pytest.fixture
def rosenbrock_opt_problem():
    # Create the Rosenbrock optimization problem
    problem = Rosenbrock(l_b=0, u_b=1.0)
    problem.preprocess_functions()

    # Define the equality constraint function (works with multiple variables)
    def eq_constraint(x):
        return array([x[0] - 1.0])  # A constraint on x[0] to make it equal to 1.0

    # Define the Jacobian for the equality constraint (partial derivatives)
    def eq_constraint_jac(x):
        jacobian = zeros((1, x.size))  # Create a Jacobian of the correct size
        jacobian[0, 0] = 1.0  # Derivative of (x[0] - 1.0) w.r.t. x[0] is 1
        return jacobian

    # Create the MDOFunction with the Jacobian
    eq_constraint_func = MDOFunction(
        func=eq_constraint, jac=eq_constraint_jac, name="c_eq"
    )

    # Add the equality constraint to the problem
    problem.add_constraint(
        eq_constraint_func,
        value=0.0,
        constraint_type=MDOFunction.ConstraintType.EQ,
    )

    return problem


@pytest.fixture
def optimizer():
    # Create an optimizer instance
    return OptimizationLibraryFactory().create("Augmented_Lagrangian_order_1")


def test_solve_sub_problem_adds_constraints_with_rosenbrock(
    rosenbrock_opt_problem, optimizer
):
    """Test that sub_problem.constraints.append is executed when constraints match."""
    x_init = rosenbrock_opt_problem.design_space.get_current_value()
    options = {
        "normalize_design_space": True,
        "sub_algorithm_name": "SLSQP",
        "sub_algorithm_settings": {},
        "initial_rho": 1.0,
        "max_iter": 10,
    }
    optimizer.execute(rosenbrock_opt_problem, **options)
    optimizer._problem = rosenbrock_opt_problem

    sub_problem_constraints = ["c_eq"]
    lambda0 = {"c_eq": 0.0}
    mu0 = {}

    _, _x_new = optimizer._BaseAugmentedLagrangian__solve_sub_problem(
        lambda0=lambda0,
        mu0=mu0,
        normalize=options["normalize_design_space"],
        sub_problem_constraints=sub_problem_constraints,
        sub_algorithm_name=options["sub_algorithm_name"],
        sub_algorithm_settings=options["sub_algorithm_settings"],
        x_init=x_init,
    )

    assert len(optimizer._sub_problems[-1].constraints) > 0
    constraint_names = [c.name for c in optimizer._sub_problems[-1].constraints]
    assert "c_eq" in constraint_names


def test_solve_sub_problem_triggers_update_options_callback(
    rosenbrock_opt_problem, optimizer
):
    """Test that sub-problem options are updated during sub-problem solving."""
    sub_problem_options = {}
    alm_options = {
        "normalize_design_space": True,
        "max_iter": 100,
        "ftol_abs": 1e-8,
        "ftol_rel": 1e-8,
        "xtol_abs": 1e-8,
        "xtol_rel": 1e-8,
    }

    def mock_update_options_callback(sub_problems, sub_problem_options):
        sub_problem_options.update({
            "normalize_design_space": True,
            "max_iter": 50,
        })

    optimizer.execute(
        rosenbrock_opt_problem,
        sub_algorithm_name="SLSQP",
        sub_algorithm_settings=sub_problem_options,
        update_options_callback=mock_update_options_callback,
        **alm_options,
    )

    optimizer._problem = rosenbrock_opt_problem
    sub_problem_constraints = ["c_eq"]
    lambda0 = {"c_eq": 0.0}
    mu0 = {}

    _, _x_new = optimizer._BaseAugmentedLagrangian__solve_sub_problem(
        lambda0=lambda0,
        mu0=mu0,
        normalize=alm_options["normalize_design_space"],
        sub_problem_constraints=sub_problem_constraints,
        sub_algorithm_name="SLSQP",
        sub_algorithm_settings=sub_problem_options,
        x_init=rosenbrock_opt_problem.design_space.get_current_value(),
    )

    assert "normalize_design_space" in sub_problem_options
    assert sub_problem_options["normalize_design_space"] is True
    assert "max_iter" in sub_problem_options
    assert sub_problem_options["max_iter"] == 50


def test_preconditioner_logging_direct(optimizer, caplog):
    """Test that LOGGER.info is called when 'precond' is in sub_algorithm_settings."""

    # Define sub_algorithm_settings with "precond" present
    sub_algorithm_settings = {"precond": True}

    with caplog.at_level(logging.INFO, logger="gemseo"):
        optimizer._check_for_preconditioner(sub_algorithm_settings)

    assert "Preconditioner Detected" in caplog.text
