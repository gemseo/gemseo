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

from gemseo import execute_algo
from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock


@pytest.mark.parametrize("problem", [Power2(), Rosenbrock(l_b=0, u_b=1.0)])
def test_kkt_norm_correctly_stored(problem):
    """Test that kkt norm is stored at each iteration requiring gradient."""
    problem.preprocess_functions()
    options = {
        "normalize_design_space": True,
        "kkt_tol_abs": 1e-5,
        "kkt_tol_rel": 1e-5,
        "max_iter": 100,
        "sub_solver_algorithm": "L-BFGS-B",
    }
    problem.reset()
    OptimizersFactory().execute(problem, "Augmented_Lagrangian_order_1", **options)
    kkt_hist = problem.database.get_function_history(problem.KKT_RESIDUAL_NORM)
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
            "algo_options": {
                "kkt_tol_abs": 1e-4,
                "sub_solver_algorithm": "SLSQP",
                "sub_problem_options": {"max_iter": 50},
            },
        },
        {
            "max_iter": 50,
            "algo_options": {
                "sub_solver_algorithm": "SLSQP",
                "max_iter": 50,
                "sub_problem_options": {"max_iter": 50},
            },
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
):
    """Test for lagrange multiplier inequality almost optimum."""
    opt = options.copy()
    problem = analytical_test_2d_ineq.formulation.opt_problem
    if reformulate_constraints_with_slack_var:
        problem = problem.get_reformulated_problem_with_slack_variables()
    execute_algo(problem, algo, "opt", **opt["algo_options"])
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
def test_2d_eq(analytical_test_2d_eq, options, algo):
    """Test for lagrange multiplier inequality almost optimum."""
    opt = options.copy()
    opt["algo"] = algo
    analytical_test_2d_eq.execute(opt)
    problem = analytical_test_2d_eq.formulation.opt_problem
    lagrange = LagrangeMultipliers(problem)
    epsilon = 1e-3
    lag = lagrange.compute(
        problem.solution.x_opt - epsilon * array([0.0, 1.0]),
        ineq_tolerance=2.5 * epsilon,
    )
    assert pytest.approx(lag["equality"][1], 10 * epsilon) == array([-1.0])


@parametrized_options
@parametrized_algo
def test_2d_multiple_eq(analytical_test_2d__multiple_eq, options, algo):
    """Test for lagrange multiplier inequality almost optimum."""
    opt = options.copy()
    opt["algo"] = algo
    analytical_test_2d__multiple_eq.execute(opt)
    problem = analytical_test_2d__multiple_eq.formulation.opt_problem
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
):
    """Test for lagrange multiplier inequality almost optimum."""
    opt = options.copy()
    opt["algo"] = algo
    opt["algo_options"]["sub_problem_constraints"] = subsolver_constraints
    opt["algo_options"]["ftol_rel"] = 1e-3
    problem = analytical_test_2d_mixed_rank_deficient.formulation.opt_problem
    if reformulate_constraints_with_slack_var:
        problem = problem.get_reformulated_problem_with_slack_variables()
    execute_algo(problem, algo, "opt", **opt["algo_options"])
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
