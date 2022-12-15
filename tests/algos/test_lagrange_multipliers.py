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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
from gemseo.utils.derivatives.derivatives_approx import comp_best_step
from numpy import array

DS_FILE = Path(__file__).parent / "sobieski_design_space.txt"
NLOPT_OPTIONS = {
    "eq_tolerance": 1e-11,
    "ftol_abs": 1e-14,
    "ftol_rel": 1e-14,
    "ineq_tolerance": 1e-11,
    "normalize_design_space": False,
    "xtol_abs": 1e-14,
    "xtol_rel": 1e-14,
}


@pytest.fixture
def problem() -> Power2:
    """The Power2 optimization problem."""
    return Power2()


@pytest.mark.parametrize("upper_bound", [False, True])
def test_lagrange_pow2_too_many_acts(problem, upper_bound):
    problem.design_space.set_lower_bound("x", array([-1.0, 0.8, -1.0]))
    if upper_bound:
        problem.design_space.set_current_value(array([0.5, 0.9, -0.5]))
        problem.design_space.set_upper_bound("x", array([1.0, 1.0, 0.9]))

    OptimizersFactory().execute(
        problem, "SLSQP", eq_tolerance=1e-6, ineq_tolerance=1e-6
    )
    lagrange = LagrangeMultipliers(problem)
    x_opt = problem.solution.x_opt
    x_n = problem.design_space.normalize_vect(x_opt)
    problem.evaluate_functions(x_n, eval_jac=True)
    lagrangian = lagrange.compute(x_opt)
    assert ("upper_bounds" in lagrangian) is upper_bound
    assert "lower_bounds" in lagrangian
    assert "equality" in lagrangian
    assert ("inequality" not in lagrangian) is upper_bound


@pytest.mark.parametrize("normalize,eps,tol", [(False, 1e-5, 1e-7), (True, 1e-3, 1e-8)])
def test_lagrangian_validation_lbound_normalize(problem, normalize, eps, tol):
    options = deepcopy(NLOPT_OPTIONS)
    options["normalize_design_space"] = normalize
    problem.design_space.set_lower_bound("x", array([-1.0, 0.8, -1.0]))
    OptimizersFactory().execute(problem, "NLOPT_SLSQP", **options)
    lagrange = LagrangeMultipliers(problem)
    lagrangian = lagrange.compute(problem.solution.x_opt)

    def obj(lb):
        problem = Power2()
        dspace = problem.design_space
        dspace.set_current_value(array([1.0, 0.9, 1.0]))
        dspace.set_lower_bound("x", array([-1.0, 0.8 + lb, -1.0]))
        OptimizersFactory().execute(problem, "NLOPT_SLSQP", **options)
        return problem.solution.f_opt

    df_fd = (obj(eps) - obj(-eps)) / (2 * eps)
    df_anal = lagrangian["lower_bounds"][1]
    err = abs((df_fd - df_anal) / df_anal)
    assert err < tol


def test_lagrangian_validation_eq(problem):
    OptimizersFactory().execute(problem, "NLOPT_SLSQP", **NLOPT_OPTIONS)

    lagrange = LagrangeMultipliers(problem)
    lagrangian = lagrange.compute(problem.solution.x_opt)

    def obj(eq_val):
        problem2 = Power2()
        problem2.constraints[-1] = problem2.constraints[-1] + eq_val
        OptimizersFactory().execute(problem2, "NLOPT_SLSQP", **NLOPT_OPTIONS)
        return problem2.solution.f_opt

    eps = 1e-5
    df_fd = (obj(eps) - obj(-eps)) / (2 * eps)
    df_anal = lagrangian["equality"][1]
    err = abs((df_fd - df_anal) / df_fd)
    assert err < 1e-6


def test_lagrangian_validation_ineq_normalize():
    options = deepcopy(NLOPT_OPTIONS)
    options["normalize_design_space"] = True

    def obj(eq_val):
        problem2 = Power2()
        problem2.constraints[-2] = problem2.constraints[-2] + eq_val
        OptimizersFactory().execute(problem2, "NLOPT_SLSQP", **options)
        return problem2.solution.f_opt

    def obj_grad(eq_val):
        problem = Power2()
        problem.constraints[-2] = problem.constraints[-2] + eq_val
        OptimizersFactory().execute(problem, "NLOPT_SLSQP", **options)
        lagrange = LagrangeMultipliers(problem)
        x_opt = problem.solution.x_opt
        lagrangian = lagrange.compute(x_opt)
        df_anal = lagrangian["inequality"][1][1]

        return df_anal

    eps = 1e-4
    obj_ref = obj(0.0)

    _, _, opt_step = comp_best_step(obj(eps), obj_ref, obj(-eps), eps, 1e-8)
    df_anal = obj_grad(0.0)

    df_fd = (obj(opt_step) - obj(-opt_step)) / (2 * opt_step)
    err = abs((df_fd - df_anal) / df_fd)
    assert err < 1e-3


@pytest.mark.parametrize("constraint_type", ["eq", "ineq"])
def test_lagrangian_constraint(constraint_type):
    disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])

    design_space = SellarDesignSpace()
    scenario = create_scenario(
        disciplines,
        formulation="MDF",
        objective_name="obj",
        design_space=design_space,
    )

    scenario.add_constraint("c_1", constraint_type)
    scenario.add_constraint("c_2", constraint_type)

    scenario.execute({"max_iter": 50, "algo": "SLSQP"})
    problem = scenario.formulation.opt_problem
    lagrange = LagrangeMultipliers(problem)

    lag = lagrange.compute(problem.solution.x_opt)

    if constraint_type == "eq":
        assert lagrange.EQUALITY in lag
        assert len(lag[lagrange.EQUALITY][-1]) == 2

    else:
        assert lagrange.INEQUALITY in lag
        for c_vals in lag.values():
            assert (c_vals[-1] > 0).all()


def test_lagrange_store(problem):
    options = deepcopy(NLOPT_OPTIONS)
    options["normalize_design_space"] = True
    OptimizersFactory().execute(problem, "NLOPT_SLSQP", **options)
    lagrange = LagrangeMultipliers(problem)
    lagrange.active_lb_names = [0]
    lagrange._store_multipliers(np.ones(10))
    lagrange.active_lb_names = []
    lagrange.active_ub_names = [0]
    lagrange._store_multipliers(-1 * np.ones(10))
    lagrange.active_lb_names = []
    lagrange.active_ub_names = []
    lagrange.active_ineq_names = [0]
    lagrange._store_multipliers(-1 * np.ones(10))


@pytest.fixture(params=[0.0, 0.1, 1.0])
def x0(request):
    return request.param


@pytest.fixture(params=[0.0, 0.1, 1.0])
def y0(request):
    return request.param


@pytest.fixture
def analytical_test_2d_ineq(x0, y0):
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline(
        name="2D_test", expressions={"f": "(x-1)**2+(y-1)**2", "g": "x+y-1"}
    )
    ds = DesignSpace()
    ds.add_variable("x", l_b=0.0, u_b=1.0, value=x0)
    ds.add_variable("y", l_b=0.0, u_b=1.0, value=y0)
    scenario = create_scenario(
        disciplines=[disc],
        formulation="DisciplinaryOpt",
        objective_name="f",
        design_space=ds,
    )
    scenario.add_constraint("g", "ineq")
    return scenario


@pytest.fixture
def analytical_test_2d_eq(x0, y0):
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline(
        name="2D_test", expressions={"f": "(x)**2+(y)**2", "g": "x+y-1"}
    )
    ds = DesignSpace()
    ds.add_variable("x", l_b=0.0, u_b=1.0, value=x0)
    ds.add_variable("y", l_b=0.0, u_b=1.0, value=y0)
    scenario = create_scenario(
        disciplines=[disc],
        formulation="DisciplinaryOpt",
        objective_name="f",
        design_space=ds,
    )
    scenario.add_constraint("g", "eq")
    return scenario


@pytest.fixture
def analytical_test_2d_mixed_rank_deficient():
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline(
        name="2D_test",
        expressions={
            "f": "x**2+y**2+z**2",
            "g": "x+y+z-1",
            "h": "(x-1.)**2+(y-1)**2+(z-1)**2-4./3.",
        },
    )
    ds = DesignSpace()
    ds.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    ds.add_variable("y", l_b=0.0, u_b=1.0, value=0.5)
    ds.add_variable("z", l_b=0.0, u_b=1.0, value=0.5)
    scenario = create_scenario(
        disciplines=[disc],
        formulation="DisciplinaryOpt",
        objective_name="f",
        design_space=ds,
    )
    scenario.add_constraint("g", "ineq")
    scenario.add_constraint("h", "eq")
    return scenario


parametrized_options = pytest.mark.parametrize(
    "options",
    [
        {
            "max_iter": 50,
            "algo_options": {"kkt_tol_abs": 1e-3, "kkt_tol_rel": 1e-3},
        },
        {
            "max_iter": 50,
        },
    ],
)
parametrized_algo_ineq = pytest.mark.parametrize(
    "algo_ineq", ["NLOPT_MMA", "SLSQP", "NLOPT_SLSQP"]
)
parametrized_algo_eq = pytest.mark.parametrize("algo_eq", ["SLSQP", "NLOPT_SLSQP"])


@parametrized_options
@parametrized_algo_ineq
def test_2d_ineq(analytical_test_2d_ineq, options, algo_ineq):
    """Test for lagrange multiplier inequality almost optimum."""
    opt = options.copy()
    opt["algo"] = algo_ineq
    analytical_test_2d_ineq.execute(opt)
    problem = analytical_test_2d_ineq.formulation.opt_problem
    lagrange = LagrangeMultipliers(problem)
    epsilon = 1e-3
    lag = lagrange.compute(
        problem.solution.x_opt - epsilon * array([0.0, 1.0]),
        ineq_tolerance=2.5 * epsilon,
    )
    assert pytest.approx(lag["inequality"][1], 1.1 * epsilon) == array([1.0])


@parametrized_options
@parametrized_algo_eq
def test_2d_eq(analytical_test_2d_eq, options, algo_eq):
    """Test for lagrange multiplier inequality almost optimum."""
    opt = options.copy()
    opt["algo"] = algo_eq
    analytical_test_2d_eq.execute(opt)
    problem = analytical_test_2d_eq.formulation.opt_problem
    lagrange = LagrangeMultipliers(problem)
    epsilon = 1e-3
    lag = lagrange.compute(
        problem.solution.x_opt - epsilon * array([0.0, 1.0]),
        ineq_tolerance=2.5 * epsilon,
    )
    assert pytest.approx(lag["equality"][1], 1.1 * epsilon) == array([-1.0])


@parametrized_options
@parametrized_algo_eq
def test_2d_mixed(analytical_test_2d_mixed_rank_deficient, options, algo_eq):
    """Test for lagrange multiplier inequality almost optimum."""
    opt = options.copy()
    opt["algo"] = algo_eq
    analytical_test_2d_mixed_rank_deficient.execute(opt)
    problem = analytical_test_2d_mixed_rank_deficient.formulation.opt_problem
    lagrange = LagrangeMultipliers(problem)
    epsilon = 1e-3
    lag_approx = lagrange.compute(
        problem.solution.x_opt + epsilon * array([0.0, 1.0, 0.0]),
        ineq_tolerance=2.5 * epsilon,
    )
    assert lag_approx["inequality"][1] > 0
