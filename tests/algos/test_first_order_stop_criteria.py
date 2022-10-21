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

import pytest as pytest
from gemseo.algos.first_order_stop_criteria import is_kkt_residual_norm_reached
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import array
from numpy import ones
from numpy import zeros


@pytest.mark.parametrize("is_optimum", [False, True])
def test_is_kkt_norm_tol_reached_rosenbrock(is_optimum):
    """Test KKT criterion on Rosenbrock problem."""
    problem = Rosenbrock(l_b=0, u_b=1.0)
    design_point = ones(2) if is_optimum else zeros(2)
    assert (
        is_kkt_residual_norm_reached(
            problem, design_point, kkt_abs_tol=1e-2, kkt_rel_tol=1e-2
        )
        == is_optimum
    )
    assert (
        problem.database.get_f_of_x(problem.KKT_RESIDUAL_NORM, design_point) is not None
    )


@pytest.mark.parametrize("is_optimum", [False, True])
def test_is_kkt_norm_tol_reached_power2(is_optimum):
    """Test KKT criterion on Power2 problem."""
    problem = Power2()
    problem.preprocess_functions()
    design_point = (
        array([0.5 ** (1.0 / 3.0), 0.5 ** (1.0 / 3.0), 0.9 ** (1.0 / 3.0)])
        if is_optimum
        else ones(3)
    )
    assert (
        is_kkt_residual_norm_reached(
            problem, design_point, kkt_abs_tol=1e-2, kkt_rel_tol=1e-2
        )
        == is_optimum
    )
    assert (
        problem.database.get_f_of_x(problem.KKT_RESIDUAL_NORM, design_point) is not None
    )


@pytest.mark.parametrize("algorithm", ["NLOPT_SLSQP", "SLSQP"])
@pytest.mark.parametrize("problem", [Power2(), Rosenbrock(l_b=0, u_b=1.0)])
def test_kkt_norm_correctly_stored(algorithm, problem):
    """Test that kkt norm is stored at each iteration requiring gradient."""
    OptimizersFactory().execute(
        problem,
        algorithm,
        normalize_design_space=True,
        kkt_tol_abs=1e-3,
        kkt_tol_rel=1e-3,
    )
    kkt_hist = problem.database.get_func_history(problem.KKT_RESIDUAL_NORM)
    obj_grad_hist = problem.database.get_func_grad_history(problem.objective.name)
    obj_hist = problem.database.get_func_history(problem.objective.name)
    assert len(kkt_hist) == obj_grad_hist.shape[0]
    assert len(obj_hist) >= len(kkt_hist)
