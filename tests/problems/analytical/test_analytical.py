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
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import contextlib

import numpy as np
import pytest
from numpy import zeros

from gemseo.algos.driver_library import MaxIterReachedException
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rastrigin import Rastrigin
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.problems.analytical.rosenbrock import RosenMF


def run_and_test_problem(problem, algo_name="SLSQP"):
    """

    :param problem: param algo_name:  (Default value = "SLSQP")
    :param algo_name:  (Default value = "SLSQP")

    """
    opt = OptimizersFactory().execute(problem, algo_name=algo_name, max_iter=800)
    x_opt, f_opt = problem.get_solution()
    assert opt.x_opt == pytest.approx(x_opt, abs=1.0e-3)
    assert opt.f_opt == pytest.approx(f_opt, abs=1.0e-3)

    x_0 = problem.get_x0_normalized()
    for func in problem.get_all_functions():
        with contextlib.suppress(MaxIterReachedException):
            func.check_grad(x_0, step=1e-9, error_max=1e-4)


def test_rastrigin():
    """"""
    problem = Rastrigin()
    run_and_test_problem(problem)


def test_rosen():
    """"""
    problem = Rosenbrock()
    run_and_test_problem(problem, "L-BFGS-B")
    Rosenbrock(initial_guess=zeros(2))
    problem = Rosenbrock(scalar_var=True)
    assert "x1" in problem.design_space.variable_names
    assert "x" not in problem.design_space.variable_names


def test_power2():
    """"""
    problem = Power2()
    run_and_test_problem(problem)


def test_rosen_mf():
    disc = RosenMF(3)
    assert disc.check_jacobian(
        {"x": np.zeros(3)},
        derr_approx="finite_differences",
        step=1e-8,
        threshold=1e-4,
    )
