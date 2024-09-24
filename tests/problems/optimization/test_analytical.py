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

from gemseo.algos.base_driver_library import MaxIterReachedException
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.problems.optimization.power_2 import Power2
from gemseo.problems.optimization.rastrigin import Rastrigin
from gemseo.problems.optimization.rosen_mf import RosenMF
from gemseo.problems.optimization.rosenbrock import Rosenbrock


def run_and_test_problem(problem, algo_name="SLSQP") -> None:
    """

    :param problem: param algo_name:  (Default value = "SLSQP")
    :param algo_name:  (Default value = "SLSQP")

    """
    opt = OptimizationLibraryFactory().execute(
        problem, algo_name=algo_name, max_iter=800
    )
    x_opt, f_opt = problem.get_solution()
    assert opt.x_opt == pytest.approx(x_opt, abs=1.0e-3)
    assert opt.f_opt == pytest.approx(f_opt, abs=1.0e-3)

    x_0 = problem.design_space.get_current_value(normalize=True)
    for func in problem.functions:
        with contextlib.suppress(MaxIterReachedException):
            func.check_grad(x_0, step=1e-9, error_max=1e-4)


def test_rastrigin() -> None:
    """"""
    problem = Rastrigin()
    run_and_test_problem(problem)


def test_rosen() -> None:
    """"""
    problem = Rosenbrock()
    run_and_test_problem(problem, "L-BFGS-B")
    Rosenbrock(initial_guess=zeros(2))
    problem = Rosenbrock(scalar_var=True)
    assert "x1" in problem.design_space
    assert "x" not in problem.design_space


def test_power2() -> None:
    """"""
    problem = Power2()
    run_and_test_problem(problem)


def test_rosen_mf() -> None:
    disc = RosenMF(3)
    assert disc.check_jacobian(
        {"x": np.zeros(3)},
        derr_approx="finite_differences",
        step=1e-8,
        threshold=1e-4,
    )
