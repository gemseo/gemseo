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

from gemseo.api import execute_algo
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.power_2 import Power2
from numpy import allclose
from numpy import vstack
from numpy.core._multiarray_umath import concatenate


def create_problem():
    """Creates a basic sellar problem with vectorized inequality constraints."""
    problem = Power2()
    ineq1 = problem.constraints[0]
    ineq2 = problem.constraints[1]
    eq = problem.constraints[2]

    def cstr(x):
        return concatenate([ineq1(x), ineq2(x)])

    def jac(x):
        return vstack([ineq1.jac(x), ineq2.jac(x)])

    func = MDOFunction(cstr, "cstr", jac=jac, f_type="ineq")
    problem.constraints = [func, eq]
    return problem


def test_exterior_penalty():
    """Tests exterior penalty compared to no aggregation."""
    algo_options = {"ineq_tolerance": 1e-2, "eq_tolerance": 1e-2}
    problem_ref = create_problem()
    execute_algo(problem_ref, algo_name="SLSQP", algo_options=algo_options)
    ref_sol = problem_ref.solution

    problem = create_problem()
    problem.apply_exterior_penalty(
        objective_scale=10.0, scale_inequality=100.0, scale_equality=100.0
    )
    execute_algo(
        problem,
        algo_name="SLSQP",
        max_iter=1000,
    )
    sol2 = problem.solution
    problem_ref.constraints[0](sol2.x_opt)

    assert allclose(ref_sol.x_opt, sol2.x_opt, rtol=1e-2)
