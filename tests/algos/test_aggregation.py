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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.algos.aggregation.aggregation_func import aggregate_iks
from gemseo.algos.aggregation.aggregation_func import aggregate_ks
from gemseo.algos.aggregation.aggregation_func import aggregate_max
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.api import execute_algo
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.power_2 import Power2
from numpy import allclose
from numpy import array
from numpy import concatenate
from numpy import vstack


def create_problem():
    """Creates a basic sellar problem with vectorized inequality constraints."""
    problem = Power2()
    ineq1 = problem.constraints[0]
    ineq2 = problem.constraints[1]
    eq = problem.constraints[2]

    def cstr(x):
        c = concatenate([ineq1(x), ineq2(x)])
        return c

    def jac(x):
        return vstack([ineq1.jac(x), ineq2.jac(x)])

    func = MDOFunction(cstr, "cstr", jac=jac, f_type="ineq")
    problem.constraints = [func, eq]
    return problem


@pytest.fixture
def sellar_problem():
    """Sellar problem fixture."""
    return create_problem()


def create_pb_alleq():
    """Creates a basic sellar problem with vectorized equality constraints only."""
    problem = Power2()
    constraints = problem.constraints

    def cstr(x):
        c = concatenate([cstr(x) for cstr in constraints])
        return c

    def jac(x):
        return vstack([cstr.jac(x) for cstr in constraints])

    func = MDOFunction(cstr, "cstr", jac=jac, f_type="eq")
    problem.constraints = [func]
    return problem


@pytest.mark.parametrize("method", ["KS", "IKS"])
def test_ks_aggreg(method):
    """Tests KS and IKS aggregation methods compared to no aggregation."""
    algo_options = {"ineq_tolerance": 1e-2, "eq_tolerance": 1e-2}
    problem_ref = create_problem()
    execute_algo(problem_ref, algo_name="SLSQP", algo_options=algo_options)
    ref_sol = problem_ref.solution

    problem = create_problem()
    problem.aggregate_constraint(0, method, rho=300.0, scale=1.0)
    execute_algo(
        problem,
        algo_name="SLSQP",
        algo_options=algo_options,
    )
    sol2 = problem.solution

    assert allclose(ref_sol.x_opt, sol2.x_opt, rtol=1e-2)


def test_wrong_method():
    """Tests unallowed type for aggregation method."""
    problem = create_pb_alleq()
    with pytest.raises(ValueError, match="Unknown method"):
        problem.aggregate_constraint(0, "unknown")


def test_groups(sellar_problem):
    """Test groups aggregation."""
    sellar_problem.aggregate_constraint(0, "KS", rho=300.0, scale=1.0, groups=(0, 1))
    assert len(sellar_problem.constraints) == 3


def test_max_aggreg(sellar_problem):
    """Tests max inequality aggregation method compared to no aggregation."""
    xopt_ref = array([0.79370053, 0.79370053, 0.96548938])
    sellar_problem.aggregate_constraint(0, aggregate_max, scale=2.0)
    execute_algo(sellar_problem, algo_name="SLSQP")
    sol2 = sellar_problem.solution

    assert allclose(sol2.x_opt, xopt_ref, rtol=1e-2)


def test_unknown_method(sellar_problem):
    """Tests error when the aggregation method is wrong."""
    with pytest.raises(ValueError, match="Unknown method"):
        sellar_problem.aggregate_constraint(0, "unknwon")


@pytest.mark.parametrize("indices", [None, [0], [0, 1]])
@pytest.mark.parametrize(
    "aggregation_meth", [aggregate_max, aggregate_ks, aggregate_iks]
)
def test_gradients_ineq(sellar_problem, aggregation_meth, indices):
    """Checks gradients of inequality aggregation methods by finite differences."""
    c = sellar_problem.constraints[0]
    f1 = aggregation_meth(c, indices=indices)
    f1.check_grad(array([0.5, 0.6, 0.2]), error_max=1e-5)


@pytest.mark.parametrize("indices", [None, [0]])
def test_gradients_eq(sellar_problem, indices):
    """Checks gradients of equality aggregation methods by finite differences."""
    c = create_pb_alleq().constraints[0]
    f4 = aggregate_sum_square(c, indices=indices)
    f4.check_grad(array([0.5, 0.6, 0.2]), error_max=1e-5)
