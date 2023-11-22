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

import re

import pytest
from numpy import allclose
from numpy import array
from numpy import complex128
from numpy import concatenate
from numpy import cos
from numpy import sin
from numpy import vstack

from gemseo import execute_algo
from gemseo.algos.aggregation.aggregation_func import aggregate_iks
from gemseo.algos.aggregation.aggregation_func import aggregate_lower_bound_ks
from gemseo.algos.aggregation.aggregation_func import aggregate_max
from gemseo.algos.aggregation.aggregation_func import aggregate_positive_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_upper_bound_ks
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.power_2 import Power2


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


@pytest.fixture()
def sellar_problem():
    """Sellar problem fixture."""
    return create_problem()


def create_pb_alleq():
    """Creates a basic sellar problem with vectorized equality constraints only."""
    problem = Power2()
    constraints = problem.constraints

    def cstr(x):
        return concatenate([cstr(x) for cstr in constraints])

    def jac(x):
        return vstack([cstr.jac(x) for cstr in constraints])

    func = MDOFunction(cstr, "cstr", jac=jac, f_type="eq")
    problem.constraints = [func]
    return problem


@pytest.mark.parametrize(
    "method", ["upper_bound_KS", "lower_bound_KS", "IKS", "POS_SUM"]
)
def test_ks_aggreg(method):
    """Tests KS and IKS aggregation methods compared to no aggregation."""
    algo_options = {"ineq_tolerance": 1e-2, "eq_tolerance": 1e-2}
    problem_ref = create_problem()
    execute_algo(problem_ref, algo_name="SLSQP", algo_options=algo_options)
    ref_sol = problem_ref.solution

    problem = create_problem()
    if method in ["upper_bound_KS", "lower_bound_KS", "IKS"]:
        problem.aggregate_constraint(0, method=method, rho=300.0, scale=1.0)
    else:
        problem.aggregate_constraint(0, method=method, scale=1.0)
    execute_algo(
        problem,
        algo_name="SLSQP",
        algo_options=algo_options,
    )
    sol2 = problem.solution

    assert allclose(ref_sol.x_opt, sol2.x_opt, rtol=1e-2)


def test_wrong_constraint_index():
    """Tests OptimizationProblem.aggregate_constraint with a wrong constraint index."""
    problem = create_pb_alleq()
    with pytest.raises(
        KeyError,
        match=re.escape(
            "The index of the constraint (10) must be lower than "
            "the number of constraints (1)."
        ),
    ):
        problem.aggregate_constraint(10)


@pytest.mark.parametrize(
    "method", ["upper_bound_KS", "lower_bound_KS", "IKS", "POS_SUM"]
)
def test_groups(sellar_problem, method):
    """Test groups aggregation."""
    if method in ["upper_bound_KS", "lower_bound_KS", "IKS"]:
        sellar_problem.aggregate_constraint(
            0, method=method, rho=300.0, scale=1.0, groups=(0, 1)
        )
    else:
        sellar_problem.aggregate_constraint(0, method=method, scale=1.0, groups=(0, 1))
    assert len(sellar_problem.constraints) == 3


def test_max_aggreg(sellar_problem):
    """Tests max inequality aggregation method compared to no aggregation."""
    xopt_ref = array([0.79370053, 0.79370053, 0.96548938])
    sellar_problem.aggregate_constraint(0, method=aggregate_max, scale=2.0)
    execute_algo(sellar_problem, algo_name="SLSQP")
    sol2 = sellar_problem.solution

    assert allclose(sol2.x_opt, xopt_ref, rtol=1e-2)


@pytest.mark.parametrize("indices", [None, [0], [0, 1]])
@pytest.mark.parametrize(
    "aggregation_meth",
    [
        aggregate_max,
        aggregate_upper_bound_ks,
        aggregate_lower_bound_ks,
        aggregate_iks,
        aggregate_positive_sum_square,
    ],
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


def jacobian_function(x):
    return array(concatenate([2 * x, cos(x), -sin(x)]).reshape((3, len(x))))


@pytest.fixture(
    scope="module",
    params=[
        (
            MDOFunction.FunctionType.INEQ,
            aggregate_max,
        ),
        (MDOFunction.FunctionType.INEQ, aggregate_lower_bound_ks),
        (MDOFunction.FunctionType.INEQ, aggregate_upper_bound_ks),
        (MDOFunction.FunctionType.INEQ, aggregate_iks),
        (MDOFunction.FunctionType.INEQ, aggregate_positive_sum_square),
        (MDOFunction.FunctionType.EQ, aggregate_sum_square),
    ],
)
def complex_real_mdo_func_aggregation(
    request,
) -> tuple[MDOFunction, MDOFunction, callable]:
    """Returns two MDOFunctions and a consistent aggregation callable for tests."""
    return (
        MDOFunction(
            lambda x: array([sum(x**2), sum(sin(x)), sum(cos(x))], complex128),
            "c",
            f_type=request.param[0],
            jac=jacobian_function,
        ),
        MDOFunction(
            lambda x: array([sum(x**2), sum(sin(x)), sum(cos(x))]),
            "r",
            f_type=request.param[0],
            jac=jacobian_function,
        ),
        request.param[1],
    )


@pytest.mark.parametrize("indices", [None, [0], [0, 1]])
def test_real_complex(complex_real_mdo_func_aggregation, indices):
    """Test that the aggregation applied to complex and real values is consistent."""
    (
        complex_mdo_function,
        real_mdo_function,
        aggregation_function,
    ) = complex_real_mdo_func_aggregation
    complex_mdo_func_agg = aggregation_function(complex_mdo_function, indices=indices)
    real_mdo_func_agg = aggregation_function(real_mdo_function, indices=indices)
    input_data = array([0.5, 0.6, 0.2])
    complex_mdo_func_agg.check_grad(x_vect=input_data, approximation_mode="ComplexStep")
    assert pytest.approx(complex_mdo_func_agg(input_data)) == real_mdo_func_agg(
        input_data
    )
