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
"""Tests for the quadratic analytical problem."""
from __future__ import annotations

import re

import numpy
import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.power_2 import Power2
from numpy.testing import assert_equal


@pytest.fixture(scope="module")
def problem() -> Power2:
    """An analytical quadratic problem."""
    return Power2()


@pytest.fixture(scope="module")
def objective(problem) -> MDOFunction:
    """The objective function."""
    return problem.objective


@pytest.fixture(scope="module")
def inequality_1(problem) -> MDOFunction:
    """The first inequality constraint."""
    return problem.constraints[0]


@pytest.fixture(scope="module")
def inequality_2(problem) -> MDOFunction:
    """The second inequality constraint."""
    return problem.constraints[1]


@pytest.fixture(scope="module")
def equality(problem) -> MDOFunction:
    """The equality constraint."""
    return problem.constraints[2]


def test_design_space(problem):
    """Check the design space of the problem."""
    design_space = DesignSpace()
    design_space.add_variable("x", 3, l_b=-1.0, u_b=1.0, value=1.0)
    assert problem.design_space == design_space


def test_initial_value():
    """Check the passing of an initial value."""
    assert_equal(
        Power2(initial_value=0.12).design_space.get_current_value(), [0.12, 0.12, 0.12]
    )


def test_constraints_names(problem):
    """Check the names of the constraints."""
    assert problem.get_constraints_names() == ["ineq1", "ineq2", "eq"]


@pytest.mark.parametrize(
    ["function", "name"],
    [
        ["objective", "pow2"],
        ["inequality_1", "ineq1"],
        ["inequality_2", "ineq2"],
        ["equality", "eq"],
    ],
)
def test_function_name(request, function, name):
    """Check the name of a function."""
    assert request.getfixturevalue(function).name == name


@pytest.mark.parametrize(
    ["function", "type_"],
    [
        ["objective", "obj"],
        ["inequality_1", "ineq"],
        ["inequality_2", "ineq"],
        ["equality", "eq"],
    ],
)
def test_function_type(request, function, type_):
    """Check the type of a function."""
    assert request.getfixturevalue(function).f_type == type_


@pytest.mark.parametrize(
    ["function", "expr", "value"],
    [
        ["objective", "x[0]**2 + x[1]**2 + x[2]**2", 14],
        ["inequality_1", "0.5 - x[0]**3", 1.5],
        ["inequality_2", "0.5 - x[1]**3", 8.5],
        ["equality", "0.9 - x[2]**3", 27.9],
    ],
)
def test_function_expression(request, function, expr, value):
    """Check the consistency between the expression of a function and its value."""
    func = request.getfixturevalue(function)
    assert func.expr == expr
    assert_equal(func(numpy.array([-1, -2, -3])), value)


@pytest.mark.parametrize(
    "function", ["objective", "inequality_1", "inequality_2", "equality"]
)
def test_function_args(request, function):
    """Check the name of the input of a function.."""
    assert request.getfixturevalue(function).args == ["x"]


@pytest.mark.parametrize(
    ["function", "gradient"],
    [
        ["objective", [-2, -4, -6]],
        ["inequality_1", [-3, 0, 0]],
        ["inequality_2", [0, -12, 0]],
        ["equality", [0, 0, -27]],
    ],
)
def test_function_gradient(request, function, gradient):
    """Check the gradient of a function."""
    assert_equal(
        request.getfixturevalue(function).jac(numpy.array([-1, -2, -3])), gradient
    )


@pytest.mark.parametrize(["exception_error", "iter_error"], [[False, 0], [True, 1]])
def test_iter_error(exception_error, iter_error):
    """Check the `iter_error` attribute."""
    power2 = Power2(exception_error)
    power2.objective(numpy.zeros(3))
    assert power2.iter_error == iter_error


def test_exception_error():
    """Check the `exception_error` mechanism."""
    power2 = Power2(True)
    power2.objective(numpy.zeros(3))
    power2.objective(numpy.zeros(3))
    power2.objective(numpy.zeros(3))
    with pytest.raises(
        ValueError, match=re.escape("pow2() has already been called three times.")
    ):
        power2.objective(numpy.zeros(3))


def test_solution(problem):
    """Check the objective value at the solution."""
    x_opt, f_opt = problem.get_solution()
    assert problem.objective(x_opt) == pytest.approx(f_opt)
