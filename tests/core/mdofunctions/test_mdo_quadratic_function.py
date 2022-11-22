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
"""Tests for the quadratic functions."""
from __future__ import annotations

import pytest
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction
from numpy import array
from numpy.testing import assert_equal


@pytest.fixture(scope="module")
def quadratic_function() -> MDOQuadraticFunction:
    """A quadratic function."""
    return MDOQuadraticFunction(
        array([[1.0, 2.0], [3.0, 4.0]]),
        "f",
        args=("x", "y"),
        linear_coeffs=array([5.0, 6.0]),
        value_at_zero=7.0,
    )


@pytest.fixture(scope="module")
def quadratic_without_linear_term() -> MDOQuadraticFunction:
    """A quadratic function without a linear term."""
    return MDOQuadraticFunction(
        array([[1.0, 2.0], [3.0, 4.0]]), "f", args=("x", "y"), value_at_zero=7.0
    )


@pytest.mark.parametrize("coefficients", ["test", array([1, 2]), array([[1, 2]])])
def test_init(coefficients):
    """Check the initialization of the quadratic function."""
    with pytest.raises(
        ValueError,
        match=(
            "Quadratic coefficients must be passed "
            "as a 2-dimensional square ndarray."
        ),
    ):
        MDOQuadraticFunction(coefficients, "f")


@pytest.mark.parametrize(
    ["function", "value", "gradient"],
    [
        ("quadratic_function", 51.0, [17.0, 27.0]),
        ("quadratic_without_linear_term", 34.0, [12.0, 21.0]),
    ],
)
def test_values(function, value, gradient, request):
    """Check the value of a quadratic function."""
    x_vect = array([1.0, 2.0])
    assert request.getfixturevalue(function)(x_vect) == value
    assert_equal(request.getfixturevalue(function).jac(x_vect), gradient)


@pytest.mark.parametrize(
    ["function", "expr"],
    [
        (
            "quadratic_function",
            "[x]'[{} {}][x] + [{}]'[x] + {}\n"
            "[y] [{} {}][y]   [{}] [y]".format(
                *(
                    MDOFunction.COEFF_FORMAT_ND.format(coefficient)
                    for coefficient in (1, 2, 5, 7, 3, 4, 6)
                )
            ),
        ),
        (
            "quadratic_without_linear_term",
            "[x]'[{} {}][x] + {}\n"
            "[y] [{} {}][y]".format(
                *(
                    MDOFunction.COEFF_FORMAT_ND.format(coefficient)
                    for coefficient in (1, 2, 7, 3, 4)
                )
            ),
        ),
    ],
)
def test_expression(function, expr, request):
    """Check the expression of a quadratic function."""
    assert request.getfixturevalue(function).expr == expr
