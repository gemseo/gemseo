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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import numpy as np
import pytest
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from numpy import array


def test_inputs():
    """Tests the formatting of the passed inputs."""
    coeffs_as_list = [1.0, 2.0]
    coeffs_as_vec = array(coeffs_as_list)
    coeffs_as_mat = array([coeffs_as_list])
    with pytest.raises(ValueError):
        MDOLinearFunction(coeffs_as_list, "f")

    MDOLinearFunction(coeffs_as_mat, "f")
    func = MDOLinearFunction(coeffs_as_vec, "f")
    assert (func.coefficients == coeffs_as_mat).all()
    with pytest.raises(ValueError):
        MDOLinearFunction(
            coeffs_as_mat,
            "f",
            value_at_zero=array([0.0, 0.0]),
        )
    MDOLinearFunction(coeffs_as_mat, "f", value_at_zero=array([0.0]))
    func = MDOLinearFunction(coeffs_as_mat, "f")
    assert (func.value_at_zero == array([0.0])).all()


def test_args_generation():
    """Tests the generation of arguments strings."""
    # No arguments strings passed
    func = MDOLinearFunction(array([1.0, 2.0, 3.0]), "f")
    args = [
        MDOLinearFunction.DEFAULT_ARGS_BASE + MDOLinearFunction.INDEX_PREFIX + str(i)
        for i in range(3)
    ]
    assert func.args == args
    # Not enough arguments strings passed
    func = MDOLinearFunction(array([1.0, 2.0, 3.0]), "f", args=["u", "v"])
    assert func.args == args
    # Only one argument string passed
    func = MDOLinearFunction(array([1.0, 2.0, 3.0]), "f", args=["u"])
    args = ["u" + MDOLinearFunction.INDEX_PREFIX + str(i) for i in range(3)]
    assert func.args == args
    # Enough arguments strings passed
    func = MDOLinearFunction(array([1.0, 2.0, 3.0]), "f", args=["u1", "u2", "v"])
    assert func.args == ["u1", "u2", "v"]


def test_linear_function():
    """Tests the MDOLinearFunction class."""
    coefs = np.array([0.0, 0.0, -1.0, 2.0, 1.0, 0.0, -9.0])
    linear_fun = MDOLinearFunction(coefs, "f")
    coeffs_str = (MDOFunction.COEFF_FORMAT_1D.format(coeff) for coeff in (2, 9))
    expr = "-x!2 + {}*x!3 + x!4 - {}*x!6".format(*coeffs_str)
    assert linear_fun.expr == expr
    assert linear_fun(np.ones(coefs.size)) == -7.0
    # Jacobian
    jac = linear_fun.jac(np.array([]))
    for i in range(jac.size):
        assert jac[i] == coefs[i]


def test_nd_expression():
    """Tests multi-valued MDOLinearFunction literal expression."""
    coefficients = array([[1.0, 2.0], [3.0, 4.0]])
    value_at_zero = array([5.0, 6.0])
    func = MDOLinearFunction(
        coefficients, "f", args=["x", "y"], value_at_zero=value_at_zero
    )
    coeffs_str = (
        MDOFunction.COEFF_FORMAT_ND.format(coeff) for coeff in (1, 2, 5, 3, 4, 6)
    )
    expr = "[{} {}][x] + [{}]\n[{} {}][y]   [{}]".format(*coeffs_str)
    assert func.expr == expr


def test_mult_linear_function():
    """Tests the multiplication of a standard MDOFunction and an MDOLinearFunction."""
    sqr = MDOFunction(
        lambda x: x[0] ** 2.0,
        name="sqr",
        jac=lambda x: 2.0 * x[0],
        expr="x_0**2.",
        args=["x"],
        dim=1,
    )

    coefs = np.array([2.0])
    linear_fun = MDOLinearFunction(coefs, "f")

    prod = sqr * linear_fun
    x_array = np.array([4.0])
    assert prod(x_array) == 128.0

    numerical_jac = prod.jac(x_array)
    assert numerical_jac[0] == 96.0

    sqr_eq = MDOFunction(
        lambda x: x[0] ** 2.0,
        name="sqr",
        jac=lambda x: 2.0 * x[0],
        expr="x_0**2.",
        args=["x"],
        dim=1,
        f_type="eq",
    )
    prod = sqr * sqr_eq


def test_linear_restriction():
    """Tests the restriction of an MDOLinear function."""
    coefficients = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    value_at_zero = array([7.0, 8.0])
    function = MDOLinearFunction(
        coefficients, "f", args=["x", "y", "z"], value_at_zero=value_at_zero
    )
    frozen_indexes = array([1, 2])
    frozen_values = array([1.0, 2.0])
    restriction = function.restrict(frozen_indexes, frozen_values)
    assert (restriction.coefficients == array([[1.0], [4.0]])).all()
    assert (restriction.value_at_zero == array([15.0, 25.0])).all()
    assert restriction.args == ["x"]
    coeffs_str = (MDOFunction.COEFF_FORMAT_ND.format(val) for val in (1, 15, 4, 25))
    expr = "[{}][x] + [{}]\n[{}]      [{}]".format(*coeffs_str)
    assert restriction.expr == expr
