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
from numpy import array
from numpy import ndarray
from scipy.sparse import coo_array
from scipy.sparse import csr_array

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction


def test_inputs():
    """Tests the formatting of the passed inputs."""
    coeffs_as_list = [1.0, 2.0]
    coeffs_as_vec = array(coeffs_as_list)
    coeffs_as_mat = array([coeffs_as_list])
    coeffs_as_sparse = coo_array(coeffs_as_vec)
    with pytest.raises(ValueError):
        MDOLinearFunction(coeffs_as_list, "f")
    MDOLinearFunction(coeffs_as_mat, "f")
    func = MDOLinearFunction(coeffs_as_vec, "f")
    assert (func.coefficients == coeffs_as_mat).all()
    func = MDOLinearFunction(coeffs_as_sparse, "f")
    assert (func.coefficients == coeffs_as_sparse).toarray().all()
    with pytest.raises(ValueError):
        MDOLinearFunction(
            coeffs_as_mat,
            "f",
            value_at_zero=array([0.0, 0.0]),
        )
    MDOLinearFunction(coeffs_as_mat, "f", value_at_zero=array([0.0]))
    func = MDOLinearFunction(coeffs_as_mat, "f")
    assert (func.value_at_zero == array([0.0])).all()


@pytest.mark.parametrize(
    "coefficients", [array([1.0, 2.0, 3.0]), coo_array(array([1.0, 2.0, 3.0]))]
)
def test_input_names_generation(coefficients):
    """Tests the generation of arguments strings."""
    # No arguments strings passed
    func = MDOLinearFunction(coefficients, "f")
    input_names = [
        MDOLinearFunction.DEFAULT_BASE_INPUT_NAME
        + MDOLinearFunction.INDEX_PREFIX
        + str(i)
        for i in range(3)
    ]
    assert func.input_names == input_names
    # Not enough arguments strings passed
    func = MDOLinearFunction(coefficients, "f", input_names=["u", "v"])
    assert func.input_names == input_names
    # Only one argument string passed
    func = MDOLinearFunction(coefficients, "f", input_names=["u"])
    input_names = ["u" + MDOLinearFunction.INDEX_PREFIX + str(i) for i in range(3)]
    assert func.input_names == input_names
    # Enough arguments strings passed
    func = MDOLinearFunction(coefficients, "f", input_names=["u1", "u2", "v"])
    assert func.input_names == ["u1", "u2", "v"]


@pytest.mark.parametrize(
    "coefs",
    [
        np.array([0.0, 0.0, -1.0, 2.0, 1.0, 0.0, -9.0]),
        csr_array(np.array([0.0, 0.0, -1.0, 2.0, 1.0, 0.0, -9.0])),
    ],
)
def test_linear_function(coefs):
    """Tests the MDOLinearFunction class."""

    linear_fun = MDOLinearFunction(coefs, "f")
    coeffs_str = (MDOFunction.COEFF_FORMAT_1D.format(coeff) for coeff in (2, 9))
    expr = "-x!2 + {}*x!3 + x!4 - {}*x!6".format(*coeffs_str)
    assert linear_fun.expr == expr
    assert linear_fun(np.ones(max(coefs.shape))) == -7.0
    # Jacobian
    jac = linear_fun.jac(np.array([]))
    if isinstance(jac, ndarray):
        assert (jac == coefs).all()
    else:
        assert (jac == coefs).toarray().all()


@pytest.mark.parametrize(
    "coefficients",
    [array([[1.0, 2.0], [3.0, 4.0]]), coo_array(array([[1.0, 2.0], [3.0, 4.0]]))],
)
def test_nd_expression(coefficients):
    """Tests multi-valued MDOLinearFunction literal expression."""
    value_at_zero = array([5.0, 6.0])
    func = MDOLinearFunction(
        coefficients, "f", input_names=["x", "y"], value_at_zero=value_at_zero
    )
    coeffs_str = (
        MDOFunction.COEFF_FORMAT_ND.format(coeff) for coeff in (1, 2, 5, 3, 4, 6)
    )
    expr = "[{} {}][x] + [{}]\n[{} {}][y]   [{}]".format(*coeffs_str)
    assert func.expr == expr


@pytest.mark.parametrize(
    "coefficients",
    [array([[1.0, 2.0], [3.0, 4.0]]), coo_array(array([[1.0, 2.0], [3.0, 4.0]]))],
)
def test_provided_expression(coefficients):
    """Tests provided expression."""
    func = MDOLinearFunction(
        coefficients,
        "f",
        input_names=["x", "y"],
        value_at_zero=array([5.0, 6.0]),
        expr="",
    )
    assert func.expr == ""
    assert (-func).expr == ""
    assert func.offset(1.0).expr == ""
    assert (
        func.restrict(frozen_indexes=array([0]), frozen_values=array([0.0])).expr == ""
    )


def test_mult_linear_function():
    """Tests the multiplication of a standard MDOFunction and an MDOLinearFunction."""
    sqr = MDOFunction(
        lambda x: x[0] ** 2.0,
        name="sqr",
        jac=lambda x: 2.0 * x[0],
        expr="x_0**2.",
        input_names=["x"],
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
        input_names=["x"],
        dim=1,
        f_type="eq",
    )
    prod = sqr * sqr_eq


@pytest.mark.parametrize(
    "coefficients",
    [
        array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        coo_array(array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
    ],
)
def test_linear_restriction(coefficients):
    """Tests the restriction of an MDOLinear function."""
    value_at_zero = array([7.0, 8.0])
    function = MDOLinearFunction(
        coefficients, "f", input_names=["x", "y", "z"], value_at_zero=value_at_zero
    )
    frozen_indexes = array([1, 2])
    frozen_values = array([1.0, 2.0])
    restriction = function.restrict(frozen_indexes, frozen_values)
    assert (restriction.coefficients == array([[1.0], [4.0]])).all()
    assert (restriction.value_at_zero == array([15.0, 25.0])).all()
    assert restriction.input_names == ["x"]
    coeffs_str = (MDOFunction.COEFF_FORMAT_ND.format(val) for val in (1, 15, 4, 25))
    expr = "[{}][x] + [{}]\n[{}]      [{}]".format(*coeffs_str)
    assert restriction.expr == expr
