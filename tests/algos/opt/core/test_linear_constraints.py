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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Benoit Pauwels
"""Build matrices from linear constraints for solvers."""
from __future__ import annotations

import pytest
from gemseo.algos.opt.core.linear_constraints import build_bounds_matrices
from gemseo.algos.opt.core.linear_constraints import build_constraints_matrices
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import inf


def test_upper_bounds_matrices():
    """Test the building of matrices for upper bound-constraints."""
    upper_bounds = array([1.0, inf, 0.0, inf])
    lhs_mat, rhs_vec = build_bounds_matrices(upper_bounds, upper=True)
    assert allclose(lhs_mat, array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]))
    assert allclose(rhs_vec, array([1.0, 0.0]))
    lhs_mat, rhs_vec = build_bounds_matrices(array([inf, inf]), upper=True)
    assert lhs_mat is None
    assert rhs_vec is None


def test_lower_bounds_matrices():
    """Test the building of matrices for lower bound-constraints."""
    lower_bounds = array([1.0, -inf, 0.0, -inf])
    lhs_mat, rhs_vec = build_bounds_matrices(lower_bounds, upper=False)
    assert allclose(lhs_mat, array([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]]))
    assert allclose(rhs_vec, array([-1.0, 0.0]))
    lhs_mat, rhs_vec = build_bounds_matrices(array([-inf, -inf]), upper=False)
    assert lhs_mat is None
    assert rhs_vec is None


def get_ineq_constraints():
    """Return two linear inequality-constraints.

    :returns: two linear inequality-contraints
    :rtype: MDOLinearFunction, MDOLinearFunction
    """
    ineq_cstr_1 = MDOLinearFunction(
        coefficients=arange(0, 4).reshape((2, 2)),
        value_at_zero=arange(0, 2),
        name="g_1",
        f_type="ineq",
    )
    ineq_cstr_2 = MDOLinearFunction(
        coefficients=arange(4, 10).reshape((3, 2)),
        value_at_zero=arange(2, 5),
        name="g_2",
        f_type="ineq",
    )
    return ineq_cstr_1, ineq_cstr_2


def get_eq_constraints():
    """Return two linear equality-constraints.

    :returns: two linear equality-contraints
    :rtype: MDOLinearFunction, MDOLinearFunction
    """
    eq_cstr_1 = MDOLinearFunction(
        coefficients=arange(10, 14).reshape((2, 2)),
        value_at_zero=arange(5, 7),
        name="h_1",
        f_type="eq",
    )
    eq_cstr_2 = MDOLinearFunction(
        coefficients=arange(14, 20).reshape((3, 2)),
        value_at_zero=arange(7, 10),
        name="h_2",
        f_type="eq",
    )
    return eq_cstr_1, eq_cstr_2


def test_constraint_check():
    """Test the checking of the constraints."""
    ineq_cstr_1, ineq_cstr_2 = get_ineq_constraints()
    # Check constraint type
    with pytest.raises(ValueError):
        build_constraints_matrices([ineq_cstr_1, ineq_cstr_2], "obj")
    # Check function type
    nonlinear_ineq_cstr = MDOFunction(lambda x: x**2, "square", f_type="ineq")
    with pytest.raises(TypeError):
        build_constraints_matrices([ineq_cstr_1, nonlinear_ineq_cstr], "ineq")


def test_inequality_constraints_matrices():
    """Test the building of matrices for inequality-constraints."""
    ineq_cstr_1, ineq_cstr_2 = get_ineq_constraints()
    eq_cstr_1, eq_cstr_2 = get_eq_constraints()
    constraints = (ineq_cstr_1, ineq_cstr_2, eq_cstr_1, eq_cstr_2)
    ineq_lhs_mat, ineq_rhs_vec = build_constraints_matrices(constraints, "ineq")
    eq_lhs_mat, eq_rhs_vec = build_constraints_matrices(constraints, "eq")
    assert allclose(ineq_lhs_mat, arange(0, 10).reshape((5, 2)))
    assert allclose(ineq_rhs_vec, -arange(0, 5))
    assert allclose(eq_lhs_mat, arange(10, 20).reshape((5, 2)))
    assert allclose(eq_rhs_vec, -arange(5, 10))
    ineq_lhs_mat, ineq_rhs_vec = build_constraints_matrices(
        [ineq_cstr_1, ineq_cstr_2], "eq"
    )
    eq_lhs_mat, eq_rhs_vec = build_constraints_matrices([eq_cstr_1, eq_cstr_2], "ineq")
    assert ineq_lhs_mat is None
    assert ineq_rhs_vec is None
    assert eq_lhs_mat is None
    assert eq_rhs_vec is None
