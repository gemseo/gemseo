# -*- coding: utf-8 -*-
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
from numpy import hstack, isfinite, vstack, zeros

from gemseo.core.mdofunctions.mdo_function import MDOLinearFunction


def build_constraints_matrices(constraints, constraint_type):
    """Build the constraints matrices associated with passed linear constraints.

    :param constraints: list of linear constraints
    :type constraints: list(MDOLinearFunction)
    :param constraint_type: type of constraint to consider
    :type constraint_type: str
    :returns: left-hand side matrix, right-hand side vector
    :rtype: ndarray or None, ndarray or None
    """
    # Check the constraint type
    valid_types = [MDOLinearFunction.TYPE_INEQ, MDOLinearFunction.TYPE_EQ]
    if constraint_type not in valid_types:
        raise ValueError(
            "{} is not among valid constraint types {}".format(
                constraint_type, " ".join(valid_types)
            )
        )

    # Filter the constraints to consider
    cstr_list = [cstr for cstr in constraints if cstr.f_type == constraint_type]
    if not cstr_list:
        return None, None

    # Check that the constraint are linear
    for cstr in cstr_list:
        if not isinstance(cstr, MDOLinearFunction):
            raise TypeError(
                'Constraint "{}" is not an MDOLinearFunction'.format(cstr.name)
            )

    # Build the constraints matrices
    lhs_matrix = vstack([cstr.coefficients for cstr in cstr_list])
    rhs_vector = hstack([-cstr.value_at_zero for cstr in cstr_list])

    return lhs_matrix, rhs_vector


def build_bounds_matrices(bounds, upper):
    """Return the constraint matrices corresponding to bounds.

    :param bounds: value of the bounds
    :type bounds: ndarray
    :param upper: if True the bounds are considered upper bounds
    :type upper: bool
    :return: left-hand side matrix, right-hand side vector
    :rtype: ndarray, ndarray
    """
    is_finite = isfinite(bounds)
    n_finite = is_finite.sum()
    if n_finite == 0:
        return None, None
    lhs_mat = zeros((n_finite, bounds.size))
    lhs_mat[(range(n_finite), is_finite)] = 1.0 if upper else -1.0
    rhs_vec = bounds[is_finite] if upper else -bounds[is_finite]
    return lhs_mat, rhs_vec
