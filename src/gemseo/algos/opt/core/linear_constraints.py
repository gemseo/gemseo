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

from typing import Iterable

from numpy import hstack
from numpy import isfinite
from numpy import ndarray
from numpy import vstack
from numpy import zeros

from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction


def build_constraints_matrices(
    constraints: Iterable[MDOLinearFunction], constraint_type: str
) -> tuple[ndarray | None, ndarray | None]:
    """Build the constraints matrices associated with passed linear constraints.

    Args:
        constraints: The linear constraints.
        constraint_type: The type of constraint to consider.

    Returns:
        The left-hand side matrix, the right-hand side vector
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
    constraints = [
        constraint for constraint in constraints if constraint.f_type == constraint_type
    ]
    if not constraints:
        return None, None

    # Check that the constraint are linear
    for constraint in constraints:
        if not isinstance(constraint, MDOLinearFunction):
            raise TypeError(
                f'The constraint "{constraint.name}" is not an MDOLinearFunction.'
            )

    # Build the constraints matrices
    lhs_matrix = vstack([constraint.coefficients for constraint in constraints])
    rhs_vector = hstack([-constraint.value_at_zero for constraint in constraints])

    return lhs_matrix, rhs_vector


def build_bounds_matrices(
    bounds: ndarray, upper: bool
) -> tuple[ndarray | None, ndarray | None]:
    """Return the constraint matrices corresponding to bound.

    Args:
        bounds: The value of the bounds.
        upper: Whether the bounds are considered as upper bounds.

    Returns:
        The left-hand side matrix, the right-hand side vector.
    """
    is_finite = isfinite(bounds)
    n_finite = is_finite.sum()
    if n_finite == 0:
        return None, None
    lhs_mat = zeros((n_finite, bounds.size))
    lhs_mat[(range(n_finite), is_finite)] = 1.0 if upper else -1.0
    rhs_vec = bounds[is_finite] if upper else -bounds[is_finite]
    return lhs_mat, rhs_vec
