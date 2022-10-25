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
#       :author: Simone Coniglio
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Implementation of the Karush-Kuhn-Tucker residual norm stopping criterion."""
from __future__ import annotations

from numpy import ndarray
from numpy.linalg import norm

from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.stop_criteria import TerminationCriterion


class KKTReached(TerminationCriterion):
    """A termination criterion based on the Karush-Kuhn-Tucker (KKT) residual norm."""


def is_kkt_residual_norm_reached(
    opt_problem: OptimizationProblem,
    x_vect: ndarray,
    kkt_abs_tol: float | None = 0.0,
    kkt_rel_tol: float | None = 0.0,
    ineq_tolerance: float = 1e-4,
    reference_residual: float = 1.0,
) -> bool:
    """Test if the KKT conditions are satisfied.

    Args:
        opt_problem: The optimization problem containing an optimization history.
        x_vect: The design point vector where the KKT conditions are tested.
        kkt_abs_tol: The absolute tolerance on the KKT condition residual.
            If ``None``, the absolute criterion is not activated.
        kkt_rel_tol: The relative tolerance on the KKT condition residual.
            If ``None``, the relative criterion is not activated.
        ineq_tolerance: The tolerance to consider a constraint as active.
        reference_residual: The reference KKT condition residual.

    Returns:
        Whether the absolute or the relative KKT residual norm criterion is reached.
    """
    if kkt_abs_tol is None:
        kkt_abs_tol = 0.0
    if kkt_rel_tol is None:
        kkt_rel_tol = 0.0
    return kkt_residual_computation(opt_problem, x_vect, ineq_tolerance) <= max(
        kkt_abs_tol, kkt_rel_tol * reference_residual
    )


def kkt_residual_computation(
    opt_problem: OptimizationProblem,
    x_vect: ndarray,
    ineq_tolerance: float = 1e-4,
) -> float:
    """Compute the KKT residual norm.

    This implementation is inspired from Svanberg Matlab implementation of
    MMA algorithm see :cite:`svanberg1998method`

    Args:
        opt_problem: The optimization problem containing an optimization history.
        x_vect: The design point vector where the KKT conditions are tested.
        ineq_tolerance: The tolerance to consider a constraint as active.

    Returns:
        The KKT residual norm.
    """
    res = opt_problem.database.get_f_of_x(opt_problem.KKT_RESIDUAL_NORM, x_vect)
    if res is not None:
        return res
    lagrange = LagrangeMultipliers(opt_problem)
    if opt_problem.has_constraints():
        lagrange.compute(x_vect, ineq_tolerance=ineq_tolerance)
        res = lagrange.kkt_residual + lagrange.constraint_violation
        opt_problem.database.store(x_vect, {opt_problem.KKT_RESIDUAL_NORM: res})
        return res
    else:
        res = norm(lagrange.get_objective_jacobian(x_vect))
        opt_problem.database.store(x_vect, {opt_problem.KKT_RESIDUAL_NORM: res})
        return res
