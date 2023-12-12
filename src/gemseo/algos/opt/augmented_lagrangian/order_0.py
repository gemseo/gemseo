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
"""Augmented Lagrangian of order 0."""

from __future__ import annotations

from numpy import heaviside
from numpy import ndarray

from gemseo.algos.opt.augmented_lagrangian.penalty_heuristic import (
    AugmentedLagrangianPenaltyHeuristic,
)
from gemseo.algos.opt.optimization_library import OptimizationAlgorithmDescription


class AugmentedLagrangianOrder0(AugmentedLagrangianPenaltyHeuristic):
    """An augmented Lagrangian algorithm of order 0.

    The Lagrange multipliers are updated thanks to the constraint values solely (no
    gradient used).
    """

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        self.descriptions = {
            "Augmented_Lagrangian_order_0": OptimizationAlgorithmDescription(
                algorithm_name="Augmented_Lagrangian_order_0",
                description=(
                    "Augmented Lagrangian algorithm for gradient-less functions."
                ),
                internal_algorithm_name="Augmented_Lagrangian",
                handle_equality_constraints=True,
                handle_inequality_constraints=True,
                require_gradient=False,
            )
        }

    def _update_lagrange_multipliers(
        self, eq_lag: dict[str, ndarray], ineq_lag: dict[str, ndarray], x_opt: ndarray
    ) -> None:  # noqa:D107
        for constraint in self.problem.constraints:
            if constraint.name in ineq_lag:
                mu_1 = (
                    ineq_lag[constraint.name]
                    + self._rho * self._function_outputs[constraint.name]
                )
                ineq_lag[constraint.name] = (mu_1) * heaviside(mu_1.real, 0.0)
            elif constraint.name in eq_lag:
                eq_lag[constraint.name] = (
                    eq_lag[constraint.name]
                    + self._rho * self._function_outputs[constraint.name]
                )
