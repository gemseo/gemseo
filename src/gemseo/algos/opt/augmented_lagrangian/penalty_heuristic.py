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
"""Augmented Lagrangian penalty update scheme."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Final

if TYPE_CHECKING:
    from gemseo.typing import RealArray

from gemseo.algos.opt.augmented_lagrangian.base_augmented_lagrangian import (
    BaseAugmentedLagrangian,
)


class AugmentedLagrangianPenaltyHeuristic(BaseAugmentedLagrangian):
    """This class implements the penalty update scheme of :cite:`birgin2014practical`.

    This class must be inherited in order to implement the function
    :func:`_update_lagrange_multipliers`.
    """

    __GAMMA: Final[str] = "gamma"
    """The name of `gamma` option, which is the increase of the penalty."""

    __TAU: Final[str] = "tau"
    """The name of `tau` option, which is the threshold for the penalty increase."""

    __MAX_RHO: Final[str] = "max_rho"
    """The name of `max_rho` option, which is the maximum penalty value."""

    def _update_penalty(  # noqa: D107
        self,
        constraint_violation_current_iteration: float | RealArray,
        objective_function_current_iteration: float | RealArray,
        constraint_violation_previous_iteration: float | RealArray,
        current_penalty: float | RealArray,
        iteration: int,
        **options: Any,
    ) -> float:
        if iteration == 0 and constraint_violation_current_iteration > 1e-9:
            gamma = max(
                abs(objective_function_current_iteration)
                / constraint_violation_current_iteration,
                options[self.__GAMMA],
            )
        elif (
            constraint_violation_current_iteration
            > options[self.__TAU] * constraint_violation_previous_iteration
        ):
            gamma = options[self.__GAMMA]
        else:
            gamma = 1.0

        return min(gamma * current_penalty, options.get(self.__MAX_RHO))
