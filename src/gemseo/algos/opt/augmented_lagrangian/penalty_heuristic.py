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

from gemseo.algos.opt.augmented_lagrangian.base import BaseAugmentedLagrangian

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy import ndarray


class AugmentedLagrangianPenaltyHeuristic(BaseAugmentedLagrangian):
    """This class implements the penalty update scheme of :cite:`birgin2014practical`.

    This class must be inherited in order to implement the function
    :func:`_update_lagrange_multipliers`.
    """

    __GAMMA: Final[str] = "gamma"
    """The name of `gamma` option, which is the increase of the penalty term."""

    __TAU: Final[str] = "tau"
    """The name of `tau` option, which is the threshold for the penalty term
    increase."""

    __MAX_RHO: Final[str] = "max_rho"
    """The name of `max_rho` option, which is the max penalty value."""

    def _get_options(
        self,
        sub_solver_algorithm: str,
        normalize_design_space: bool = True,
        max_iter: int = 999,
        stop_crit_n_x: int = 3,
        ftol_rel: float = 1e-9,
        ftol_abs: float = 1e-9,
        xtol_rel: float = 1e-9,
        xtol_abs: float = 1e-9,
        max_fun_eval: int = 999,
        eq_tolerance: float = 1e-2,
        ineq_tolerance: float = 1e-4,
        kkt_tol_abs: float | None = None,
        kkt_tol_rel: float | None = None,
        sub_problem_options: Mapping[str, Any] | None = None,
        sub_problem_constraints: Iterable[str] = (),
        initial_rho: float = 10.0,
        tau: float = 0.9,
        gamma: float = 1.5,
        max_rho: float = 10000,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Args:
            tau: The threshold for the penalty term increase.
            gamma: The increase of the penalty term.
            max_rho: The max penalty value.
        """  # noqa: D205, D212
        if sub_problem_options is None:
            sub_problem_options = {}
        return self._process_options(
            tau=tau,
            gamma=gamma,
            max_rho=max_rho,
            sub_solver_algorithm=sub_solver_algorithm,
            normalize_design_space=normalize_design_space,
            max_iter=max_iter,
            stop_crit_n_x=stop_crit_n_x,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            max_fun_eval=max_fun_eval,
            eq_tolerance=eq_tolerance,
            ineq_tolerance=ineq_tolerance,
            kkt_tol_abs=kkt_tol_abs,
            kkt_tol_rel=kkt_tol_rel,
            sub_problem_options=sub_problem_options,
            sub_problem_constraints=sub_problem_constraints,
            initial_rho=initial_rho,
        )

    def _update_penalty(  # noqa: D107
        self,
        constraint_violation_current_iteration: ndarray | float,
        objective_function_current_iteration: ndarray | float,
        constraint_violation_previous_iteration: ndarray | float,
        current_penalty: ndarray | float,
        iteration: int,
        **options: Any,
    ) -> float | ndarray:
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
