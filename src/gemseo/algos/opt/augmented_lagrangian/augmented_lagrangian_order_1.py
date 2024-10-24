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
"""Augmented Lagrangian of order 1."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.lagrange_multipliers import LagrangeMultipliers
from gemseo.algos.opt.augmented_lagrangian.penalty_heuristic import (
    AugmentedLagrangianPenaltyHeuristic,
)
from gemseo.algos.opt.augmented_lagrangian.settings.augmented_lagrangian_order_1_settings import (  # noqa: E501
    AugmentedLagrangianOrder1Settings,
)
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription

if TYPE_CHECKING:
    from gemseo import OptimizationProblem
    from gemseo import OptimizationResult
    from gemseo.typing import NumberArray


class AugmentedLagrangianOrder1(AugmentedLagrangianPenaltyHeuristic):
    """An augmented Lagrangian algorithm of order 1.

    The Lagrange multipliers are updated using gradient information
    computed using the :class:`.LagrangeMultipliers` class.
    """

    __lagrange_multiplier_calculator: LagrangeMultipliers
    """The Lagrange multiplier calculator."""

    ALGORITHM_INFOS: ClassVar[dict[str, OptimizationAlgorithmDescription]] = {
        "Augmented_Lagrangian_order_1": OptimizationAlgorithmDescription(
            algorithm_name="Augmented_Lagrangian_order_1",
            description="Augmented Lagrangian algorithm using gradient information",
            internal_algorithm_name="Augmented_Lagrangian",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            require_gradient=True,
            Settings=AugmentedLagrangianOrder1Settings,
        ),
    }

    def __init__(self, algo_name: str = "Augmented_Lagrangian_order_1") -> None:  # noqa:D107
        super().__init__(algo_name)
        self.__lagrange_multiplier_calculator = None

    def _post_run(
        self,
        problem: OptimizationProblem,
        result: OptimizationResult,
        max_design_space_dimension_to_log: int,
        **options: Any,
    ) -> None:
        super()._post_run(problem, result, max_design_space_dimension_to_log, **options)
        # Reset this cached attribute since an algorithm shall be stateless to take
        # full advantage of the algorithm factory cache.
        self.__lagrange_multiplier_calculator = None

    def _update_lagrange_multipliers(
        self,
        eq_lag: dict[str, NumberArray],
        ineq_lag: dict[str, NumberArray],
        x_opt: NumberArray,
    ) -> None:  # noqa:D107
        if self.__lagrange_multiplier_calculator is None:
            self.__lagrange_multiplier_calculator = LagrangeMultipliers(self._problem)
        lag_ms = self.__lagrange_multiplier_calculator.compute(x_opt)
        for constraint in self._problem.constraints:
            if constraint.name in ineq_lag and LagrangeMultipliers.INEQUALITY in lag_ms:
                for var_compo_name, lag_value in zip(
                    lag_ms[LagrangeMultipliers.INEQUALITY][0],
                    lag_ms[LagrangeMultipliers.INEQUALITY][1],
                ):
                    if constraint.name in var_compo_name:
                        if DesignSpace.SEP in var_compo_name:
                            var_component_index = int(
                                var_compo_name.replace(constraint.name, "").replace(
                                    DesignSpace.SEP, ""
                                )
                            )
                            ineq_lag[constraint.name][var_component_index] = lag_value
                        else:
                            ineq_lag[constraint.name] = lag_value

            elif constraint.name in eq_lag and LagrangeMultipliers.EQUALITY in lag_ms:
                for var_compo_name, lag_value in zip(
                    lag_ms[LagrangeMultipliers.EQUALITY][0],
                    lag_ms[LagrangeMultipliers.EQUALITY][1],
                ):
                    if constraint.name in var_compo_name:
                        if DesignSpace.SEP in var_compo_name:
                            var_component_index = int(
                                var_compo_name.replace(constraint.name, "").replace(
                                    DesignSpace.SEP, ""
                                )
                            )
                            eq_lag[constraint.name][var_component_index] = lag_value
                        else:
                            eq_lag[constraint.name] = lag_value
