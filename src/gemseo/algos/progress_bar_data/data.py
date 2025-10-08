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
"""Progress bar data of an optimization problem."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy import ndarray

from gemseo.algos.progress_bar_data.base import BaseProgressBarData

if TYPE_CHECKING:
    from gemseo.algos.hashable_ndarray import HashableNdarray
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.typing import StrKeyMapping

_NOT_EVALUATED_MESSAGE: Final[str] = "Not evaluated"
"""The message to be logged instead of the objective value when not evaluated."""


class ProgressBarData(BaseProgressBarData):
    """The data of an optimization problem to be displayed in the progress bar."""

    __change_objective_sign: bool
    """Whether to change the sign of the objective value before logging it."""

    __OBJ_KEY: Final[str] = "obj"
    """The key of the progress bar data corresponding to the objective."""

    __FEAS_KEY: Final[str] = "feas"
    """The key of the progress bar data corresponding to the feasibility."""

    def __init__(self, problem: OptimizationProblem) -> None:  # noqa: D107
        super().__init__(problem)
        self.__change_objective_sign = not (
            problem.minimize_objective or problem.use_standardized_objective
        )

    def __get_objective_value(
        self, input_value: HashableNdarray
    ) -> ndarray | float | str:
        """Return the objective value.

        Args:
            input_value: The input value related to this objective value.

        Returns:
            The objective value.
        """
        obj = self._problem.database.get_function_value(
            self._problem.objective.name, input_value
        )
        if obj is None:
            return _NOT_EVALUATED_MESSAGE

        if self.__change_objective_sign:
            obj = -obj

        if isinstance(obj, ndarray) and len(obj) == 1:
            obj = obj[0]

        return obj

    def get(self, input_value: HashableNdarray | None) -> StrKeyMapping:  # noqa: D102
        if input_value is None or not self._problem.database:
            return {}

        feasible = self._problem.history.check_design_point_is_feasible(input_value)[0]
        return {
            self.__OBJ_KEY: self.__get_objective_value(input_value),
            self.__FEAS_KEY: str(feasible),
        }
