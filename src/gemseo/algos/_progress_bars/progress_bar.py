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
"""A progress bar suffixed by metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import ndarray

from gemseo.algos._progress_bars.base_progress_bar import BaseProgressBar
from gemseo.algos._progress_bars.custom_tqdm_progress_bar import CustomTqdmProgressBar
from gemseo.algos.optimization_problem import OptimizationProblem

if TYPE_CHECKING:
    from gemseo.algos.evaluation_problem import EvaluationProblem


class ProgressBar(BaseProgressBar):
    """A progress bar suffixed by metadata."""

    _problem: EvaluationProblem
    """The evaluation problem."""

    _tqdm_progress_bar: CustomTqdmProgressBar
    """The tqdm-based progress bar."""

    __is_current_iteration_logged: bool
    """Whether the current iteration is logged."""

    __is_optimization_problem: bool
    """Whether the problem is an optimization problem."""

    __change_objective_sign: bool
    """Whether to change the sign of the objective value before logging it.

    Used only when the evaluation problem is an optimization problem.
    """

    def __init__(
        self,
        max_iter: int,
        problem: EvaluationProblem,
        description: str = "",
    ) -> None:
        """
        Args:
            max_iter: The maximum number of iterations.
            problem: The problem for which the driver will evaluate the functions.
            description: The text prefixing the progress bar.
        """  # noqa: D205 D212 D415
        self._problem = problem
        self._tqdm_progress_bar = CustomTqdmProgressBar(
            total=max_iter,
            desc=description,
            ascii=False,
        )
        self._tqdm_progress_bar.n = problem.evaluation_counter.current
        self.__is_current_iteration_logged = True
        if isinstance(problem, OptimizationProblem):
            self.__is_optimization_problem = True
            self.__change_objective_sign = (
                not problem.minimize_objective
                and not problem.use_standardized_objective
            )
        else:
            self.__is_optimization_problem = False

    def set_objective_value(self, x_vect: ndarray | None) -> None:  # noqa: D102
        if x_vect is None:
            if self.__is_current_iteration_logged:
                return

            x_vect = self._problem.database.get_x_vect(
                self._problem.evaluation_counter.current or -1
            )

        self._set_objective_value(x_vect)

    def _set_objective_value(self, x_vect: ndarray) -> None:
        """Set the objective value.

        Args:
            x_vect: The design variable values.
        """
        if self.__is_optimization_problem:
            self._problem: OptimizationProblem
            obj = self._problem.database.get_function_value(
                self._problem.objective.name, x_vect
            )
        else:
            obj = None

        if obj is None:
            self.__is_current_iteration_logged = not self.__is_current_iteration_logged
            if self.__is_current_iteration_logged:
                self._tqdm_progress_bar.n += 1
                obj = "Not evaluated"
        else:
            self.__is_current_iteration_logged = True
            self._tqdm_progress_bar.n += 1
            if self.__change_objective_sign:
                obj = -obj

            if isinstance(obj, ndarray) and len(obj) == 1:
                obj = obj[0]

        if self.__is_current_iteration_logged:
            kwargs = {"obj": obj} if self.__is_optimization_problem else {}
            self._tqdm_progress_bar.set_postfix(refresh=True, **kwargs)

    def finalize_iter_observer(self):  # noqa: D102
        if not self.__is_current_iteration_logged:
            self.set_objective_value(
                self._problem.database.get_x_vect(
                    self._problem.evaluation_counter.current or -1
                )
            )
        self._tqdm_progress_bar.leave = False
        self._tqdm_progress_bar.close()
