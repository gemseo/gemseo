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

if TYPE_CHECKING:
    from gemseo.algos.opt_problem import OptimizationProblem


class ProgressBar(BaseProgressBar):
    """A progress bar suffixed by metadata."""

    _problem: OptimizationProblem
    """The optimization problem."""

    _tqdm_progress_bar: CustomTqdmProgressBar
    """The tqdm-based progress bar."""

    __is_current_iteration_logged: bool
    """Whether the current iteration is logged."""

    __change_objective_sign: bool
    """Whether to change the sign of the objective value before logging it."""

    def __init__(
        self,
        max_iter: int,
        first_iter: int,
        problem: OptimizationProblem,
        description: str = "",
    ) -> None:
        """
        Args:
            max_iter: The maximum number of iterations.
            first_iter: The first iteration.
            problem: The problem for which the driver will evaluate the functions.
            description: The text prefixing the progress bar.
        """  # noqa: D205 D212 D415
        self._problem = problem
        self._tqdm_progress_bar = CustomTqdmProgressBar(
            total=max_iter,
            desc=description,
            ascii=False,
        )
        self._tqdm_progress_bar.n = first_iter
        self.__is_current_iteration_logged = True
        self.__change_objective_sign = (
            not problem.minimize_objective and not problem.use_standardized_objective
        )

    def set_objective_value(  # noqa D102
        self, x_vect: ndarray | None, current_iter_must_not_be_logged: bool = False
    ) -> None:
        if current_iter_must_not_be_logged:
            if not self.__is_current_iteration_logged:
                self._set_objective_value(
                    self._problem.database.get_x_vect(self._problem.current_iter or -1)
                )
        else:
            self._set_objective_value(x_vect)

    def _set_objective_value(self, x_vect: ndarray | None) -> None:
        """Set the objective value.

        Args:
            x_vect: The design variable values.
                If ``None``, consider the objective at the last iteration.
        """
        if x_vect is None:
            obj = self._problem.objective.last_eval
        else:
            obj = self._problem.database.get_function_value(
                self._problem.objective.name, x_vect
            )

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
            self._tqdm_progress_bar.set_postfix(refresh=True, obj=obj)

    def finalize_iter_observer(self):  # noqa D102
        if not self.__is_current_iteration_logged:
            self.set_objective_value(
                self._problem.database.get_x_vect(self._problem.current_iter or -1)
            )
        self._tqdm_progress_bar.leave = False
        self._tqdm_progress_bar.close()
