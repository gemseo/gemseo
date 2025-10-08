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

from gemseo.algos._progress_bar.base import BaseProgressBar
from gemseo.algos._progress_bar.custom import CustomTqdmProgressBar
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.progress_bar_data.data import ProgressBarData
from gemseo.algos.progress_bar_data.factory import PROGRESS_BAR_DATA_FACTORY

if TYPE_CHECKING:
    from gemseo.algos.evaluation_problem import EvaluationProblem
    from gemseo.algos.hashable_ndarray import HashableNdarray
    from gemseo.algos.progress_bar_data.base import BaseProgressBarData
    from gemseo.algos.progress_bar_data.factory import ProgressBarDataName


class ProgressBar(BaseProgressBar):
    """A progress bar suffixed by metadata."""

    _problem: EvaluationProblem
    """The evaluation problem."""

    _tqdm_progress_bar: CustomTqdmProgressBar
    """The tqdm-based progress bar."""

    __data: BaseProgressBarData | None
    """The data to be displayed in the progress bar, if any."""

    def __init__(
        self,
        max_iter: int,
        problem: EvaluationProblem,
        description: str = "",
        progress_bar_data_name: ProgressBarDataName = ProgressBarData.__name__,
    ) -> None:
        """
        Args:
            max_iter: The maximum number of iterations.
            problem: The problem for which the driver will evaluate the functions.
            description: The text prefixing the progress bar.
            progress_bar_data_name: The name
                of a :class:`.BaseProgressBarData` class
                to define the data of an optimization problem
                to be displayed in the progress bar.
        """  # noqa: D205 D212 D415
        self._problem = problem
        self._tqdm_progress_bar = CustomTqdmProgressBar(
            total=max_iter,
            desc=description,
            ascii=False,
        )
        self._tqdm_progress_bar.n = problem.evaluation_counter.current
        if isinstance(problem, OptimizationProblem):
            self.__data = PROGRESS_BAR_DATA_FACTORY.create(
                progress_bar_data_name, self._problem
            )
        else:
            # Do not log information
            # when the EvaluationProblem is not an OptimizationProblem.
            self.__data = None

    def update(self, input_value: HashableNdarray | None) -> None:
        self._tqdm_progress_bar.n += 1
        progress_bar_data = {} if self.__data is None else self.__data.get(input_value)
        self._tqdm_progress_bar.set_postfix(refresh=True, **progress_bar_data)

    def close(self) -> None:
        self._tqdm_progress_bar.leave = False
        self._tqdm_progress_bar.close()
