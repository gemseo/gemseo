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
#        :author: Pierre-Jean Barjhoux
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A constraints plot."""

from __future__ import annotations

import operator
from typing import ClassVar
from typing import Final

from numpy import arange
from numpy import newaxis

from gemseo.post.base_post import BasePost
from gemseo.post.basic_history_settings import BasicHistory_Settings
from gemseo.post.dataset.lines import Lines


class BasicHistory(BasePost[BasicHistory_Settings]):
    """Plot the history of selected constraint, objective and observable functions.

    This post-processor requires the names of these selected outputs.
    """

    Settings: ClassVar[type[BasicHistory_Settings]] = BasicHistory_Settings

    __ITERATION_NAME: Final[str] = ",;:!"
    """The name for the variable iteration in the dataset.

    A name that a user cannot chose for its own variables. Only used in the background.
    """

    def _plot(self, settings: BasicHistory_Settings) -> None:  # noqa: D205, D212, D415
        problem = self.optimization_problem
        dataset = problem.to_dataset(opt_naming=False)
        dataset.add_variable(
            self.__ITERATION_NAME, arange(1, len(dataset) + 1)[:, newaxis]
        )

        variable_names = list(settings.variable_names)
        if self._obj_name in variable_names:
            if problem.use_standardized_objective and not problem.minimize_objective:
                obj_index = variable_names.index(self._obj_name)
                variable_names[obj_index] = self._neg_obj_name

            if self._change_obj:
                dataset.transform_data(operator.neg, variable_names=self._neg_obj_name)
                dataset.rename_variable(self._neg_obj_name, self._obj_name)

        if settings.normalize:
            dataset = dataset.get_normalized()

        plot = Lines(
            dataset,
            abscissa_variable=self.__ITERATION_NAME,
            variables=problem.get_function_names(variable_names),
            use_integer_xticks=True,
        )
        plot.font_size = 12
        plot.xlabel = "Iterations"
        plot.fig_size_x = settings.fig_size[0]
        plot.fig_size_y = settings.fig_size[1]
        plot.title = "History plot"
        self._add_figure(plot)
