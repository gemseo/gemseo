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

import logging
from typing import Sequence

from gemseo.post.dataset.lines import Lines
from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class BasicHistory(OptPostProcessor):
    """Plot the history of selected constraint, objective and observable functions.

    This post-processor requires the names of these selected outputs.
    """

    DEFAULT_FIG_SIZE = (11.0, 6.0)

    def _plot(
        self,
        variable_names: Sequence[str],
    ) -> None:
        """
        Args:
            variable_names: The names of the variables.
        """  # noqa: D205, D212, D415
        problem = self.opt_problem
        dataset = problem.export_to_dataset(opt_naming=False, by_group=False)
        if self._obj_name in variable_names:
            if problem.use_standardized_objective and not problem.minimize_objective:
                obj_index = variable_names.index(self._obj_name)
                variable_names[obj_index] = self._neg_obj_name

            if self._change_obj:
                dataset.transform_variable(self._neg_obj_name, lambda x: -x)
                dataset.rename_variable(self._neg_obj_name, self._obj_name)

        plot = Lines(dataset)
        plot.font_size = 12
        plot.xlabel = "Iterations"
        plot.fig_size_x = self.DEFAULT_FIG_SIZE[0]
        plot.fig_size_y = self.DEFAULT_FIG_SIZE[1]
        plot.title = "History plot"
        figures = plot.execute(
            save=False, variables=problem.get_function_names(variable_names)
        )
        for figure in figures:
            self._add_figure(figure)
