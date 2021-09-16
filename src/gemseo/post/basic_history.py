# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

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
        data_list,  # type: Sequence[str]
    ):  # type: (...) -> None
        """
        Args:
            data_list: The names of the variables.
        """
        dataset = self.opt_problem.export_to_dataset(
            "OptimizationProblem", opt_naming=False, by_group=False
        )
        plot = Lines(dataset)
        plot.font_size = 12
        plot.xlabel = "Iterations"
        plot.figsize_x = self.DEFAULT_FIG_SIZE[0]
        plot.figsize_y = self.DEFAULT_FIG_SIZE[1]
        plot.title = "History plot"
        figures = plot.execute(save=False, show=False, variables=data_list)
        for figure in figures:
            self._add_figure(figure)
