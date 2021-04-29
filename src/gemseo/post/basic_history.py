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
"""
A constraints plot
******************
"""
from __future__ import absolute_import, division, unicode_literals

import logging

from gemseo.post.dataset.lines import Lines
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.py23_compat import Path

LOGGER = logging.getLogger(__name__)


class BasicHistory(OptPostProcessor):
    """The **BasicHistory** post processing plots any of the constraint or objective
    functions w.r.t. optimization iterations or sampling snapshots.

    The plot method requires the list of variable names to plot. It is possible either
    to save the plot, to show the plot or both.
    """

    def _plot(
        self,
        data_list,
        show=False,
        save=False,
        file_path="basic_history",
        extension="pdf",
    ):
        """Plots the optimization history: 1 plot for the constraints.

        :param data_list: list of variable names
        :type data_list: list(str)
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param extension: file extension
        :type extension: str
        """
        dataset = self.opt_problem.export_to_dataset(
            "OptimizationProblem", opt_naming=False, by_group=False
        )
        plot = Lines(dataset)
        plot.font_size = 12
        plot.xlabel = "Iterations"
        plot.figsize_x = 11
        plot.figsize_y = 6
        plot.title = "History plot"
        fpath = Path(file_path).with_suffix(".{}".format(extension))
        plot.execute(save=save, show=show, variables=data_list, file_path=fpath)
