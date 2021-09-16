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
"""A radar plot of constraints."""
from __future__ import division, unicode_literals

import logging
from typing import Sequence

from numpy import vstack, zeros

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.radar_chart import RadarChart as RadarChartPost
from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class RadarChart(OptPostProcessor):
    """Plot on radar style chart a list of constraint functions.

    This class has the responsability of plotting on radar style chart a list
    of constraint functions at a given iteration.

    By default, the iteration is the last one.
    It is possible either to save the plot, to show the plot or both.
    """

    def _plot(
        self,
        constraints_list,  # type: Sequence[str]
        iteration=-1,  # type: int
    ):  # type: (...) -> None
        """
        Args:
            constraints_list: The names of the constraints.
            iteration: The number of iteration to post-process.

        Raises:
            ValueError: If a given element of `constraints_list` is not a
                function. If the `iteration` is larger than the maximum
                iteration or less than -1.
        """
        # retrieve the constraints values
        add_dv = False
        all_constr_names = self.opt_problem.get_constraints_names()

        for func in constraints_list:
            if func not in all_constr_names:
                raise ValueError(
                    "Cannot build radar chart; "
                    "function {} is not among constraints names"
                    " or does not exist.".format(func)
                )

        cstr_values, cstr_names, _ = self.database.get_history_array(
            constraints_list, add_dv=add_dv
        )

        if iteration < -1 or iteration >= len(self.database):
            raise ValueError(
                "iteration should be either equal to -1 or positive and lower than "
                "maximum iteration = {}".format(len(self.database))
            )
        cstr_values = cstr_values[iteration, :].ravel()

        dataset = Dataset("Constraints")
        values = vstack((cstr_values, zeros(len(cstr_values))))
        dataset.add_group(
            dataset.DEFAULT_GROUP,
            values,
            cstr_names,
            {name: 1 for name in cstr_names},
        )
        dataset.row_names = ["computed constraints", "limit constraint"]

        radar = RadarChartPost(dataset)
        radar.linestyle = {"computed constraints": "-", "limit constraint": "--"}
        radar.color = {"computed constraints": "k", "limit constraint": "r"}
        if iteration == -1:
            radar.title = "Constraints at last iteration"
        else:
            radar.title = "Constraints at iteration {}".format(iteration)
        figures = radar.execute(save=False, show=False, display_zero=False)
        for figure in figures:
            self._add_figure(figure)
