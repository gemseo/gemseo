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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Draw a bar plot from a :class:`.Dataset`. """
from __future__ import division, unicode_literals

from typing import List, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import arange, linspace

from gemseo.post.dataset.dataset_plot import DatasetPlot, DatasetPlotPropertyType


class BarPlot(DatasetPlot):
    """Barplot visualization."""

    def _plot(
        self,
        properties,  # type: Mapping[str,DatasetPlotPropertyType]
        n_digits=1,  # type: int
    ):  # type: (...) -> List[Figure]
        """
        Args:
            n_digits: The number of digits to print the different bar values.
        """
        # radar solid grid lines
        all_data, _, sizes = self.dataset.get_all_data(False, False)
        variables_names = self.dataset.columns_names
        dimension = sum(sizes.values())
        series_names = self.dataset.row_names

        if self.color is None:
            colormap = plt.cm.get_cmap(self.colormap)
            self.color = {
                name: colormap(color)
                for name, color in zip(series_names, linspace(0, 1, len(all_data)))
            }

        fig, axe = plt.subplots()
        axe.tick_params(labelsize=self.font_size)

        discretization = arange(dimension)
        width = 0.75 / len(all_data)
        subplots = []
        positions = [
            discretization + index * width + width / 2 for index in range(dimension)
        ]
        for position, name, data in zip(positions, series_names, all_data):
            data = data.tolist()
            subplots.append(
                axe.bar(
                    position,
                    data,
                    width,
                    label=name,
                    color=self.color[name],
                )
            )

        for rects in subplots:
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    pos = 3
                else:
                    pos = -12
                axe.annotate(
                    "{}".format(round(height, n_digits)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, pos),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        axe.set_xticks(discretization)
        axe.set_xticklabels(variables_names)
        axe.set_title(self.title, fontsize=self.font_size * 1.2)
        axe.legend(fontsize=self.font_size)
        return [fig]
