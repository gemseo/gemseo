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
r"""Draw a bar plot from a :class:`.Dataset`."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import arange
from numpy import linspace

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


class BarPlot(DatasetPlot):
    """Barplot visualization."""

    def __init__(
        self,
        dataset: Dataset,
        n_digits: int = 1,
    ) -> None:
        """
        Args:
            n_digits: The number of digits to print the different bar values.
        """  # noqa: D205, D212, D415
        super().__init__(dataset, n_digits=n_digits)

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        # radar solid grid lines
        all_data, _, sizes = self.dataset.get_all_data(False)
        variables_names = self.dataset.columns_names
        dimension = sum(sizes.values())
        series_names = self.dataset.row_names

        if not self.color:
            colormap = plt.cm.get_cmap(self.colormap)
            self.color = [colormap(color) for color in linspace(0, 1, len(all_data))]

        fig, axes = self._get_figure_and_axes(fig, axes)

        axes.tick_params(labelsize=self.font_size)

        discretization = arange(dimension)
        width = 0.75 / len(all_data)
        subplots = []
        positions = [
            discretization + index * width + width / 2 for index in range(dimension)
        ]
        for position, name, data, color in zip(
            positions, series_names, all_data, self.color
        ):
            subplots.append(
                axes.bar(
                    position,
                    data.tolist(),
                    width,
                    label=name,
                    color=color,
                )
            )

        for rects in subplots:
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    pos = 3
                else:
                    pos = -12
                axes.annotate(
                    f"{round(height, self._param.n_digits)}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, pos),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        axes.set_xticks(discretization)
        axes.set_xticklabels(variables_names)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title, fontsize=self.font_size * 1.2)
        axes.legend(fontsize=self.font_size)
        axes.set_axisbelow(True)
        axes.grid()
        return [fig]
