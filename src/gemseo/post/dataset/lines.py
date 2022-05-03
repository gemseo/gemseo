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
"""Draw lines from a :class:`.Dataset`.

A :class:`.Lines` plot represents variables vs samples using lines.
"""
from __future__ import annotations

from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


class Lines(DatasetPlot):
    """Plot sampled variables as lines."""

    def __init__(
        self,
        dataset: Dataset,
        variables: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            variables: The names of the variables to plot.
        """
        super().__init__(dataset, variables=variables)

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        x_data = range(len(self.dataset))
        variables = self._param.variables
        if variables is None:
            y_data = self.dataset.get_all_data(False, True)
            variables = y_data.keys()
        else:
            y_data = self.dataset[variables]

        self._set_color(len(variables))
        self._set_linestyle(len(variables), "-")

        fig, axes = self._get_figure_and_axes(fig, axes)
        for index, (name, value) in enumerate(y_data.items()):
            axes.plot(
                x_data,
                value,
                linestyle=self.linestyle[index],
                color=self.color[index],
                label=name,
            )
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)
        axes.legend(loc=self.legend_location)
        return [fig]
