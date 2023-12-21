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
"""Visualize a variable versus two others using matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.tri as mtri

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from gemseo.datasets.dataset import Dataset


class ZvsXY(MatplotlibPlot):
    """Visualize a variable versus two others using matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        axes: Axes | None,
        x_values: ArrayLike,
        y_values: ArrayLike,
        z_values: ArrayLike,
        other_datasets: list[Dataset],
    ) -> list[Figure]:
        """
        Args:
            x_values: The values of the points on the x-axis.
            y_values: The values of the points on the y-axis.
            z_values: The values of the points on the z-axis.
            other_datasets: Other datasets.
        """  # noqa: D205, D212, D415
        fig, axes = self._get_figure_and_axes(fig, axes)
        grid = mtri.Triangulation(x_values, y_values)
        options = {"cmap": self._common_settings.colormap}
        if self._specific_settings.levels is not None:
            options["levels"] = self._specific_settings.levels

        plot_contour = (
            axes.tricontourf if self._specific_settings.fill else axes.tricontour
        )
        tcf = plot_contour(grid, z_values, **options)
        if self._specific_settings.add_points:
            axes.scatter(x_values, y_values, color=self._common_settings.color[0])

        if other_datasets:
            x, x_comp = self._specific_settings.x
            y, y_comp = self._specific_settings.y
            for index, dataset in enumerate(other_datasets):
                x_data = dataset.get_view(
                    variable_names=x, components=x_comp
                ).to_numpy()
                y_data = dataset.get_view(
                    variable_names=y, components=y_comp
                ).to_numpy()
                axes.scatter(
                    x_data, y_data, color=self._common_settings.color[index + 1]
                )

        axes.set_xlabel(self._common_settings.xlabel)
        axes.set_ylabel(self._common_settings.ylabel)
        axes.set_title(self._common_settings.title)
        fig.colorbar(tcf, ax=axes)
        return [fig]
