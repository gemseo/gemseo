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
r"""Draw a variable versus two others from a :class:`.Dataset`.

A :class:`.ZvsXY` plot represents the variable :math:`z` with respect to
:math:`x` and :math:`y` as a surface plot, based on a set of points
:points :math:`\{x_i,y_i,z_i\}_{1\leq i \leq n}`. This interpolation is
relies on the Delaunay triangulation of :math:`\{x_i,y_i\}_{1\leq i \leq n}`
"""
from __future__ import annotations

from typing import Iterable
from typing import Sequence

import matplotlib.tri as mtri
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.post.dataset.dataset_plot import VariableType


class ZvsXY(DatasetPlot):
    """Plot surface z versus x,y."""

    def __init__(
        self,
        dataset: Dataset,
        x: VariableType,
        y: VariableType,
        z: VariableType,
        add_points: bool = False,
        fill: bool = True,
        levels: int | Sequence[int] | None = None,
        other_datasets: Iterable[Dataset] | None = None,
    ) -> None:
        """
        Args:
            x: The name of the variable on the x-axis,
                with its optional component if not ``0``,
                e.g. ``("foo", 3)`` for the fourth component of the variable ``"foo"``.
            y: The name of the variable on the y-axis,
                with its optional component if not ``0``,
                e.g. ``("bar", 3)`` for the fourth component of the variable ``"bar"``.
            z: The name of the variable on the z-axis,
                with its optional component if not ``0``,
                e.g. ``("baz", 3)`` for the fourth component of the variable ``"baz"``.
            add_points: Whether to display the entries of the dataset as points
                above the surface.
            fill: Whether to generate a filled contour plot.
            levels: Either the number of contour lines
                or the values of the contour lines in increasing order.
                If ``None``, select them automatically.
            other_datasets: Additional datasets to be plotted as points
                above the surface.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset=dataset,
            x=self._force_variable_to_tuple(x),
            y=self._force_variable_to_tuple(y),
            z=self._force_variable_to_tuple(z),
            add_points=add_points,
            other_datasets=other_datasets,
            fill=fill,
            levels=levels,
        )

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        other_datasets = self._param.other_datasets
        x, x_comp = self._param.x
        y, y_comp = self._param.y
        z, z_comp = self._param.z

        n_series = 1
        if other_datasets:
            n_series += len(other_datasets)

        self._set_color(n_series)

        x_data = self.dataset.get_view(variable_names=x).to_numpy()[:, x_comp]
        y_data = self.dataset.get_view(variable_names=y).to_numpy()[:, y_comp]
        z_data = self.dataset.get_view(variable_names=z).to_numpy()[:, z_comp]

        fig, axes = self._get_figure_and_axes(fig, axes)

        grid = mtri.Triangulation(x_data, y_data)

        levels = self._param.levels
        options = {"cmap": self.colormap}

        if levels is not None:
            options["levels"] = levels

        if self._param.fill:
            tcf = axes.tricontourf(grid, z_data, **options)
        else:
            tcf = axes.tricontour(grid, z_data, **options)

        if self._param.add_points:
            axes.scatter(x_data, y_data, color=self.color[0])

        if other_datasets:
            for index, dataset in enumerate(other_datasets):
                x_data = dataset.get_view(variable_names=x).to_numpy()[:, x_comp]
                y_data = dataset.get_view(variable_names=y).to_numpy()[:, y_comp]
                axes.scatter(x_data, y_data, color=self.color[index + 1])

        if not self.xlabel:
            self.xlabel = self._get_component_name(
                x, x_comp, self.dataset.variable_names_to_n_components
            )

        if not self.ylabel:
            self.ylabel = self._get_component_name(
                y, y_comp, self.dataset.variable_names_to_n_components
            )

        if not self.zlabel:
            self.zlabel = self._get_component_name(
                z, z_comp, self.dataset.variable_names_to_n_components
            )

        if not self.title:
            self.title = self.zlabel

        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)

        fig.colorbar(tcf, ax=axes)
        return [fig]
