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
"""Draw surfaces from a :class:`.Dataset`.

A :class:`.Surfaces` plot represents samples
of a functional variable :math:`z(x,y)` discretized over a 2D mesh.
Both evaluations of :math:`z` and mesh are stored in a :class:`.Dataset`,
:math:`z` as a parameter and the mesh as a metadata.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


class Surfaces(DatasetPlot):
    """Plot surfaces y_i over the mesh x."""

    def __init__(
        self,
        dataset: Dataset,
        mesh: str,
        variable: str,
        samples: Sequence[int] | None = None,
        add_points: bool = False,
        fill: bool = True,
        levels: int | Sequence[int] = None,
    ) -> None:
        """
        Args:
            mesh: The name of the dataset metadata corresponding to the mesh.
            variable: The name of the variable for the x-axis.
            samples: The indices of the samples to plot. If None, plot all samples.
            add_points: If True then display the samples over the surface plot.
            fill: Whether to generate a filled contour plot.
            levels: Either the number of contour lines
                or the values of the contour lines in increasing order.
                If ``None``, select them automatically.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            mesh=mesh,
            variable=variable,
            samples=samples,
            add_points=add_points,
            fill=fill,
            levels=levels,
        )

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        mesh = self._param.mesh
        variable = self._param.variable
        samples = self._param.samples
        x_data = self.dataset.metadata[mesh][:, 0]
        y_data = self.dataset.metadata[mesh][:, 1]
        if samples is not None:
            samples = self.dataset[variable][samples, :]
        else:
            samples = self.dataset[variable]

        options = {"cmap": self.colormap}
        levels = self._param.levels
        if levels is not None:
            options["levels"] = levels

        figs = []
        for sample, sample_name in zip(samples, self.dataset.row_names):
            fig = plt.figure(figsize=self.fig_size)
            axes = fig.add_subplot(1, 1, 1)
            triangle = mtri.Triangulation(x_data, y_data)
            if self._param.fill:
                tcf = axes.tricontourf(triangle, sample, **options)
            else:
                tcf = axes.tricontour(triangle, sample, **options)

            if self._param.add_points:
                axes.scatter(x_data, y_data, color=self.color or None)

            axes.set_xlabel(self.xlabel)
            axes.set_ylabel(self.ylabel)
            axes.set_title(f"{self.title or self.zlabel or variable} - {sample_name}")
            fig.colorbar(tcf)
            figs.append(fig)

        return figs
