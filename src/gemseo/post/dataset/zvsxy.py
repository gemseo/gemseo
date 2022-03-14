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
r"""Draw a variable versus two others from a :class:`.Dataset`.

A :class:`.ZvsXY` plot represents the variable :math:`z` with respect to
:math:`x` and :math:`y` as a surface plot, based on a set of points
:points :math:`\{x_i,y_i,z_i\}_{1\leq i \leq n}`. This interpolation is
relies on the Delaunay triangulation of :math:`\{x_i,y_i\}_{1\leq i \leq n}`
"""
from __future__ import division, unicode_literals

from typing import List

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.figure import Figure

from gemseo.post.dataset.dataset_plot import DatasetPlot, DatasetPlotPropertyType


class ZvsXY(DatasetPlot):
    """Plot surface z versus x,y."""

    def __init__(
        self,
        dataset,  # Dataset
        x,  # type: str
        y,  # type: str
        z,  # type: str
        x_comp=0,  # type: int
        y_comp=0,  # type: int
        z_comp=0,  # type: int
        add_points=False,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            x: The name of the variable on the x-axis.
            y: The name of the variable on the y-axis.
            z: The name of the variable on the z-axis.
            x_comp: The component of x.
            y_comp: The component of y.
            z_comp: The component of z.
            add_points: If True, display samples over the surface plot.
        """
        super().__init__(
            dataset=dataset,
            x=x,
            y=y,
            z=z,
            x_comp=x_comp,
            y_comp=y_comp,
            z_comp=z_comp,
            add_points=add_points,
        )

    def _plot(
        self,
        **properties,  # type: DatasetPlotPropertyType
    ):  # type: (...) -> List[Figure]
        color = properties.get(self.COLOR) or "blue"
        colormap = properties.get(self.COLORMAP) or "Blues"
        x_data = self.dataset[self._param.x][self._param.x][:, self._param.x_comp]
        y_data = self.dataset[self._param.y][self._param.y][:, self._param.y_comp]
        z_data = self.dataset[self._param.z][self._param.z][:, self._param.z_comp]

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        grid = mtri.Triangulation(x_data, y_data)
        tcf = axes.tricontourf(grid, z_data, cmap=colormap)
        if self._param.add_points:
            axes.scatter(x_data, y_data, color=color)
        if self.dataset.sizes[self._param.x] == 1:
            axes.set_xlabel(self.xlabel or self._param.x)
        else:
            axes.set_xlabel(
                self.xlabel or "{}({})".format(self._param.x, self._param.x_comp)
            )
        if self.dataset.sizes[self._param.y] == 1:
            axes.set_ylabel(self.ylabel or self._param.y)
        else:
            axes.set_ylabel(
                self.ylabel or "{}({})".format(self._param.y, self._param.y_comp)
            )
        if self.dataset.sizes[self._param.z] == 1:
            axes.set_title(self.zlabel or self._param.z)
        else:
            axes.set_title(
                self.zlabel or "{}({})".format(self._param.z, self._param.z_comp)
            )
        fig.colorbar(tcf)
        return [fig]
