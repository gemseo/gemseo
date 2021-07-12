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

from typing import List, Mapping

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.figure import Figure

from gemseo.post.dataset.dataset_plot import DatasetPlot


class ZvsXY(DatasetPlot):
    """Plot surface z versus x,y."""

    def _plot(
        self,
        properties,  # type: Mapping
        x,  # type: str
        y,  # type: str
        z,  # type: str
        x_comp=0,  # type: int
        y_comp=0,  # type: int
        z_comp=0,  # type: int
        add_points=False,  # type: bool
    ):  # type: (...) -> List[Figure]
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
        color = properties.get(self.COLOR) or "blue"
        colormap = properties.get(self.COLORMAP) or "Blues"
        x_data = self.dataset[x][x][:, x_comp]
        y_data = self.dataset[y][y][:, y_comp]
        z_data = self.dataset[z][z][:, z_comp]

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        grid = mtri.Triangulation(x_data, y_data)
        tcf = axes.tricontourf(grid, z_data, cmap=colormap)
        if add_points:
            axes.scatter(x_data, y_data, color=color)
        if self.dataset.sizes[x] == 1:
            axes.set_xlabel(self.xlabel or x)
        else:
            axes.set_xlabel(self.xlabel or "{}({})".format(x, x_comp))
        if self.dataset.sizes[y] == 1:
            axes.set_ylabel(self.ylabel or y)
        else:
            axes.set_ylabel(self.ylabel or "{}({})".format(y, y_comp))
        if self.dataset.sizes[z] == 1:
            axes.set_title(self.zlabel or z)
        else:
            axes.set_title(self.zlabel or "{}({})".format(z, z_comp))
        fig.colorbar(tcf)
        return [fig]
