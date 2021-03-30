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
r"""
ZvsXY plot
==========

A :class:`.ZvsXY` plot represents the variable :math:`z` with respect to
:math:`x` and :math:`y` as a surface plot, based on a set of points
:points :math:`\{x_i,y_i,z_i\}_{1\leq i \leq n}`. This interpolation is
relies on the Delaunay triangulation of :math:`\{x_i,y_i\}_{1\leq i \leq n}`
"""
from __future__ import absolute_import, division, unicode_literals

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from future import standard_library

from gemseo.post.dataset.dataset_plot import DatasetPlot

standard_library.install_aliases()


class ZvsXY(DatasetPlot):
    """ Plot surface z versus x,y. """

    def _plot(
        self,
        x,
        y,
        z,
        x_comp=0,
        y_comp=0,
        z_comp=0,
        colormap="Blues",
        add_points=False,
        color="blue",
    ):
        """Surface.

        :param x: name of the variable on the x-axis
        :type x: str
        :param y: name of the variable on the y-axis
        :type z: str
        :param z: name of the variable on the z-axis
        :type z: str
        :param x_comp: x component. Default: 0.
        :type x_comp: int
        :param y_comp: y component. Default: 0.
        :type y_comp: int
        :param z_comp: z component. Default: 0.
        :type z_comp: int
        :param colormap: matplotlib colormap. Default: 'Blues'.
        :type colormap: str
        :param add_points: display points over the surface plot.
            Default: False.
        :type add_points: bool
        :param color: point color. Default: 'blue'.
        :type color: str
        """
        x_data = self.dataset[x][x][:, x_comp]
        y_data = self.dataset[y][y][:, y_comp]
        z_data = self.dataset[z][z][:, z_comp]

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        triang = mtri.Triangulation(x_data, y_data)
        tcf = axes.tricontourf(triang, z_data, cmap=colormap)
        if add_points:
            axes.scatter(x_data, y_data, color=color)
        if self.dataset.sizes[x] == 1:
            axes.set_xlabel(x)
        else:
            axes.set_xlabel(x + "(" + str(x_comp) + ")")
        if self.dataset.sizes[y] == 1:
            axes.set_ylabel(y)
        else:
            axes.set_ylabel(y + "(" + str(y_comp) + ")")
        if self.dataset.sizes[z] == 1:
            axes.set_title(z)
        else:
            axes.set_title(z + "(" + str(z_comp) + ")")
        fig.colorbar(tcf)
        fig = plt.gcf()
        return fig
