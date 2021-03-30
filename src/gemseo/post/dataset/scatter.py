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
"""
Scatter plot
============

A :class:`Scatter` plot represents a set of points
:math:`\{x_i,y_i\}_{1\leq i \leq n}` as markers on a classical plot
where the color of points can be heterogeneous.
"""
from __future__ import absolute_import, division, unicode_literals

import matplotlib.pyplot as plt
from future import standard_library

from gemseo.post.dataset.dataset_plot import DatasetPlot

standard_library.install_aliases()


class Scatter(DatasetPlot):
    """ Plot curve y versus x. """

    def _plot(self, x, y, x_comp=0, y_comp=0, color="blue"):
        """Surface.

        :param x: name of the variable on the x-axis
        :type x: str
        :param y: name of the variable on the y-axis
        :type z: str
        :param x_comp: x component. Default: 0.
        :type x_comp: int
        :param y_comp: y component. Default: 0.
        :type y_comp: int
        :param color: point color name. Possibly a list of color names
           whose length is equal to the number of points. Default: 'blue'.
        :type color: str or list(str)
        """
        x_data = self.dataset[x][x][:, x_comp]
        y_data = self.dataset[y][y][:, y_comp]

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.scatter(x_data, y_data, color=color)
        if self.dataset.sizes[x] == 1:
            axes.set_xlabel(x)
        else:
            axes.set_xlabel(x + "(" + str(x_comp) + ")")
        if self.dataset.sizes[y] == 1:
            axes.set_ylabel(y)
        else:
            axes.set_ylabel(y + "(" + str(y_comp) + ")")
        fig = plt.gcf()
        return fig
