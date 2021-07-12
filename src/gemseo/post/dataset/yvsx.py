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
"""Draw a variable versus another from a :class:`.Dataset`.

A :class:`.YvsX` plot represents samples of a couple :math:`(x,y)` as a set of points
whose values are stored in a :class:`.Dataset`. The user can select the style of line or
markers, as well as the color.
"""
from __future__ import division, unicode_literals

from typing import List, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gemseo.post.dataset.dataset_plot import DatasetPlot


class YvsX(DatasetPlot):
    """Plot curve y versus x."""

    def _plot(
        self,
        properties,  # type: Mapping
        x,  # type: str
        y,  # type: str
        x_comp=0,  # type: int
        y_comp=0,  # type: int
    ):  # type: (...) -> List[Figure]
        """
        Args:
            x: The name of the variable on the x-axis.
            y: The name of the variable on the y-axis.
            x_comp: The component of x.
            y_comp: The component of y.
        """
        color = properties.get(self.COLOR) or "blue"
        style = properties.get(self.LINESTYLE) or "o"
        x_data = self.dataset[x][x][:, x_comp]
        y_data = self.dataset[y][y][:, y_comp]

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(x_data, y_data, style, color=color)
        if self.dataset.sizes[x] == 1:
            axes.set_xlabel(self.xlabel or x)
        else:
            axes.set_xlabel(self.xlabel or "{}({})".format(x, x_comp))
        if self.dataset.sizes[y] == 1:
            axes.set_ylabel(self.ylabel or y)
        else:
            axes.set_ylabel("{}({})".format(y, y_comp))
        return [fig]
