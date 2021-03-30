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
Surface plot
============

A :class:`.Surfaces` plot represents samples of a functional variable
:math:`z(x,y)` discretized over a 2D mesh. Both evaluations of :math:`z`
and mesh are stored in a :class:`.Dataset`, :math:`z` as a parameter
and the mesh as a metadata.
"""
from __future__ import absolute_import, division, unicode_literals

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from future import standard_library

from gemseo.post.dataset.dataset_plot import DatasetPlot

standard_library.install_aliases()


class Surfaces(DatasetPlot):
    """ Plot surfaces y_i over the mesh x. """

    def _plot(
        self,
        mesh,
        variable,
        samples=None,
        colormap="Blues",
        add_points=False,
        color="blue",
    ):
        """Curve.

        :param mesh: name of the mesh stored in Dataset.metadata.
        :type mesh: str
        :param variable: variable name for the x-axis.
        :type variable: float
        :param samples: samples indices. If None, plot all samples.
            Default: None.
        :type samples: list(int)
        :param colormap: colormap. Default: 'Blues'.
        :type color: str
        :param add_points: display points over the surface plot.
            Default: False.
        :type add_points: bool
        :param color: point color. Default: blue.
        :type color: str
        """
        x_data = self.dataset.metadata[mesh][:, 0]
        y_data = self.dataset.metadata[mesh][:, 1]
        if samples is not None:
            outputs = self.dataset[variable][variable][samples, :]
        else:
            outputs = self.dataset[variable][variable]

        sample = 0
        fig = []
        for z_data in outputs:
            fig.append(plt.figure())
            axes = fig[-1].add_subplot(1, 1, 1)
            triang = mtri.Triangulation(x_data, y_data)
            tcf = axes.tricontourf(triang, z_data, cmap=colormap)
            if add_points:
                axes.scatter(x_data, y_data, color=color)
            axes.set_title(variable)
            fig[-1].colorbar(tcf)
            fig[-1] = plt.gcf()
            sample += 1
        return fig
