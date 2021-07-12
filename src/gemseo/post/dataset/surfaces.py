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
"""Draw surfaces from a :class:`.Dataset`.

A :class:`.Surfaces` plot represents samples
of a functional variable :math:`z(x,y)` discretized over a 2D mesh.
Both evaluations of :math:`z` and mesh are stored in a :class:`.Dataset`,
:math:`z` as a parameter and the mesh as a metadata.
"""
from __future__ import division, unicode_literals

from typing import List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.figure import Figure

from gemseo.post.dataset.dataset_plot import DatasetPlot


class Surfaces(DatasetPlot):
    """Plot surfaces y_i over the mesh x."""

    def _plot(
        self,
        properties,  # type: Mapping
        mesh,  # type: str
        variable,  # type: str
        samples=None,  # type:Optional[Sequence[int]]
        add_points=False,  # type: bool
    ):  # type: (...) -> List[Figure]
        """
        Args:
            mesh: The name of the dataset metadata corresponding to the mesh.
            variable: The name of the variable for the x-axis.
            samples: The indices of the samples to plot. If None, plot all samples.
            add_points: If True then display the samples over the surface plot.
        """
        color = properties.get(self.COLOR) or "blue"
        colormap = properties.get(self.COLORMAP) or "Blues"
        x_data = self.dataset.metadata[mesh][:, 0]
        y_data = self.dataset.metadata[mesh][:, 1]
        if samples is not None:
            outputs = self.dataset[variable][variable][samples, :]
        else:
            outputs = self.dataset[variable][variable]

        sample = 0
        fig = []
        for z_data, variable_component in zip(outputs, self.dataset.row_names):
            fig.append(plt.figure())
            axes = fig[-1].add_subplot(1, 1, 1)
            triangle = mtri.Triangulation(x_data, y_data)
            tcf = axes.tricontourf(triangle, z_data, cmap=colormap)
            if add_points:
                axes.scatter(x_data, y_data, color=color)
            axes.set_title("{} - {}".format(variable, variable_component))
            fig[-1].colorbar(tcf)
            fig[-1] = plt.gcf()
            sample += 1
        return fig
