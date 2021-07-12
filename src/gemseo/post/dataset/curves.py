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
"""Draw curves from a :class:`.Dataset`.

A :class:`.Curves` plot represents samples of a functional variable
:math:`y(x)` discretized over a 1D mesh. Both evaluations of :math:`y`
and mesh are stored in a :class:`.Dataset`, :math:`y` as a parameter
and the mesh as a metadata.
"""
from __future__ import division, unicode_literals

from typing import List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gemseo.post.dataset.dataset_plot import DatasetPlot


class Curves(DatasetPlot):
    """Plot curves y_i over the mesh x."""

    def _plot(
        self,
        properties,  # type: Mapping
        mesh,  # type: str
        variable,  # type: str
        samples=None,  # type: Optional[Sequence[int]]
    ):  # type: (...) -> List[Figure]
        """
        Args:
            mesh: The name of the dataset metadata corresponding to the mesh.
            variable: The name of the variable for the x-axis.
            samples: The indices of the samples to plot.
                If None, plot all the samples.
        """

        def lines_gen():
            """Linestyle generator."""
            yield "-"
            for i in range(1, self.dataset.n_samples):
                yield 0, (i, 1, 1, 1)

        if samples is not None:
            output = self.dataset[variable][variable][samples, :].T
        else:
            output = self.dataset[variable][variable].T
            samples = range(output.shape[1])
        n_samples = output.shape[1]

        self._set_color(properties, n_samples)
        self._set_linestyle(properties, n_samples, [line for line in lines_gen()])

        data = (output.T, self.linestyle, self.color, samples)
        for output, line_style, color, sample in zip(*data):
            plt.plot(
                self.dataset.metadata[mesh],
                output,
                linestyle=line_style,
                color=color,
                label=self.dataset.row_names[sample],
            )
        plt.xlabel(self.xlabel or mesh)
        plt.ylabel(self.ylabel or "{}({})".format(variable, mesh))
        plt.legend(loc=self.legend_location)
        fig = plt.gcf()
        fig.set_size_inches(*self.figsize)
        return [fig]
