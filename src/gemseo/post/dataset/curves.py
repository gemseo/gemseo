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
from __future__ import annotations

from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


class Curves(DatasetPlot):
    """Plot curves y_i over the mesh x."""

    def __init__(
        self,
        dataset: Dataset,
        mesh: str,
        variable: str,
        samples: Sequence[int] | None = None,
    ) -> None:
        """
        Args:
            mesh: The name of the dataset metadata corresponding to the mesh.
            variable: The name of the variable for the x-axis.
            samples: The indices of the samples to plot.
                If None, plot all the samples.
        """  # noqa: D205, D212, D415
        super().__init__(dataset, mesh=mesh, variable=variable, samples=samples)

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        def lines_gen():
            """Linestyle generator."""
            yield "-"
            for i in range(1, self.dataset.n_samples):
                yield 0, (i, 1, 1, 1)

        variable = self._param.variable
        samples = self._param.samples
        if samples is not None:
            output = self.dataset[variable][samples, :].T
        else:
            output = self.dataset[variable].T
            samples = range(output.shape[1])
        n_samples = output.shape[1]

        self._set_color(n_samples)
        self._set_linestyle(n_samples, [line for line in lines_gen()])

        data = (output.T, self.linestyle, self.color, samples)
        mesh = self._param.mesh

        fig, axes = self._get_figure_and_axes(fig, axes)
        for output, line_style, color, sample in zip(*data):
            axes.plot(
                self.dataset.metadata[mesh],
                output,
                linestyle=line_style,
                color=color,
                label=self.dataset.row_names[sample],
            )
        axes.set_xlabel(self.xlabel or mesh)
        axes.set_ylabel(self.ylabel or f"{variable}({mesh})")
        axes.set_title(self.title)
        axes.legend(loc=self.legend_location)
        return [fig]
