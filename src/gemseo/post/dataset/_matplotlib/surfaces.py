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
"""Surfaces based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Surfaces(MatplotlibPlot):
    """Surfaces based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        axes: Axes | None,
    ) -> list[Figure]:
        mesh = self._specific_settings.mesh
        variable = self._specific_settings.variable
        samples = self._specific_settings.samples
        x_data = self._common_dataset.misc[mesh][:, 0]
        y_data = self._common_dataset.misc[mesh][:, 1]
        data = self._common_dataset.get_view(variable_names=variable).to_numpy()

        samples = data[samples, :] if samples is not None else data

        options = {"cmap": self._common_settings.colormap}
        levels = self._specific_settings.levels
        if levels is not None:
            options["levels"] = levels

        figs = []
        for sample, sample_name in zip(samples, self._common_dataset.index):
            fig = plt.figure(figsize=self._common_settings.fig_size)
            axes = fig.add_subplot(1, 1, 1)
            func = axes.tricontourf if self._specific_settings.fill else axes.tricontour
            tcf = func(mtri.Triangulation(x_data, y_data), sample, **options)
            if self._specific_settings.add_points:
                axes.scatter(x_data, y_data, color=self._common_settings.color or None)

            axes.set_xlabel(self._common_settings.xlabel)
            axes.set_ylabel(self._common_settings.ylabel)
            main_title = (
                self._common_settings.title or self._common_settings.zlabel or variable
            )
            axes.set_title(f"{main_title} - {sample_name}")
            fig.colorbar(tcf)
            figs.append(fig)

        return figs
