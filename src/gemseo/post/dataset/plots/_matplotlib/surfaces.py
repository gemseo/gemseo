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

from gemseo.post.dataset.plots._matplotlib.plot import MatplotlibPlot
from gemseo.post.dataset.surfaces_settings import Surfaces_Settings

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Surfaces(MatplotlibPlot[Surfaces_Settings]):
    """Surfaces based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
    ) -> list[Figure]:
        settings = self._settings
        mesh = settings.mesh
        variable = settings.variable
        samples = settings.samples
        x_data = self._common_dataset.misc[mesh][:, 0]
        y_data = self._common_dataset.misc[mesh][:, 1]
        data = self._common_dataset.get_view(variable_names=variable).to_numpy()

        samples = data[samples, :] if samples else data

        options = {"cmap": settings.colormap}
        levels = settings.levels
        if levels:
            options["levels"] = levels

        figs = []
        for sample, sample_name in zip(
            samples, self._common_dataset.index, strict=False
        ):
            fig = plt.figure(figsize=settings.fig_size)
            ax = fig.add_subplot(1, 1, 1)
            func = ax.tricontourf if settings.fill else ax.tricontour
            tcf = func(mtri.Triangulation(x_data, y_data), sample, **options)
            if settings.add_points:
                ax.scatter(x_data, y_data, color=settings.color or None)

            ax.set_xlabel(settings.xlabel)
            ax.set_ylabel(settings.ylabel)
            main_title = settings.title or settings.zlabel or variable
            ax.set_title(f"{main_title} - {sample_name}")
            fig.colorbar(tcf)
            figs.append(fig)

        return figs
