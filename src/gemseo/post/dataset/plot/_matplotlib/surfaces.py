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
from numpy import isfinite

from gemseo.post.dataset.plot._matplotlib.plot import MatplotlibPlot
from gemseo.post.dataset.surfaces_settings import Surfaces_Settings

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Surfaces(MatplotlibPlot[Surfaces_Settings]):
    """Surfaces based on matplotlib.

    The triangles of the mesh with a non-finite value at a vertex are left out,
    which shows as a gap;
    when every triangle is left out,
    the figure has neither surface nor colorbar.
    """

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

        sample_names = self._common_dataset.index
        if samples:
            samples = list(samples)
            data = data[samples, :]
            sample_names = sample_names[samples]

        options = {"cmap": settings.colormap}
        levels = settings.levels
        if levels:
            options["levels"] = levels

        figs = []
        triangulation = mtri.Triangulation(x_data, y_data)
        for sample, sample_name in zip(data, sample_names, strict=True):
            fig = plt.figure(figsize=settings.fig_size)
            ax = fig.add_subplot(1, 1, 1)
            func = ax.tricontourf if settings.fill else ax.tricontour
            # matplotlib rejects non-finite values in the triangulation;
            # masking the triangles with such a vertex leaves a gap.
            # The mask replaces the one of the previous sample.
            mask = (~isfinite(sample))[triangulation.triangles].any(axis=1)
            triangulation.set_mask(mask)
            is_drawable = not mask.all()
            if is_drawable:
                tcf = func(triangulation, sample, **options)

            if settings.add_points:
                ax.scatter(x_data, y_data, color=settings.color or None)

            ax.set_xlabel(settings.xlabel)
            ax.set_ylabel(settings.ylabel)
            main_title = settings.title or settings.zlabel or variable
            ax.set_title(f"{main_title} - {sample_name}")
            if is_drawable:
                fig.colorbar(tcf)

            figs.append(fig)

        return figs
