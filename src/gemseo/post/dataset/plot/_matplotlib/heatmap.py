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
"""A heat map of a dataset, either rectangular or symmetric, using matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.colors import SymLogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import LogFormatterSciNotation
from numpy import arange
from numpy import isnan

from gemseo.post.dataset.heatmap_settings import Heatmap_Settings
from gemseo.post.dataset.plot._heatmap_utils import compute_centered_bounds
from gemseo.post.dataset.plot._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gemseo.util.typing import RealArray


class Heatmap(MatplotlibPlot[Heatmap_Settings]):
    """A heat map of a dataset, either rectangular or symmetric, using matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
        data: RealArray,
        variable_names: tuple[str, ...],
    ) -> list[Figure]:
        """
        Args:
            data: The data to be plotted.
            variable_names: The names of the variables.
        """  # noqa: D205, D212, D415
        settings = self._settings
        fig, ax = self._get_figure_and_axes(fig, ax)
        norm = None
        if settings.use_log:
            maximum = abs(data).max()
            norm = SymLogNorm(vmin=-maximum, vmax=maximum, linthresh=1.0)
        elif settings.symmetric and settings.center is not None:
            bounds = compute_centered_bounds(data, settings.center)
            if bounds is not None:
                norm = TwoSlopeNorm(
                    vcenter=settings.center, vmin=bounds[0], vmax=bounds[1]
                )

        img = ax.imshow(
            data,
            cmap=settings.colormap,
            norm=norm,
            alpha=settings.opacity,
            **settings.matplotlib_options,
        )

        if settings.symmetric:
            ticks = arange(len(variable_names))
            ax.set_xticks(ticks)
            ax.set_xticklabels(variable_names)
            ax.set_yticks(ticks)
            ax.set_yticklabels(variable_names)
        else:
            ax.set_xticks(arange(data.shape[1]))
            ax.set_xticklabels([str(value) for value in self._common_dataset.index])
            ax.set_yticks(arange(len(variable_names)))
            ax.set_yticklabels(variable_names)

        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)
        ax.set_title(settings.title)

        if settings.annotate:
            for i, row in enumerate(data):
                for j, value in enumerate(row):
                    ax.text(
                        j,
                        i,
                        "NaN" if isnan(value) else format(value, settings.annotate_fmt),
                        ha="center",
                        va="center",
                    )

        fig.colorbar(
            img,
            ax=ax,
            format=LogFormatterSciNotation() if settings.use_log else None,
        )
        return [fig]
