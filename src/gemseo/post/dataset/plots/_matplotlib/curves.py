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
"""Curves based on maptlotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.curves_settings import Curves_Settings
from gemseo.post.dataset.plots._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


class Curves(MatplotlibPlot[Curves_Settings]):
    """Curves based on maptlotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
        y_values: ArrayLike,
        labels: list[str],
    ) -> list[Figure]:
        """
        Args:
            y_values: The values of the points of the curves on the y-axis
                (one curve per row).
            labels: The labels of the curves.
        """  # noqa: D205 D212 D415
        settings = self._settings
        fig, ax = self._get_figure_and_axes(fig, ax)
        settings.set_colors(settings.color)
        settings.set_linestyles(
            settings.linestyle
            or ["-"] + [(0, (i, 1, 1, 1)) for i in range(1, settings.n_items)]
        )
        mesh_name = settings.mesh
        mesh = self._common_dataset.misc[mesh_name]
        for sub_y_values, line_style, color, label in zip(
            y_values,
            settings.linestyle,
            settings.color,
            labels,
            strict=False,
        ):
            ax.plot(mesh, sub_y_values, linestyle=line_style, color=color, label=label)

        ax.grid(visible=settings.grid)
        ax.set_xlabel(settings.xlabel or mesh_name)
        ax.set_ylabel(settings.ylabel or f"{settings.variable}({mesh_name})")
        ax.set_title(settings.title)
        ax.legend(loc=settings.legend_location)
        return [fig]
