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
"""Lines based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.ticker import MaxNLocator

from gemseo.post.dataset.lines_settings import Lines_Settings
from gemseo.post.dataset.plots._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


class Lines(MatplotlibPlot[Lines_Settings]):
    """Lines based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
        x_values: ArrayLike,
        y_name_to_value: Mapping[str, ArrayLike],
        default_xlabel: str,
        n_lines: int,
    ) -> list[Figure]:
        """
        Args:
            x_values: The values on the x-axis.
            y_name_to_value: The variable names bound to the values on the y-axis.
            default_xlabel: The default x-label.
            n_lines: The number of lines.
        """  # noqa: D205 D212 D415
        settings = self._settings
        fig, ax = self._get_figure_and_axes(fig, ax, fig_size=settings.fig_size)
        settings.set_colors(settings.color)
        settings.set_linestyles(settings.linestyle or "-")
        settings.set_markers(settings.marker or "o")
        line_index = -1
        for y_name, y_values in y_name_to_value.items():
            for yi_name, yi_values in zip(
                self._common_dataset.get_columns(y_name), y_values, strict=False
            ):
                line_index += 1
                linestyle = settings.linestyle[line_index]
                color = settings.color[line_index]
                ax.plot(
                    x_values, yi_values, linestyle=linestyle, color=color, label=yi_name
                )
                if settings.add_markers:
                    ax.scatter(
                        x_values,
                        yi_values,
                        color=color,
                        marker=settings.marker[line_index],
                    )

        ax.grid(visible=settings.grid)
        ax.set_xlabel(settings.xlabel or default_xlabel)
        ax.set_ylabel(settings.ylabel)
        ax.set_title(settings.title)
        ax.legend(loc=settings.legend_location)
        if settings.set_xticks_from_data:
            ax.set_xticks(x_values)

        if settings.use_integer_xticks:
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        return [fig]
