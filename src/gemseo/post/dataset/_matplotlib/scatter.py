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
"""Scatter based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot
from gemseo.post.dataset._trend import TREND_FUNCTIONS
from gemseo.post.dataset._trend import Trend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


class Scatter(MatplotlibPlot):
    """Scatter based on matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        axes: Axes | None,
        x_values: ArrayLike,
        y_values: ArrayLike,
    ) -> list[Figure]:
        """
        Args:
            x_values: The values of the points on the x-axis.
            y_values: The values of the points on the y-axis.
        """  # noqa: D205, D212, D415
        fig, axes = self._get_figure_and_axes(fig, axes)
        scatter = axes.scatter(x_values, y_values, color=self._common_settings.color)
        scatter.set_zorder(3)
        trend_function_creator = self._specific_settings.trend
        if trend_function_creator != Trend.NONE:
            if not isinstance(trend_function_creator, Callable):
                trend_function_creator = TREND_FUNCTIONS[trend_function_creator]

            indices = x_values[:, 0].argsort()
            x_values = x_values[indices]
            y_values = y_values[indices]
            trend_function = trend_function_creator(x_values[:, 0], y_values[:, 0])
            axes.plot(
                x_values,
                trend_function(x_values),
                color="gray",
                linestyle="--",
            )

        axes.set_xlabel(self._common_settings.xlabel)
        axes.set_ylabel(self._common_settings.ylabel)
        axes.set_title(self._common_settings.title)
        return [fig]
