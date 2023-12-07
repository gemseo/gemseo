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
"""Visualize a variable versus another using matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


class YvsX(MatplotlibPlot):
    """Visualize a variable versus another using matplotlib."""

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
        axes.plot(
            x_values,
            y_values,
            self._common_settings.linestyle,
            color=self._common_settings.color,
        )
        axes.set_xlabel(self._common_settings.xlabel)
        axes.set_ylabel(self._common_settings.ylabel)
        axes.set_title(self._common_settings.title)
        return [fig]
