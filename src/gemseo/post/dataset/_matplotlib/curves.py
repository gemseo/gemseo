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

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


class Curves(MatplotlibPlot):
    """Curves based on maptlotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        axes: Axes | None,
        y_values: ArrayLike,
        labels: Iterable[str],
    ) -> list[Figure]:
        """
        Args:
            y_values: The values of the points of the curves on the y-axis
                (one curve per row).
            labels: The labels of the curves.
        """  # noqa: D205 D212 D415
        fig, axes = self._get_figure_and_axes(fig, axes)
        mesh_name = self._specific_settings.mesh
        mesh = self._common_dataset.misc[mesh_name]
        for sub_y_values, line_style, color, label in zip(
            y_values,
            self._common_settings.linestyle,
            self._common_settings.color,
            labels,
        ):
            axes.plot(
                mesh, sub_y_values, linestyle=line_style, color=color, label=label
            )
        axes.set_xlabel(self._common_settings.xlabel or mesh_name)
        axes.set_ylabel(
            self._common_settings.ylabel
            or f"{self._specific_settings.variable}({mesh_name})"
        )
        axes.set_title(self._common_settings.title)
        axes.legend(loc=self._common_settings.legend_location)
        return [fig]
