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
"""Data for a plot."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

from numpy import linspace
from pydantic import BaseModel
from pydantic import Field

from gemseo.utils.compatibility.matplotlib import get_color_map
from gemseo.utils.matplotlib_figure import FigSizeType


class PlotSettings(BaseModel):
    """The settings of a plot."""

    color: str | Sequence[str] = ""
    """The color.

    Either a global one or one per item if ``n_items`` is non-zero.

    If empty, use a default one.
    """

    colormap: str = "rainbow"
    """The color map."""

    fig_size: FigSizeType = (6.4, 4.8)
    """The figure size."""

    font_size: int = 10
    """The font size."""

    legend_location: str = "best"
    """The location of the legend."""

    linestyle: str | Sequence[str] = ""
    """The line style.

    Either a global one or one per item if ``n_items`` is non-zero.

    If empty, use a default one.
    """

    marker: str | Sequence[str] = ""
    """The marker.

    Either a global one or one per item if ``n_items`` is non-zero.
    If empty, use a default one.
    """

    xtick_rotation: float = 0.0
    """The rotation angle in degrees for the x-ticks."""

    title: str = ""
    """The title of the plot."""

    xlabel: str = ""
    """The label for the x-axis."""

    xmin: float | None = None
    """The minimum value on the x-axis.

    If ``None``, compute it from data.
    """

    xmax: float | None = None
    """The maximum value on the x-axis.".

    If ``None``, compute it from data.
    """

    ylabel: str = ""
    """The label for the y-axis."""

    ymin: float | None = None
    """The minimum value on the y-axis.

    If ``None``, compute it from data.
    """

    ymax: float | None = None
    """The maximum value on the y-axis.

    If ``None``, compute it from data.
    """

    zlabel: str = ""
    """The label for the z-axis."""

    zmin: float | None = None
    """The minimum value on the z-axis.

    If ``None``, compute it from data.
    """

    zmax: float | None = None
    """The maximum value on the z-axis.

    If ``None``, compute it from data.
    """

    rmin: float | None = None
    """The minimum value on the r-axis.

    If ``None``, compute it from data.
    """

    rmax: float | None = None
    """The maximum value on the r-axis.

    If ``None``, compute it from data.
    """

    labels: Mapping[str, str] = Field(default_factory=dict)
    """The labels for the variables."""

    n_items: int = 0
    """The number of items.

    The item definition is specific to the plot type and is used to define properties,
    e.g. color and line style, for each item.

    For example, items can correspond to curves or series of points.

    By default, a graph has no item.
    """

    grid: bool = True
    """Whether to add a grid."""

    def set_colors(self, color: str | list[str]) -> None:
        """Set one color per item if ``n_items`` is non-zero or a unique one.

        Args:
            color: The color(s).
        """
        self.color = color
        if not self.n_items:
            return

        if not self.color:
            color_map = get_color_map(self.colormap)
            self.color = [color_map(c) for c in linspace(0, 1, self.n_items)]

        if isinstance(self.color, str):
            self.color = [self.color] * self.n_items

    def set_linestyles(self, linestyle: str | Sequence[str]) -> None:
        """Set the line style(s) if ``n_items`` is non-zero or a unique one.

        Args:
            linestyle: The line style(s).
        """
        self.linestyle = linestyle
        if not self.n_items:
            return

        if isinstance(self.linestyle, str):
            self.linestyle = [self.linestyle] * self.n_items

    def set_markers(self, marker: str | Sequence[str]) -> None:
        """Set the marker(s) if ``n_items`` is non-zero or a unique one.

        Args:
            marker: The marker(s).
        """
        self.marker = marker
        if not self.n_items:
            return

        if isinstance(self.marker, str):
            self.marker = [self.marker] * self.n_items
