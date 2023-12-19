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

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemseo.utils.matplotlib_figure import FigSizeType


@dataclass
class PlotSettings:
    """The settings of a plot."""

    color: str | list[str] = ""
    """The color(s) for the series.

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

    linestyle: str | list[str] = ""
    """The line style(s) for the series.

    If empty, use a default one.
    """

    marker: str | list[str] = ""
    """The marker(s) for the series.

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

    labels: dict[str, str] = field(default_factory=dict)
    """The labels for the variables."""
