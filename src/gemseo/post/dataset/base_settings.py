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
"""Base settings for dataset visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from numpy import linspace
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt

from gemseo.utils.pydantic import BaseSettings

if TYPE_CHECKING:
    from collections.abc import Sequence


class BaseDatasetPlotSettings(BaseSettings, validate_assignment=True):
    """The base settings for dataset visualizations."""

    color: str | tuple = Field(
        default="",
        description="The color. "
        "Either a global one or one per item if `n_items` is non-zero. "
        "If empty, use a default one.",
    )

    colormap: str = Field(default="rainbow", description="""The color map.""")

    fig_size: tuple[PositiveFloat, PositiveFloat] = Field(
        default=(6.4, 4.8), description="The figure size."
    )

    font_size: PositiveInt = Field(default=10, description="The font size.")

    legend_location: str = Field(
        default="best", description="The location of the legend."
    )

    linestyle: str | tuple[str | tuple[int, tuple[int, int, int, int]], ...] = Field(
        default="",
        description="The line style. "
        "Either a global one or one per item if `n_items` is non-zero. "
        "If empty, use a default one.",
    )

    marker: str | tuple[str, ...] = Field(
        default="",
        description="The marker. "
        "Either a global one or one per item if `n_items` is non-zero. "
        "If empty, use a default one.",
    )

    title: str = Field(default="", description="The title of the plot.")

    xtick_rotation: float = Field(
        default=0.0, description="The rotation angle in degrees for the x-ticks."
    )

    zlabel: str = Field(default="", description="The label for the z-axis.")

    zmin: float | None = Field(
        default=None,
        description="The minimum value on the z-axis. If `None`, compute it from data.",
    )

    zmax: float | None = Field(
        default=None,
        description="The maximum value on the z-axis. If `None`, compute it from data.",
    )

    labels: dict[str, str] = Field(
        default_factory=dict, description="The labels for the variables."
    )

    n_items: NonNegativeInt = Field(
        default=0,
        description="The number of items. "
        "The item definition is specific to the plot type "
        "and is used to define properties, "
        "e.g. color and line style, for each item. "
        "For example, items can correspond to curves or series of points. "
        "By default, a graph has no item.",
    )

    grid: bool = Field(default=True, description="Whether to add a grid.")

    def set_colors(self, color: str | list[str]) -> None:
        """Set one color per item if `n_items` is non-zero or a unique one.

        Args:
            color: The color(s).
        """
        self.color = color
        if not self.n_items:
            return

        if not self.color:
            color_map = plt.colormaps[self.colormap]
            self.color = [color_map(c) for c in linspace(0, 1, self.n_items)]

        if isinstance(self.color, str):
            self.color = [self.color] * self.n_items

    def set_linestyles(self, linestyle: str | Sequence[str]) -> None:
        """Set the line style(s) if `n_items` is non-zero or a unique one.

        Args:
            linestyle: The line style(s).
        """
        self.linestyle = linestyle
        if not self.n_items:
            return

        if isinstance(self.linestyle, str):
            self.linestyle = [self.linestyle] * self.n_items

    def set_markers(self, marker: str | Sequence[str]) -> None:
        """Set the marker(s) if `n_items` is non-zero or a unique one.

        Args:
            marker: The marker(s).
        """
        self.marker = marker
        if not self.n_items:
            return

        if isinstance(self.marker, str):
            self.marker = [self.marker] * self.n_items
