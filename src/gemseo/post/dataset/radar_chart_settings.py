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

"""The settings for `RadarChart`."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002

from gemseo.post.dataset.base_polar_settings import BasePolarDatasetPlotSettings


class RadarChart_Settings(BasePolarDatasetPlotSettings):  # noqa: N801
    """The settings for `RadarChart`."""

    display_zero: bool = Field(
        default=True,
        description="Whether to display the line where the output is equal to zero.",
    )

    connect: bool = Field(
        default=False,
        description="Whether to connect the elements of a series with a line.",
    )

    radial_ticks: bool = Field(
        default=False, description="Whether to align the ticks names with the radius."
    )

    n_levels: PositiveInt = Field(default=6, description="The number of grid levels.")

    scientific_notation: bool = Field(
        default=True,
        description="Whether to format the grid levels with scientific notation.",
    )
