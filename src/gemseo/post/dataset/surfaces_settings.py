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

"""The settings for `Surfaces`."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002

from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings


class Surfaces_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `Surfaces`."""

    mesh: str = Field(
        description="The name of the dataset metadata corresponding to the mesh."
    )
    variable: str = Field(description="The name of the variable for the x-axis.")

    samples: tuple[int, ...] = Field(
        default=(),
        description="The indices of the samples to plot. If empty, plot all samples.",
    )

    add_points: bool = Field(
        default=False, description="Whether to display samples over the surface plot."
    )

    fill: bool = Field(
        default=True, description="Whether to generate a filled contour plot."
    )

    levels: PositiveInt | tuple[float, ...] = Field(
        default=(),
        description="Either the number of contour lines "
        "or the values of the contour lines in increasing order. "
        "If empty, select them automatically.",
    )
