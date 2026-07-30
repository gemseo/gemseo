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

"""The settings for `BarPlot`."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002

from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings


class BarPlot_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `BarPlot`."""

    n_digits: PositiveInt = Field(
        default=1, description="The number of digits to print the different bar values."
    )

    annotate: bool = Field(
        default=True,
        description="Whether to add annotations of the height value on each bar.",
    )

    annotation_rotation: float = Field(
        default=0.0, description="The angle by which annotations are rotated."
    )
