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

"""The settings for `ColorEvolution`."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002
from pydantic import field_validator

from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings


class ColorEvolution_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `ColorEvolution`."""

    variables: tuple[str, ...] = Field(
        default=(),
        description="The variables of interest. If empty, use all the variables.",
    )

    use_log: bool = Field(
        default=False, description="Whether to use a symmetric logarithmic scale."
    )

    opacity: NonNegativeFloat = Field(
        default=0.6,
        description="The level of opacity (0 = transparent; 1 = opaque).",
        le=1.0,
    )

    options: dict[str, bool | float | str | None] = Field(
        default_factory=dict,
        description="The options for the matplotlib function `imshow()`.",
    )

    @field_validator("options", mode="before")
    @classmethod
    def __validate_y(
        cls, options: dict[str, bool | float | str | None]
    ) -> dict[str, bool | float | str | None]:
        return {"interpolation": "nearest", "aspect": "auto"} | options
