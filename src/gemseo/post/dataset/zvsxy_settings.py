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

"""The settings for `ZvsXY`."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeInt  # noqa: TC002
from pydantic import field_validator

from gemseo.datasets.dataset import Dataset  # noqa: TC001
from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings


class ZvsXY_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `ZvsXY`."""

    x: str | tuple[str, NonNegativeInt] = Field(
        description="The name of the variable on the x-axis "
        "with its optional component if not `0`, "
        "e.g. `('foo', 3)` for the fourth component of the variable `'foo'`."
    )

    y: str | tuple[str, NonNegativeInt] = Field(
        description="The name of the variable on the y-axis "
        "with its optional component if not `0`, "
        "e.g. `('bar', 3)` for the fourth component of the variable `'bar'`."
    )

    z: str | tuple[str, NonNegativeInt] = Field(
        description="The name of the variable on the z-axis "
        "with its optional component if not `0`, "
        "e.g. `('baz', 3)` for the fourth component of the variable `'baz'`."
    )

    add_points: bool = Field(
        default=False,
        description="Whether to add dataset entries as points above the surface.",
    )

    fill: bool = Field(default=True, description="Whether to fill the contour plot.")

    levels: int | tuple[int, ...] = Field(
        default=(),
        description="Either the number of contour lines "
        "or the values of the contour lines in increasing order. "
        "If empty, select them automatically.",
    )

    other_datasets: tuple[Dataset, ...] = Field(
        default=(),
        description="Additional datasets to be plotted as points above the surface.",
    )

    @field_validator("x", mode="before")
    @classmethod
    def __validate_x(
        cls, x: str | tuple[str, NonNegativeInt]
    ) -> tuple[str, NonNegativeInt]:
        if isinstance(x, str):
            return x, 0
        return x

    @field_validator("y", mode="before")
    @classmethod
    def __validate_y(
        cls, y: str | tuple[str, NonNegativeInt]
    ) -> tuple[str, NonNegativeInt]:
        if isinstance(y, str):
            return y, 0
        return y

    @field_validator("z", mode="before")
    @classmethod
    def __validate_z(
        cls, z: str | tuple[str, NonNegativeInt]
    ) -> tuple[str, NonNegativeInt]:
        if isinstance(z, str):
            return z, 0
        return z
