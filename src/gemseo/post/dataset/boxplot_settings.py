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

"""The settings for `Boxplot`."""

from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002
from pydantic import field_validator

from gemseo.dataset.dataset import Dataset  # noqa: TC001
from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings

_orientation_options: Final[frozenset[str]] = frozenset({"orientation", "vert"})
"""The boxplot options setting the orientation, which `use_vertical_bars` owns."""

_tick_labels_options: Final[frozenset[str]] = frozenset({"labels"})
"""The boxplot options renamed to `tick_labels` by matplotlib."""


class Boxplot_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `Boxplot`."""

    datasets: tuple[Dataset, ...] = Field(
        default=(), description="The other datasets to plot."
    )

    variables: tuple[str, ...] = Field(
        default=(),
        description="The names of the variables to plot. "
        "If empty, use all the variables.",
    )

    center: bool = Field(
        default=False,
        description="Whether to center the variables so that they have a zero mean.",
    )

    scale: bool = Field(
        default=False,
        description="Whether to scale the variables so that they have a unit variance.",
    )

    use_vertical_bars: bool = Field(
        default=True, description="Whether to use vertical bars."
    )

    add_confidence_interval: bool = Field(
        default=False,
        description="Whether to add the confidence interval around the median.",
    )

    add_outliers: bool = Field(default=True, description="Whether to add the outliers.")

    opacity_level: NonNegativeFloat = Field(
        default=0.25,
        description="The opacity level for the faces, between 0 and 1.",
        le=1.0,
    )

    options: dict = Field(
        default_factory=dict,
        description="The additional keyword arguments passed to "
        "[boxplot][matplotlib.axes.Axes.boxplot]. "
        "The orientation is set by `use_vertical_bars`, "
        "so `orientation` and `vert` are not allowed. "
        "`labels` is not allowed either; use `tick_labels` instead.",
    )

    @field_validator("options")
    @classmethod
    def __validate_options(cls, options: dict) -> dict:
        names = _orientation_options.intersection(options)
        if names:
            msg = (
                f"The boxplot options {sorted(names)} are not supported; "
                "use the setting use_vertical_bars to set the orientation."
            )
            raise ValueError(msg)

        names = _tick_labels_options.intersection(options)
        if names:
            msg = (
                f"The boxplot options {sorted(names)} are not supported; "
                "use the matplotlib option tick_labels instead."
            )
            raise ValueError(msg)

        return options
