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

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002

from gemseo.datasets.dataset import Dataset  # noqa: TC001
from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings


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
        default_factory=dict, description="The options of the wrapped boxplot function."
    )
