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

"""The settings for `Lines`."""

from __future__ import annotations

from pydantic import Field

from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings


class Lines_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `Lines`."""

    variables: tuple[str, ...] = Field(
        default=(),
        description="The names of the variables to plot. "
        "If empty, use all the variables.",
    )

    abscissa_variable: str = Field(
        default="",
        description="The name of the variable used in abscissa. "
        "The observations of the `variables` are plotted "
        "as a function of the observations of this `abscissa_variable`. "
        "If empty, "
        "the observations of the `variables` are plotted "
        "as a function of the indices of the observations.",
    )

    add_markers: bool = Field(
        default=False, description="Whether to mark the observations with dots."
    )

    set_xticks_from_data: bool = Field(
        default=False,
        description="Whether to use the values of `abscissa_variable` "
        "as locations of abscissa ticks.",
    )

    use_integer_xticks: bool = Field(
        default=False, description="Whether to use integer xticks."
    )

    plot_abscissa_variable: bool = Field(
        default=False, description="Whether to plot the abscissa variable."
    )
